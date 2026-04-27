"""TDD tests for fetch_race_card() cache invalidation logic.

Covers:
1. >50% frame_number=0 in non-scratched entries → cache invalidated, re-scrape triggered
2. Valid frame numbers → cache used, no scrape
3. force_refresh=True → cache bypassed entirely
4. Scratched entries excluded from frame=0 check
5. Edge: all entries scratched → use cache (zero non-scratched to check)
6. Edge: exactly 50% frame=0 → use cache (threshold is strictly >50%)
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Helpers to build lightweight mock DB objects
# ---------------------------------------------------------------------------

def _make_race(race_id: str = "202606030111", age_days: int = 1) -> MagicMock:
    """Return a mock Race ORM object within CACHE_TTL."""
    race = MagicMock()
    race.race_id = race_id
    race.race_name = "テスト記念"
    race.race_number = 11
    race.grade = "GII"
    race.distance = 2500
    race.surface = "芝"
    race.course_detail = "右回り"
    race.start_time = "15:45"
    race.racecourse_code = "06"
    race.date = "20260601"
    race.head_count = 4
    # scraped_at is within 30-day TTL
    race.scraped_at = datetime.utcnow() - timedelta(days=age_days)
    return race


def _make_entry(
    horse_number: int,
    frame_number: int,
    is_scratched: bool = False,
) -> MagicMock:
    entry = MagicMock()
    entry.race_id = "202606030111"
    entry.horse_number = horse_number
    entry.frame_number = frame_number
    entry.horse_name = f"ウマ{horse_number}"
    entry.horse_id = f"20200000{horse_number:02d}"
    entry.sire_name = ""
    entry.dam_name = ""
    entry.coat_color = ""
    entry.weight_carried = 57.0
    entry.age = "牡4"
    entry.jockey_name = "ルメール"
    entry.jockey_id = ""
    entry.trainer_name = "矢作"
    entry.trainer_id = ""
    entry.horse_weight = "480(0)"
    entry.odds = 5.0
    entry.popularity = 1
    entry.is_scratched = is_scratched
    entry.brood_mare_sire = ""
    entry.past_races_json = "[]"
    return entry


# ---------------------------------------------------------------------------
# Minimal scraped data returned from parse_race_card()
# ---------------------------------------------------------------------------

def _scraped_data(race_id: str = "202606030111", num_entries: int = 4) -> dict:
    return {
        "race_info": {
            "raceName": "テスト記念",
            "raceNumber": 11,
            "grade": "GII",
            "distance": 2500,
            "surface": "芝",
            "courseDetail": "右回り",
            "startTime": "15:45",
            "date": "20260601",
        },
        "entries": [
            {
                "frameNumber": i,
                "horseNumber": i,
                "horseName": f"ウマ{i}",
                "horseId": f"20200000{i:02d}",
                "isScratched": False,
            }
            for i in range(1, num_entries + 1)
        ],
    }


# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

_MOD = "backend.scraper.netkeiba"
_GET_SESSION = f"{_MOD}.get_session"
_MAKE_SESSION = f"{_MOD}._make_session"
_PARSE_RC = f"{_MOD}.parse_race_card"
_CACHE = f"{_MOD}._cache_race_card"
_FETCH_PEDIGREE = f"{_MOD}._fetch_pedigree_from_shutuba_past"
_FETCH_RESULT = f"{_MOD}._fetch_result_data"
_TIME_SLEEP = f"{_MOD}.time.sleep"


# ---------------------------------------------------------------------------
# Test: 1 — valid frame numbers → cache is returned, no scrape
# ---------------------------------------------------------------------------

class TestCacheUsedWhenFramesAreValid:
    """When all non-scratched entries have non-zero frame numbers,
    the cached data should be returned without any HTTP request."""

    def test_returns_cached_dict_with_valid_frames(self):
        race_id = "202606030111"
        mock_race = _make_race(race_id)
        entries = [
            _make_entry(1, frame_number=1),
            _make_entry(2, frame_number=1),
            _make_entry(3, frame_number=2),
            _make_entry(4, frame_number=2),
        ]

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_race
        mock_db.query.return_value.filter.return_value.all.return_value = entries

        with patch(_GET_SESSION, return_value=mock_db), \
             patch(_MAKE_SESSION) as mock_session:

            from backend.scraper.netkeiba import fetch_race_card
            result = fetch_race_card(race_id)

        assert result is not None
        assert "race_info" in result
        assert "entries" in result
        # HTTP session must NOT have been used (no scrape)
        mock_session.assert_not_called()

    def test_single_entry_with_valid_frame_uses_cache(self):
        race_id = "202606030112"
        mock_race = _make_race(race_id)
        entries = [_make_entry(1, frame_number=3)]

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_race
        mock_db.query.return_value.filter.return_value.all.return_value = entries

        with patch(_GET_SESSION, return_value=mock_db), \
             patch(_MAKE_SESSION) as mock_session:
            from backend.scraper.netkeiba import fetch_race_card
            result = fetch_race_card(race_id)

        assert result is not None
        mock_session.assert_not_called()

    def test_exactly_50_percent_zero_frames_uses_cache(self):
        """Exactly 50% zero frames is NOT above the threshold — cache must be used."""
        race_id = "202606030113"
        mock_race = _make_race(race_id)
        # 2 out of 4 = exactly 50%
        entries = [
            _make_entry(1, frame_number=0),
            _make_entry(2, frame_number=0),
            _make_entry(3, frame_number=2),
            _make_entry(4, frame_number=3),
        ]

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_race
        mock_db.query.return_value.filter.return_value.all.return_value = entries

        with patch(_GET_SESSION, return_value=mock_db), \
             patch(_MAKE_SESSION) as mock_session:
            from backend.scraper.netkeiba import fetch_race_card
            result = fetch_race_card(race_id)

        assert result is not None
        mock_session.assert_not_called()

    def test_minority_zero_frames_uses_cache(self):
        """1 out of 4 zero frames = 25% — well below threshold, use cache."""
        race_id = "202606030114"
        mock_race = _make_race(race_id)
        entries = [
            _make_entry(1, frame_number=0),
            _make_entry(2, frame_number=1),
            _make_entry(3, frame_number=2),
            _make_entry(4, frame_number=3),
        ]

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_race
        mock_db.query.return_value.filter.return_value.all.return_value = entries

        with patch(_GET_SESSION, return_value=mock_db), \
             patch(_MAKE_SESSION) as mock_session:
            from backend.scraper.netkeiba import fetch_race_card
            result = fetch_race_card(race_id)

        assert result is not None
        mock_session.assert_not_called()


# ---------------------------------------------------------------------------
# Test: 2 — >50% frame=0 → cache invalidated, scrape performed
# ---------------------------------------------------------------------------

class TestCacheInvalidatedWhenMajorityFramesAreZero:
    """When >50% of non-scratched entries have frame_number=0,
    the cache must be invalidated and a fresh HTTP scrape performed."""

    def _setup_mocks(self, race_id: str, entries: list):
        """Wire up DB mock and HTTP session mock for an invalidation scenario."""
        mock_race = _make_race(race_id)
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_race
        mock_db.query.return_value.filter.return_value.all.return_value = entries

        scraped = _scraped_data(race_id)
        mock_http_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "<html></html>"
        mock_resp.apparent_encoding = "UTF-8"
        mock_http_session.get.return_value = mock_resp

        return mock_db, mock_http_session, scraped

    def test_majority_zero_triggers_rescrape(self):
        race_id = "202606030120"
        entries = [
            _make_entry(1, frame_number=0),
            _make_entry(2, frame_number=0),
            _make_entry(3, frame_number=0),
            _make_entry(4, frame_number=1),  # only 1 with frame
        ]
        mock_db, mock_http, scraped = self._setup_mocks(race_id, entries)

        with patch(_GET_SESSION, return_value=mock_db), \
             patch(_MAKE_SESSION, return_value=mock_http), \
             patch(_PARSE_RC, return_value=scraped), \
             patch(_FETCH_PEDIGREE, return_value={}), \
             patch(_FETCH_RESULT, return_value={}), \
             patch(_CACHE), \
             patch(_TIME_SLEEP):
            from backend.scraper.netkeiba import fetch_race_card
            result = fetch_race_card(race_id)

        # HTTP session was created (scrape happened)
        mock_http.get.assert_called()
        assert result is not None
        assert result["entries"]

    def test_all_zero_frames_triggers_rescrape(self):
        """When every non-scratched entry has frame=0, re-scrape must happen."""
        race_id = "202606030121"
        entries = [
            _make_entry(1, frame_number=0),
            _make_entry(2, frame_number=0),
            _make_entry(3, frame_number=0),
        ]
        mock_db, mock_http, scraped = self._setup_mocks(race_id, entries)

        with patch(_GET_SESSION, return_value=mock_db), \
             patch(_MAKE_SESSION, return_value=mock_http), \
             patch(_PARSE_RC, return_value=scraped), \
             patch(_FETCH_PEDIGREE, return_value={}), \
             patch(_FETCH_RESULT, return_value={}), \
             patch(_CACHE), \
             patch(_TIME_SLEEP):
            from backend.scraper.netkeiba import fetch_race_card
            result = fetch_race_card(race_id)

        mock_http.get.assert_called()
        assert result is not None

    def test_just_over_50_percent_triggers_rescrape(self):
        """3 out of 5 = 60% > 50% — must trigger re-scrape."""
        race_id = "202606030122"
        entries = [
            _make_entry(1, frame_number=0),
            _make_entry(2, frame_number=0),
            _make_entry(3, frame_number=0),
            _make_entry(4, frame_number=2),
            _make_entry(5, frame_number=3),
        ]
        mock_db, mock_http, scraped = self._setup_mocks(race_id, entries)

        with patch(_GET_SESSION, return_value=mock_db), \
             patch(_MAKE_SESSION, return_value=mock_http), \
             patch(_PARSE_RC, return_value=scraped), \
             patch(_FETCH_PEDIGREE, return_value={}), \
             patch(_FETCH_RESULT, return_value={}), \
             patch(_CACHE), \
             patch(_TIME_SLEEP):
            from backend.scraper.netkeiba import fetch_race_card
            result = fetch_race_card(race_id)

        mock_http.get.assert_called()
        assert result is not None

    def test_rescrape_result_is_returned_not_stale_cache(self):
        """The data returned after invalidation must be the fresh scraped data."""
        race_id = "202606030123"
        entries = [
            _make_entry(1, frame_number=0),
            _make_entry(2, frame_number=0),
        ]
        mock_db, mock_http, scraped = self._setup_mocks(race_id, entries)
        # Give the scraped data a distinctive race name to tell it apart
        scraped["race_info"]["raceName"] = "FRESHLY_SCRAPED"

        with patch(_GET_SESSION, return_value=mock_db), \
             patch(_MAKE_SESSION, return_value=mock_http), \
             patch(_PARSE_RC, return_value=scraped), \
             patch(_FETCH_PEDIGREE, return_value={}), \
             patch(_FETCH_RESULT, return_value={}), \
             patch(_CACHE), \
             patch(_TIME_SLEEP):
            from backend.scraper.netkeiba import fetch_race_card
            result = fetch_race_card(race_id)

        assert result["race_info"]["raceName"] == "FRESHLY_SCRAPED"


# ---------------------------------------------------------------------------
# Test: 3 — force_refresh=True bypasses cache entirely
# ---------------------------------------------------------------------------

class TestForceRefreshBypassesCache:
    """With force_refresh=True, get_session / DB query should never be called
    for the initial cache lookup, and an HTTP scrape must always happen."""

    def test_force_refresh_does_not_read_cache(self):
        race_id = "202606030130"
        scraped = _scraped_data(race_id)

        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "<html></html>"
        mock_resp.apparent_encoding = "UTF-8"
        mock_http.get.return_value = mock_resp

        # We patch get_session to track whether it is called
        mock_db = MagicMock()

        with patch(_GET_SESSION, return_value=mock_db) as mock_get_session, \
             patch(_MAKE_SESSION, return_value=mock_http), \
             patch(_PARSE_RC, return_value=scraped), \
             patch(_FETCH_PEDIGREE, return_value={}), \
             patch(_FETCH_RESULT, return_value={}), \
             patch(_CACHE), \
             patch(_TIME_SLEEP):
            from backend.scraper.netkeiba import fetch_race_card
            result = fetch_race_card(race_id, force_refresh=True)

        # DB was only touched for the _cache_race_card write (which we patched).
        # The cache-lookup block (get_session called inside `if not force_refresh`)
        # must NOT have run.
        assert mock_get_session.call_count == 0
        assert result is not None

    def test_force_refresh_triggers_http_get(self):
        race_id = "202606030131"
        scraped = _scraped_data(race_id)

        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "<html></html>"
        mock_resp.apparent_encoding = "UTF-8"
        mock_http.get.return_value = mock_resp

        with patch(_GET_SESSION), \
             patch(_MAKE_SESSION, return_value=mock_http), \
             patch(_PARSE_RC, return_value=scraped), \
             patch(_FETCH_PEDIGREE, return_value={}), \
             patch(_FETCH_RESULT, return_value={}), \
             patch(_CACHE), \
             patch(_TIME_SLEEP):
            from backend.scraper.netkeiba import fetch_race_card
            fetch_race_card(race_id, force_refresh=True)

        # At least the shutuba.html URL was fetched
        called_urls = [str(c.args[0]) for c in mock_http.get.call_args_list]
        assert any("shutuba" in u for u in called_urls)

    def test_force_refresh_works_even_with_valid_cached_entry(self):
        """force_refresh must bypass even a perfectly healthy cache."""
        race_id = "202606030132"
        scraped = _scraped_data(race_id)
        scraped["race_info"]["raceName"] = "FRESH_FORCED"

        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "<html></html>"
        mock_resp.apparent_encoding = "UTF-8"
        mock_http.get.return_value = mock_resp

        # Even if the DB has a valid race, force_refresh must skip it
        mock_db = MagicMock()
        valid_race = _make_race(race_id)
        valid_entries = [_make_entry(i, frame_number=i) for i in range(1, 5)]
        mock_db.query.return_value.filter.return_value.first.return_value = valid_race
        mock_db.query.return_value.filter.return_value.all.return_value = valid_entries

        with patch(_GET_SESSION, return_value=mock_db) as mock_get_session, \
             patch(_MAKE_SESSION, return_value=mock_http), \
             patch(_PARSE_RC, return_value=scraped), \
             patch(_FETCH_PEDIGREE, return_value={}), \
             patch(_FETCH_RESULT, return_value={}), \
             patch(_CACHE), \
             patch(_TIME_SLEEP):
            from backend.scraper.netkeiba import fetch_race_card
            result = fetch_race_card(race_id, force_refresh=True)

        # Cache lookup must have been skipped
        assert mock_get_session.call_count == 0
        assert result["race_info"]["raceName"] == "FRESH_FORCED"


# ---------------------------------------------------------------------------
# Test: 4 — scratched entries excluded from frame=0 check
# ---------------------------------------------------------------------------

class TestScratchedEntriesExcludedFromFrameCheck:
    """is_scratched=True entries must not count toward the frame=0 ratio."""

    def test_scratched_entry_with_zero_frame_not_counted(self):
        """A scratched entry with frame=0 must not push the ratio above threshold."""
        race_id = "202606030140"
        mock_race = _make_race(race_id)
        entries = [
            _make_entry(1, frame_number=1),           # active, frame OK
            _make_entry(2, frame_number=2),           # active, frame OK
            _make_entry(3, frame_number=0, is_scratched=True),  # scratched, frame=0
            _make_entry(4, frame_number=0, is_scratched=True),  # scratched, frame=0
        ]

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_race
        mock_db.query.return_value.filter.return_value.all.return_value = entries

        with patch(_GET_SESSION, return_value=mock_db), \
             patch(_MAKE_SESSION) as mock_session:
            from backend.scraper.netkeiba import fetch_race_card
            result = fetch_race_card(race_id)

        # Only 2 non-scratched, both have valid frames → use cache
        assert result is not None
        mock_session.assert_not_called()

    def test_mix_of_scratched_valid_and_scratched_zero(self):
        """Multiple scratched entries with frame=0 must not affect the threshold."""
        race_id = "202606030141"
        mock_race = _make_race(race_id)
        entries = [
            _make_entry(1, frame_number=3),
            _make_entry(2, frame_number=4),
            _make_entry(3, frame_number=5),
            # Scratched horses that JRA never assigned frame numbers to
            _make_entry(4, frame_number=0, is_scratched=True),
            _make_entry(5, frame_number=0, is_scratched=True),
            _make_entry(6, frame_number=0, is_scratched=True),
        ]

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_race
        mock_db.query.return_value.filter.return_value.all.return_value = entries

        with patch(_GET_SESSION, return_value=mock_db), \
             patch(_MAKE_SESSION) as mock_session:
            from backend.scraper.netkeiba import fetch_race_card
            result = fetch_race_card(race_id)

        assert result is not None
        mock_session.assert_not_called()

    def test_scratched_and_non_scratched_zero_frame_triggers_rescrape(self):
        """If non-scratched entries are also missing frame numbers, still re-scrape."""
        race_id = "202606030142"
        mock_race = _make_race(race_id)
        entries = [
            _make_entry(1, frame_number=0),            # active, frame=0
            _make_entry(2, frame_number=0),            # active, frame=0
            _make_entry(3, frame_number=1),            # active, frame OK
            _make_entry(4, frame_number=0, is_scratched=True),  # scratched
        ]
        scraped = _scraped_data(race_id)

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_race
        mock_db.query.return_value.filter.return_value.all.return_value = entries

        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "<html></html>"
        mock_resp.apparent_encoding = "UTF-8"
        mock_http.get.return_value = mock_resp

        with patch(_GET_SESSION, return_value=mock_db), \
             patch(_MAKE_SESSION, return_value=mock_http), \
             patch(_PARSE_RC, return_value=scraped), \
             patch(_FETCH_PEDIGREE, return_value={}), \
             patch(_FETCH_RESULT, return_value={}), \
             patch(_CACHE), \
             patch(_TIME_SLEEP):
            from backend.scraper.netkeiba import fetch_race_card
            result = fetch_race_card(race_id)

        # 2/3 non-scratched have frame=0 → 66% > 50% → must re-scrape
        mock_http.get.assert_called()
        assert result is not None


# ---------------------------------------------------------------------------
# Test: 5 — all entries scratched → use cache
# ---------------------------------------------------------------------------

class TestAllEntriesScratchedUseCache:
    """When every entry is scratched, non_scratched list is empty.
    The condition `if non_scratched and ...` short-circuits to False,
    so the cached data must be returned."""

    def test_all_scratched_uses_cache(self):
        race_id = "202606030150"
        mock_race = _make_race(race_id)
        entries = [
            _make_entry(1, frame_number=0, is_scratched=True),
            _make_entry(2, frame_number=0, is_scratched=True),
            _make_entry(3, frame_number=0, is_scratched=True),
        ]

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_race
        mock_db.query.return_value.filter.return_value.all.return_value = entries

        with patch(_GET_SESSION, return_value=mock_db), \
             patch(_MAKE_SESSION) as mock_session:
            from backend.scraper.netkeiba import fetch_race_card
            result = fetch_race_card(race_id)

        assert result is not None
        # No HTTP scrape should happen
        mock_session.assert_not_called()

    def test_all_scratched_cache_content_is_returned(self):
        """Verify the dict structure returned is the formatted cache."""
        race_id = "202606030151"
        mock_race = _make_race(race_id)
        mock_race.race_name = "取消レース"
        entries = [
            _make_entry(1, frame_number=0, is_scratched=True),
        ]

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_race
        mock_db.query.return_value.filter.return_value.all.return_value = entries

        with patch(_GET_SESSION, return_value=mock_db), \
             patch(_MAKE_SESSION):
            from backend.scraper.netkeiba import fetch_race_card
            result = fetch_race_card(race_id)

        assert result is not None
        assert result["race_info"]["raceName"] == "取消レース"
        assert len(result["entries"]) == 1
        assert result["entries"][0]["isScratched"] is True


# ---------------------------------------------------------------------------
# Test: 6 — exactly 50% frame=0 uses cache (not >50%)
# ---------------------------------------------------------------------------

class TestExactly50PercentFrameZeroUsesCache:
    """The invalidation condition is `zero_frames > len(non_scratched) * 0.5`,
    i.e., strictly greater-than.  At exactly 50% the cache must be used."""

    @pytest.mark.parametrize("total,zeros", [
        (2, 1),    # 1/2 = 50.0%
        (4, 2),    # 2/4 = 50.0%
        (10, 5),   # 5/10 = 50.0%
        (100, 50), # 50/100 = 50.0%
    ])
    def test_exact_50_percent_uses_cache(self, total: int, zeros: int):
        race_id = f"2026060301{total:02d}"
        mock_race = _make_race(race_id)
        entries = (
            [_make_entry(i, frame_number=0) for i in range(1, zeros + 1)]
            + [_make_entry(i, frame_number=i - zeros) for i in range(zeros + 1, total + 1)]
        )

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_race
        mock_db.query.return_value.filter.return_value.all.return_value = entries

        with patch(_GET_SESSION, return_value=mock_db), \
             patch(_MAKE_SESSION) as mock_session:
            from backend.scraper.netkeiba import fetch_race_card
            result = fetch_race_card(race_id)

        assert result is not None, (
            f"Expected cache hit at {zeros}/{total} = 50%, but got None"
        )
        mock_session.assert_not_called()

    @pytest.mark.parametrize("total,zeros", [
        (2, 2),    # 2/2 = 100%
        (4, 3),    # 3/4 = 75%
        (10, 6),   # 6/10 = 60%
        (100, 51), # 51/100 = 51%
    ])
    def test_above_50_percent_triggers_rescrape(self, total: int, zeros: int):
        race_id = f"2026060302{total:02d}"
        mock_race = _make_race(race_id)
        entries = (
            [_make_entry(i, frame_number=0) for i in range(1, zeros + 1)]
            + [_make_entry(i, frame_number=i - zeros) for i in range(zeros + 1, total + 1)]
        )
        scraped = _scraped_data(race_id)

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_race
        mock_db.query.return_value.filter.return_value.all.return_value = entries

        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "<html></html>"
        mock_resp.apparent_encoding = "UTF-8"
        mock_http.get.return_value = mock_resp

        with patch(_GET_SESSION, return_value=mock_db), \
             patch(_MAKE_SESSION, return_value=mock_http), \
             patch(_PARSE_RC, return_value=scraped), \
             patch(_FETCH_PEDIGREE, return_value={}), \
             patch(_FETCH_RESULT, return_value={}), \
             patch(_CACHE), \
             patch(_TIME_SLEEP):
            from backend.scraper.netkeiba import fetch_race_card
            result = fetch_race_card(race_id)

        mock_http.get.assert_called(), (
            f"Expected re-scrape at {zeros}/{total} > 50%, but HTTP was not called"
        )
        assert result is not None


# ---------------------------------------------------------------------------
# Test: Cache not hit at all — no cached race in DB
# ---------------------------------------------------------------------------

class TestNoCachedRaceFallsThrough:
    """When there is no cached Race row, fetch_race_card must scrape."""

    def test_missing_race_row_triggers_scrape(self):
        race_id = "202606030160"
        scraped = _scraped_data(race_id)

        mock_db = MagicMock()
        # No race found in DB
        mock_db.query.return_value.filter.return_value.first.return_value = None

        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "<html></html>"
        mock_resp.apparent_encoding = "UTF-8"
        mock_http.get.return_value = mock_resp

        with patch(_GET_SESSION, return_value=mock_db), \
             patch(_MAKE_SESSION, return_value=mock_http), \
             patch(_PARSE_RC, return_value=scraped), \
             patch(_FETCH_PEDIGREE, return_value={}), \
             patch(_FETCH_RESULT, return_value={}), \
             patch(_CACHE), \
             patch(_TIME_SLEEP):
            from backend.scraper.netkeiba import fetch_race_card
            result = fetch_race_card(race_id)

        mock_http.get.assert_called()
        assert result is not None

    def test_expired_cache_triggers_scrape(self):
        """A race cached 31 days ago (beyond 30-day TTL) must be re-scraped."""
        race_id = "202606030161"
        old_race = _make_race(race_id, age_days=31)
        scraped = _scraped_data(race_id)

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = old_race

        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "<html></html>"
        mock_resp.apparent_encoding = "UTF-8"
        mock_http.get.return_value = mock_resp

        with patch(_GET_SESSION, return_value=mock_db), \
             patch(_MAKE_SESSION, return_value=mock_http), \
             patch(_PARSE_RC, return_value=scraped), \
             patch(_FETCH_PEDIGREE, return_value={}), \
             patch(_FETCH_RESULT, return_value={}), \
             patch(_CACHE), \
             patch(_TIME_SLEEP):
            from backend.scraper.netkeiba import fetch_race_card
            result = fetch_race_card(race_id)

        mock_http.get.assert_called()
        assert result is not None


# ---------------------------------------------------------------------------
# Test: parse_race_card returns empty/None — fetch_race_card returns None
# ---------------------------------------------------------------------------

class TestScrapeFailurePropagation:
    """When the scraper gets no data (empty page, error), None must be returned."""

    def test_empty_entries_returns_none(self):
        race_id = "202606030170"

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None

        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "<html></html>"
        mock_resp.apparent_encoding = "UTF-8"
        mock_http.get.return_value = mock_resp

        with patch(_GET_SESSION, return_value=mock_db), \
             patch(_MAKE_SESSION, return_value=mock_http), \
             patch(_PARSE_RC, return_value={"race_info": {}, "entries": []}), \
             patch(_FETCH_PEDIGREE, return_value={}), \
             patch(_FETCH_RESULT, return_value={}), \
             patch(_CACHE), \
             patch(_TIME_SLEEP):
            from backend.scraper.netkeiba import fetch_race_card
            result = fetch_race_card(race_id)

        assert result is None

    def test_parse_returns_none_returns_none(self):
        race_id = "202606030171"

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None

        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "<html></html>"
        mock_resp.apparent_encoding = "UTF-8"
        mock_http.get.return_value = mock_resp

        with patch(_GET_SESSION, return_value=mock_db), \
             patch(_MAKE_SESSION, return_value=mock_http), \
             patch(_PARSE_RC, return_value=None), \
             patch(_FETCH_PEDIGREE, return_value={}), \
             patch(_FETCH_RESULT, return_value={}), \
             patch(_CACHE), \
             patch(_TIME_SLEEP):
            from backend.scraper.netkeiba import fetch_race_card
            result = fetch_race_card(race_id)

        assert result is None

    def test_network_error_returns_none(self):
        import requests as req_mod
        race_id = "202606030172"

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None

        mock_http = MagicMock()
        mock_http.get.side_effect = req_mod.RequestException("timeout")

        with patch(_GET_SESSION, return_value=mock_db), \
             patch(_MAKE_SESSION, return_value=mock_http), \
             patch(_TIME_SLEEP):
            from backend.scraper.netkeiba import fetch_race_card
            result = fetch_race_card(race_id)

        assert result is None


# ---------------------------------------------------------------------------
# Test: DB session is always closed (resource leak prevention)
# ---------------------------------------------------------------------------

class TestDbSessionAlwaysClosed:
    """get_session().close() must be called even when cache path short-circuits."""

    def test_db_closed_on_cache_hit(self):
        race_id = "202606030180"
        mock_race = _make_race(race_id)
        entries = [_make_entry(1, frame_number=1), _make_entry(2, frame_number=2)]

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_race
        mock_db.query.return_value.filter.return_value.all.return_value = entries

        with patch(_GET_SESSION, return_value=mock_db), \
             patch(_MAKE_SESSION):
            from backend.scraper.netkeiba import fetch_race_card
            fetch_race_card(race_id)

        mock_db.close.assert_called_once()

    def test_db_closed_on_cache_invalidation_path(self):
        """DB must also be closed when cache is invalidated before scraping."""
        race_id = "202606030181"
        mock_race = _make_race(race_id)
        entries = [_make_entry(1, frame_number=0), _make_entry(2, frame_number=0)]
        scraped = _scraped_data(race_id)

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_race
        mock_db.query.return_value.filter.return_value.all.return_value = entries

        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "<html></html>"
        mock_resp.apparent_encoding = "UTF-8"
        mock_http.get.return_value = mock_resp

        with patch(_GET_SESSION, return_value=mock_db), \
             patch(_MAKE_SESSION, return_value=mock_http), \
             patch(_PARSE_RC, return_value=scraped), \
             patch(_FETCH_PEDIGREE, return_value={}), \
             patch(_FETCH_RESULT, return_value={}), \
             patch(_CACHE), \
             patch(_TIME_SLEEP):
            from backend.scraper.netkeiba import fetch_race_card
            fetch_race_card(race_id)

        mock_db.close.assert_called_once()
