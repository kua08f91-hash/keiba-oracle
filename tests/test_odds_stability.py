"""TDD tests for odds stability fixes.

RED -> GREEN cycle covering:
1. _fetch_live_win_odds  -- netkeiba API with 3-retry logic
2. _apply_odds_to_entries -- in-place mutation of entries list
3. _save_odds_to_db      -- DB persistence with rollback guard
4. Cache TTL logic       -- CACHE_TTL_NO_ODDS (5 min) and frame=0 invalidation
"""
from __future__ import annotations

import os
import sys
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock, call

import pytest
import requests as _requests_lib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =====================================================================
# Helpers
# =====================================================================

def _make_response(data: dict, status_code: int = 200) -> MagicMock:
    """Build a mock requests.Response whose .text is the JSON of *data*."""
    r = MagicMock()
    r.status_code = status_code
    r.text = json.dumps(data)
    return r


def _tansho_payload(horse_odds: dict) -> dict:
    """Build a netkeiba JSON payload for the given {horse_number: (odds, popularity)} dict."""
    tansho = {}
    for hn, (odds_val, pop) in horse_odds.items():
        tansho[str(hn)] = [str(odds_val), "0", str(pop)]
    return {"data": {"odds": {"1": tansho}}}


# =====================================================================
# 1. _fetch_live_win_odds
# =====================================================================

class TestFetchLiveWinOdds:
    """Unit tests for backend.main._fetch_live_win_odds."""

    def test_success_first_attempt_returns_odds_dict(self):
        """Successful first request returns populated odds dict immediately."""
        from backend.main import _fetch_live_win_odds

        payload = _tansho_payload({1: (3.2, 1), 2: (5.0, 2), 3: (10.5, 3)})
        mock_resp = _make_response(payload)

        with patch("requests.get", return_value=mock_resp) as mock_get, \
             patch("time.sleep"):
            result = _fetch_live_win_odds("202605230601")

        assert result == {
            1: {"odds": 3.2, "popularity": 1},
            2: {"odds": 5.0, "popularity": 2},
            3: {"odds": 10.5, "popularity": 3},
        }
        # Only one HTTP call on immediate success
        assert mock_get.call_count == 1

    def test_first_two_attempts_fail_third_succeeds(self):
        """Transient failures on attempts 1 and 2 are retried; 3rd succeeds."""
        from backend.main import _fetch_live_win_odds

        payload = _tansho_payload({1: (4.0, 1)})
        responses = [
            Exception("timeout"),
            Exception("connection reset"),
            _make_response(payload),
        ]

        with patch("requests.get", side_effect=responses) as mock_get, \
             patch("time.sleep") as mock_sleep:
            result = _fetch_live_win_odds("202605230601")

        assert result == {1: {"odds": 4.0, "popularity": 1}}
        assert mock_get.call_count == 3
        # sleep called between each failed attempt (attempts 0 and 1)
        assert mock_sleep.call_count == 2

    def test_all_three_attempts_fail_returns_empty_dict(self):
        """When all 3 attempts raise exceptions, return empty dict (no propagation)."""
        from backend.main import _fetch_live_win_odds

        with patch("requests.get", side_effect=Exception("DNS failure")), \
             patch("time.sleep"):
            result = _fetch_live_win_odds("202605230601")

        assert result == {}

    def test_all_three_attempts_fail_no_exception_propagated(self):
        """Failures must be silently swallowed — callers must not catch exceptions."""
        from backend.main import _fetch_live_win_odds

        with patch("requests.get", side_effect=RuntimeError("boom")), \
             patch("time.sleep"):
            # Must not raise
            result = _fetch_live_win_odds("202605230601")

        assert isinstance(result, dict)

    def test_empty_tansho_data_returns_empty_after_retries(self):
        """netkeiba returns valid JSON but empty tansho section -> retries, then empty dict."""
        from backend.main import _fetch_live_win_odds

        empty_payload = {"data": {"odds": {"1": {}}}}
        mock_resp = _make_response(empty_payload)

        with patch("requests.get", return_value=mock_resp) as mock_get, \
             patch("time.sleep") as mock_sleep:
            result = _fetch_live_win_odds("202605230601")

        assert result == {}
        # All 3 attempts exhausted because result is empty each time
        assert mock_get.call_count == 3
        # sleep between each of the first two failed attempts
        assert mock_sleep.call_count == 2

    def test_parse_error_in_odds_values_skips_bad_horse_returns_rest(self):
        """Malformed vals for one horse are skipped; valid horses still returned."""
        from backend.main import _fetch_live_win_odds

        payload = {
            "data": {
                "odds": {
                    "1": {
                        "1": ["3.2", "0", "1"],         # valid
                        "2": ["not_a_float", "0", "2"],  # bad odds value
                        "3": ["7.8", "0", "3"],          # valid
                        "4": ["9.0"],                    # too short (IndexError)
                    }
                }
            }
        }
        mock_resp = _make_response(payload)

        with patch("requests.get", return_value=mock_resp), \
             patch("time.sleep"):
            result = _fetch_live_win_odds("202605230601")

        # Only valid entries present; bad ones silently skipped
        assert 1 in result
        assert 3 in result
        assert result[1] == {"odds": 3.2, "popularity": 1}
        assert result[3] == {"odds": 7.8, "popularity": 3}
        assert 2 not in result
        assert 4 not in result

    def test_missing_data_key_in_response_returns_empty(self):
        """Response JSON missing 'data' key -> empty dict, no crash."""
        from backend.main import _fetch_live_win_odds

        mock_resp = _make_response({"status": "ok"})

        with patch("requests.get", return_value=mock_resp), \
             patch("time.sleep"):
            result = _fetch_live_win_odds("202605230601")

        assert result == {}

    def test_non_list_vals_in_tansho_skipped(self):
        """Tansho entries where vals is not a list are silently skipped."""
        from backend.main import _fetch_live_win_odds

        payload = {
            "data": {
                "odds": {
                    "1": {
                        "1": {"odds": 3.2},    # dict instead of list
                        "2": ["5.0", "0", "2"],  # valid
                    }
                }
            }
        }
        mock_resp = _make_response(payload)

        with patch("requests.get", return_value=mock_resp), \
             patch("time.sleep"):
            result = _fetch_live_win_odds("202605230601")

        assert 2 in result
        assert 1 not in result

    def test_request_uses_correct_race_id_in_url(self):
        """The race_id is embedded in the URL sent to netkeiba."""
        from backend.main import _fetch_live_win_odds

        payload = _tansho_payload({1: (2.0, 1)})
        mock_resp = _make_response(payload)
        race_id = "202606010501"

        with patch("requests.get", return_value=mock_resp) as mock_get, \
             patch("time.sleep"):
            _fetch_live_win_odds(race_id)

        call_url = mock_get.call_args[0][0]
        assert race_id in call_url

    def test_no_sleep_after_last_failed_attempt(self):
        """sleep is NOT called after the 3rd (final) failed attempt."""
        from backend.main import _fetch_live_win_odds

        sleep_calls = []

        def fake_sleep(secs):
            sleep_calls.append(secs)

        with patch("requests.get", side_effect=Exception("fail")), \
             patch("time.sleep", side_effect=fake_sleep):
            _fetch_live_win_odds("202605230601")

        # Only 2 sleeps between 3 attempts, not 3
        assert len(sleep_calls) == 2


# =====================================================================
# 2. _apply_odds_to_entries
# =====================================================================

class TestApplyOddsToEntries:
    """Unit tests for backend.main._apply_odds_to_entries."""

    def _make_entry(self, horse_number: int, odds=None, popularity=None) -> dict:
        return {
            "horseNumber": horse_number,
            "horseName": f"Horse{horse_number}",
            "odds": odds,
            "popularity": popularity,
        }

    def test_applies_odds_to_matching_horse_numbers(self):
        """odds dict entries are written to the matching entry in-place."""
        from backend.main import _apply_odds_to_entries

        entries = [
            self._make_entry(1),
            self._make_entry(2),
            self._make_entry(3),
        ]
        odds = {
            1: {"odds": 3.2, "popularity": 1},
            2: {"odds": 5.0, "popularity": 2},
            3: {"odds": 10.0, "popularity": 3},
        }

        _apply_odds_to_entries(entries, odds)

        assert entries[0]["odds"] == 3.2
        assert entries[0]["popularity"] == 1
        assert entries[1]["odds"] == 5.0
        assert entries[1]["popularity"] == 2
        assert entries[2]["odds"] == 10.0
        assert entries[2]["popularity"] == 3

    def test_horses_not_in_odds_dict_are_unchanged(self):
        """Entries whose horseNumber is absent from odds are not modified."""
        from backend.main import _apply_odds_to_entries

        entries = [
            self._make_entry(1, odds=None, popularity=None),
            self._make_entry(5, odds=None, popularity=None),  # not in odds
        ]
        odds = {1: {"odds": 3.2, "popularity": 1}}

        _apply_odds_to_entries(entries, odds)

        assert entries[1]["odds"] is None
        assert entries[1]["popularity"] is None

    def test_existing_odds_overwritten_by_new_data(self):
        """Pre-existing odds values in entries are replaced with live data."""
        from backend.main import _apply_odds_to_entries

        entries = [self._make_entry(1, odds=99.0, popularity=18)]
        odds = {1: {"odds": 3.2, "popularity": 1}}

        _apply_odds_to_entries(entries, odds)

        assert entries[0]["odds"] == 3.2
        assert entries[0]["popularity"] == 1

    def test_empty_odds_dict_does_not_change_entries(self):
        """Empty odds dict -> no entry is modified."""
        from backend.main import _apply_odds_to_entries

        entries = [self._make_entry(1, odds=5.0, popularity=2)]
        original_odds = entries[0]["odds"]
        original_pop = entries[0]["popularity"]

        _apply_odds_to_entries(entries, {})

        assert entries[0]["odds"] == original_odds
        assert entries[0]["popularity"] == original_pop

    def test_empty_entries_does_not_crash(self):
        """Empty entries list with a non-empty odds dict -> no crash."""
        from backend.main import _apply_odds_to_entries

        odds = {1: {"odds": 3.2, "popularity": 1}}
        # Must not raise
        _apply_odds_to_entries([], odds)

    def test_both_empty_no_crash(self):
        """Empty entries AND empty odds -> no crash."""
        from backend.main import _apply_odds_to_entries

        _apply_odds_to_entries([], {})

    def test_mutates_in_place_same_list_object(self):
        """The function mutates the same list object; no new list is returned."""
        from backend.main import _apply_odds_to_entries

        entries = [self._make_entry(1)]
        original_id = id(entries)
        odds = {1: {"odds": 4.0, "popularity": 1}}

        result = _apply_odds_to_entries(entries, odds)

        # Return value is None (in-place mutation only)
        assert result is None
        assert id(entries) == original_id

    def test_partial_odds_dict_only_updates_matching(self):
        """Odds dict covering only some horses leaves others untouched."""
        from backend.main import _apply_odds_to_entries

        entries = [
            self._make_entry(1),
            self._make_entry(2),
            self._make_entry(3),
        ]
        odds = {2: {"odds": 7.5, "popularity": 3}}

        _apply_odds_to_entries(entries, odds)

        assert entries[0]["odds"] is None   # horse 1: untouched
        assert entries[1]["odds"] == 7.5    # horse 2: updated
        assert entries[2]["odds"] is None   # horse 3: untouched

    def test_large_field_all_horses_updated(self):
        """All 18 horses in a full field receive their odds correctly."""
        from backend.main import _apply_odds_to_entries

        entries = [self._make_entry(i) for i in range(1, 19)]
        odds = {i: {"odds": float(i) * 1.5, "popularity": i} for i in range(1, 19)}

        _apply_odds_to_entries(entries, odds)

        for i, entry in enumerate(entries, start=1):
            assert entry["odds"] == i * 1.5
            assert entry["popularity"] == i


# =====================================================================
# 3. _save_odds_to_db
# =====================================================================

class TestSaveOddsToDb:
    """Unit tests for backend.main._save_odds_to_db."""

    def _make_horse_entry(self, race_id: str, horse_number: int,
                          odds=None, popularity=None) -> MagicMock:
        he = MagicMock()
        he.race_id = race_id
        he.horse_number = horse_number
        he.odds = odds
        he.popularity = popularity
        return he

    def _make_db_session(self, horse_entries: list) -> MagicMock:
        """Build a mock DB session that returns *horse_entries* from query."""
        db = MagicMock()
        query_chain = MagicMock()
        query_chain.filter.return_value = query_chain
        query_chain.all.return_value = horse_entries
        db.query.return_value = query_chain
        return db

    def test_saves_odds_and_popularity_to_matching_entries(self):
        """Matching HorseEntry rows receive new odds and popularity values."""
        from backend.main import _save_odds_to_db

        he1 = self._make_horse_entry("202605230601", 1)
        he2 = self._make_horse_entry("202605230601", 2)
        db = self._make_db_session([he1, he2])

        odds = {
            1: {"odds": 3.2, "popularity": 1},
            2: {"odds": 5.0, "popularity": 2},
        }

        with patch("backend.main.get_session", return_value=db):
            _save_odds_to_db("202605230601", odds)

        assert he1.odds == 3.2
        assert he1.popularity == 1
        assert he2.odds == 5.0
        assert he2.popularity == 2

    def test_commits_transaction_on_success(self):
        """db.commit() is called exactly once when save succeeds."""
        from backend.main import _save_odds_to_db

        he = self._make_horse_entry("202605230601", 1)
        db = self._make_db_session([he])
        odds = {1: {"odds": 3.2, "popularity": 1}}

        with patch("backend.main.get_session", return_value=db):
            _save_odds_to_db("202605230601", odds)

        db.commit.assert_called_once()
        db.rollback.assert_not_called()

    def test_rollback_on_error(self):
        """When commit raises, rollback is called and exception is swallowed."""
        from backend.main import _save_odds_to_db

        he = self._make_horse_entry("202605230601", 1)
        db = self._make_db_session([he])
        db.commit.side_effect = Exception("DB write failed")
        odds = {1: {"odds": 3.2, "popularity": 1}}

        with patch("backend.main.get_session", return_value=db):
            # Must not propagate
            _save_odds_to_db("202605230601", odds)

        db.rollback.assert_called_once()

    def test_db_session_always_closed(self):
        """db.close() is called in all code paths (success and error)."""
        from backend.main import _save_odds_to_db

        # Success path
        he = self._make_horse_entry("202605230601", 1)
        db_ok = self._make_db_session([he])
        odds = {1: {"odds": 3.2, "popularity": 1}}

        with patch("backend.main.get_session", return_value=db_ok):
            _save_odds_to_db("202605230601", odds)

        db_ok.close.assert_called_once()

        # Error path
        db_err = self._make_db_session([he])
        db_err.commit.side_effect = Exception("disk full")

        with patch("backend.main.get_session", return_value=db_err):
            _save_odds_to_db("202605230601", odds)

        db_err.close.assert_called_once()

    def test_empty_odds_dict_skips_all_db_operations(self):
        """Empty odds dict returns immediately without touching the DB."""
        from backend.main import _save_odds_to_db

        db = MagicMock()

        with patch("backend.main.get_session", return_value=db):
            _save_odds_to_db("202605230601", {})

        db.query.assert_not_called()
        db.commit.assert_not_called()
        db.close.assert_not_called()

    def test_non_existent_race_id_no_crash(self):
        """No HorseEntry rows for race_id -> function completes without error."""
        from backend.main import _save_odds_to_db

        db = self._make_db_session([])  # empty result set
        odds = {1: {"odds": 3.2, "popularity": 1}}

        with patch("backend.main.get_session", return_value=db):
            # Must not raise
            _save_odds_to_db("999999999999", odds)

        db.commit.assert_called_once()

    def test_horses_not_in_odds_dict_not_modified(self):
        """HorseEntry rows whose horse_number is absent from odds are left alone."""
        from backend.main import _save_odds_to_db

        he1 = self._make_horse_entry("202605230601", 1, odds=None)
        he2 = self._make_horse_entry("202605230601", 7, odds=None)  # not in odds
        db = self._make_db_session([he1, he2])
        odds = {1: {"odds": 3.2, "popularity": 1}}

        with patch("backend.main.get_session", return_value=db):
            _save_odds_to_db("202605230601", odds)

        assert he2.odds is None
        assert he2.popularity is None

    def test_query_error_triggers_rollback_not_crash(self):
        """If the query itself raises, rollback is called and exception is swallowed."""
        from backend.main import _save_odds_to_db

        db = MagicMock()
        db.query.side_effect = Exception("table locked")
        odds = {1: {"odds": 3.2, "popularity": 1}}

        with patch("backend.main.get_session", return_value=db):
            _save_odds_to_db("202605230601", odds)

        db.rollback.assert_called_once()
        db.close.assert_called_once()


# =====================================================================
# 4. Cache TTL logic in netkeiba.py
# =====================================================================

class TestCacheTtlLogic:
    """Unit tests for the CACHE_TTL_NO_ODDS short-circuit in fetch_race_card.

    The function checks:
    - odds present, within normal TTL  -> return cached
    - no odds, cache age < 5 min       -> return cached (short TTL still fresh)
    - no odds, cache age >= 5 min      -> invalidate (return None / re-scrape path)
    - frame=0 majority                 -> always invalidate regardless of TTL
    """

    # ------ helpers ------

    def _utcnow_minus(self, delta: timedelta) -> datetime:
        return datetime.utcnow() - delta

    def _make_race(self, race_id: str, scraped_at: datetime,
                   date: str = "20260525") -> MagicMock:
        race = MagicMock()
        race.race_id = race_id
        race.scraped_at = scraped_at
        race.date = date
        return race

    def _make_entries(self, count: int, has_odds: bool = True,
                      frame_number: int = 1) -> list:
        """Return *count* mock HorseEntry objects."""
        entries = []
        for i in range(count):
            e = MagicMock()
            e.is_scratched = False
            e.odds = 3.0 if has_odds else None
            e.frame_number = frame_number
            entries.append(e)
        return entries

    def _make_db_session(self, race: MagicMock,
                         entries: list) -> MagicMock:
        db = MagicMock()

        race_chain = MagicMock()
        race_chain.filter.return_value = race_chain
        race_chain.first.return_value = race

        entry_chain = MagicMock()
        entry_chain.filter.return_value = entry_chain
        entry_chain.all.return_value = entries

        from backend.database.models import Race as RaceModel, HorseEntry

        def side_effect(model):
            if model is RaceModel:
                return race_chain
            if model is HorseEntry:
                return entry_chain
            return MagicMock()

        db.query.side_effect = side_effect
        return db

    # ------ tests ------

    def test_cached_with_odds_within_raceday_ttl_returns_cache(self):
        """Race day cache with odds present and age < 30 min -> return cached."""
        from backend.scraper.netkeiba import fetch_race_card

        race_id = "202605250601"
        # Scraped 10 minutes ago
        scraped_at = self._utcnow_minus(timedelta(minutes=10))
        today_str = datetime.now().strftime("%Y%m%d")

        race = self._make_race(race_id, scraped_at, date=today_str)
        entries = self._make_entries(8, has_odds=True)
        db = self._make_db_session(race, entries)

        with patch("backend.scraper.netkeiba.get_session", return_value=db), \
             patch("backend.scraper.netkeiba._format_cached",
                   return_value={"raceInfo": {}, "entries": []}) as mock_fmt, \
             patch("backend.scraper.netkeiba.time"):
            result = fetch_race_card(race_id)

        mock_fmt.assert_called_once()
        assert result is not None

    def test_cached_no_odds_age_under_5min_returns_cache(self):
        """Cache without odds but age < 5 min -> still return cached (short-lived freshness)."""
        from backend.scraper.netkeiba import fetch_race_card

        race_id = "202605250601"
        scraped_at = self._utcnow_minus(timedelta(minutes=2))  # 2 min old
        today_str = datetime.now().strftime("%Y%m%d")

        race = self._make_race(race_id, scraped_at, date=today_str)
        entries = self._make_entries(8, has_odds=False)  # all odds=None
        db = self._make_db_session(race, entries)

        with patch("backend.scraper.netkeiba.get_session", return_value=db), \
             patch("backend.scraper.netkeiba._format_cached",
                   return_value={"raceInfo": {}, "entries": []}) as mock_fmt, \
             patch("backend.scraper.netkeiba.time"):
            result = fetch_race_card(race_id)

        # Should use cached path because age < CACHE_TTL_NO_ODDS
        mock_fmt.assert_called_once()
        assert result is not None

    def test_cached_no_odds_age_over_5min_invalidates_cache(self):
        """Cache without odds and age > 5 min -> do NOT return cached; trigger re-scrape."""
        from backend.scraper.netkeiba import fetch_race_card

        race_id = "202605250601"
        scraped_at = self._utcnow_minus(timedelta(minutes=10))  # 10 min old
        today_str = datetime.now().strftime("%Y%m%d")

        race = self._make_race(race_id, scraped_at, date=today_str)
        entries = self._make_entries(8, has_odds=False)  # all odds=None
        db = self._make_db_session(race, entries)

        # Stub _format_cached to detect if it gets called (it should NOT for invalidated cache)
        with patch("backend.scraper.netkeiba.get_session", return_value=db), \
             patch("backend.scraper.netkeiba._format_cached") as mock_fmt, \
             patch("backend.scraper.netkeiba._make_session") as mock_make_sess, \
             patch("backend.scraper.netkeiba.time"):
            # Make the live scrape fail via RequestException so fetch_race_card catches it
            mock_http_sess = MagicMock()
            mock_http_sess.get.side_effect = _requests_lib.exceptions.ConnectionError("blocked")
            mock_make_sess.return_value = mock_http_sess

            result = fetch_race_card(race_id)

        # Cache path must NOT have been used
        mock_fmt.assert_not_called()
        # Result is None because live scrape also failed
        assert result is None

    def test_cached_no_odds_exactly_at_5min_boundary_invalidates(self):
        """Cache age of exactly 5 min (== TTL) should NOT be returned (>= threshold)."""
        from backend.scraper import netkeiba as nk_module
        from backend.scraper.netkeiba import fetch_race_card, CACHE_TTL_NO_ODDS

        race_id = "202605250601"
        # age == CACHE_TTL_NO_ODDS exactly
        scraped_at = self._utcnow_minus(CACHE_TTL_NO_ODDS)
        today_str = datetime.now().strftime("%Y%m%d")

        race = self._make_race(race_id, scraped_at, date=today_str)
        entries = self._make_entries(8, has_odds=False)
        db = self._make_db_session(race, entries)

        with patch("backend.scraper.netkeiba.get_session", return_value=db), \
             patch("backend.scraper.netkeiba._format_cached") as mock_fmt, \
             patch("backend.scraper.netkeiba._make_session") as mock_make_sess, \
             patch("backend.scraper.netkeiba.time"):
            mock_http_sess = MagicMock()
            mock_http_sess.get.side_effect = _requests_lib.exceptions.ConnectionError("blocked")
            mock_make_sess.return_value = mock_http_sess

            fetch_race_card(race_id)

        mock_fmt.assert_not_called()

    def test_cached_with_frame_zero_majority_always_invalidates(self):
        """Majority frame=0 entries -> cache invalidated regardless of age or odds."""
        from backend.scraper.netkeiba import fetch_race_card

        race_id = "202605250601"
        # Very fresh cache (1 minute old) — would normally be valid
        scraped_at = self._utcnow_minus(timedelta(minutes=1))
        today_str = datetime.now().strftime("%Y%m%d")

        race = self._make_race(race_id, scraped_at, date=today_str)
        # 8 entries all with frame_number=0 and odds present
        entries = self._make_entries(8, has_odds=True, frame_number=0)
        db = self._make_db_session(race, entries)

        with patch("backend.scraper.netkeiba.get_session", return_value=db), \
             patch("backend.scraper.netkeiba._format_cached") as mock_fmt, \
             patch("backend.scraper.netkeiba._make_session") as mock_make_sess, \
             patch("backend.scraper.netkeiba.time"):
            mock_http_sess = MagicMock()
            mock_http_sess.get.side_effect = _requests_lib.exceptions.ConnectionError("blocked")
            mock_make_sess.return_value = mock_http_sess

            result = fetch_race_card(race_id)

        # Frame=0 majority must trigger invalidation, not return cached
        mock_fmt.assert_not_called()
        assert result is None

    def test_cached_with_frame_zero_minority_uses_normal_ttl(self):
        """Only a few entries with frame=0 (minority) -> normal TTL applies; cache returned."""
        from backend.scraper.netkeiba import fetch_race_card

        race_id = "202605250601"
        scraped_at = self._utcnow_minus(timedelta(minutes=5))
        today_str = datetime.now().strftime("%Y%m%d")

        race = self._make_race(race_id, scraped_at, date=today_str)

        # 2 with frame=0, 6 with frame=1 (minority zero-frames, majority have odds)
        entries_with_frame = self._make_entries(6, has_odds=True, frame_number=1)
        entries_no_frame = self._make_entries(2, has_odds=True, frame_number=0)
        entries = entries_with_frame + entries_no_frame

        db = self._make_db_session(race, entries)

        with patch("backend.scraper.netkeiba.get_session", return_value=db), \
             patch("backend.scraper.netkeiba._format_cached",
                   return_value={"raceInfo": {}, "entries": []}) as mock_fmt, \
             patch("backend.scraper.netkeiba.time"):
            result = fetch_race_card(race_id)

        mock_fmt.assert_called_once()
        assert result is not None

    def test_non_raceday_cache_uses_30day_ttl(self):
        """For non-race-day dates, the 30-day TTL is used (old cache still valid)."""
        from backend.scraper.netkeiba import fetch_race_card

        race_id = "202601010101"
        # 7 days old — within 30-day TTL, but well outside 30-min raceday TTL
        scraped_at = self._utcnow_minus(timedelta(days=7))

        race = self._make_race(race_id, scraped_at, date="20260101")  # past date
        entries = self._make_entries(8, has_odds=True)
        db = self._make_db_session(race, entries)

        with patch("backend.scraper.netkeiba.get_session", return_value=db), \
             patch("backend.scraper.netkeiba._format_cached",
                   return_value={"raceInfo": {}, "entries": []}) as mock_fmt, \
             patch("backend.scraper.netkeiba.time"):
            result = fetch_race_card(race_id)

        mock_fmt.assert_called_once()
        assert result is not None

    def test_raceday_cache_expired_at_31min_triggers_rescrape(self):
        """Race-day cache older than 30 min is stale -> re-scrape is attempted."""
        from backend.scraper.netkeiba import fetch_race_card

        race_id = "202605250601"
        scraped_at = self._utcnow_minus(timedelta(minutes=31))
        today_str = datetime.now().strftime("%Y%m%d")

        race = self._make_race(race_id, scraped_at, date=today_str)
        entries = self._make_entries(8, has_odds=True)
        db = self._make_db_session(race, entries)

        with patch("backend.scraper.netkeiba.get_session", return_value=db), \
             patch("backend.scraper.netkeiba._format_cached") as mock_fmt, \
             patch("backend.scraper.netkeiba._make_session") as mock_make_sess, \
             patch("backend.scraper.netkeiba.time"):
            mock_http_sess = MagicMock()
            mock_http_sess.get.side_effect = _requests_lib.exceptions.ConnectionError("blocked")
            mock_make_sess.return_value = mock_http_sess

            result = fetch_race_card(race_id)

        # Cache was not returned — re-scrape was triggered (which then failed)
        mock_fmt.assert_not_called()
        assert result is None

    def test_cache_ttl_no_odds_constant_is_5_minutes(self):
        """Sanity check: CACHE_TTL_NO_ODDS must be exactly 5 minutes."""
        from backend.scraper.netkeiba import CACHE_TTL_NO_ODDS

        assert CACHE_TTL_NO_ODDS == timedelta(minutes=5)

    def test_force_refresh_bypasses_cache_entirely(self):
        """force_refresh=True skips cache check and goes straight to live scrape."""
        from backend.scraper.netkeiba import fetch_race_card

        race_id = "202605250601"

        with patch("backend.scraper.netkeiba.get_session") as mock_get_sess, \
             patch("backend.scraper.netkeiba._make_session") as mock_make_sess, \
             patch("backend.scraper.netkeiba.time"):
            mock_http_sess = MagicMock()
            mock_http_sess.get.side_effect = _requests_lib.exceptions.ConnectionError("blocked")
            mock_make_sess.return_value = mock_http_sess

            result = fetch_race_card(race_id, force_refresh=True)

        # DB was never consulted for cached data
        mock_get_sess.assert_not_called()
        assert result is None
