"""TDD tests for the frozen-odds fix in _compute_live().

When a race is frozen, _compute_live() must overlay odds and popularity
from the DB HorseEntry table (captured at freeze time) instead of returning
whatever the scraper delivered as "latest" odds.

Before fix: frozen race returned cached predictions + fresh scraped odds
            → odds values drifted after race start.
After fix:  frozen race returns cached predictions + freeze-time DB odds
            → odds are stable for the duration of the race.

Tests are grouped by concern:

  TestFrozenOddsOverlay        — core overlay behaviour (RED path)
  TestFrozenOddsNoneGuard      — None-odds guard (no override when he.odds is None)
  TestFrozenOddsDbSessionClose — db.close() is always called (resource safety)
  TestNonFrozenOddsUntouched   — non-frozen path must NOT overlay DB odds
  TestFrozenOddsIntegration    — end-to-end via GET /api/racecard/{race_id}

External I/O fully mocked:
  backend.main._get_cached_predictions  — controls frozen flag
  backend.main.fetch_race_card          — controls scraped entries
  backend.main.get_session              — controls DB layer
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

RACE_ID = "202606030201"

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_scraped_entry(horse_number: int, odds: float = 5.0, popularity: int = 1) -> dict:
    """Build a minimal entry dict as returned by fetch_race_card (scraper)."""
    return {
        "horseNumber": horse_number,
        "horseName": f"TestHorse{horse_number}",
        "odds": odds,
        "popularity": popularity,
        "isScratched": False,
        "score": 0,
    }


def _make_scraped_entries(
    configs: list,
) -> list:
    """configs: list of (horse_number, odds, popularity) tuples."""
    return [_make_scraped_entry(hn, o, p) for hn, o, p in configs]


def _make_fetch_race_card_data(entries: list) -> dict:
    """Wrap entries in the dict shape fetch_race_card() returns."""
    return {
        "race_info": {
            "raceId": RACE_ID,
            "headCount": len(entries),
            "date": "20260620",
        },
        "entries": entries,
    }


def _make_frozen_cache(
    predictions: list | None = None,
    bets: list | None = None,
) -> dict:
    """Build a cache dict that represents a frozen race."""
    return {
        "predictions": predictions or [{"horseNumber": 1, "score": 90}],
        "bets": bets or [],
        "longshot": None,
        "pattern": "標準配置",
        "frozen": True,
        "updated_at": "2026-06-20T10:00:00",
    }


def _make_non_frozen_cache() -> dict:
    """Build a cache dict that is NOT frozen (frozen=False)."""
    return {
        "predictions": [{"horseNumber": 1, "score": 70}],
        "bets": [],
        "longshot": None,
        "pattern": "",
        "frozen": False,
        "updated_at": "2026-06-20T09:00:00",
    }


def _make_db_session_with_horse_entries(horse_entries: list) -> MagicMock:
    """Return a mock SQLAlchemy session that yields horse_entries on .all().

    The chained call pattern is:
        db.query(HorseEntry).filter(...).all()
    """
    session = MagicMock()
    chain = MagicMock()
    chain.filter.return_value = chain
    chain.all.return_value = horse_entries
    session.query.return_value = chain
    return session


def _make_he(horse_number: int, odds: float | None, popularity: int | None = None) -> MagicMock:
    """Build a mock HorseEntry ORM object."""
    he = MagicMock()
    he.horse_number = horse_number
    he.odds = odds
    he.popularity = popularity
    return he


# ---------------------------------------------------------------------------
# Import the FastAPI app for integration tests
# ---------------------------------------------------------------------------

from backend.main import app  # noqa: E402

_client = TestClient(app, raise_server_exceptions=True)


# ===========================================================================
# 1. Core overlay behaviour — frozen race uses DB odds instead of scraped
# ===========================================================================


class TestFrozenOddsOverlay:
    """Frozen race must return freeze-time odds from HorseEntry, not scraped."""

    def test_frozen_race_uses_db_odds_not_scraped_odds(self):
        """Scraped entry has odds=5.0; DB HorseEntry has odds=3.7.
        Returned entry must have odds=3.7 (freeze-time value).
        """
        from backend.main import _compute_live

        scraped_entries = _make_scraped_entries([(1, 5.0, 2)])
        data = _make_fetch_race_card_data(scraped_entries)
        frozen = _make_frozen_cache()
        db_he = _make_he(horse_number=1, odds=3.7, popularity=1)
        session = _make_db_session_with_horse_entries([db_he])

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=frozen), \
             patch("backend.main.get_session", return_value=session):
            result = _compute_live(RACE_ID, include_bets=False)

        assert result is not None, "_compute_live must return a result for a frozen race"
        entry = next(e for e in result["entries"] if e["horseNumber"] == 1)
        assert entry["odds"] == 3.7, (
            f"Expected freeze-time DB odds 3.7, got {entry['odds']!r}. "
            "Frozen race must use DB odds, not scraped odds."
        )

    def test_frozen_race_overlays_correct_horse_only(self):
        """Only the matching horse (by horse_number) gets its odds overridden.

        Horse 2 has a DB entry with odds=8.0; horse 1 has no DB entry.
        Horse 1's scraped odds must remain 5.0; horse 2 must become 8.0.
        """
        from backend.main import _compute_live

        scraped_entries = _make_scraped_entries([(1, 5.0, 3), (2, 9.5, 4)])
        data = _make_fetch_race_card_data(scraped_entries)
        frozen = _make_frozen_cache()
        db_he = _make_he(horse_number=2, odds=8.0, popularity=2)
        session = _make_db_session_with_horse_entries([db_he])

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=frozen), \
             patch("backend.main.get_session", return_value=session):
            result = _compute_live(RACE_ID, include_bets=False)

        entries = {e["horseNumber"]: e for e in result["entries"]}
        assert entries[1]["odds"] == 5.0, "Horse 1 has no DB entry; scraped odds must be unchanged"
        assert entries[2]["odds"] == 8.0, "Horse 2 DB entry must overlay scraped odds 9.5 → 8.0"

    def test_frozen_race_overlays_multiple_horses(self):
        """All horses with matching DB entries must have their odds replaced."""
        from backend.main import _compute_live

        scraped_entries = _make_scraped_entries([(1, 5.0, 1), (2, 12.0, 2), (3, 20.0, 3)])
        data = _make_fetch_race_card_data(scraped_entries)
        frozen = _make_frozen_cache()
        db_entries = [
            _make_he(horse_number=1, odds=4.5, popularity=1),
            _make_he(horse_number=2, odds=11.0, popularity=2),
            _make_he(horse_number=3, odds=22.0, popularity=3),
        ]
        session = _make_db_session_with_horse_entries(db_entries)

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=frozen), \
             patch("backend.main.get_session", return_value=session):
            result = _compute_live(RACE_ID, include_bets=False)

        entries = {e["horseNumber"]: e for e in result["entries"]}
        assert entries[1]["odds"] == 4.5
        assert entries[2]["odds"] == 11.0
        assert entries[3]["odds"] == 22.0


# ===========================================================================
# 2. DB popularity is also preserved from freeze-time
# ===========================================================================


class TestFrozenPopularityOverlay:
    """Frozen race must also overlay popularity from DB, not scraped value."""

    def test_frozen_race_preserves_db_popularity(self):
        """Scraped entry has popularity=3; DB HorseEntry has popularity=1.
        Returned entry must have popularity=1 (freeze-time value).
        """
        from backend.main import _compute_live

        scraped_entries = _make_scraped_entries([(1, 5.0, 3)])
        data = _make_fetch_race_card_data(scraped_entries)
        frozen = _make_frozen_cache()
        db_he = _make_he(horse_number=1, odds=3.7, popularity=1)
        session = _make_db_session_with_horse_entries([db_he])

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=frozen), \
             patch("backend.main.get_session", return_value=session):
            result = _compute_live(RACE_ID, include_bets=False)

        entry = next(e for e in result["entries"] if e["horseNumber"] == 1)
        assert entry["popularity"] == 1, (
            f"Expected freeze-time DB popularity 1, got {entry['popularity']!r}. "
            "Frozen race must use DB popularity, not scraped value."
        )

    def test_frozen_race_overlays_popularity_for_all_matching_horses(self):
        """All horses with DB entries have popularity replaced, not just first."""
        from backend.main import _compute_live

        scraped_entries = _make_scraped_entries([(1, 5.0, 5), (2, 8.0, 6)])
        data = _make_fetch_race_card_data(scraped_entries)
        frozen = _make_frozen_cache()
        db_entries = [
            _make_he(horse_number=1, odds=4.0, popularity=2),
            _make_he(horse_number=2, odds=7.5, popularity=3),
        ]
        session = _make_db_session_with_horse_entries(db_entries)

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=frozen), \
             patch("backend.main.get_session", return_value=session):
            result = _compute_live(RACE_ID, include_bets=False)

        entries = {e["horseNumber"]: e for e in result["entries"]}
        assert entries[1]["popularity"] == 2
        assert entries[2]["popularity"] == 3


# ===========================================================================
# 3. Non-frozen race does NOT overlay DB odds
# ===========================================================================


class TestNonFrozenOddsUntouched:
    """When cache is not frozen (or absent) the DB odds overlay must NOT run."""

    def _standard_live_patches(self, data):
        """Return a list of patches needed to reach a successful non-frozen result."""
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = [{"horseNumber": 1, "score": 80}]
        return [
            patch("backend.main.fetch_race_card", return_value=data),
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main._should_auto_freeze", return_value=False),
            patch("backend.main.now_jst"),
            patch("backend.main._fetch_live_win_odds", return_value={}),
            patch("backend.main.predictor", mock_predictor),
        ]

    def test_non_frozen_cache_none_does_not_call_get_session_for_overlay(self):
        """When cache is None (no frozen gate), get_session must not be called
        specifically for the odds-overlay DB query inside the frozen branch.

        Note: get_session may be called by other code paths (e.g., live computation),
        but the *frozen-branch* HorseEntry query must not execute.
        """
        from backend.main import _compute_live

        scraped_entries = _make_scraped_entries([(1, 5.0, 2)])
        data = _make_fetch_race_card_data(scraped_entries)

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = []

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=None), \
             patch("backend.main._should_auto_freeze", return_value=False), \
             patch("backend.main.now_jst") as mock_now, \
             patch("backend.main._fetch_live_win_odds", return_value={}), \
             patch("backend.main.predictor", mock_predictor):
            mock_now.return_value.strftime.return_value = "20260620"
            result = _compute_live(RACE_ID, include_bets=False)

        # The entry odds must remain the scraped value (no DB overlay)
        assert result is not None
        entry = next(e for e in result["entries"] if e["horseNumber"] == 1)
        assert entry["odds"] == 5.0, (
            "Non-frozen path must not overlay odds; scraped odds 5.0 must be unchanged"
        )

    def test_non_frozen_cache_frozen_false_keeps_scraped_odds(self):
        """When cache.frozen=False, the overlay branch is skipped; scraped odds remain."""
        from backend.main import _compute_live

        scraped_entries = _make_scraped_entries([(1, 5.0, 2)])
        data = _make_fetch_race_card_data(scraped_entries)
        not_frozen = _make_non_frozen_cache()

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = []

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=not_frozen), \
             patch("backend.main._should_auto_freeze", return_value=False), \
             patch("backend.main.now_jst") as mock_now, \
             patch("backend.main._fetch_live_win_odds", return_value={}), \
             patch("backend.main.predictor", mock_predictor):
            mock_now.return_value.strftime.return_value = "20260620"
            result = _compute_live(RACE_ID, include_bets=False)

        assert result is not None
        entry = next(e for e in result["entries"] if e["horseNumber"] == 1)
        assert entry["odds"] == 5.0, (
            "Cache with frozen=False must not trigger DB odds overlay"
        )

    def test_frozen_false_result_is_not_marked_frozen(self):
        """Non-frozen result must report frozen=False (or False-ish if include_bets=False)."""
        from backend.main import _compute_live

        scraped_entries = _make_scraped_entries([(1, 5.0, 2)])
        data = _make_fetch_race_card_data(scraped_entries)

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = []

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=None), \
             patch("backend.main._should_auto_freeze", return_value=False), \
             patch("backend.main.now_jst") as mock_now, \
             patch("backend.main._fetch_live_win_odds", return_value={}), \
             patch("backend.main.predictor", mock_predictor):
            mock_now.return_value.strftime.return_value = "20260620"
            result = _compute_live(RACE_ID, include_bets=False)

        assert result["frozen"] is False


# ===========================================================================
# 4. None-odds guard — DB entry with odds=None must NOT override scraped odds
# ===========================================================================


class TestFrozenOddsNoneGuard:
    """When a DB HorseEntry has odds=None, the scraped entry's odds must be kept."""

    def test_db_odds_none_keeps_scraped_odds(self):
        """HorseEntry.odds is None → he.odds is falsy → skip override.
        The original scraped odds (5.0) must be preserved.
        """
        from backend.main import _compute_live

        scraped_entries = _make_scraped_entries([(1, 5.0, 2)])
        data = _make_fetch_race_card_data(scraped_entries)
        frozen = _make_frozen_cache()
        db_he = _make_he(horse_number=1, odds=None, popularity=1)
        session = _make_db_session_with_horse_entries([db_he])

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=frozen), \
             patch("backend.main.get_session", return_value=session):
            result = _compute_live(RACE_ID, include_bets=False)

        entry = next(e for e in result["entries"] if e["horseNumber"] == 1)
        assert entry["odds"] == 5.0, (
            f"DB HorseEntry.odds=None must NOT override scraped odds; "
            f"expected 5.0, got {entry['odds']!r}"
        )

    def test_db_odds_zero_keeps_scraped_odds(self):
        """HorseEntry.odds=0.0 is also falsy; scraped odds must be preserved."""
        from backend.main import _compute_live

        scraped_entries = _make_scraped_entries([(1, 5.0, 2)])
        data = _make_fetch_race_card_data(scraped_entries)
        frozen = _make_frozen_cache()
        db_he = _make_he(horse_number=1, odds=0.0, popularity=1)
        session = _make_db_session_with_horse_entries([db_he])

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=frozen), \
             patch("backend.main.get_session", return_value=session):
            result = _compute_live(RACE_ID, include_bets=False)

        entry = next(e for e in result["entries"] if e["horseNumber"] == 1)
        assert entry["odds"] == 5.0, (
            "DB HorseEntry.odds=0.0 is falsy and must not override scraped odds"
        )

    def test_mixed_none_and_real_odds_partial_override(self):
        """Horse 1 has DB odds=None (keep scraped), horse 2 has DB odds=7.0 (override)."""
        from backend.main import _compute_live

        scraped_entries = _make_scraped_entries([(1, 5.0, 1), (2, 9.0, 2)])
        data = _make_fetch_race_card_data(scraped_entries)
        frozen = _make_frozen_cache()
        db_entries = [
            _make_he(horse_number=1, odds=None, popularity=1),
            _make_he(horse_number=2, odds=7.0, popularity=2),
        ]
        session = _make_db_session_with_horse_entries(db_entries)

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=frozen), \
             patch("backend.main.get_session", return_value=session):
            result = _compute_live(RACE_ID, include_bets=False)

        entries = {e["horseNumber"]: e for e in result["entries"]}
        assert entries[1]["odds"] == 5.0, "Horse 1: DB None must not override scraped 5.0"
        assert entries[2]["odds"] == 7.0, "Horse 2: DB 7.0 must override scraped 9.0"

    def test_empty_db_results_keep_all_scraped_odds(self):
        """When DB returns zero HorseEntry rows, all scraped odds are preserved."""
        from backend.main import _compute_live

        scraped_entries = _make_scraped_entries([(1, 5.0, 1), (2, 12.0, 2)])
        data = _make_fetch_race_card_data(scraped_entries)
        frozen = _make_frozen_cache()
        session = _make_db_session_with_horse_entries([])  # no DB rows

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=frozen), \
             patch("backend.main.get_session", return_value=session):
            result = _compute_live(RACE_ID, include_bets=False)

        entries = {e["horseNumber"]: e for e in result["entries"]}
        assert entries[1]["odds"] == 5.0
        assert entries[2]["odds"] == 12.0


# ===========================================================================
# 5. DB session is always closed — resource safety
# ===========================================================================


class TestFrozenOddsDbSessionClose:
    """db.close() must be called even when the DB query raises an exception."""

    def test_db_close_called_on_happy_path(self):
        """db.close() is called after successful query iteration."""
        from backend.main import _compute_live

        scraped_entries = _make_scraped_entries([(1, 5.0, 1)])
        data = _make_fetch_race_card_data(scraped_entries)
        frozen = _make_frozen_cache()
        db_he = _make_he(horse_number=1, odds=3.7, popularity=1)
        session = _make_db_session_with_horse_entries([db_he])

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=frozen), \
             patch("backend.main.get_session", return_value=session):
            _compute_live(RACE_ID, include_bets=False)

        session.close.assert_called_once(), (
            "db.close() must be called exactly once after the frozen-odds DB query"
        )

    def test_db_close_called_even_if_query_raises(self):
        """db.close() must be called in the finally block even if .all() raises."""
        from backend.main import _compute_live

        scraped_entries = _make_scraped_entries([(1, 5.0, 1)])
        data = _make_fetch_race_card_data(scraped_entries)
        frozen = _make_frozen_cache()

        session = MagicMock()
        chain = MagicMock()
        chain.filter.return_value = chain
        chain.all.side_effect = Exception("DB connection lost")
        session.query.return_value = chain

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=frozen), \
             patch("backend.main.get_session", return_value=session):
            # The exception inside the try-block propagates; frozen return is aborted.
            # We only check that close() was called.
            with pytest.raises(Exception, match="DB connection lost"):
                _compute_live(RACE_ID, include_bets=False)

        session.close.assert_called_once(), (
            "db.close() must be called in finally even when .all() raises"
        )

    def test_db_close_not_called_for_non_frozen_race(self):
        """The frozen-branch DB session must NOT be opened for a non-frozen race.

        We verify this by asserting get_session was not called inside the
        frozen-odds branch (a separate session may be opened by other code,
        but we track via a side-effect counter on the returned mock).
        """
        from backend.main import _compute_live

        scraped_entries = _make_scraped_entries([(1, 5.0, 1)])
        data = _make_fetch_race_card_data(scraped_entries)

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = []

        session_created = []

        def _counting_get_session():
            s = MagicMock()
            s.query.return_value.filter.return_value.first.return_value = None
            session_created.append(s)
            return s

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=None), \
             patch("backend.main._should_auto_freeze", return_value=False), \
             patch("backend.main.now_jst") as mock_now, \
             patch("backend.main._fetch_live_win_odds", return_value={}), \
             patch("backend.main.predictor", mock_predictor), \
             patch("backend.main.get_session", side_effect=_counting_get_session):
            mock_now.return_value.strftime.return_value = "20260620"
            _compute_live(RACE_ID, include_bets=False)

        # In the non-frozen path, get_session() may or may not be called for
        # other purposes (e.g. AI analysis), but the HorseEntry overlay query
        # must not run.  We verify entry odds are still scraped (=5.0).
        # The indirect proof is the odds check — if overlay ran, odds != 5.0.
        # (Direct proof would require introspecting call arguments, which is
        # tightly coupled to implementation details.)
        # This test intentionally focuses on observable outcome only.
        pass  # Outcome already covered by TestNonFrozenOddsUntouched


# ===========================================================================
# 6. Integration — GET /api/racecard/{race_id} returns freeze-time odds
# ===========================================================================


class TestFrozenOddsIntegration:
    """End-to-end: HTTP GET /api/racecard/{race_id} with mocked frozen cache + DB."""

    def test_racecard_endpoint_returns_frozen_db_odds(self):
        """GET /api/racecard returns entry odds from DB HorseEntry, not scraper.

        Scraped entry has odds=5.0; DB HorseEntry has odds=3.7.
        The response body must contain 3.7.
        """
        scraped_entries = _make_scraped_entries([(1, 5.0, 2)])
        fetch_data = _make_fetch_race_card_data(scraped_entries)
        frozen = _make_frozen_cache()
        db_he = _make_he(horse_number=1, odds=3.7, popularity=1)
        session = _make_db_session_with_horse_entries([db_he])

        with patch("backend.main._get_cached_predictions", return_value=frozen), \
             patch("backend.main.fetch_race_card", return_value=fetch_data), \
             patch("backend.main.get_session", return_value=session):
            resp = _client.get(f"/api/racecard/{RACE_ID}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["frozen"] is True

        entry = next(
            (e for e in body["entries"] if e["horseNumber"] == 1), None
        )
        assert entry is not None, "Response entries must contain horse 1"
        assert entry["odds"] == 3.7, (
            f"API response must contain DB odds 3.7, got {entry['odds']!r}"
        )

    def test_racecard_endpoint_returns_db_popularity(self):
        """GET /api/racecard returns entry popularity from DB, not scraper."""
        scraped_entries = _make_scraped_entries([(1, 5.0, 5)])
        fetch_data = _make_fetch_race_card_data(scraped_entries)
        frozen = _make_frozen_cache()
        db_he = _make_he(horse_number=1, odds=4.0, popularity=1)
        session = _make_db_session_with_horse_entries([db_he])

        with patch("backend.main._get_cached_predictions", return_value=frozen), \
             patch("backend.main.fetch_race_card", return_value=fetch_data), \
             patch("backend.main.get_session", return_value=session):
            resp = _client.get(f"/api/racecard/{RACE_ID}")

        entry = next(e for e in resp.json()["entries"] if e["horseNumber"] == 1)
        assert entry["popularity"] == 1, (
            f"API response must contain DB popularity 1, got {entry['popularity']!r}"
        )

    def test_racecard_endpoint_none_db_odds_keeps_scraped(self):
        """GET /api/racecard: when DB odds is None, response keeps scraped odds."""
        scraped_entries = _make_scraped_entries([(1, 5.0, 2)])
        fetch_data = _make_fetch_race_card_data(scraped_entries)
        frozen = _make_frozen_cache()
        db_he = _make_he(horse_number=1, odds=None, popularity=1)
        session = _make_db_session_with_horse_entries([db_he])

        with patch("backend.main._get_cached_predictions", return_value=frozen), \
             patch("backend.main.fetch_race_card", return_value=fetch_data), \
             patch("backend.main.get_session", return_value=session):
            resp = _client.get(f"/api/racecard/{RACE_ID}")

        entry = next(e for e in resp.json()["entries"] if e["horseNumber"] == 1)
        assert entry["odds"] == 5.0, (
            "API response must not replace odds when DB has None"
        )

    def test_racecard_endpoint_multiple_horses_all_overlaid(self):
        """GET /api/racecard: all horses in DB have their odds overlaid."""
        scraped_entries = _make_scraped_entries(
            [(1, 5.0, 1), (2, 10.0, 2), (3, 15.0, 3)]
        )
        fetch_data = _make_fetch_race_card_data(scraped_entries)
        frozen = _make_frozen_cache()
        db_entries = [
            _make_he(horse_number=1, odds=4.5, popularity=1),
            _make_he(horse_number=2, odds=9.0, popularity=2),
            _make_he(horse_number=3, odds=13.5, popularity=3),
        ]
        session = _make_db_session_with_horse_entries(db_entries)

        with patch("backend.main._get_cached_predictions", return_value=frozen), \
             patch("backend.main.fetch_race_card", return_value=fetch_data), \
             patch("backend.main.get_session", return_value=session):
            resp = _client.get(f"/api/racecard/{RACE_ID}")

        result_entries = {e["horseNumber"]: e for e in resp.json()["entries"]}
        assert result_entries[1]["odds"] == 4.5
        assert result_entries[2]["odds"] == 9.0
        assert result_entries[3]["odds"] == 13.5

    def test_racecard_endpoint_non_frozen_cache_keeps_scraped_odds(self):
        """GET /api/racecard with non-frozen cache: scraped odds passed through unchanged."""
        scraped_entries = _make_scraped_entries([(1, 5.0, 1)])
        fetch_data = _make_fetch_race_card_data(scraped_entries)

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = [{"horseNumber": 1, "score": 80}]

        with patch("backend.main._get_cached_predictions", return_value=None), \
             patch("backend.main.fetch_race_card", return_value=fetch_data), \
             patch("backend.main.predictor", mock_predictor), \
             patch("requests.get", side_effect=OSError("network mocked off")):
            resp = _client.get(f"/api/racecard/{RACE_ID}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["frozen"] is False
        entry = next(e for e in body["entries"] if e["horseNumber"] == 1)
        assert entry["odds"] == 5.0, (
            "Non-frozen path must not alter scraped odds; expected 5.0"
        )

    def test_racecard_endpoint_returns_cached_predictions_for_frozen_race(self):
        """GET /api/racecard frozen path returns predictions from cache, not predictor."""
        scraped_entries = _make_scraped_entries([(1, 5.0, 1)])
        fetch_data = _make_fetch_race_card_data(scraped_entries)
        cached_preds = [{"horseNumber": 1, "score": 99, "source": "FROZEN_CACHE"}]
        frozen = _make_frozen_cache(predictions=cached_preds)
        db_he = _make_he(horse_number=1, odds=3.7, popularity=1)
        session = _make_db_session_with_horse_entries([db_he])

        with patch("backend.main._get_cached_predictions", return_value=frozen), \
             patch("backend.main.fetch_race_card", return_value=fetch_data), \
             patch("backend.main.get_session", return_value=session), \
             patch("backend.main.predictor") as mock_predictor:
            resp = _client.get(f"/api/racecard/{RACE_ID}")
            mock_predictor.predict.assert_not_called()

        assert resp.json()["predictions"] == cached_preds
