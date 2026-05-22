"""TDD tests for live-odds-first logic in /api/racecard/ and /api/optimized-bets/.

Behaviour under test (introduced after freeze-gate refactor):

  /api/racecard/{race_id}
    - frozen cache   → return cached predictions, frozen=True
    - non-frozen     → live computation, frozen=False
    - no cache       → live computation, frozen=False
    - stale entries (< 5) → force_refresh called
    - headCount mismatch > 2 → force_refresh called

  /api/optimized-bets/{race_id}
    - frozen cache   → return cached bets, frozen=True
    - non-frozen     → live computation, frozen=False
    - no cache       → live computation, frozen=False

External I/O is fully mocked:
  - backend.main._get_cached_predictions
  - backend.main.fetch_race_card
  - backend.main.get_session       (DB session)
  - requests.get                   (netkeiba live-odds HTTP call)
  - backend.main.predictor.predict (ML model)
"""
from __future__ import annotations

import json
import sys
import os
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Shared test data helpers
# ---------------------------------------------------------------------------

RACE_ID = "202604050811"   # 12-digit valid race id

def _make_race_info(head_count: int = 8) -> dict:
    return {
        "raceId": RACE_ID,
        "raceName": "テストレース",
        "raceNumber": 8,
        "grade": None,
        "distance": 1800,
        "surface": "芝",
        "courseDetail": "右回り",
        "startTime": "15:00",
        "racecourseCode": "05",
        "date": "20260427",
        "headCount": head_count,
        "trackCondition": "良",
    }


def _make_entry(number: int, is_scratched: bool = False) -> dict:
    return {
        "horseNumber": number,
        "frameNumber": ((number - 1) // 2) + 1,
        "horseName": f"テストホース{number}",
        "horseId": f"20200100{number:02d}",
        "sireName": "ゴールドシップ",
        "damName": "テストダム",
        "broodmareSire": "",
        "age": "牡4",
        "weightCarried": 57.0,
        "jockeyName": "ルメール",
        "trainerName": "矢作",
        "horseWeight": "480(+2)",
        "odds": 5.0,
        "popularity": number,
        "isScratched": is_scratched,
        "pastRaces": [{"pos": 1, "track": "東京"}],
    }


def _make_entries(count: int = 8, scratched: list[int] | None = None) -> list:
    scratched = scratched or []
    return [_make_entry(i, is_scratched=(i in scratched)) for i in range(1, count + 1)]


def _make_fetch_race_card_data(
    head_count: int = 8, entry_count: int = 8, scratched: list[int] | None = None
) -> dict:
    return {
        "race_info": _make_race_info(head_count),
        "entries": _make_entries(entry_count, scratched),
    }


def _make_cached(
    frozen: bool,
    predictions: list | None = None,
    bets: list | None = None,
    longshot: dict | None = None,
    pattern: str = "standard",
    updated_at: str = "2026-04-27T09:00:00",
) -> dict:
    return {
        "predictions": predictions or [{"horseNumber": 1, "score": 80}],
        "bets": bets or [{"type": "umaren", "horses": [1, 2], "odds": 25.0}],
        "longshot": longshot,
        "pattern": pattern,
        "frozen": frozen,
        "updated_at": updated_at,
    }


def _make_mock_db_session(horse_entries: list | None = None) -> MagicMock:
    """Return a mock DB session compatible with the HorseEntry query in get_race_card."""
    session = MagicMock()
    query_chain = MagicMock()
    query_chain.filter.return_value = query_chain
    query_chain.all.return_value = horse_entries or []
    session.query.return_value = query_chain
    return session


def _silent_requests_get(*_args, **_kwargs):
    """Simulate a network error so the live-odds HTTP block is a no-op."""
    raise OSError("network mocked off")


# ---------------------------------------------------------------------------
# Import the app AFTER helpers are defined so patch targets resolve correctly
# ---------------------------------------------------------------------------

from backend.main import app   # noqa: E402

client = TestClient(app, raise_server_exceptions=True)


# ===========================================================================
# /api/racecard/{race_id} — frozen cache
# ===========================================================================

class TestRacecardFrozenCache:
    """When _get_cached_predictions returns frozen=True the endpoint must
    return the DB-locked predictions without touching the predictor."""

    def test_returns_200_with_frozen_true(self):
        """Frozen cache path returns HTTP 200 and frozen=True in payload."""
        cached = _make_cached(frozen=True)
        fetch_data = _make_fetch_race_card_data()
        db_session = _make_mock_db_session()

        with (
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.get_session", return_value=db_session),
        ):
            resp = client.get(f"/api/racecard/{RACE_ID}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["frozen"] is True

    def test_returns_cached_predictions_not_live(self):
        """Frozen path returns the cached predictions list, not a freshly computed one."""
        cached_preds = [{"horseNumber": 3, "score": 99, "source": "DB_CACHE"}]
        cached = _make_cached(frozen=True, predictions=cached_preds)
        fetch_data = _make_fetch_race_card_data()
        db_session = _make_mock_db_session()

        with (
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.get_session", return_value=db_session),
            patch("backend.main.predictor") as mock_predictor,
        ):
            resp = client.get(f"/api/racecard/{RACE_ID}")

        assert resp.json()["predictions"] == cached_preds
        mock_predictor.predict.assert_not_called()

    def test_returns_updated_at_from_cache(self):
        """updatedAt in the response equals the timestamp stored in the cache."""
        ts = "2026-04-27T08:30:00"
        cached = _make_cached(frozen=True, updated_at=ts)
        fetch_data = _make_fetch_race_card_data()
        db_session = _make_mock_db_session()

        with (
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.get_session", return_value=db_session),
        ):
            resp = client.get(f"/api/racecard/{RACE_ID}")

        assert resp.json()["updatedAt"] == ts

    def test_db_entry_odds_are_overlaid_onto_entries(self):
        """Frozen path must overlay DB HorseEntry.odds onto the returned entries."""
        cached = _make_cached(frozen=True)
        fetch_data = _make_fetch_race_card_data(entry_count=3)

        # Simulate two DB HorseEntry rows with fresh odds
        he1 = MagicMock()
        he1.horse_number = 1
        he1.odds = 3.7
        he1.popularity = 1

        he2 = MagicMock()
        he2.horse_number = 2
        he2.odds = 6.5
        he2.popularity = 2

        db_session = _make_mock_db_session(horse_entries=[he1, he2])

        with (
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.get_session", return_value=db_session),
        ):
            resp = client.get(f"/api/racecard/{RACE_ID}")

        entries = {e["horseNumber"]: e for e in resp.json()["entries"]}
        assert entries[1]["odds"] == pytest.approx(3.7)
        assert entries[1]["popularity"] == 1
        assert entries[2]["odds"] == pytest.approx(6.5)


# ===========================================================================
# /api/racecard/{race_id} — non-frozen / no cache → live computation
# ===========================================================================

class TestRacecardLiveComputation:
    """When cached is None or frozen=False the endpoint must run live computation."""

    def test_no_cache_calls_predictor_and_returns_frozen_false(self):
        """No DB cache: predictor.predict must be called, frozen=False."""
        fetch_data = _make_fetch_race_card_data()
        live_preds = [{"horseNumber": 1, "score": 75}]

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.predictor") as mock_pred,
            patch("requests.get", side_effect=_silent_requests_get),
        ):
            mock_pred.predict.return_value = live_preds
            resp = client.get(f"/api/racecard/{RACE_ID}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["frozen"] is False
        mock_pred.predict.assert_called_once()

    def test_non_frozen_cache_calls_predictor(self):
        """Cache exists but frozen=False: endpoint must ignore cache and compute live."""
        cached = _make_cached(frozen=False)
        fetch_data = _make_fetch_race_card_data()
        live_preds = [{"horseNumber": 2, "score": 88}]

        with (
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.predictor") as mock_pred,
            patch("requests.get", side_effect=_silent_requests_get),
        ):
            mock_pred.predict.return_value = live_preds
            resp = client.get(f"/api/racecard/{RACE_ID}")

        assert resp.status_code == 200
        assert resp.json()["frozen"] is False
        mock_pred.predict.assert_called_once()

    def test_live_path_returns_updated_at_none(self):
        """Live computation path always returns updatedAt=None."""
        fetch_data = _make_fetch_race_card_data()

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.predictor") as mock_pred,
            patch("requests.get", side_effect=_silent_requests_get),
        ):
            mock_pred.predict.return_value = []
            resp = client.get(f"/api/racecard/{RACE_ID}")

        assert resp.json()["updatedAt"] is None

    def test_live_path_returns_predictor_output_as_predictions(self):
        """The predictions field must equal the exact output of predictor.predict."""
        fetch_data = _make_fetch_race_card_data()
        expected = [{"horseNumber": 5, "score": 55, "rank": 1}]

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.predictor") as mock_pred,
            patch("requests.get", side_effect=_silent_requests_get),
        ):
            mock_pred.predict.return_value = expected
            resp = client.get(f"/api/racecard/{RACE_ID}")

        assert resp.json()["predictions"] == expected

    def test_racecard_404_when_fetch_returns_none(self):
        """If fetch_race_card returns None the endpoint returns 404."""
        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=None),
        ):
            resp = client.get(f"/api/racecard/{RACE_ID}")

        assert resp.status_code == 404

    def test_racecard_400_for_short_race_id(self):
        """Race IDs shorter than 10 characters return 400."""
        resp = client.get("/api/racecard/12345")
        assert resp.status_code == 400


# ===========================================================================
# /api/racecard — stale entry detection
# ===========================================================================

class TestRacecardStaleEntryDetection:
    """The endpoint must call fetch_race_card(force_refresh=True) when entries
    look stale (too few, or headCount mismatch > 2)."""

    def test_fewer_than_5_entries_triggers_force_refresh(self):
        """Only 4 entries returned: endpoint must call force_refresh once."""
        # First call returns 4 entries; second (force_refresh) returns 8
        stale_data = _make_fetch_race_card_data(head_count=8, entry_count=4)
        fresh_data = _make_fetch_race_card_data(head_count=8, entry_count=8)

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card",
                  side_effect=[stale_data, fresh_data]) as mock_frc,
            patch("backend.main.predictor") as mock_pred,
            patch("requests.get", side_effect=_silent_requests_get),
        ):
            mock_pred.predict.return_value = []
            resp = client.get(f"/api/racecard/{RACE_ID}")

        assert resp.status_code == 200
        assert mock_frc.call_count == 2
        # Second call must have force_refresh=True
        _, kwargs = mock_frc.call_args
        assert kwargs.get("force_refresh") is True

    def test_exactly_5_entries_does_not_trigger_force_refresh(self):
        """Exactly 5 entries is the lower threshold; no refresh needed."""
        normal_data = _make_fetch_race_card_data(head_count=5, entry_count=5)

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=normal_data) as mock_frc,
            patch("backend.main.predictor") as mock_pred,
            patch("requests.get", side_effect=_silent_requests_get),
        ):
            mock_pred.predict.return_value = []
            client.get(f"/api/racecard/{RACE_ID}")

        mock_frc.assert_called_once()

    def test_head_count_mismatch_greater_than_2_triggers_force_refresh(self):
        """headCount=12, non-scratched=8 → diff=4 > 2 → force_refresh."""
        # 10 total entries, none scratched, but headCount says 14 → diff 4 > 2
        stale_data = _make_fetch_race_card_data(head_count=14, entry_count=10)
        fresh_data = _make_fetch_race_card_data(head_count=14, entry_count=14)

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card",
                  side_effect=[stale_data, fresh_data]) as mock_frc,
            patch("backend.main.predictor") as mock_pred,
            patch("requests.get", side_effect=_silent_requests_get),
        ):
            mock_pred.predict.return_value = []
            resp = client.get(f"/api/racecard/{RACE_ID}")

        assert resp.status_code == 200
        assert mock_frc.call_count == 2
        _, kwargs = mock_frc.call_args
        assert kwargs.get("force_refresh") is True

    def test_head_count_mismatch_of_2_does_not_trigger_force_refresh(self):
        """headCount=10, non-scratched=8 → diff=2, not > 2, no refresh."""
        normal_data = _make_fetch_race_card_data(head_count=10, entry_count=8)

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=normal_data) as mock_frc,
            patch("backend.main.predictor") as mock_pred,
            patch("requests.get", side_effect=_silent_requests_get),
        ):
            mock_pred.predict.return_value = []
            client.get(f"/api/racecard/{RACE_ID}")

        mock_frc.assert_called_once()

    def test_scratched_horses_excluded_from_head_count_comparison(self):
        """Scratched horses must not count toward non_scratched for mismatch check.

        8 entries, 2 scratched → 6 non_scratched; headCount=8 → diff=2, no refresh.
        """
        data = _make_fetch_race_card_data(head_count=8, entry_count=8, scratched=[7, 8])

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=data) as mock_frc,
            patch("backend.main.predictor") as mock_pred,
            patch("requests.get", side_effect=_silent_requests_get),
        ):
            mock_pred.predict.return_value = []
            client.get(f"/api/racecard/{RACE_ID}")

        mock_frc.assert_called_once()

    def test_force_refresh_result_is_used_in_response(self):
        """After force_refresh the fresh entries must appear in the response."""
        stale_data = _make_fetch_race_card_data(head_count=8, entry_count=3)
        # Fresh result has 8 entries with horse numbers 1-8
        fresh_data = _make_fetch_race_card_data(head_count=8, entry_count=8)

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card",
                  side_effect=[stale_data, fresh_data]),
            patch("backend.main.predictor") as mock_pred,
            patch("requests.get", side_effect=_silent_requests_get),
        ):
            mock_pred.predict.return_value = []
            resp = client.get(f"/api/racecard/{RACE_ID}")

        assert len(resp.json()["entries"]) == 8

    def test_head_count_zero_suppresses_mismatch_check(self):
        """headCount=0 means unknown; mismatch check must be skipped, no refresh."""
        # 8 entries, headCount=0 → (0 > 0) is False so no mismatch branch
        data = _make_fetch_race_card_data(head_count=0, entry_count=8)

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=data) as mock_frc,
            patch("backend.main.predictor") as mock_pred,
            patch("requests.get", side_effect=_silent_requests_get),
        ):
            mock_pred.predict.return_value = []
            client.get(f"/api/racecard/{RACE_ID}")

        mock_frc.assert_called_once()


# ===========================================================================
# /api/optimized-bets/{race_id} — frozen cache
# ===========================================================================

class TestOptimizedBetsFrozenCache:
    """When cache is frozen the /api/optimized-bets/ endpoint returns cached bets."""

    def test_frozen_cache_returns_200_with_frozen_true(self):
        """Frozen cache: HTTP 200 with frozen=True."""
        cached = _make_cached(frozen=True)

        with patch("backend.main._get_cached_predictions", return_value=cached):
            resp = client.get(f"/api/optimized-bets/{RACE_ID}")

        assert resp.status_code == 200
        assert resp.json()["frozen"] is True

    def test_frozen_cache_returns_cached_bets(self):
        """Frozen cache: the bets field equals the cached bets exactly."""
        cached_bets = [{"type": "wide", "horses": [3, 7], "odds": 18.5}]
        cached = _make_cached(frozen=True, bets=cached_bets)

        with patch("backend.main._get_cached_predictions", return_value=cached):
            resp = client.get(f"/api/optimized-bets/{RACE_ID}")

        assert resp.json()["bets"] == cached_bets

    def test_frozen_cache_returns_longshot_from_cache(self):
        """Frozen cache: longshot field is taken from DB cache."""
        longshot = {"type": "sanrentan", "horses": [4, 9, 12], "odds": 280.0}
        cached = _make_cached(frozen=True, longshot=longshot)

        with patch("backend.main._get_cached_predictions", return_value=cached):
            resp = client.get(f"/api/optimized-bets/{RACE_ID}")

        assert resp.json()["longshot"] == longshot

    def test_frozen_cache_returns_pattern_from_cache(self):
        """Frozen cache: pattern field is taken from DB cache."""
        cached = _make_cached(frozen=True, pattern="upset")

        with patch("backend.main._get_cached_predictions", return_value=cached):
            resp = client.get(f"/api/optimized-bets/{RACE_ID}")

        assert resp.json()["pattern"] == "upset"

    def test_frozen_cache_returns_updated_at_from_cache(self):
        """Frozen cache: updatedAt equals the cache timestamp."""
        ts = "2026-04-27T07:55:00"
        cached = _make_cached(frozen=True, updated_at=ts)

        with patch("backend.main._get_cached_predictions", return_value=cached):
            resp = client.get(f"/api/optimized-bets/{RACE_ID}")

        assert resp.json()["updatedAt"] == ts

    def test_frozen_cache_does_not_call_fetch_race_card(self):
        """Frozen cache: fetch_race_card (live scrape) must not be called."""
        cached = _make_cached(frozen=True)

        with (
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main.fetch_race_card") as mock_frc,
        ):
            client.get(f"/api/optimized-bets/{RACE_ID}")

        mock_frc.assert_not_called()

    def test_frozen_cache_returns_correct_race_id(self):
        """raceId in the response must equal the requested race_id."""
        cached = _make_cached(frozen=True)

        with patch("backend.main._get_cached_predictions", return_value=cached):
            resp = client.get(f"/api/optimized-bets/{RACE_ID}")

        assert resp.json()["raceId"] == RACE_ID


# ===========================================================================
# /api/optimized-bets/{race_id} — live computation
# ===========================================================================

class TestOptimizedBetsLiveComputation:
    """When cache is absent or not frozen the endpoint must compute live bets."""

    def _patch_live_bets(self, cached_val, bets_return=None):
        """Context-manager stack for live-bet tests."""
        from contextlib import ExitStack
        from unittest.mock import patch as _patch

        fetch_data = _make_fetch_race_card_data()
        bets_return = bets_return or []

        stack = ExitStack()
        stack.enter_context(
            _patch("backend.main._get_cached_predictions", return_value=cached_val)
        )
        mock_frc = stack.enter_context(
            _patch("backend.main.fetch_race_card", return_value=fetch_data)
        )
        mock_pred = stack.enter_context(_patch("backend.main.predictor"))
        mock_pred.predict.return_value = []
        mock_opt = stack.enter_context(
            _patch("backend.main.optimize_bets", return_value=bets_return)
        )
        stack.enter_context(
            _patch("backend.main._fetch_live_combination_odds", return_value={})
        )
        stack.enter_context(
            _patch("backend.main.estimate_from_entries", return_value={},
                   create=True)
        )
        return stack, mock_frc, mock_pred, mock_opt

    def test_no_cache_calls_optimize_bets(self):
        """No cache: optimize_bets must be called to produce live bets."""
        fetch_data = _make_fetch_race_card_data()

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.predictor") as mock_pred,
            patch("backend.main.optimize_bets", return_value=[]) as mock_opt,
            patch("backend.main._fetch_live_combination_odds", return_value={}),
            patch("backend.scraper.odds.estimate_from_entries", return_value={}),
        ):
            mock_pred.predict.return_value = []
            resp = client.get(f"/api/optimized-bets/{RACE_ID}")

        assert resp.status_code == 200
        assert resp.json()["frozen"] is False
        mock_opt.assert_called_once()

    def test_non_frozen_cache_ignores_cache_and_computes_live(self):
        """Cache frozen=False: live computation, not cached bets."""
        cached = _make_cached(frozen=False, bets=[{"type": "sanrentan"}])
        fetch_data = _make_fetch_race_card_data()
        live_bets = [{"type": "umaren", "horses": [1, 3], "odds": 30.0}]

        with (
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.predictor") as mock_pred,
            patch("backend.main.optimize_bets", return_value=live_bets),
            patch("backend.main._fetch_live_combination_odds", return_value={}),
            patch("backend.scraper.odds.estimate_from_entries", return_value={}),
        ):
            mock_pred.predict.return_value = []
            resp = client.get(f"/api/optimized-bets/{RACE_ID}")

        body = resp.json()
        assert body["frozen"] is False
        assert body["bets"] == live_bets

    def test_live_path_returns_updated_at_none(self):
        """Live computation: updatedAt must be None."""
        fetch_data = _make_fetch_race_card_data()

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.predictor") as mock_pred,
            patch("backend.main.optimize_bets", return_value=[]),
            patch("backend.main._fetch_live_combination_odds", return_value={}),
            patch("backend.scraper.odds.estimate_from_entries", return_value={}),
        ):
            mock_pred.predict.return_value = []
            resp = client.get(f"/api/optimized-bets/{RACE_ID}")

        assert resp.json()["updatedAt"] is None

    def test_optimized_bets_404_when_fetch_returns_none(self):
        """fetch_race_card returning None → 404 from the live path."""
        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=None),
        ):
            resp = client.get(f"/api/optimized-bets/{RACE_ID}")

        assert resp.status_code == 404

    def test_optimized_bets_400_for_short_race_id(self):
        """Race IDs shorter than 10 characters return 400."""
        resp = client.get("/api/optimized-bets/123456789")
        assert resp.status_code == 400

    def test_live_path_returns_race_id_in_response(self):
        """raceId in the live response must equal the requested race_id."""
        fetch_data = _make_fetch_race_card_data()

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.predictor") as mock_pred,
            patch("backend.main.optimize_bets", return_value=[]),
            patch("backend.main._fetch_live_combination_odds", return_value={}),
            patch("backend.scraper.odds.estimate_from_entries", return_value={}),
        ):
            mock_pred.predict.return_value = []
            resp = client.get(f"/api/optimized-bets/{RACE_ID}")

        assert resp.json()["raceId"] == RACE_ID


# ===========================================================================
# _get_cached_predictions contract (unit)
# ===========================================================================

class TestGetCachedPredictionsUnit:
    """Unit tests for the _get_cached_predictions helper to verify it correctly
    maps DB rows to the dict the endpoint logic expects."""

    def _mock_session_with_cache(self, cache_obj):
        session = MagicMock()
        query_chain = MagicMock()
        query_chain.filter.return_value = query_chain
        query_chain.first.return_value = cache_obj
        session.query.return_value = query_chain
        return session

    def test_returns_none_when_no_cache_row(self):
        from backend.main import _get_cached_predictions
        session = self._mock_session_with_cache(None)
        with patch("backend.main.get_session", return_value=session):
            result = _get_cached_predictions(RACE_ID)
        assert result is None

    def test_returns_none_when_predictions_json_empty(self):
        from backend.main import _get_cached_predictions
        cache = MagicMock()
        cache.predictions_json = ""
        session = self._mock_session_with_cache(cache)
        with patch("backend.main.get_session", return_value=session):
            result = _get_cached_predictions(RACE_ID)
        assert result is None

    def test_frozen_true_propagated(self):
        from backend.main import _get_cached_predictions
        cache = MagicMock()
        cache.predictions_json = json.dumps([{"horseNumber": 1, "score": 80}])
        cache.bets_json = json.dumps([])
        cache.longshot_json = None
        cache.pattern = "standard"
        cache.frozen = True
        cache.updated_at = None
        session = self._mock_session_with_cache(cache)
        with patch("backend.main.get_session", return_value=session):
            result = _get_cached_predictions(RACE_ID)
        assert result is not None
        assert result["frozen"] is True

    def test_frozen_false_propagated(self):
        from backend.main import _get_cached_predictions
        cache = MagicMock()
        cache.predictions_json = json.dumps([{"horseNumber": 1, "score": 70}])
        cache.bets_json = json.dumps([])
        cache.longshot_json = None
        cache.pattern = ""
        cache.frozen = False
        cache.updated_at = None
        session = self._mock_session_with_cache(cache)
        with patch("backend.main.get_session", return_value=session):
            result = _get_cached_predictions(RACE_ID)
        assert result is not None
        assert result["frozen"] is False

    def test_bets_json_deserialised(self):
        from backend.main import _get_cached_predictions
        bets = [{"type": "wide", "horses": [2, 5], "odds": 12.0}]
        cache = MagicMock()
        cache.predictions_json = json.dumps([])
        cache.bets_json = json.dumps(bets)
        cache.longshot_json = None
        cache.pattern = ""
        cache.frozen = True
        cache.updated_at = None
        session = self._mock_session_with_cache(cache)
        with patch("backend.main.get_session", return_value=session):
            result = _get_cached_predictions(RACE_ID)
        assert result["bets"] == bets

    def test_missing_bets_json_returns_empty_list(self):
        from backend.main import _get_cached_predictions
        cache = MagicMock()
        cache.predictions_json = json.dumps([{"horseNumber": 1}])
        cache.bets_json = None
        cache.longshot_json = None
        cache.pattern = ""
        cache.frozen = True
        cache.updated_at = None
        session = self._mock_session_with_cache(cache)
        with patch("backend.main.get_session", return_value=session):
            result = _get_cached_predictions(RACE_ID)
        assert result["bets"] == []

    def test_db_exception_returns_none(self):
        """Any DB error must be swallowed and return None."""
        from backend.main import _get_cached_predictions
        session = MagicMock()
        session.query.side_effect = Exception("DB connection lost")
        with patch("backend.main.get_session", return_value=session):
            result = _get_cached_predictions(RACE_ID)
        assert result is None
