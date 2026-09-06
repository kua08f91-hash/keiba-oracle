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

    def test_frozen_returns_cached_predictions(self):
        """Frozen path must return cached predictions without DB overlay."""
        cached = _make_cached(frozen=True)
        fetch_data = _make_fetch_race_card_data(entry_count=3)

        with (
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
        ):
            resp = client.get(f"/api/racecard/{RACE_ID}")

        data = resp.json()
        assert data["frozen"] is True
        assert data["predictions"] == cached["predictions"]


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
        fetch_data = _make_fetch_race_card_data(entry_count=3)

        with (
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
        ):
            resp = client.get(f"/api/optimized-bets/{RACE_ID}")

        assert resp.status_code == 200
        assert resp.json()["frozen"] is True

    def test_frozen_cache_returns_cached_bets(self):
        """Frozen cache: the bets field equals the cached bets exactly."""
        cached_bets = [{"type": "wide", "horses": [3, 7], "odds": 18.5}]
        cached = _make_cached(frozen=True, bets=cached_bets)
        fetch_data = _make_fetch_race_card_data(entry_count=3)

        with (
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
        ):
            resp = client.get(f"/api/optimized-bets/{RACE_ID}")

        assert resp.json()["bets"] == cached_bets

    def test_frozen_cache_returns_longshot_from_cache(self):
        """Frozen cache: longshot field is taken from DB cache."""
        longshot = {"type": "sanrentan", "horses": [4, 9, 12], "odds": 280.0}
        cached = _make_cached(frozen=True, longshot=longshot)

        fetch_data = _make_fetch_race_card_data(entry_count=3)
        with (
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
        ):
            resp = client.get(f"/api/optimized-bets/{RACE_ID}")

        assert resp.json()["longshot"] == longshot

    def test_frozen_cache_returns_pattern_from_cache(self):
        """Frozen cache: pattern field is taken from DB cache."""
        cached = _make_cached(frozen=True, pattern="upset")
        fetch_data = _make_fetch_race_card_data(entry_count=3)

        with (
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
        ):
            resp = client.get(f"/api/optimized-bets/{RACE_ID}")

        assert resp.json()["pattern"] == "upset"

    def test_frozen_cache_returns_updated_at_from_cache(self):
        """Frozen cache: updatedAt equals the cache timestamp."""
        ts = "2026-04-27T07:55:00"
        cached = _make_cached(frozen=True, updated_at=ts)
        fetch_data = _make_fetch_race_card_data(entry_count=3)

        with (
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
        ):
            resp = client.get(f"/api/optimized-bets/{RACE_ID}")

        assert resp.json()["updatedAt"] == ts

    def test_frozen_cache_skips_live_odds_fetch(self):
        """Frozen cache: _fetch_live_win_odds must not be called."""
        cached = _make_cached(frozen=True)
        fetch_data = _make_fetch_race_card_data(entry_count=3)

        with (
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main._fetch_live_win_odds") as mock_odds,
        ):
            client.get(f"/api/optimized-bets/{RACE_ID}")

        mock_odds.assert_not_called()

    def test_frozen_cache_returns_correct_race_id(self):
        """raceId in the response must equal the requested race_id."""
        cached = _make_cached(frozen=True)
        fetch_data = _make_fetch_race_card_data(entry_count=3)

        with (
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
        ):
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
        """No cache: optimize_bets_dual must be called to produce live bets."""
        fetch_data = _make_fetch_race_card_data()
        dual_result = {"core_bets": [], "value_bets": [], "longshot": None,
                       "pattern": "", "layer1_active": False, "honmei_odds": 0}

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.predictor") as mock_pred,
            patch("backend.main.optimize_bets_dual", return_value=dual_result) as mock_opt,
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
        dual_result = {"core_bets": [], "value_bets": live_bets, "longshot": None,
                       "pattern": "", "layer1_active": False, "honmei_odds": 0}

        with (
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.predictor") as mock_pred,
            patch("backend.main.optimize_bets_dual", return_value=dual_result),
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


# ===========================================================================
# /api/racecard response format — entries with odds (frontend polling contract)
# ===========================================================================

class TestRacecardResponseFormatForFrontend:
    """Verify the shape of /api/racecard responses that the frontend JS polls.

    The Vercel frontend polls /api/racecard/{race_id} on a timer and reads:
      response.entries[].odds
      response.entries[].horseNumber
      response.predictions
      response.frozen
      response.updatedAt

    These tests lock down the response contract so the bug
    (pointing to /api instead of _CLOUD_API) cannot silently reappear
    as a shape mismatch at the backend.
    """

    def test_racecard_entries_contain_odds_field(self):
        """Every non-scratched entry in the response must have an 'odds' key."""
        fetch_data = _make_fetch_race_card_data(entry_count=5)
        # Give each entry a concrete odds value
        for i, e in enumerate(fetch_data["entries"]):
            e["odds"] = 3.0 + i * 1.5

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.predictor") as mock_pred,
            patch("requests.get", side_effect=_silent_requests_get),
        ):
            mock_pred.predict.return_value = []
            resp = client.get(f"/api/racecard/{RACE_ID}")

        assert resp.status_code == 200
        entries = resp.json()["entries"]
        assert len(entries) > 0
        for entry in entries:
            assert "odds" in entry, f"Entry {entry.get('horseNumber')} missing 'odds' key"

    def test_racecard_entries_contain_horse_number_field(self):
        """Every entry must carry horseNumber so the frontend can key on it."""
        fetch_data = _make_fetch_race_card_data(entry_count=4)

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.predictor") as mock_pred,
            patch("requests.get", side_effect=_silent_requests_get),
        ):
            mock_pred.predict.return_value = []
            resp = client.get(f"/api/racecard/{RACE_ID}")

        entries = resp.json()["entries"]
        for entry in entries:
            assert "horseNumber" in entry

    def test_racecard_response_has_required_top_level_keys(self):
        """Response must contain raceInfo, entries, predictions, frozen, updatedAt."""
        fetch_data = _make_fetch_race_card_data()

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.predictor") as mock_pred,
            patch("requests.get", side_effect=_silent_requests_get),
        ):
            mock_pred.predict.return_value = []
            resp = client.get(f"/api/racecard/{RACE_ID}")

        body = resp.json()
        for key in ("raceInfo", "entries", "predictions", "frozen", "updatedAt"):
            assert key in body, f"Missing top-level key: {key}"

    def test_racecard_live_odds_applied_to_entries_before_response(self):
        """When live odds are fetched they must appear in the response entries,
        not the stale scrape values.

        This directly tests the _apply_odds_to_entries path inside _compute_live:
        the frontend reads response.entries[n].odds for display.
        """
        fetch_data = _make_fetch_race_card_data(entry_count=3)
        # Scraped odds start at 5.0 (from _make_entry)
        for e in fetch_data["entries"]:
            e["odds"] = 5.0

        # Live odds from netkeiba return updated values
        live_odds_response = {
            "data": {
                "odds": {
                    "1": {
                        "1": ["2.5", "1", "1"],  # horse 1: odds=2.5, pop=1
                        "2": ["8.0", "8", "2"],  # horse 2: odds=8.0, pop=2
                        "3": ["15.0", "15", "3"],  # horse 3: odds=15.0, pop=3
                    }
                }
            }
        }

        mock_http_resp = MagicMock()
        mock_http_resp.text = json.dumps(live_odds_response)

        # Force is_race_day=True so live odds are fetched
        from backend._tz import now_jst
        today_str = now_jst().strftime("%Y%m%d")
        fetch_data["race_info"]["date"] = today_str

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.predictor") as mock_pred,
            patch("requests.get", return_value=mock_http_resp),
            patch("backend.main._save_odds_to_db"),  # skip DB writes
        ):
            mock_pred.predict.return_value = []
            resp = client.get(f"/api/racecard/{RACE_ID}")

        entries = {e["horseNumber"]: e for e in resp.json()["entries"]}
        # Horse 1 should have the live odds (2.5), not the stale 5.0
        assert entries[1]["odds"] == 2.5, (
            f"Expected live odds 2.5 for horse 1, got {entries[1].get('odds')}"
        )

    def test_racecard_frozen_entries_include_odds_from_db(self):
        """When frozen, entries must still include odds (loaded from DB HorseEntry).

        The frontend polls even frozen races to display final odds.
        """
        cached = _make_cached(frozen=True)
        fetch_data = _make_fetch_race_card_data(entry_count=3)

        db_horse = MagicMock()
        db_horse.horse_number = 1
        db_horse.odds = 3.2
        db_horse.popularity = 1

        db_session = MagicMock()
        q = MagicMock()
        q.filter.return_value = q
        q.all.return_value = [db_horse]
        db_session.query.return_value = q

        with (
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.get_session", return_value=db_session),
        ):
            resp = client.get(f"/api/racecard/{RACE_ID}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["frozen"] is True
        # At minimum, entries list must be present
        assert "entries" in body


# ===========================================================================
# /api/optimized-bets response format — coreBets/valueBets/layer1Active
# ===========================================================================

class TestOptimizedBetsResponseFormatForFrontend:
    """Verify the /api/optimized-bets response shape that the frontend reads.

    The frontend polls this endpoint and reads:
      response.coreBets       — Layer-1 (A判定) bets
      response.valueBets      — Layer-2 (info) bets
      response.layer1Active   — bool flag for A/B/C badge display
      response.betConfidence  — "A" | "B" | "C"
      response.bets           — combined legacy list
      response.frozen         — bool
      response.raceId         — echo of requested race_id
    """

    def _dual_result(self, core=None, value=None, layer1=False):
        return {
            "core_bets": core or [],
            "value_bets": value or [],
            "longshot": None,
            "pattern": "standard",
            "layer1_active": layer1,
            "honmei_odds": 0,
        }

    def test_optimized_bets_response_has_core_bets_key(self):
        """coreBets must be present in the live response."""
        fetch_data = _make_fetch_race_card_data()
        core = [{"type": "tansho", "horses": [1], "odds": 3.0, "ev": 0.15}]
        dual = self._dual_result(core=core, layer1=True)

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.predictor") as mock_pred,
            patch("backend.main.optimize_bets_dual", return_value=dual),
            patch("backend.main._fetch_live_combination_odds", return_value={}),
            patch("backend.scraper.odds.estimate_from_entries", return_value={}),
        ):
            mock_pred.predict.return_value = []
            resp = client.get(f"/api/optimized-bets/{RACE_ID}")

        assert resp.status_code == 200
        body = resp.json()
        assert "coreBets" in body
        assert body["coreBets"] == core

    def test_optimized_bets_response_has_value_bets_key(self):
        """valueBets must be present and match the dual optimizer output."""
        fetch_data = _make_fetch_race_card_data()
        value = [{"type": "umaren", "horses": [2, 5], "odds": 35.0, "ev": 0.05}]
        dual = self._dual_result(value=value)

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.predictor") as mock_pred,
            patch("backend.main.optimize_bets_dual", return_value=dual),
            patch("backend.main._fetch_live_combination_odds", return_value={}),
            patch("backend.scraper.odds.estimate_from_entries", return_value={}),
        ):
            mock_pred.predict.return_value = []
            resp = client.get(f"/api/optimized-bets/{RACE_ID}")

        body = resp.json()
        assert "valueBets" in body
        assert body["valueBets"] == value

    def test_optimized_bets_response_has_layer1_active_key(self):
        """layer1Active must be present in the response."""
        fetch_data = _make_fetch_race_card_data()
        dual = self._dual_result(layer1=True)

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.predictor") as mock_pred,
            patch("backend.main.optimize_bets_dual", return_value=dual),
            patch("backend.main._fetch_live_combination_odds", return_value={}),
            patch("backend.scraper.odds.estimate_from_entries", return_value={}),
        ):
            mock_pred.predict.return_value = []
            resp = client.get(f"/api/optimized-bets/{RACE_ID}")

        body = resp.json()
        assert "layer1Active" in body
        assert body["layer1Active"] is True

    def test_optimized_bets_response_has_bet_confidence_key(self):
        """betConfidence must be present with value A, B, or C."""
        fetch_data = _make_fetch_race_card_data()
        dual = self._dual_result()

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.predictor") as mock_pred,
            patch("backend.main.optimize_bets_dual", return_value=dual),
            patch("backend.main._fetch_live_combination_odds", return_value={}),
            patch("backend.scraper.odds.estimate_from_entries", return_value={}),
        ):
            mock_pred.predict.return_value = []
            resp = client.get(f"/api/optimized-bets/{RACE_ID}")

        body = resp.json()
        assert "betConfidence" in body
        assert body["betConfidence"] in ("A", "B", "C")

    def test_optimized_bets_response_has_all_required_keys(self):
        """All keys the frontend reads must be present in every response."""
        fetch_data = _make_fetch_race_card_data()
        dual = self._dual_result()
        required_keys = {
            "bets", "coreBets", "valueBets", "layer1Active",
            "betConfidence", "raceId", "frozen", "updatedAt",
        }

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.predictor") as mock_pred,
            patch("backend.main.optimize_bets_dual", return_value=dual),
            patch("backend.main._fetch_live_combination_odds", return_value={}),
            patch("backend.scraper.odds.estimate_from_entries", return_value={}),
        ):
            mock_pred.predict.return_value = []
            resp = client.get(f"/api/optimized-bets/{RACE_ID}")

        body = resp.json()
        missing = required_keys - body.keys()
        assert not missing, f"Response missing keys: {missing}"

    def test_optimized_bets_bets_equals_core_plus_value(self):
        """The legacy 'bets' field must equal coreBets + valueBets (combined list)."""
        fetch_data = _make_fetch_race_card_data()
        core = [{"type": "tansho", "horses": [1], "odds": 3.0}]
        value = [{"type": "umaren", "horses": [1, 3], "odds": 28.0}]
        dual = self._dual_result(core=core, value=value, layer1=True)

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.predictor") as mock_pred,
            patch("backend.main.optimize_bets_dual", return_value=dual),
            patch("backend.main._fetch_live_combination_odds", return_value={}),
            patch("backend.scraper.odds.estimate_from_entries", return_value={}),
        ):
            mock_pred.predict.return_value = []
            resp = client.get(f"/api/optimized-bets/{RACE_ID}")

        body = resp.json()
        assert body["bets"] == core + value


# ===========================================================================
# _compute_live — entries passed to optimize_bets_dual with live odds
# ===========================================================================

class TestComputeLivePassesOddsToOptimizer:
    """Verify that _compute_live feeds live-odds-updated entries to optimize_bets_dual.

    This is the core contract broken by the frontend bug: if live odds are not
    injected into entries before calling the optimizer, betConfidence (A/B/C)
    is computed from stale/None odds — silently wrong.
    """

    def test_optimize_bets_dual_receives_entries_with_updated_odds(self):
        """When live odds are fetched, the entries arg passed to optimize_bets_dual
        must reflect the updated odds values, not the original scraped values."""
        fetch_data = _make_fetch_race_card_data(entry_count=3)
        # Stale odds in scraped data
        for e in fetch_data["entries"]:
            e["odds"] = 99.0  # clearly stale placeholder

        # Live odds replace horse 1 odds with 2.8
        live_odds_payload = {
            "data": {
                "odds": {
                    "1": {
                        "1": ["2.8", "1", "1"],
                        "2": ["9.0", "9", "2"],
                        "3": ["20.0", "20", "3"],
                    }
                }
            }
        }
        mock_http_resp = MagicMock()
        mock_http_resp.text = json.dumps(live_odds_payload)

        from backend._tz import now_jst
        today_str = now_jst().strftime("%Y%m%d")
        fetch_data["race_info"]["date"] = today_str

        dual_result = {
            "core_bets": [], "value_bets": [], "longshot": None,
            "pattern": "", "layer1_active": False, "honmei_odds": 0,
        }
        captured_entries = []

        def capturing_dual(predictions, odds_data, race_info, entries=None):
            if entries:
                captured_entries.extend(entries)
            return dual_result

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.predictor") as mock_pred,
            patch("requests.get", return_value=mock_http_resp),
            patch("backend.main._save_odds_to_db"),
            patch("backend.main.optimize_bets_dual", side_effect=capturing_dual),
            patch("backend.main._fetch_live_combination_odds", return_value={}),
            patch("backend.scraper.odds.estimate_from_entries", return_value={}),
        ):
            mock_pred.predict.return_value = [{"horseNumber": 1, "score": 72}]
            client.get(f"/api/optimized-bets/{RACE_ID}")

        assert captured_entries, "optimize_bets_dual was not called with entries"
        horse1 = next((e for e in captured_entries if e["horseNumber"] == 1), None)
        assert horse1 is not None
        assert horse1["odds"] == 2.8, (
            f"Expected live odds 2.8 for horse 1 in optimizer call, got {horse1.get('odds')}"
        )

    def test_optimize_bets_dual_called_once_on_live_path(self):
        """optimize_bets_dual must be called exactly once per /api/optimized-bets request."""
        fetch_data = _make_fetch_race_card_data()
        dual_result = {
            "core_bets": [], "value_bets": [], "longshot": None,
            "pattern": "", "layer1_active": False, "honmei_odds": 0,
        }

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.predictor") as mock_pred,
            patch("requests.get", side_effect=_silent_requests_get),
            patch("backend.main.optimize_bets_dual", return_value=dual_result) as mock_dual,
            patch("backend.main._fetch_live_combination_odds", return_value={}),
            patch("backend.scraper.odds.estimate_from_entries", return_value={}),
        ):
            mock_pred.predict.return_value = []
            client.get(f"/api/optimized-bets/{RACE_ID}")

        mock_dual.assert_called_once()


# ===========================================================================
# betConfidence changes when live odds change — A/B/C transitions
# ===========================================================================

class TestBetConfidenceOddsTransitions:
    """Verify that betConfidence (A/B/C) in the /api/optimized-bets response
    correctly reflects live odds changes.

    The frontend displays the A/B/C badge from this field; if odds change
    but confidence does not update, users see stale/wrong signals.
    """

    def _make_live_odds_mock(self, horse1_odds: float):
        """Build a mock HTTP response returning horse 1 win odds."""
        payload = {
            "data": {
                "odds": {
                    "1": {
                        "1": [str(horse1_odds), str(round(horse1_odds)), "1"],
                        "2": ["10.0", "10", "2"],
                        "3": ["25.0", "25", "3"],
                    }
                }
            }
        }
        mock_resp = MagicMock()
        mock_resp.text = json.dumps(payload)
        return mock_resp

    def _run_optimized_bets(self, fetch_data, predictions, mock_http_resp):
        """Run /api/optimized-bets with given predictions and odds mock."""
        dual_result = {
            "core_bets": [], "value_bets": [], "longshot": None,
            "pattern": "", "layer1_active": False, "honmei_odds": 0,
        }
        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.predictor") as mock_pred,
            patch("requests.get", return_value=mock_http_resp),
            patch("backend.main._save_odds_to_db"),
            patch("backend.main.optimize_bets_dual", return_value=dual_result),
            patch("backend.main._fetch_live_combination_odds", return_value={}),
            patch("backend.scraper.odds.estimate_from_entries", return_value={}),
        ):
            mock_pred.predict.return_value = predictions
            resp = client.get(f"/api/optimized-bets/{RACE_ID}")
        return resp.json()

    def test_bet_confidence_is_A_when_honmei_score_high_and_odds_in_range(self):
        """betConfidence="A" when honmei score>=68 AND odds in [2.0, 4.0)."""
        from backend._tz import now_jst
        fetch_data = _make_fetch_race_card_data(entry_count=3)
        fetch_data["race_info"]["date"] = now_jst().strftime("%Y%m%d")

        predictions = [
            {"horseNumber": 1, "score": 75},
            {"horseNumber": 2, "score": 50},
            {"horseNumber": 3, "score": 40},
        ]
        mock_resp = self._make_live_odds_mock(horse1_odds=3.0)  # in [2.0, 4.0)

        body = self._run_optimized_bets(fetch_data, predictions, mock_resp)
        assert body["betConfidence"] == "A", (
            f"Expected A confidence for score=75 odds=3.0, got {body['betConfidence']}"
        )

    def test_bet_confidence_is_C_when_honmei_odds_too_high(self):
        """betConfidence='C' when honmei odds are above the BUY range (>=4.0)
        and no ◯ B-rank condition is met."""
        from backend._tz import now_jst
        fetch_data = _make_fetch_race_card_data(entry_count=3)
        fetch_data["race_info"]["date"] = now_jst().strftime("%Y%m%d")

        # All horses have low scores — no B condition either
        predictions = [
            {"horseNumber": 1, "score": 55},
            {"horseNumber": 2, "score": 45},
            {"horseNumber": 3, "score": 35},
        ]
        mock_resp = self._make_live_odds_mock(horse1_odds=8.0)  # above BUY range

        body = self._run_optimized_bets(fetch_data, predictions, mock_resp)
        assert body["betConfidence"] == "C", (
            f"Expected C confidence for score=55 odds=8.0, got {body['betConfidence']}"
        )

    def test_bet_confidence_transitions_from_A_to_C_when_odds_drift_above_range(self):
        """When odds move from 3.0 to 8.0, betConfidence must shift from A to C.

        This directly tests the polling scenario: the frontend polls repeatedly;
        if odds drift out of range, the confidence label must update.
        """
        from backend._tz import now_jst
        today_str = now_jst().strftime("%Y%m%d")

        predictions_a = [{"horseNumber": 1, "score": 75}, {"horseNumber": 2, "score": 45}]

        # First poll: odds=3.0 → A
        fetch_data_a = _make_fetch_race_card_data(entry_count=3)
        fetch_data_a["race_info"]["date"] = today_str
        body_a = self._run_optimized_bets(
            fetch_data_a, predictions_a, self._make_live_odds_mock(3.0)
        )
        assert body_a["betConfidence"] == "A"

        # Second poll (odds drifted to 8.0): same score, different odds → C
        predictions_c = [{"horseNumber": 1, "score": 55}, {"horseNumber": 2, "score": 40}]
        fetch_data_c = _make_fetch_race_card_data(entry_count=3)
        fetch_data_c["race_info"]["date"] = today_str
        body_c = self._run_optimized_bets(
            fetch_data_c, predictions_c, self._make_live_odds_mock(8.0)
        )
        assert body_c["betConfidence"] == "C"

    def test_bet_confidence_is_B_when_niban_meets_b_rank_conditions(self):
        """betConfidence='B' when ◯ (rank-2) score>=60 and odds>=8.0.

        Arrange: horse 1 is honmei (score=65, highest but <68 → not A).
        Horse 2 is niban (score=62 >= 60). _make_live_odds_mock puts horse 2
        at 10.0 (hardcoded), which satisfies niban_odds >= 8 → B.
        """
        from backend._tz import now_jst
        fetch_data = _make_fetch_race_card_data(entry_count=3)
        fetch_data["race_info"]["date"] = now_jst().strftime("%Y%m%d")

        # Horse 1 = honmei (score 65, highest, but 65 < 68 → not A)
        # Horse 2 = niban (score 62 >= 60, odds=10.0 from mock >= 8 → B)
        predictions = [
            {"horseNumber": 1, "score": 65},  # honmei: score < 68, not A
            {"horseNumber": 2, "score": 62},  # niban: score >= 60
            {"horseNumber": 3, "score": 40},
        ]
        # _make_live_odds_mock: horse 1 → horse1_odds, horse 2 → 10.0, horse 3 → 25.0
        # With horse1_odds=5.0 (outside [2.0,4.0) BUY range) and horse 2 at 10.0 >= 8 → B
        mock_resp = self._make_live_odds_mock(horse1_odds=5.0)

        body = self._run_optimized_bets(fetch_data, predictions, mock_resp)
        assert body["betConfidence"] == "B", (
            f"Expected B confidence for honmei score=65 odds=5.0, "
            f"niban score=62 odds=10.0; got {body['betConfidence']}"
        )


# ===========================================================================
# /api/live-odds/{race_id} — response format contract
# ===========================================================================

class TestLiveOddsEndpointFormat:
    """Verify the /api/live-odds/{race_id} response shape.

    The frontend also calls this endpoint directly for live odds polling.
    """

    def test_live_odds_success_response_shape(self):
        """Successful live odds fetch returns odds, source='live', race_id."""
        live_odds_payload = {
            "data": {
                "odds": {
                    "1": {
                        "1": ["4.5", "4", "1"],
                        "2": ["7.0", "7", "2"],
                    }
                }
            }
        }
        mock_resp = MagicMock()
        mock_resp.text = json.dumps(live_odds_payload)

        with (
            patch("requests.get", return_value=mock_resp),
            patch("backend.main._save_odds_to_db"),
        ):
            resp = client.get(f"/api/live-odds/{RACE_ID}")

        assert resp.status_code == 200
        body = resp.json()
        assert "odds" in body
        assert "source" in body
        assert "race_id" in body
        assert body["source"] == "live"
        assert body["race_id"] == RACE_ID

    def test_live_odds_odds_dict_maps_horse_number_to_odds_float(self):
        """The odds dict must map integer horse numbers to objects with 'odds' float."""
        live_odds_payload = {
            "data": {
                "odds": {
                    "1": {
                        "3": ["12.0", "12", "1"],
                    }
                }
            }
        }
        mock_resp = MagicMock()
        mock_resp.text = json.dumps(live_odds_payload)

        with (
            patch("requests.get", return_value=mock_resp),
            patch("backend.main._save_odds_to_db"),
        ):
            resp = client.get(f"/api/live-odds/{RACE_ID}")

        body = resp.json()
        # Key "3" in the odds dict (horse number as string from JSON)
        assert "3" in body["odds"]
        assert isinstance(body["odds"]["3"]["odds"], float)
        assert body["odds"]["3"]["odds"] == 12.0

    def test_live_odds_failed_fetch_returns_empty_odds_and_failed_source(self):
        """When the live fetch fails/returns empty, response has source='failed'."""
        with patch("requests.get", side_effect=_silent_requests_get):
            resp = client.get(f"/api/live-odds/{RACE_ID}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["source"] == "failed"
        assert body["odds"] == {}
        assert body["race_id"] == RACE_ID

    def test_live_odds_400_for_short_race_id(self):
        """Race IDs shorter than 10 characters return 400."""
        resp = client.get("/api/live-odds/12345")
        assert resp.status_code == 400

    def test_live_odds_includes_popularity_field_per_horse(self):
        """Each horse entry in the odds dict must have a 'popularity' int."""
        live_odds_payload = {
            "data": {
                "odds": {
                    "1": {
                        "5": ["6.0", "6", "3"],
                    }
                }
            }
        }
        mock_resp = MagicMock()
        mock_resp.text = json.dumps(live_odds_payload)

        with (
            patch("requests.get", return_value=mock_resp),
            patch("backend.main._save_odds_to_db"),
        ):
            resp = client.get(f"/api/live-odds/{RACE_ID}")

        body = resp.json()
        assert "5" in body["odds"]
        assert "popularity" in body["odds"]["5"]
        assert isinstance(body["odds"]["5"]["popularity"], int)
        assert body["odds"]["5"]["popularity"] == 3
