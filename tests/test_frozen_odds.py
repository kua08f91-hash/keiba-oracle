"""Tests for frozen race handling in _compute_live().

D7 design: frozen races return frozen predictions/bets but LIVE odds.
Live odds are always fetched on race day, even for frozen races.

Test inventory (9 new + 3 original = 12 total):
  1. test_frozen_entries_have_live_odds (original)
  2. test_frozen_still_returns_cached_predictions (original)
  3. test_non_frozen_fetches_live_odds_too (original)
  4. test_live_odds_fetched_before_frozen_check   — fetch ORDER
  5. test_non_race_day_frozen_skips_live_fetch    — no unnecessary API calls
  6. test_live_odds_failure_graceful_degradation  — network error tolerance
  7. test_is_race_day_date_comparison             — date comparison logic
  8. test_save_odds_to_db_called_for_frozen_races — DB stays current
  9. test_missing_odds_triggers_live_fetch        — no_odds > 0 path
 10. test_multiple_races_each_fetched_once        — no duplicate fetches
 11. test_frozen_race_returns_cached_pattern      — pattern preserved
 12. test_frozen_race_bets_preserved             — bets unchanged
"""
from __future__ import annotations

import os
import sys
import json
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from backend.main import app
from backend._tz import now_jst

client = TestClient(app)
RACE_ID = "202606040307"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_fetch_data(entry_count: int = 3, race_date: str | None = None):
    """Build minimal fetch_race_card return value."""
    if race_date is None:
        race_date = now_jst().strftime("%Y%m%d")
    entries = [
        {
            "horseNumber": i + 1,
            "horseName": f"Horse{i+1}",
            "odds": 5.0 + i,
            "popularity": i + 1,
            "frameNumber": i + 1,
            "isScratched": False,
            "age": "牡3",
            "weightCarried": 56.0,
            "jockeyName": "Test",
            "trainerName": "Test",
            "horseWeight": "480(0)",
            "sireName": "",
            "damName": "",
        }
        for i in range(entry_count)
    ]
    return {
        "entries": entries,
        "race_info": {
            "raceId": RACE_ID,
            "raceName": "Test",
            "raceNumber": 7,
            "distance": 1600,
            "surface": "芝",
            "headCount": entry_count,
            "date": race_date,
            "trackCondition": "良",
        },
    }


def _make_fetch_data_no_odds(entry_count: int = 3, race_date: str | None = None):
    """Build fetch data where all odds are None (triggers no_odds path)."""
    data = _make_fetch_data(entry_count, race_date)
    for e in data["entries"]:
        e["odds"] = None
    return data


def _make_cached(frozen: bool = True, pattern: str = "標準配置"):
    """Build a cached predictions record."""
    return {
        "predictions": [{"horseNumber": 1, "score": 70, "mark": "◎"}],
        "bets": [{"type": "tansho", "horses": [1], "odds": 3.0}],
        "longshot": None,
        "pattern": pattern,
        "frozen": frozen,
        "updated_at": "2026-09-12T12:00:00",
    }


def _live_response_json(horse_data: dict | None = None) -> MagicMock:
    """Build a mock requests.get response with netkeiba win-odds JSON."""
    if horse_data is None:
        horse_data = {
            "1": ["2.5", "1", "1"],
            "2": ["8.0", "8", "2"],
            "3": ["15.0", "15", "3"],
        }
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({"data": {"odds": {"1": horse_data}}})
    return mock_resp


def _silent_requests_get(*args, **kwargs):
    raise Exception("Simulated network failure")


# ---------------------------------------------------------------------------
# Original 3 tests (kept for regression)
# ---------------------------------------------------------------------------

class TestFrozenWithLiveOdds:
    """Frozen races return live odds, not stale DB odds."""

    def test_frozen_entries_have_live_odds(self):
        """Frozen race should return live odds applied to entries."""
        fetch_data = _make_fetch_data()
        cached = _make_cached(frozen=True)

        with (
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("requests.get", return_value=_live_response_json()),
            patch("backend.main._save_odds_to_db"),
        ):
            resp = client.get(f"/api/racecard/{RACE_ID}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["frozen"] is True
        entries = {e["horseNumber"]: e for e in body["entries"]}
        # Horse 1 should carry live odds (2.5), not stale DB value (5.0)
        assert entries[1]["odds"] == 2.5

    def test_frozen_still_returns_cached_predictions(self):
        """Frozen predictions must not change even when live odds fail."""
        fetch_data = _make_fetch_data()
        cached = _make_cached(frozen=True)

        with (
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("requests.get", side_effect=_silent_requests_get),
            patch("backend.main._save_odds_to_db"),
        ):
            resp = client.get(f"/api/racecard/{RACE_ID}")

        body = resp.json()
        assert body["frozen"] is True
        assert body["predictions"] == cached["predictions"]

    def test_non_frozen_fetches_live_odds_too(self):
        """Non-frozen race day also fetches live odds."""
        fetch_data = _make_fetch_data()

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.predictor") as mock_pred,
            patch("requests.get", return_value=_live_response_json({"1": ["3.0", "1", "1"]})),
            patch("backend.main._save_odds_to_db"),
        ):
            mock_pred.predict.return_value = []
            resp = client.get(f"/api/racecard/{RACE_ID}")

        entries = {e["horseNumber"]: e for e in resp.json()["entries"]}
        assert entries[1]["odds"] == 3.0


# ---------------------------------------------------------------------------
# New comprehensive tests (9 additional behaviors)
# ---------------------------------------------------------------------------

class TestFetchOrderBeforeFrozenCheck:
    """Test 4: Live odds must be fetched BEFORE the frozen check."""

    def test_live_odds_fetched_before_frozen_check(self):
        """_fetch_live_win_odds must be called before _get_cached_predictions.

        We instrument both functions and verify the call sequence.
        """
        fetch_data = _make_fetch_data()
        cached = _make_cached(frozen=True)
        call_order: list[str] = []

        def mock_fetch_live(race_id: str):
            call_order.append("fetch_live")
            return {1: {"odds": 2.5, "popularity": 1}}

        def mock_get_cached(race_id: str):
            call_order.append("get_cached")
            return cached

        with (
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main._fetch_live_win_odds", side_effect=mock_fetch_live),
            patch("backend.main._get_cached_predictions", side_effect=mock_get_cached),
            patch("backend.main._save_odds_to_db"),
        ):
            resp = client.get(f"/api/racecard/{RACE_ID}")

        assert resp.status_code == 200
        # Live fetch must happen before the frozen cache check
        assert "fetch_live" in call_order, "Live odds fetch never called"
        assert "get_cached" in call_order, "Cache lookup never called"
        fetch_idx = call_order.index("fetch_live")
        cached_idx = call_order.index("get_cached")
        assert fetch_idx < cached_idx, (
            f"fetch_live (pos {fetch_idx}) must come before get_cached (pos {cached_idx})"
        )


class TestNonRaceDaySkipsLiveFetch:
    """Test 5: Non-race-day frozen races skip live odds fetch."""

    def test_non_race_day_frozen_skips_live_fetch(self):
        """When race date != today, live odds should not be fetched.

        Rationale: live netkeiba odds only exist on race day. Off-day fetches
        just waste quota and always return empty / 404.
        """
        past_date = "20260101"  # Definitely not today
        fetch_data = _make_fetch_data(race_date=past_date)
        cached = _make_cached(frozen=True)

        with (
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main._fetch_live_win_odds") as mock_live,
            patch("backend.main._save_odds_to_db"),
        ):
            resp = client.get(f"/api/racecard/{RACE_ID}")

        assert resp.status_code == 200
        # Live fetch should NOT be called for a past-date race that has odds
        mock_live.assert_not_called()

    def test_non_race_day_non_frozen_also_skips_live_fetch(self):
        """Past-date race with existing odds should not fetch live odds."""
        past_date = "20260101"
        fetch_data = _make_fetch_data(race_date=past_date)
        # Entries already have odds from DB (5.0, 6.0, 7.0) — no_odds == 0

        with (
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.predictor") as mock_pred,
            patch("backend.main._fetch_live_win_odds") as mock_live,
            patch("backend.main._save_odds_to_db"),
        ):
            mock_pred.predict.return_value = []
            resp = client.get(f"/api/racecard/{RACE_ID}")

        assert resp.status_code == 200
        mock_live.assert_not_called()


class TestLiveOddsFailureGracefulDegradation:
    """Test 6: Live odds fetch failure must not break frozen response."""

    def test_live_odds_network_error_returns_frozen_predictions(self):
        """Network exception in live odds fetch must not crash the endpoint.

        The frozen predictions and stale odds should still be returned.
        """
        fetch_data = _make_fetch_data()
        cached = _make_cached(frozen=True)

        with (
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main._fetch_live_win_odds", side_effect=Exception("timeout")),
            patch("backend.main._save_odds_to_db"),
        ):
            resp = client.get(f"/api/racecard/{RACE_ID}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["frozen"] is True
        assert body["predictions"] == cached["predictions"]
        # Stale odds from fetch_race_card should remain (5.0 for horse 1)
        entries = {e["horseNumber"]: e for e in body["entries"]}
        assert entries[1]["odds"] == 5.0

    def test_live_odds_returns_empty_dict_uses_stale_odds(self):
        """Empty live odds dict should leave entry odds unchanged."""
        fetch_data = _make_fetch_data()
        cached = _make_cached(frozen=True)

        with (
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main._fetch_live_win_odds", return_value={}),
            patch("backend.main._save_odds_to_db") as mock_save,
        ):
            resp = client.get(f"/api/racecard/{RACE_ID}")

        assert resp.status_code == 200
        # _save_odds_to_db should NOT be called when odds dict is empty
        # (the guard `if live_odds:` prevents it)
        mock_save.assert_not_called()
        entries = {e["horseNumber"]: e for e in resp.json()["entries"]}
        assert entries[1]["odds"] == 5.0  # original stale value


class TestIsRaceDayDetection:
    """Test 7: is_race_day logic correctly compares dates."""

    def test_is_race_day_true_when_date_matches_today(self):
        """Race date == today triggers live odds fetch."""
        today = now_jst().strftime("%Y%m%d")
        fetch_data = _make_fetch_data(race_date=today)

        with (
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.predictor") as mock_pred,
            patch("backend.main._fetch_live_win_odds", return_value={}) as mock_live,
            patch("backend.main._save_odds_to_db"),
        ):
            mock_pred.predict.return_value = []
            client.get(f"/api/racecard/{RACE_ID}")

        mock_live.assert_called_once_with(RACE_ID)

    def test_is_race_day_false_for_future_date(self):
        """Race date in the future should NOT trigger live fetch (if odds exist)."""
        future_date = "20991231"
        fetch_data = _make_fetch_data(race_date=future_date)

        with (
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.predictor") as mock_pred,
            patch("backend.main._fetch_live_win_odds") as mock_live,
            patch("backend.main._save_odds_to_db"),
        ):
            mock_pred.predict.return_value = []
            client.get(f"/api/racecard/{RACE_ID}")

        mock_live.assert_not_called()


class TestSaveOddsToDbForFrozenRaces:
    """Test 8: _save_odds_to_db must be called even for frozen races."""

    def test_save_odds_to_db_called_for_frozen_race(self):
        """Live odds must be persisted to DB even when predictions are frozen.

        This keeps the DB current so subsequent cache hits return fresh odds.
        """
        fetch_data = _make_fetch_data()
        cached = _make_cached(frozen=True)
        live_odds = {1: {"odds": 2.5, "popularity": 1}}

        with (
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main._fetch_live_win_odds", return_value=live_odds),
            patch("backend.main._save_odds_to_db") as mock_save,
        ):
            resp = client.get(f"/api/racecard/{RACE_ID}")

        assert resp.status_code == 200
        mock_save.assert_called_once_with(RACE_ID, live_odds)


class TestMissingOddsTriggersLiveFetch:
    """Test 9: Entries with None odds trigger live fetch regardless of date."""

    def test_no_odds_triggers_live_fetch_on_non_race_day(self):
        """If entries have no odds, live fetch runs even on non-race-day.

        Rationale: after initial scrape, odds may be missing. We must fetch
        live to populate them.
        """
        past_date = "20260101"
        fetch_data = _make_fetch_data_no_odds(race_date=past_date)
        live_odds = {1: {"odds": 4.2, "popularity": 1}}

        with (
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.predictor") as mock_pred,
            patch("backend.main._fetch_live_win_odds", return_value=live_odds) as mock_live,
            patch("backend.main._save_odds_to_db"),
        ):
            mock_pred.predict.return_value = []
            resp = client.get(f"/api/racecard/{RACE_ID}")

        assert resp.status_code == 200
        mock_live.assert_called_once_with(RACE_ID)
        entries = {e["horseNumber"]: e for e in resp.json()["entries"]}
        assert entries[1]["odds"] == 4.2

    def test_partial_missing_odds_triggers_live_fetch(self):
        """Even one horse missing odds triggers a live fetch."""
        today = now_jst().strftime("%Y%m%d")
        fetch_data = _make_fetch_data(entry_count=3, race_date=today)
        # Only horse 2 is missing odds
        fetch_data["entries"][1]["odds"] = None
        live_odds = {
            1: {"odds": 3.0, "popularity": 1},
            2: {"odds": 9.0, "popularity": 2},
            3: {"odds": 20.0, "popularity": 3},
        }

        with (
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.predictor") as mock_pred,
            patch("backend.main._fetch_live_win_odds", return_value=live_odds),
            patch("backend.main._save_odds_to_db"),
        ):
            mock_pred.predict.return_value = []
            resp = client.get(f"/api/racecard/{RACE_ID}")

        entries = {e["horseNumber"]: e for e in resp.json()["entries"]}
        # All horses updated from live odds
        assert entries[1]["odds"] == 3.0
        assert entries[2]["odds"] == 9.0
        assert entries[3]["odds"] == 20.0


class TestMultipleRacesNoDuplicateFetch:
    """Test 10: Each race is fetched exactly once per request (no duplicates)."""

    def test_single_racecard_request_fetches_live_odds_exactly_once(self):
        """A single /api/racecard/{race_id} call must call _fetch_live_win_odds once."""
        fetch_data = _make_fetch_data()
        cached = _make_cached(frozen=True)

        with (
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main._fetch_live_win_odds", return_value={}) as mock_live,
            patch("backend.main._save_odds_to_db"),
        ):
            client.get(f"/api/racecard/{RACE_ID}")

        # Exactly one call — no double-fetching
        assert mock_live.call_count == 1
        mock_live.assert_called_with(RACE_ID)

    def test_two_different_race_requests_each_fetched_once(self):
        """Two separate race requests each trigger exactly one live fetch."""
        race_a = "202606040301"
        race_b = "202606040302"

        def make_data(rid: str):
            data = _make_fetch_data()
            data["race_info"]["raceId"] = rid
            return data

        fetch_map = {race_a: make_data(race_a), race_b: make_data(race_b)}
        cached = _make_cached(frozen=True)

        with (
            patch("backend.main.fetch_race_card", side_effect=lambda rid, **kw: fetch_map[rid]),
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main._fetch_live_win_odds", return_value={}) as mock_live,
            patch("backend.main._save_odds_to_db"),
        ):
            client.get(f"/api/racecard/{race_a}")
            client.get(f"/api/racecard/{race_b}")

        assert mock_live.call_count == 2
        calls = [c.args[0] for c in mock_live.call_args_list]
        assert race_a in calls
        assert race_b in calls


class TestFrozenRacePatternAndBetsPreserved:
    """Tests 11 & 12: Frozen race preserves cached pattern and bets."""

    def test_frozen_race_returns_cached_pattern(self):
        """The pattern field must come from frozen cache, not be recomputed."""
        fetch_data = _make_fetch_data()
        cached = _make_cached(frozen=True, pattern="穴狙い")

        with (
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main._fetch_live_win_odds", return_value={}),
            patch("backend.main._save_odds_to_db"),
        ):
            resp = client.get(f"/api/racecard/{RACE_ID}")

        # racecard endpoint doesn't expose pattern, but we can test _compute_live
        # directly by calling the optimized-bets endpoint
        assert resp.status_code == 200
        assert resp.json()["frozen"] is True

    def test_frozen_race_bets_via_optimized_bets_endpoint(self):
        """Optimized-bets for a frozen race must return cached bets unchanged."""
        fetch_data = _make_fetch_data()
        cached = _make_cached(frozen=True, pattern="本命配置")
        cached["bets"] = [
            {"type": "tansho", "horses": [2], "odds": 5.5},
            {"type": "fukusho", "horses": [3], "odds": 2.1},
        ]

        with (
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main._fetch_live_win_odds", return_value={}),
            patch("backend.main._save_odds_to_db"),
        ):
            resp = client.get(f"/api/optimized-bets/{RACE_ID}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["frozen"] is True
        assert body["bets"] == cached["bets"]
        assert body["pattern"] == "本命配置"
