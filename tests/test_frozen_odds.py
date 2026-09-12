"""Tests for frozen race handling in _compute_live().

D7 design: frozen races return frozen predictions/bets but LIVE odds.
Live odds are always fetched on race day, even for frozen races.
"""
from __future__ import annotations

import os
import sys
import json
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)
RACE_ID = "202606040307"


def _make_fetch_data(entry_count=3):
    entries = [
        {"horseNumber": i + 1, "horseName": f"Horse{i+1}", "odds": 5.0 + i,
         "popularity": i + 1, "frameNumber": i + 1, "isScratched": False,
         "age": "牡3", "weightCarried": 56.0, "jockeyName": "Test",
         "trainerName": "Test", "horseWeight": "480(0)", "sireName": "", "damName": ""}
        for i in range(entry_count)
    ]
    return {
        "entries": entries,
        "race_info": {
            "raceId": RACE_ID, "raceName": "Test", "raceNumber": 7,
            "distance": 1600, "surface": "芝", "headCount": entry_count,
            "date": "20260912", "trackCondition": "良",
        },
    }


def _make_cached(frozen=True):
    return {
        "predictions": [{"horseNumber": 1, "score": 70, "mark": "◎"}],
        "bets": [{"type": "tansho", "horses": [1], "odds": 3.0}],
        "longshot": None,
        "pattern": "標準配置",
        "frozen": frozen,
        "updated_at": "2026-09-12T12:00:00",
    }


def _silent_requests_get(*args, **kwargs):
    raise Exception("No network")


class TestFrozenWithLiveOdds:
    """Frozen races return live odds, not stale DB odds."""

    def test_frozen_entries_have_live_odds(self):
        """Frozen race should return live odds applied to entries."""
        fetch_data = _make_fetch_data()
        cached = _make_cached(frozen=True)

        # Live odds from netkeiba
        live_response = MagicMock()
        live_response.text = json.dumps({
            "data": {"odds": {"1": {
                "1": ["2.5", "1", "1"],
                "2": ["8.0", "8", "2"],
                "3": ["15.0", "15", "3"],
            }}}
        })

        from backend._tz import now_jst
        fetch_data["race_info"]["date"] = now_jst().strftime("%Y%m%d")

        with (
            patch("backend.main._get_cached_predictions", return_value=cached),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("requests.get", return_value=live_response),
            patch("backend.main._save_odds_to_db"),
        ):
            resp = client.get(f"/api/racecard/{RACE_ID}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["frozen"] is True
        entries = {e["horseNumber"]: e for e in body["entries"]}
        # Should have live odds (2.5), not stale (5.0)
        assert entries[1]["odds"] == 2.5

    def test_frozen_still_returns_cached_predictions(self):
        """Frozen predictions/bets must NOT change even with live odds."""
        fetch_data = _make_fetch_data()
        cached = _make_cached(frozen=True)

        from backend._tz import now_jst
        fetch_data["race_info"]["date"] = now_jst().strftime("%Y%m%d")

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

        live_response = MagicMock()
        live_response.text = json.dumps({
            "data": {"odds": {"1": {
                "1": ["3.0", "1", "1"],
            }}}
        })

        from backend._tz import now_jst
        fetch_data["race_info"]["date"] = now_jst().strftime("%Y%m%d")

        with (
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main.fetch_race_card", return_value=fetch_data),
            patch("backend.main.predictor") as mock_pred,
            patch("requests.get", return_value=live_response),
            patch("backend.main._save_odds_to_db"),
        ):
            mock_pred.predict.return_value = []
            resp = client.get(f"/api/racecard/{RACE_ID}")

        entries = {e["horseNumber"]: e for e in resp.json()["entries"]}
        assert entries[1]["odds"] == 3.0
