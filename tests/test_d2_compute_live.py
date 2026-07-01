"""TDD tests for D2 strategy parameters in optimize_bets() and _compute_live() in main.py.

Section 1: D2 Strategy Parameters
  - umaren boundary at 15.0x (accepted) and 14.9x (rejected)
  - AI top-7 filtering: rank 6/7 included, rank 8 excluded
  - MAX_BETS=5 still enforced
  - TYPE_MAX=2 per type still enforced

Section 2: _compute_live() behavior
  - Returns None when fetch_race_card returns None
  - Returns frozen cached data when cache is frozen
  - Calls _fetch_live_win_odds on race day
  - Does NOT call _fetch_live_win_odds on non-race-day with odds present
  - Calls _fetch_live_win_odds when odds are missing (regardless of date)
  - include_bets=True includes bets, longshot, pattern in response
  - include_bets=False returns empty bets
  - betConfidence is included in non-frozen response
  - Auto-freeze triggers when _should_auto_freeze returns True
"""
from __future__ import annotations

import os
import sys
import pytest
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_predictions(horse_configs):
    """Build a predictions list from (horseNumber, score) tuples."""
    return [
        {"horseNumber": hn, "score": score, "isScratched": False}
        for hn, score in horse_configs
    ]


def _odds_entry(horses, odds):
    """Build a single odds entry dict."""
    return {"horses": horses, "odds": float(odds), "payout": int(odds * 100)}


def _race_info(head_count=16):
    """Build a minimal race_info dict."""
    return {
        "raceId": "202606030201",
        "headCount": head_count,
        "date": "20260620",
    }


def _make_race_card_data(race_date="20260620", head_count=16, n_horses=8):
    """Build a minimal fetch_race_card() response dict."""
    entries = [
        {
            "horseNumber": i,
            "score": 0,
            "isScratched": False,
            "odds": float(i * 2),
            "popularity": i,
        }
        for i in range(1, n_horses + 1)
    ]
    return {
        "race_info": {
            "raceId": "202606030201",
            "headCount": head_count,
            "date": race_date,
        },
        "entries": entries,
    }


# ======================================================================
# Section 1: D2 Strategy Parameters in optimize_bets()
# ======================================================================

class TestD2UmarenBoundary:
    """D2 changed umaren lower bound from 20.0x to 15.0x.

    Tests that 15.0x is now accepted (was rejected under D1) and 14.9x
    is still correctly rejected as below the minimum.
    """

    # Predictions with 7 horses so AI top-7 covers all
    _PREDICTIONS = _make_predictions(
        [(1, 90), (2, 80), (3, 70), (4, 60), (5, 50), (6, 40), (7, 30)]
    )

    def test_umaren_at_15x_is_accepted(self):
        """umaren odds = 15.0 is exactly at the D2 lower bound → bet selected."""
        from backend.predictor.bet_optimizer import optimize_bets

        odds_data = {
            "umaren": [_odds_entry([1, 2], 15.0)],
            "umatan": [],
            "wide": [],
        }
        bets = optimize_bets(self._PREDICTIONS, odds_data, _race_info(), mc_samples=100)

        umaren_bets = [b for b in bets if b["type"] == "umaren"]
        assert len(umaren_bets) >= 1, (
            "umaren at 15.0x should be accepted under D2 (lower bound lowered from 20x to 15x)"
        )
        assert umaren_bets[0]["odds"] == 15.0

    def test_umaren_at_14_9x_is_rejected(self):
        """umaren odds = 14.9 is below 15.0 lower bound → not selected."""
        from backend.predictor.bet_optimizer import optimize_bets

        odds_data = {
            "umaren": [_odds_entry([1, 2], 14.9)],
            "umatan": [],
            "wide": [],
        }
        bets = optimize_bets(self._PREDICTIONS, odds_data, _race_info(), mc_samples=100)

        umaren_bets = [b for b in bets if b["type"] == "umaren"]
        assert len(umaren_bets) == 0, (
            "umaren at 14.9x is below D2 minimum (15.0x) and should be rejected"
        )

    def test_umaren_at_20x_still_accepted(self):
        """umaren at 20.0x remains inside [15.0, 30.0) → still accepted."""
        from backend.predictor.bet_optimizer import optimize_bets

        odds_data = {
            "umaren": [_odds_entry([1, 2], 20.0)],
            "umatan": [],
            "wide": [],
        }
        bets = optimize_bets(self._PREDICTIONS, odds_data, _race_info(), mc_samples=100)

        umaren_bets = [b for b in bets if b["type"] == "umaren"]
        assert len(umaren_bets) >= 1, "20.0x was valid under D1 and should remain valid under D2"

    def test_umaren_at_30x_is_rejected(self):
        """umaren upper bound is exclusive: 30.0x → rejected."""
        from backend.predictor.bet_optimizer import optimize_bets

        odds_data = {
            "umaren": [_odds_entry([1, 2], 30.0)],
            "umatan": [],
            "wide": [],
        }
        bets = optimize_bets(self._PREDICTIONS, odds_data, _race_info(), mc_samples=100)

        umaren_bets = [b for b in bets if b["type"] == "umaren"]
        assert len(umaren_bets) == 0, "umaren at exactly 30.0x should be rejected (exclusive upper bound)"

    def test_umaren_at_29_9x_is_accepted(self):
        """umaren at 29.9x is just below 30.0 upper exclusive bound → accepted."""
        from backend.predictor.bet_optimizer import optimize_bets

        odds_data = {
            "umaren": [_odds_entry([1, 2], 29.9)],
            "umatan": [],
            "wide": [],
        }
        bets = optimize_bets(self._PREDICTIONS, odds_data, _race_info(), mc_samples=100)

        umaren_bets = [b for b in bets if b["type"] == "umaren"]
        assert len(umaren_bets) >= 1, "29.9x is inside [15.0, 30.0) → should be accepted"


class TestD2TopNFiltering:
    """D2 expanded AI candidate pool from top-5 to top-7.

    Tests that horses ranked 6 and 7 (AI rank, by score) are included
    as valid candidates, and horse ranked 8 is excluded.
    """

    def _build_predictions_8_horses(self):
        """8 horses ranked by score: rank 1 = score 90, rank 8 = score 10."""
        return _make_predictions([
            (1, 90), (2, 80), (3, 70), (4, 60),
            (5, 50), (6, 40), (7, 30), (8, 10),
        ])

    def test_ai_rank_6_horse_included_in_umatan_candidates(self):
        """Horse ranked 6th by AI score can appear in selected umatan bets."""
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = self._build_predictions_8_horses()
        # Horse 1 (rank 1) → horse 6 (rank 6): a valid ◎-anchor umatan
        odds_data = {
            "umatan": [_odds_entry([1, 6], 80.0)],   # rank6 as 2nd horse
            "umaren": [],
            "wide": [],
        }
        bets = optimize_bets(predictions, odds_data, _race_info(), mc_samples=100)

        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) >= 1, (
            "Rank-6 horse should be included in D2 top-7 candidate pool"
        )
        horses_in_bets = [h for b in umatan_bets for h in b["horses"]]
        assert 6 in horses_in_bets, "Horse 6 (AI rank 6) should appear in selected umatan bet"

    def test_ai_rank_7_horse_included_in_umatan_candidates(self):
        """Horse ranked 7th by AI score can appear in selected umatan bets."""
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = self._build_predictions_8_horses()
        odds_data = {
            "umatan": [_odds_entry([1, 7], 90.0)],   # rank7 as 2nd horse
            "umaren": [],
            "wide": [],
        }
        bets = optimize_bets(predictions, odds_data, _race_info(), mc_samples=100)

        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) >= 1, (
            "Rank-7 horse should be included in D2 top-7 candidate pool"
        )
        horses_in_bets = [h for b in umatan_bets for h in b["horses"]]
        assert 7 in horses_in_bets, "Horse 7 (AI rank 7) should appear in selected umatan bet"

    def test_ai_rank_8_horse_excluded_from_candidates(self):
        """Horse ranked 8th by AI score must NOT appear in any selected bets."""
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = self._build_predictions_8_horses()
        # Only provide odds that involve rank-8 horse
        odds_data = {
            "umatan": [_odds_entry([1, 8], 200.0)],   # rank8 as 2nd horse
            "umaren": [],
            "wide": [],
        }
        bets = optimize_bets(predictions, odds_data, _race_info(), mc_samples=100)

        horses_in_bets = [h for b in bets for h in b["horses"]]
        assert 8 not in horses_in_bets, (
            "Horse 8 (AI rank 8) is outside D2 top-7 and must not appear in any bet"
        )

    def test_rank_8_only_odds_returns_no_bets(self):
        """When only rank-8 horse appears in odds, no bets are selected."""
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = self._build_predictions_8_horses()
        odds_data = {
            "umatan": [_odds_entry([8, 7], 150.0)],  # both not ◎, rank8 present
            "umaren": [_odds_entry([7, 8], 20.0)],
            "wide": [],
        }
        bets = optimize_bets(predictions, odds_data, _race_info(), mc_samples=100)

        # Rank-8 should not be in any bet
        horses_in_bets = [h for b in bets for h in b["horses"]]
        assert 8 not in horses_in_bets, (
            "Horse 8 must never be included even when it appears in odds"
        )


class TestD2MaxBetsEnforced:
    """MAX_BETS=5 still enforced under D2 strategy."""

    def test_at_most_5_bets_returned(self):
        """optimize_bets never returns more than 5 bets under D2."""
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = _make_predictions([
            (1, 90), (2, 80), (3, 70), (4, 60), (5, 50), (6, 40), (7, 30)
        ])
        # Provide many valid candidates across all types
        odds_data = {
            "umatan": [
                _odds_entry([1, 2], 80.0),
                _odds_entry([1, 3], 120.0),
                _odds_entry([2, 3], 100.0),
            ],
            "umaren": [
                _odds_entry([1, 4], 15.5),
                _odds_entry([2, 4], 18.0),
                _odds_entry([3, 5], 22.0),
            ],
            "wide": [
                _odds_entry([1, 5], 12.0),
                _odds_entry([2, 5], 14.0),
                _odds_entry([3, 6], 11.0),
            ],
        }
        bets = optimize_bets(predictions, odds_data, _race_info(), mc_samples=100)

        assert len(bets) <= 5, f"D2 must still respect MAX_BETS=5, got {len(bets)}"

    def test_ranks_are_sequential_from_one(self):
        """Bet ranks are 1, 2, 3, ... (sequential, no gaps)."""
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = _make_predictions([
            (1, 90), (2, 80), (3, 70), (4, 60), (5, 50), (6, 40), (7, 30)
        ])
        odds_data = {
            "umatan": [_odds_entry([1, 2], 80.0), _odds_entry([1, 3], 120.0)],
            "umaren": [_odds_entry([1, 4], 15.5)],
            "wide": [_odds_entry([1, 5], 12.0)],
        }
        bets = optimize_bets(predictions, odds_data, _race_info(), mc_samples=100)

        for i, bet in enumerate(bets):
            assert bet["rank"] == i + 1, (
                f"Bet at index {i} should have rank {i + 1}, got {bet['rank']}"
            )


class TestD2TypeMaxEnforced:
    """TYPE_MAX=2 per bet type still enforced under D2 strategy."""

    def test_at_most_2_umatan_bets(self):
        """No more than 2 umatan bets even when 3+ valid candidates exist."""
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = _make_predictions([
            (1, 90), (2, 80), (3, 70), (4, 60), (5, 50), (6, 40), (7, 30)
        ])
        odds_data = {
            "umatan": [
                _odds_entry([1, 2], 60.0),
                _odds_entry([1, 3], 80.0),
                _odds_entry([1, 4], 100.0),
            ],
            "umaren": [],
            "wide": [],
        }
        bets = optimize_bets(predictions, odds_data, _race_info(), mc_samples=100)

        umatan_count = sum(1 for b in bets if b["type"] == "umatan")
        assert umatan_count <= 2, (
            f"TYPE_MAX=2: expected at most 2 umatan bets, got {umatan_count}"
        )

    def test_at_most_2_umaren_bets(self):
        """No more than 2 umaren bets even when 3+ valid candidates exist."""
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = _make_predictions([
            (1, 90), (2, 80), (3, 70), (4, 60), (5, 50), (6, 40), (7, 30)
        ])
        odds_data = {
            "umatan": [],
            "umaren": [
                _odds_entry([1, 2], 15.5),
                _odds_entry([1, 3], 18.0),
                _odds_entry([2, 3], 22.0),
            ],
            "wide": [],
        }
        bets = optimize_bets(predictions, odds_data, _race_info(), mc_samples=100)

        umaren_count = sum(1 for b in bets if b["type"] == "umaren")
        assert umaren_count <= 2, (
            f"TYPE_MAX=2: expected at most 2 umaren bets, got {umaren_count}"
        )

    def test_at_most_2_wide_bets(self):
        """No more than 2 wide bets even when 3+ valid candidates exist."""
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = _make_predictions([
            (1, 90), (2, 80), (3, 70), (4, 60), (5, 50), (6, 40), (7, 30)
        ])
        odds_data = {
            "umatan": [],
            "umaren": [],
            "wide": [
                _odds_entry([1, 2], 12.0),
                _odds_entry([1, 3], 15.0),
                _odds_entry([2, 3], 20.0),
            ],
        }
        bets = optimize_bets(predictions, odds_data, _race_info(), mc_samples=100)

        wide_count = sum(1 for b in bets if b["type"] == "wide")
        assert wide_count <= 2, (
            f"TYPE_MAX=2: expected at most 2 wide bets, got {wide_count}"
        )


# ======================================================================
# Section 2: _compute_live() tests
# ======================================================================

# Minimal prediction list returned by the mock predictor
_MOCK_PREDICTIONS = _make_predictions([
    (1, 90), (2, 80), (3, 70), (4, 60), (5, 50), (6, 40), (7, 30)
])

# Minimal odds_data returned by mock combination odds
_MOCK_ODDS_DATA = {
    "umatan": [_odds_entry([1, 2], 80.0)],
    "umaren": [_odds_entry([1, 2], 16.0)],
    "wide": [_odds_entry([1, 2], 12.0)],
}


def _make_mock_predictor(predictions=None):
    """Build a mock predictor.predict() that returns predictions."""
    mock = MagicMock()
    mock.predict.return_value = predictions or _MOCK_PREDICTIONS
    return mock


class TestComputeLiveReturnNone:
    """_compute_live returns None when fetch_race_card fails."""

    def test_returns_none_when_fetch_race_card_returns_none(self):
        """If fetch_race_card() returns None, _compute_live returns None."""
        from backend.main import _compute_live

        with patch("backend.main.fetch_race_card", return_value=None):
            result = _compute_live("202606030201", include_bets=False)

        assert result is None, (
            "_compute_live must return None when fetch_race_card returns None"
        )

    def test_returns_none_when_fetch_race_card_returns_empty_dict(self):
        """If fetch_race_card() returns falsy, _compute_live returns None."""
        from backend.main import _compute_live

        with patch("backend.main.fetch_race_card", return_value={}):
            result = _compute_live("202606030201", include_bets=False)

        assert result is None, (
            "_compute_live must return None when fetch_race_card returns empty dict"
        )


class TestComputeLiveFrozenCache:
    """_compute_live returns frozen cached data without re-computing."""

    def _make_frozen_cache(self):
        return {
            "predictions": _MOCK_PREDICTIONS,
            "bets": [{"type": "umatan", "horses": [1, 2], "odds": 80.0, "rank": 1}],
            "longshot": None,
            "pattern": "標準配置",
            "frozen": True,
            "updated_at": "2026-06-20T10:00:00",
        }

    def test_returns_frozen_data_when_cache_is_frozen(self):
        """When cache.frozen=True, return cached predictions without live fetch."""
        from backend.main import _compute_live

        data = _make_race_card_data()
        frozen = self._make_frozen_cache()

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=frozen), \
             patch("backend.main._fetch_live_win_odds") as mock_live:

            result = _compute_live("202606030201", include_bets=False)

        assert result is not None
        assert result["frozen"] is True, "Result should report frozen=True"
        assert result["predictions"] == _MOCK_PREDICTIONS, "Should return cached predictions"
        mock_live.assert_not_called(), "_fetch_live_win_odds must NOT be called for frozen races"

    def test_frozen_response_contains_bets_from_cache(self):
        """Frozen response includes bets from cache (even with include_bets=False)."""
        from backend.main import _compute_live

        data = _make_race_card_data()
        frozen = self._make_frozen_cache()

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=frozen):

            result = _compute_live("202606030201", include_bets=False)

        assert result["bets"] == frozen["bets"], "Frozen cache bets should pass through"

    def test_frozen_response_contains_entries_from_race_card(self):
        """Frozen response still includes entries from the race card (not cache)."""
        from backend.main import _compute_live

        data = _make_race_card_data()
        frozen = self._make_frozen_cache()

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=frozen):

            result = _compute_live("202606030201", include_bets=False)

        assert result["entries"] == data["entries"], "Entries should come from race card"


class TestComputeLiveFetchLiveWinOdds:
    """_compute_live calls _fetch_live_win_odds under the correct conditions."""

    def _base_patches(self, data, is_today=True, has_odds=True):
        """Return a dict of common patch targets for _compute_live tests."""
        patches = {
            "backend.main.fetch_race_card": data,
            "backend.main._get_cached_predictions": None,
            "backend.main._should_auto_freeze": False,
        }
        return patches

    def test_calls_live_win_odds_on_race_day(self):
        """On race day (date == today), _fetch_live_win_odds is called."""
        from backend.main import _compute_live

        today = "20260620"
        data = _make_race_card_data(race_date=today)
        # Remove odds from entries so no_odds > 0 too — race day is sufficient
        for e in data["entries"]:
            e["odds"] = None

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=None), \
             patch("backend.main._should_auto_freeze", return_value=False), \
             patch("backend.main.now_jst") as mock_now, \
             patch("backend.main._fetch_live_win_odds", return_value={}) as mock_live, \
             patch("backend.main.predictor", _make_mock_predictor()):

            mock_now.return_value.strftime.return_value = today
            _compute_live("202606030201", include_bets=False)

        mock_live.assert_called_once_with("202606030201")

    def test_does_not_call_live_win_odds_on_non_race_day_with_odds(self):
        """On non-race-day with all entries having odds, live odds are NOT fetched."""
        from backend.main import _compute_live

        race_date = "20260601"   # Past date, not today
        today = "20260620"
        data = _make_race_card_data(race_date=race_date)
        # Ensure all non-scratched entries have odds already set
        for e in data["entries"]:
            e["odds"] = 5.0
            e["isScratched"] = False

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=None), \
             patch("backend.main._should_auto_freeze", return_value=False), \
             patch("backend.main.now_jst") as mock_now, \
             patch("backend.main._fetch_live_win_odds") as mock_live, \
             patch("backend.main.predictor", _make_mock_predictor()):

            mock_now.return_value.strftime.return_value = today
            _compute_live("202606030201", include_bets=False)

        mock_live.assert_not_called(), (
            "_fetch_live_win_odds must NOT be called when not race-day and all entries have odds"
        )

    def test_calls_live_win_odds_when_odds_missing_regardless_of_date(self):
        """When any entry is missing odds, _fetch_live_win_odds is called regardless of date."""
        from backend.main import _compute_live

        race_date = "20260601"   # Not today
        today = "20260620"
        data = _make_race_card_data(race_date=race_date)
        # Make some entries missing odds
        data["entries"][0]["odds"] = None
        data["entries"][0]["isScratched"] = False
        # Others have odds
        for e in data["entries"][1:]:
            e["odds"] = 5.0

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=None), \
             patch("backend.main._should_auto_freeze", return_value=False), \
             patch("backend.main.now_jst") as mock_now, \
             patch("backend.main._fetch_live_win_odds", return_value={}) as mock_live, \
             patch("backend.main.predictor", _make_mock_predictor()):

            mock_now.return_value.strftime.return_value = today
            _compute_live("202606030201", include_bets=False)

        mock_live.assert_called_once_with("202606030201"), (
            "_fetch_live_win_odds must be called when entries are missing odds"
        )


class TestComputeLiveIncludeBets:
    """_compute_live respects the include_bets flag."""

    def _patches_no_bets_needed(self, data, today="20260620"):
        """Standard patch context for non-bet tests."""
        return [
            patch("backend.main.fetch_race_card", return_value=data),
            patch("backend.main._get_cached_predictions", return_value=None),
            patch("backend.main._should_auto_freeze", return_value=False),
            patch("backend.main.now_jst"),
            patch("backend.main._fetch_live_win_odds", return_value={}),
            patch("backend.main.predictor", _make_mock_predictor()),
        ]

    def test_include_bets_false_returns_empty_bets(self):
        """When include_bets=False and no auto-freeze, bets list is empty."""
        from backend.main import _compute_live

        data = _make_race_card_data(race_date="20260601")
        for e in data["entries"]:
            e["odds"] = 5.0

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=None), \
             patch("backend.main._should_auto_freeze", return_value=False), \
             patch("backend.main.now_jst") as mock_now, \
             patch("backend.main._fetch_live_win_odds", return_value={}), \
             patch("backend.main.predictor", _make_mock_predictor()):

            mock_now.return_value.strftime.return_value = "20260620"
            result = _compute_live("202606030201", include_bets=False)

        assert result is not None
        assert result["bets"] == [], (
            "include_bets=False with no auto-freeze should return empty bets list"
        )

    def test_include_bets_true_returns_bets_list(self):
        """When include_bets=True, bets list is populated (may be empty if no valid odds)."""
        from backend.main import _compute_live

        data = _make_race_card_data(race_date="20260601")
        for e in data["entries"]:
            e["odds"] = 5.0

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=None), \
             patch("backend.main._should_auto_freeze", return_value=False), \
             patch("backend.main.now_jst") as mock_now, \
             patch("backend.main._fetch_live_win_odds", return_value={}), \
             patch("backend.main._fetch_live_combination_odds", return_value=_MOCK_ODDS_DATA), \
             patch("backend.main.predictor", _make_mock_predictor()):

            mock_now.return_value.strftime.return_value = "20260620"
            result = _compute_live("202606030201", include_bets=True)

        assert result is not None
        assert isinstance(result["bets"], list), "bets should be a list when include_bets=True"

    def test_include_bets_true_contains_pattern_in_response(self):
        """When include_bets=True, pattern key is populated."""
        from backend.main import _compute_live

        data = _make_race_card_data(race_date="20260601")
        for e in data["entries"]:
            e["odds"] = 5.0

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=None), \
             patch("backend.main._should_auto_freeze", return_value=False), \
             patch("backend.main.now_jst") as mock_now, \
             patch("backend.main._fetch_live_win_odds", return_value={}), \
             patch("backend.main._fetch_live_combination_odds", return_value=_MOCK_ODDS_DATA), \
             patch("backend.main.predictor", _make_mock_predictor()):

            mock_now.return_value.strftime.return_value = "20260620"
            result = _compute_live("202606030201", include_bets=True)

        assert "pattern" in result, "Response must include pattern key"
        assert isinstance(result["pattern"], str), "pattern must be a string"

    def test_include_bets_true_contains_longshot_key(self):
        """When include_bets=True, longshot key is present in response."""
        from backend.main import _compute_live

        data = _make_race_card_data(race_date="20260601")
        for e in data["entries"]:
            e["odds"] = 5.0

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=None), \
             patch("backend.main._should_auto_freeze", return_value=False), \
             patch("backend.main.now_jst") as mock_now, \
             patch("backend.main._fetch_live_win_odds", return_value={}), \
             patch("backend.main._fetch_live_combination_odds", return_value=_MOCK_ODDS_DATA), \
             patch("backend.main.predictor", _make_mock_predictor()):

            mock_now.return_value.strftime.return_value = "20260620"
            result = _compute_live("202606030201", include_bets=True)

        assert "longshot" in result, "Response must include longshot key when include_bets=True"


class TestComputeLiveBetConfidence:
    """betConfidence is always present in non-frozen _compute_live responses."""

    def test_bet_confidence_in_response_include_bets_false(self):
        """betConfidence key is present even when include_bets=False."""
        from backend.main import _compute_live

        data = _make_race_card_data(race_date="20260601")
        for e in data["entries"]:
            e["odds"] = 5.0

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=None), \
             patch("backend.main._should_auto_freeze", return_value=False), \
             patch("backend.main.now_jst") as mock_now, \
             patch("backend.main._fetch_live_win_odds", return_value={}), \
             patch("backend.main.predictor", _make_mock_predictor()):

            mock_now.return_value.strftime.return_value = "20260620"
            result = _compute_live("202606030201", include_bets=False)

        assert result is not None
        assert "betConfidence" in result, (
            "betConfidence must be included in all non-frozen _compute_live responses"
        )

    def test_bet_confidence_value_is_a_b_or_c(self):
        """betConfidence value is one of 'A', 'B', or 'C'."""
        from backend.main import _compute_live

        data = _make_race_card_data(race_date="20260601")
        for e in data["entries"]:
            e["odds"] = 5.0

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=None), \
             patch("backend.main._should_auto_freeze", return_value=False), \
             patch("backend.main.now_jst") as mock_now, \
             patch("backend.main._fetch_live_win_odds", return_value={}), \
             patch("backend.main.predictor", _make_mock_predictor()):

            mock_now.return_value.strftime.return_value = "20260620"
            result = _compute_live("202606030201", include_bets=False)

        assert result["betConfidence"] in ("A", "B", "C"), (
            f"betConfidence must be A, B, or C, got {result['betConfidence']!r}"
        )

    def test_bet_confidence_in_response_include_bets_true(self):
        """betConfidence key is present when include_bets=True."""
        from backend.main import _compute_live

        data = _make_race_card_data(race_date="20260601")
        for e in data["entries"]:
            e["odds"] = 5.0

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=None), \
             patch("backend.main._should_auto_freeze", return_value=False), \
             patch("backend.main.now_jst") as mock_now, \
             patch("backend.main._fetch_live_win_odds", return_value={}), \
             patch("backend.main._fetch_live_combination_odds", return_value=_MOCK_ODDS_DATA), \
             patch("backend.main.predictor", _make_mock_predictor()):

            mock_now.return_value.strftime.return_value = "20260620"
            result = _compute_live("202606030201", include_bets=True)

        assert "betConfidence" in result
        assert result["betConfidence"] in ("A", "B", "C")


class TestComputeLiveAutoFreeze:
    """Auto-freeze triggers correctly when _should_auto_freeze returns True."""

    def test_auto_freeze_calls_auto_freeze_and_cache(self):
        """When _should_auto_freeze=True, _auto_freeze_and_cache is called."""
        from backend.main import _compute_live

        data = _make_race_card_data(race_date="20260601")
        for e in data["entries"]:
            e["odds"] = 5.0

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=None), \
             patch("backend.main._should_auto_freeze", return_value=True), \
             patch("backend.main.now_jst") as mock_now, \
             patch("backend.main._fetch_live_win_odds", return_value={}), \
             patch("backend.main._fetch_live_combination_odds", return_value=_MOCK_ODDS_DATA), \
             patch("backend.main._auto_freeze_and_cache") as mock_freeze, \
             patch("backend.main.predictor", _make_mock_predictor()):

            mock_now.return_value.strftime.return_value = "20260620"
            result = _compute_live("202606030201", include_bets=False)

        mock_freeze.assert_called_once(), (
            "_auto_freeze_and_cache must be called when _should_auto_freeze returns True"
        )

    def test_auto_freeze_called_with_race_id(self):
        """_auto_freeze_and_cache is called with the correct race_id."""
        from backend.main import _compute_live

        data = _make_race_card_data(race_date="20260601")
        for e in data["entries"]:
            e["odds"] = 5.0

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=None), \
             patch("backend.main._should_auto_freeze", return_value=True), \
             patch("backend.main.now_jst") as mock_now, \
             patch("backend.main._fetch_live_win_odds", return_value={}), \
             patch("backend.main._fetch_live_combination_odds", return_value=_MOCK_ODDS_DATA), \
             patch("backend.main._auto_freeze_and_cache") as mock_freeze, \
             patch("backend.main.predictor", _make_mock_predictor()):

            mock_now.return_value.strftime.return_value = "20260620"
            _compute_live("202606030201", include_bets=False)

        call_args = mock_freeze.call_args
        assert call_args is not None
        assert call_args[0][0] == "202606030201", (
            "_auto_freeze_and_cache must be called with the correct race_id"
        )

    def test_auto_freeze_not_called_when_should_auto_freeze_false(self):
        """_auto_freeze_and_cache is NOT called when _should_auto_freeze=False."""
        from backend.main import _compute_live

        data = _make_race_card_data(race_date="20260601")
        for e in data["entries"]:
            e["odds"] = 5.0

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=None), \
             patch("backend.main._should_auto_freeze", return_value=False), \
             patch("backend.main.now_jst") as mock_now, \
             patch("backend.main._fetch_live_win_odds", return_value={}), \
             patch("backend.main._auto_freeze_and_cache") as mock_freeze, \
             patch("backend.main.predictor", _make_mock_predictor()):

            mock_now.return_value.strftime.return_value = "20260620"
            _compute_live("202606030201", include_bets=False)

        mock_freeze.assert_not_called(), (
            "_auto_freeze_and_cache must not be called when auto-freeze threshold is not met"
        )

    def test_frozen_key_true_when_auto_freeze_triggers(self):
        """When auto-freeze triggers, result['frozen'] is True."""
        from backend.main import _compute_live

        data = _make_race_card_data(race_date="20260601")
        for e in data["entries"]:
            e["odds"] = 5.0

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=None), \
             patch("backend.main._should_auto_freeze", return_value=True), \
             patch("backend.main.now_jst") as mock_now, \
             patch("backend.main._fetch_live_win_odds", return_value={}), \
             patch("backend.main._fetch_live_combination_odds", return_value=_MOCK_ODDS_DATA), \
             patch("backend.main._auto_freeze_and_cache"), \
             patch("backend.main.predictor", _make_mock_predictor()):

            mock_now.return_value.strftime.return_value = "20260620"
            result = _compute_live("202606030201", include_bets=False)

        assert result["frozen"] is True, (
            "frozen should be True in response when auto-freeze triggers"
        )

    def test_frozen_key_false_when_no_auto_freeze_and_not_include_bets(self):
        """When auto-freeze does not trigger and include_bets=False, frozen=False."""
        from backend.main import _compute_live

        data = _make_race_card_data(race_date="20260601")
        for e in data["entries"]:
            e["odds"] = 5.0

        with patch("backend.main.fetch_race_card", return_value=data), \
             patch("backend.main._get_cached_predictions", return_value=None), \
             patch("backend.main._should_auto_freeze", return_value=False), \
             patch("backend.main.now_jst") as mock_now, \
             patch("backend.main._fetch_live_win_odds", return_value={}), \
             patch("backend.main.predictor", _make_mock_predictor()):

            mock_now.return_value.strftime.return_value = "20260620"
            result = _compute_live("202606030201", include_bets=False)

        assert result["frozen"] is False, (
            "frozen should be False when include_bets=False and auto-freeze not triggered"
        )
