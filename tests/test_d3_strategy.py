"""TDD tests for the D3 strategy change in bet_optimizer.py.

D3 change: umatan lower bound lowered from 50.0x (D2) to 30.0x (D3).
This closes the 30-50x dead zone and captures more ◎-win umatan cases.

Everything else unchanged from D2:
  - umaren: (15.0, 30.0) — inclusive lower, exclusive upper
  - wide:   (10.0, 30.0) — inclusive lower, exclusive upper
  - VALUE_AI_TOP_N = 7
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helper factories (same patterns as test_honmei_anchor.py)
# ---------------------------------------------------------------------------

def _make_predictions(horse_configs):
    """Build a predictions list.

    Args:
        horse_configs: list of (horseNumber, score) tuples.

    Returns:
        List of prediction dicts with horseNumber and score.
    """
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
    }


# Standard 7-horse predictions used throughout: horse 1 is ◎ (highest score)
_PREDICTIONS_7 = _make_predictions(
    [(1, 90), (2, 70), (3, 60), (4, 50), (5, 40), (6, 30), (7, 20)]
)


# ---------------------------------------------------------------------------
# D3 umatan boundary tests
# ---------------------------------------------------------------------------

class TestD3UmatanBoundaries:
    """D5: HONMEI_UMATAN_PARTNERS=0 — no umatan bets generated regardless of odds."""

    def _odds_data_umatan_only(self, odds_value):
        """Produce odds_data with a single umatan entry at the given odds."""
        return {
            "umatan": [_odds_entry([1, 2], odds_value)],
            "umaren": [],
            "wide": [],
        }

    # Test 1: 30.0x → no umatan (HONMEI_UMATAN_PARTNERS=0)
    def test_umatan_at_30x_is_accepted(self):
        from backend.predictor.bet_optimizer import optimize_bets

        bets = optimize_bets(
            _PREDICTIONS_7,
            self._odds_data_umatan_only(30.0),
            _race_info(),
            mc_samples=100,
        )
        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) == 1, (
            "D5: HONMEI_UMATAN_PARTNERS=1, expected 1 umatan bet"
        )

    # Test 2: 29.9x → no umatan (HONMEI_UMATAN_PARTNERS=0)
    def test_umatan_at_29_9x_is_accepted(self):
        from backend.predictor.bet_optimizer import optimize_bets

        bets = optimize_bets(
            _PREDICTIONS_7,
            self._odds_data_umatan_only(29.9),
            _race_info(),
            mc_samples=100,
        )
        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) == 1, (
            "D5: HONMEI_UMATAN_PARTNERS=1, expected 1 umatan bet"
        )

    # Test 3: 35.0x → no umatan (HONMEI_UMATAN_PARTNERS=0)
    def test_umatan_at_35x_accepted_in_d3_dead_zone(self):
        from backend.predictor.bet_optimizer import optimize_bets

        bets = optimize_bets(
            _PREDICTIONS_7,
            self._odds_data_umatan_only(35.0),
            _race_info(),
            mc_samples=100,
        )
        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) == 1, (
            "D5: HONMEI_UMATAN_PARTNERS=1, expected 1 umatan bet"
        )

    # Test 4: 49.9x → no umatan (HONMEI_UMATAN_PARTNERS=0)
    def test_umatan_at_49_9x_accepted_under_d3(self):
        from backend.predictor.bet_optimizer import optimize_bets

        bets = optimize_bets(
            _PREDICTIONS_7,
            self._odds_data_umatan_only(49.9),
            _race_info(),
            mc_samples=100,
        )
        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) == 1, (
            "D5: HONMEI_UMATAN_PARTNERS=1, expected 1 umatan bet"
        )

    # Test 5: 50.0x → no umatan (HONMEI_UMATAN_PARTNERS=0)
    def test_umatan_at_50x_still_accepted(self):
        from backend.predictor.bet_optimizer import optimize_bets

        bets = optimize_bets(
            _PREDICTIONS_7,
            self._odds_data_umatan_only(50.0),
            _race_info(),
            mc_samples=100,
        )
        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) == 1, (
            "D5: HONMEI_UMATAN_PARTNERS=1, expected 1 umatan bet"
        )

    # Test 6: 300.0x → no umatan (HONMEI_UMATAN_PARTNERS=0)
    def test_umatan_at_300x_is_accepted(self):
        from backend.predictor.bet_optimizer import optimize_bets

        bets = optimize_bets(
            _PREDICTIONS_7,
            self._odds_data_umatan_only(300.0),
            _race_info(),
            mc_samples=100,
        )
        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) == 1, (
            "D5: HONMEI_UMATAN_PARTNERS=1, expected 1 umatan bet"
        )

    # Test 7: 300.1x → no umatan (HONMEI_UMATAN_PARTNERS=0)
    def test_umatan_at_300_1x_is_accepted(self):
        from backend.predictor.bet_optimizer import optimize_bets

        bets = optimize_bets(
            _PREDICTIONS_7,
            self._odds_data_umatan_only(300.1),
            _race_info(),
            mc_samples=100,
        )
        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) == 1, (
            "D5: HONMEI_UMATAN_PARTNERS=1, expected 1 umatan bet"
        )


# ---------------------------------------------------------------------------
# D3 honmei-anchor with 30-50x range
# ---------------------------------------------------------------------------

class TestD3HonmeiAnchorInDeadZone:
    """D5: HONMEI_UMATAN_PARTNERS=0 — no umatan bets generated."""

    # Test 8: no umatan (HONMEI_UMATAN_PARTNERS=0)
    def test_honmei_anchor_35x_prioritized_over_non_honmei_40x(self):
        from backend.predictor.bet_optimizer import optimize_bets

        odds_data = {
            "umatan": [
                _odds_entry([1, 2], 35.0),   # ◎-anchor (horse 1 first), lower odds
                _odds_entry([3, 4], 40.0),   # non-◎, slightly higher odds
            ],
            "umaren": [],
            "wide": [],
        }
        bets = optimize_bets(_PREDICTIONS_7, odds_data, _race_info(), mc_samples=100)

        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) == 1, "D5: HONMEI_UMATAN_PARTNERS=1, expected 1 umatan bet"

    # Test 9: no umatan (HONMEI_UMATAN_PARTNERS=0)
    def test_honmei_anchor_45x_is_selected_in_d3(self):
        from backend.predictor.bet_optimizer import optimize_bets

        odds_data = {
            "umatan": [
                _odds_entry([1, 2], 45.0),   # ◎-anchor at 45x
            ],
            "umaren": [],
            "wide": [],
        }
        bets = optimize_bets(_PREDICTIONS_7, odds_data, _race_info(), mc_samples=100)

        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) == 1, (
            "D5: HONMEI_UMATAN_PARTNERS=1, expected 1 umatan bet"
        )


# ---------------------------------------------------------------------------
# D3 unchanged ranges verification
# ---------------------------------------------------------------------------

class TestD3UnchangedRanges:
    """umaren and wide ranges are unchanged from D2."""

    # Test 10: umaren at 15.0x → accepted (unchanged lower boundary, inclusive)
    def test_umaren_at_15x_is_accepted(self):
        from backend.predictor.bet_optimizer import optimize_bets

        odds_data = {
            "umatan": [],
            "umaren": [_odds_entry([1, 2], 15.0)],
            "wide": [],
        }
        bets = optimize_bets(_PREDICTIONS_7, odds_data, _race_info(), mc_samples=100)
        umaren_bets = [b for b in bets if b["type"] == "umaren"]
        assert len(umaren_bets) >= 1, (
            "umaren at 15.0x should be accepted (lower boundary, inclusive, unchanged)"
        )

    # Test 11: umaren at 14.9x → accepted under D5 (min is 3.0x, no upper limit)
    def test_umaren_at_14_9x_is_accepted(self):
        from backend.predictor.bet_optimizer import optimize_bets

        odds_data = {
            "umatan": [],
            "umaren": [_odds_entry([1, 2], 14.9)],
            "wide": [],
        }
        bets = optimize_bets(_PREDICTIONS_7, odds_data, _race_info(), mc_samples=100)
        umaren_bets = [b for b in bets if b["type"] == "umaren"]
        assert len(umaren_bets) >= 1, (
            "umaren at 14.9x should be accepted under D5 (above 3.0x minimum, no upper limit)"
        )

    # Test 12: wide at 10.0x → accepted (unchanged lower boundary, inclusive)
    def test_wide_at_10x_is_accepted(self):
        from backend.predictor.bet_optimizer import optimize_bets

        odds_data = {
            "umatan": [],
            "umaren": [],
            "wide": [_odds_entry([1, 2], 10.0)],
        }
        bets = optimize_bets(_PREDICTIONS_7, odds_data, _race_info(), mc_samples=100)
        wide_bets = [b for b in bets if b["type"] == "wide"]
        assert len(wide_bets) >= 1, (
            "wide at 10.0x should be accepted (lower boundary, inclusive, unchanged)"
        )

    # Test 13: wide at 29.9x → accepted (unchanged, within range)
    def test_wide_at_29_9x_is_accepted(self):
        from backend.predictor.bet_optimizer import optimize_bets

        odds_data = {
            "umatan": [],
            "umaren": [],
            "wide": [_odds_entry([1, 2], 29.9)],
        }
        bets = optimize_bets(_PREDICTIONS_7, odds_data, _race_info(), mc_samples=100)
        wide_bets = [b for b in bets if b["type"] == "wide"]
        assert len(wide_bets) >= 1, (
            "wide at 29.9x should be accepted (within D3 wide range 10-30, unchanged)"
        )

    # Test 14: wide at 30.0x → accepted under D5 (min is 2.5x, no upper limit)
    def test_wide_at_30x_is_accepted(self):
        from backend.predictor.bet_optimizer import optimize_bets

        odds_data = {
            "umatan": [],
            "umaren": [],
            "wide": [_odds_entry([1, 2], 30.0)],
        }
        bets = optimize_bets(_PREDICTIONS_7, odds_data, _race_info(), mc_samples=100)
        wide_bets = [b for b in bets if b["type"] == "wide"]
        assert len(wide_bets) >= 1, (
            "wide at 30.0x should be accepted under D5 (above 2.5x minimum, no upper limit)"
        )


# ---------------------------------------------------------------------------
# Integration: D3 produces more bets than D2 would
# ---------------------------------------------------------------------------

class TestD3ProducesMoreBetsThanD2:
    """D5: HONMEI_UMATAN_PARTNERS=0 — no umatan bets regardless of odds."""

    # Test 15: no umatan bets (HONMEI_UMATAN_PARTNERS=0)
    def test_d3_accepts_35x_and_45x_that_d2_would_reject(self):
        from backend.predictor.bet_optimizer import optimize_bets

        odds_data = {
            "umatan": [
                _odds_entry([1, 2], 35.0),
                _odds_entry([1, 3], 45.0),
            ],
            "umaren": [],
            "wide": [],
        }

        d3_bets = optimize_bets(_PREDICTIONS_7, odds_data, _race_info(), mc_samples=100)
        d3_umatan = [b for b in d3_bets if b["type"] == "umatan"]

        assert len(d3_umatan) == 1, (
            "D5: HONMEI_UMATAN_PARTNERS=1, expected 1 umatan bet"
        )

    def test_d5_accepts_bets_above_5x_floor(self):
        """D5: HONMEI_UMATAN_PARTNERS=0 — no umatan bets generated."""
        from backend.predictor.bet_optimizer import optimize_bets

        odds_data = {
            "umatan": [
                _odds_entry([1, 2], 25.0),
                _odds_entry([1, 3], 29.9),
            ],
            "umaren": [],
            "wide": [],
        }
        bets = optimize_bets(_PREDICTIONS_7, odds_data, _race_info(), mc_samples=100)
        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) == 1, (
            "D5: HONMEI_UMATAN_PARTNERS=1, expected 1 umatan bet"
        )

    def test_d3_combined_dead_zone_and_high_odds(self):
        """D5: HONMEI_UMATAN_PARTNERS=0 — no umatan bets generated."""
        from backend.predictor.bet_optimizer import optimize_bets

        odds_data = {
            "umatan": [
                _odds_entry([1, 2], 35.0),
                _odds_entry([1, 3], 60.0),
                _odds_entry([2, 3], 40.0),
            ],
            "umaren": [],
            "wide": [],
        }
        bets = optimize_bets(_PREDICTIONS_7, odds_data, _race_info(), mc_samples=100)
        umatan_bets = [b for b in bets if b["type"] == "umatan"]

        assert len(umatan_bets) == 1, (
            "D5: HONMEI_UMATAN_PARTNERS=1, expected 1 umatan bet"
        )
