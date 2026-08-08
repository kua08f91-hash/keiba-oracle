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
    """Verify D3 umatan range (30.0, 300.0) — both bounds inclusive."""

    def _odds_data_umatan_only(self, odds_value):
        """Produce odds_data with a single umatan entry at the given odds."""
        return {
            "umatan": [_odds_entry([1, 2], odds_value)],
            "umaren": [],
            "wide": [],
        }

    # Test 1: 30.0x → accepted (D3 lower boundary, inclusive)
    def test_umatan_at_30x_is_accepted(self):
        from backend.predictor.bet_optimizer import optimize_bets

        bets = optimize_bets(
            _PREDICTIONS_7,
            self._odds_data_umatan_only(30.0),
            _race_info(),
            mc_samples=100,
        )
        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) >= 1, (
            "umatan at 30.0x should be accepted under D3 (lower boundary, inclusive)"
        )
        assert any(b["odds"] == 30.0 for b in umatan_bets), (
            "The accepted umatan bet should have odds == 30.0"
        )

    # Test 2: 29.9x → accepted (D5 minimum is 5.0x, no upper limit)
    def test_umatan_at_29_9x_is_accepted(self):
        from backend.predictor.bet_optimizer import optimize_bets

        bets = optimize_bets(
            _PREDICTIONS_7,
            self._odds_data_umatan_only(29.9),
            _race_info(),
            mc_samples=100,
        )
        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) >= 1, (
            "umatan at 29.9x should be accepted under D5 (above 5.0x minimum, no upper limit)"
        )

    # Test 3: 35.0x → accepted (was in dead zone under D2, now valid under D3)
    def test_umatan_at_35x_accepted_in_d3_dead_zone(self):
        from backend.predictor.bet_optimizer import optimize_bets

        bets = optimize_bets(
            _PREDICTIONS_7,
            self._odds_data_umatan_only(35.0),
            _race_info(),
            mc_samples=100,
        )
        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) >= 1, (
            "umatan at 35.0x should be accepted under D3 "
            "(was dead zone 30-50x under D2, now valid)"
        )

    # Test 4: 49.9x → accepted (was rejected under D2, now valid under D3)
    def test_umatan_at_49_9x_accepted_under_d3(self):
        from backend.predictor.bet_optimizer import optimize_bets

        bets = optimize_bets(
            _PREDICTIONS_7,
            self._odds_data_umatan_only(49.9),
            _race_info(),
            mc_samples=100,
        )
        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) >= 1, (
            "umatan at 49.9x should be accepted under D3 "
            "(was below D2 minimum of 50.0, now in D3 range)"
        )

    # Test 5: 50.0x → accepted (still valid, was also valid under D2)
    def test_umatan_at_50x_still_accepted(self):
        from backend.predictor.bet_optimizer import optimize_bets

        bets = optimize_bets(
            _PREDICTIONS_7,
            self._odds_data_umatan_only(50.0),
            _race_info(),
            mc_samples=100,
        )
        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) >= 1, (
            "umatan at 50.0x should still be accepted under D3 (was valid in D2 too)"
        )

    # Test 6: 300.0x → accepted (upper boundary, inclusive)
    def test_umatan_at_300x_is_accepted(self):
        from backend.predictor.bet_optimizer import optimize_bets

        bets = optimize_bets(
            _PREDICTIONS_7,
            self._odds_data_umatan_only(300.0),
            _race_info(),
            mc_samples=100,
        )
        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) >= 1, (
            "umatan at 300.0x should be accepted (upper boundary, inclusive)"
        )

    # Test 7: 300.1x → accepted (D5 has no upper limit)
    def test_umatan_at_300_1x_is_accepted(self):
        from backend.predictor.bet_optimizer import optimize_bets

        bets = optimize_bets(
            _PREDICTIONS_7,
            self._odds_data_umatan_only(300.1),
            _race_info(),
            mc_samples=100,
        )
        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) >= 1, (
            "umatan at 300.1x should be accepted under D5 (no upper limit)"
        )


# ---------------------------------------------------------------------------
# D3 honmei-anchor with 30-50x range
# ---------------------------------------------------------------------------

class TestD3HonmeiAnchorInDeadZone:
    """◎-anchor umatan bets in the new 30-50x range behave correctly."""

    # Test 8: ◎-anchor at 35x is prioritized over non-◎ umatan at 40x
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
        assert len(umatan_bets) >= 1, "Expected at least one umatan bet"

        # The first umatan bet (lowest rank) must be the ◎-anchor
        first_umatan = min(umatan_bets, key=lambda b: b["rank"])
        assert first_umatan["horses"][0] == 1, (
            f"First selected umatan should be ◎-anchor (horse 1 as 1st), "
            f"got horses={first_umatan['horses']}, odds={first_umatan['odds']}"
        )

    # Test 9: ◎-anchor at 45x is selected (was impossible under D2)
    def test_honmei_anchor_45x_is_selected_in_d3(self):
        from backend.predictor.bet_optimizer import optimize_bets

        odds_data = {
            "umatan": [
                _odds_entry([1, 2], 45.0),   # ◎-anchor at 45x — D2 dead zone
            ],
            "umaren": [],
            "wide": [],
        }
        bets = optimize_bets(_PREDICTIONS_7, odds_data, _race_info(), mc_samples=100)

        honmei_umatan = [
            b for b in bets
            if b["type"] == "umatan" and b["horses"][0] == 1
        ]
        assert len(honmei_umatan) >= 1, (
            "◎-anchor umatan at 45x should be selected under D3 "
            "(was in dead zone 30-50x under D2, impossible then)"
        )
        assert honmei_umatan[0]["odds"] == 45.0, (
            f"Selected bet odds should be 45.0, got {honmei_umatan[0]['odds']}"
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
    """D3 lower bound (30x) means bets that D2 (50x min) would reject are now accepted."""

    # Test 15: Given umatan at 35x and 45x, D3 produces bets (D2 would not)
    def test_d3_accepts_35x_and_45x_that_d2_would_reject(self):
        from backend.predictor.bet_optimizer import optimize_bets

        # These odds are in the former D2 dead zone (30-50x range)
        odds_data = {
            "umatan": [
                _odds_entry([1, 2], 35.0),   # D2: rejected (< 50x)  D3: accepted (>= 30x)
                _odds_entry([1, 3], 45.0),   # D2: rejected (< 50x)  D3: accepted (>= 30x)
            ],
            "umaren": [],
            "wide": [],
        }

        d3_bets = optimize_bets(_PREDICTIONS_7, odds_data, _race_info(), mc_samples=100)
        d3_umatan = [b for b in d3_bets if b["type"] == "umatan"]

        assert len(d3_umatan) >= 1, (
            "D3 should produce at least one umatan bet for odds 35x and 45x "
            "(these were in the D2 dead zone and would have been rejected)"
        )

        # Verify the accepted bets have the correct odds values
        accepted_odds = {b["odds"] for b in d3_umatan}
        assert accepted_odds.issubset({35.0, 45.0}), (
            f"Only the D3-valid bets (35x, 45x) should appear, got {accepted_odds}"
        )

    def test_d5_accepts_bets_above_5x_floor(self):
        """D5 minimum for umatan is 5x — bets above 5x are accepted."""
        from backend.predictor.bet_optimizer import optimize_bets

        odds_data = {
            "umatan": [
                _odds_entry([1, 2], 25.0),   # D5: accepted (>= 5x)
                _odds_entry([1, 3], 29.9),   # D5: accepted (>= 5x)
            ],
            "umaren": [],
            "wide": [],
        }
        bets = optimize_bets(_PREDICTIONS_7, odds_data, _race_info(), mc_samples=100)
        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) >= 1, (
            "umatan at 25x and 29.9x should be accepted under D5 (above 5x minimum)"
        )

    def test_d3_combined_dead_zone_and_high_odds(self):
        """D3 accepts both former dead zone (35x) and previously valid (60x) in one race."""
        from backend.predictor.bet_optimizer import optimize_bets

        odds_data = {
            "umatan": [
                _odds_entry([1, 2], 35.0),   # D3 dead zone, now accepted
                _odds_entry([1, 3], 60.0),   # Always valid (above 50x in D2 too)
                _odds_entry([2, 3], 40.0),   # non-◎, in former dead zone — accepted D3
            ],
            "umaren": [],
            "wide": [],
        }
        bets = optimize_bets(_PREDICTIONS_7, odds_data, _race_info(), mc_samples=100)
        umatan_bets = [b for b in bets if b["type"] == "umatan"]

        # At most TYPE_MAX=2 umatan bets
        assert len(umatan_bets) <= 2, (
            f"Expected at most TYPE_MAX=2 umatan bets, got {len(umatan_bets)}"
        )
        # At least 1 umatan bet (multiple valid candidates available)
        assert len(umatan_bets) >= 1, (
            "Expected at least 1 umatan bet from mixed odds (35x, 40x, 60x)"
        )
        # ◎-anchor (horse 1) bets must come before non-◎ bets
        honmei_bets = [b for b in umatan_bets if b["horses"][0] == 1]
        non_honmei_bets = [b for b in umatan_bets if b["horses"][0] != 1]
        if honmei_bets and non_honmei_bets:
            assert min(b["rank"] for b in honmei_bets) < min(b["rank"] for b in non_honmei_bets), (
                "◎-anchor umatan bets should have lower rank numbers (selected first)"
            )
