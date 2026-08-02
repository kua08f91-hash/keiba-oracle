"""TDD tests for ◎-anchor umatan priority logic in optimize_bets().

Tests that the ◎ (honmei, highest-scored horse) is preferred as the 1st horse
in ordered umatan pairs, ahead of all other bet types and all other umatan
combinations, while still respecting TYPE_MAX=2 and MAX_BETS=5 limits.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_predictions(horse_configs):
    """Build a predictions list.

    Args:
        horse_configs: list of (horseNumber, score) tuples, sorted
            descending by score to mimic scoring engine output.

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


# ---------------------------------------------------------------------------
# VALUE_RANGES used by optimize_bets (D2 strategy)
#   umatan: 50-300x  (inclusive both ends)
#   umaren: 15-30x   (exclusive upper: odds < 30)
#   wide:   10-30x   (exclusive upper: odds < 30)
# ---------------------------------------------------------------------------

UMATAN_LOW = 55.0    # in range (50-300)
UMATAN_HIGH = 200.0  # in range (50-300)
UMATAN_TOP = 250.0   # in range (50-300)
UMAREN_OK = 20.0     # in range (15-30)
WIDE_OK = 15.0       # in range (10-30)


# ---------------------------------------------------------------------------
# Test 1: ◎-anchor umatan selected before non-◎ umatan
# ---------------------------------------------------------------------------

class TestHonmeiUmatanSelectedFirst:
    """◎→X umatan is chosen over Y→Z umatan even if Y→Z has higher odds."""

    def _build_odds(self, honmei_hn, second_hn, other_h1, other_h2):
        """Create odds_data with one ◎-anchor and one non-◎ umatan.

        The non-◎ umatan has higher odds so it would normally be selected
        first in a pure highest-odds strategy.
        """
        return {
            "umatan": [
                _odds_entry([honmei_hn, second_hn], UMATAN_LOW),   # ◎-anchor, lower odds
                _odds_entry([other_h1, other_h2], UMATAN_HIGH),     # non-◎, higher odds
            ],
            "umaren": [],
            "wide": [],
        }

    def test_honmei_anchor_selected_despite_lower_odds(self):
        from backend.predictor.bet_optimizer import optimize_bets

        # Horse 1 = ◎ (highest score)
        predictions = _make_predictions([(1, 90), (2, 70), (3, 60), (4, 50),
                                         (5, 40), (6, 30), (7, 20)])
        honmei_hn = 1
        odds_data = self._build_odds(
            honmei_hn=honmei_hn,
            second_hn=2,
            other_h1=3,
            other_h2=4,
        )
        bets = optimize_bets(predictions, odds_data, _race_info(), mc_samples=100)

        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) >= 1, "Expected at least one umatan bet"

        # The first umatan bet must be the ◎-anchor
        first_umatan = umatan_bets[0]
        assert first_umatan["horses"][0] == honmei_hn, (
            f"First umatan should be ◎-anchor (horse {honmei_hn} as 1st), "
            f"got horses={first_umatan['horses']}"
        )

    def test_honmei_anchor_rank_precedes_non_honmei(self):
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = _make_predictions([(1, 90), (2, 70), (3, 60), (4, 50),
                                         (5, 40), (6, 30), (7, 20)])
        odds_data = {
            "umatan": [
                _odds_entry([1, 2], UMATAN_LOW),    # ◎-anchor 1→2
                _odds_entry([3, 4], UMATAN_HIGH),   # non-◎ 3→4, higher odds
            ],
            "umaren": [],
            "wide": [],
        }
        bets = optimize_bets(predictions, odds_data, _race_info(), mc_samples=100)

        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        honmei_bets = [b for b in umatan_bets if b["horses"][0] == 1]
        non_honmei_bets = [b for b in umatan_bets if b["horses"][0] != 1]

        if honmei_bets and non_honmei_bets:
            # ◎-anchor must appear at a lower rank (selected earlier)
            assert honmei_bets[0]["rank"] < non_honmei_bets[0]["rank"], (
                "◎-anchor umatan should have lower rank number (selected first)"
            )


# ---------------------------------------------------------------------------
# Test 2: ◎-anchor umatan respects TYPE_MAX=2
# ---------------------------------------------------------------------------

class TestHonmeiUmatanTypeMax:
    """At most 2 ◎-anchor umatan bets are selected (TYPE_MAX=2)."""

    def test_only_two_honmei_umatan_selected_from_three(self):
        from backend.predictor.bet_optimizer import optimize_bets

        # Horse 1 = ◎; three viable ◎-anchor pairs available
        predictions = _make_predictions([(1, 90), (2, 70), (3, 60), (4, 50),
                                         (5, 40), (6, 30), (7, 20)])
        odds_data = {
            "umatan": [
                _odds_entry([1, 2], 100.0),   # ◎-anchor
                _odds_entry([1, 3], 150.0),   # ◎-anchor
                _odds_entry([1, 4], 200.0),   # ◎-anchor
            ],
            "umaren": [],
            "wide": [],
        }
        bets = optimize_bets(predictions, odds_data, _race_info(), mc_samples=100)

        honmei_umatan = [
            b for b in bets
            if b["type"] == "umatan" and b["horses"][0] == 1
        ]
        assert len(honmei_umatan) <= 2, (
            f"TYPE_MAX=2: expected at most 2 ◎-anchor umatan, got {len(honmei_umatan)}"
        )

    def test_type_max_is_exactly_two_when_three_available(self):
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = _make_predictions([(1, 90), (2, 70), (3, 60), (4, 50),
                                         (5, 40), (6, 30), (7, 20)])
        odds_data = {
            "umatan": [
                _odds_entry([1, 2], 100.0),
                _odds_entry([1, 3], 150.0),
                _odds_entry([1, 4], 200.0),
            ],
            "umaren": [],
            "wide": [],
        }
        bets = optimize_bets(predictions, odds_data, _race_info(), mc_samples=100)

        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) <= 2, (
            f"Total umatan bets should not exceed TYPE_MAX=2, got {len(umatan_bets)}"
        )


# ---------------------------------------------------------------------------
# Test 3: Non-◎ umatan fills remaining umatan slot
# ---------------------------------------------------------------------------

class TestNonHonmeiUmatanFillsSlot:
    """After 1 ◎-anchor umatan, a non-◎ umatan fills the remaining umatan slot."""

    def test_one_honmei_plus_one_non_honmei_umatan(self):
        from backend.predictor.bet_optimizer import optimize_bets

        # Horse 1 = ◎; only 1 ◎-anchor + 2 non-◎ umatan available
        predictions = _make_predictions([(1, 90), (2, 70), (3, 60), (4, 50),
                                         (5, 40), (6, 30), (7, 20)])
        odds_data = {
            "umatan": [
                _odds_entry([1, 2], 100.0),   # ◎-anchor (sole)
                _odds_entry([3, 4], 150.0),   # non-◎
                _odds_entry([4, 5], 120.0),   # non-◎
            ],
            "umaren": [],
            "wide": [],
        }
        bets = optimize_bets(predictions, odds_data, _race_info(), mc_samples=100)

        honmei_umatan = [b for b in bets if b["type"] == "umatan" and b["horses"][0] == 1]
        non_honmei_umatan = [b for b in bets if b["type"] == "umatan" and b["horses"][0] != 1]

        assert len(honmei_umatan) == 1, "Expected exactly 1 ◎-anchor umatan"
        assert len(non_honmei_umatan) == 1, (
            "Expected 1 non-◎ umatan to fill the remaining umatan slot"
        )

    def test_non_honmei_umatan_fills_after_honmei_exhausted(self):
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = _make_predictions([(1, 90), (2, 70), (3, 60), (4, 50),
                                         (5, 40), (6, 30), (7, 20)])
        odds_data = {
            "umatan": [
                _odds_entry([1, 2], 80.0),    # ◎-anchor only
                _odds_entry([3, 5], 200.0),   # non-◎, high odds
            ],
            "umaren": [],
            "wide": [],
        }
        bets = optimize_bets(predictions, odds_data, _race_info(), mc_samples=100)
        all_umatan = [b for b in bets if b["type"] == "umatan"]

        # Both should appear (total 2 = TYPE_MAX)
        assert len(all_umatan) == 2, (
            f"Expected 2 total umatan (1 honmei + 1 non-honmei), got {len(all_umatan)}"
        )


# ---------------------------------------------------------------------------
# Test 4: Other bet types (umaren, wide) are still selected normally
# ---------------------------------------------------------------------------

class TestOtherBetTypesUnaffected:
    """umaren and wide bets are selected regardless of umatan priority."""

    def test_umaren_selected_after_honmei_umatan(self):
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = _make_predictions([(1, 90), (2, 70), (3, 60), (4, 50),
                                         (5, 40), (6, 30), (7, 20)])
        odds_data = {
            "umatan": [
                _odds_entry([1, 2], 100.0),   # ◎-anchor
            ],
            "umaren": [
                _odds_entry([1, 3], UMAREN_OK),  # within 15-30x
            ],
            "wide": [],
        }
        bets = optimize_bets(predictions, odds_data, _race_info(), mc_samples=100)

        types = [b["type"] for b in bets]
        assert "umatan" in types, "umatan bet expected"
        assert "umaren" in types, "umaren should still be selected after umatan"

    def test_wide_selected_after_honmei_umatan(self):
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = _make_predictions([(1, 90), (2, 70), (3, 60), (4, 50),
                                         (5, 40), (6, 30), (7, 20)])
        odds_data = {
            "umatan": [
                _odds_entry([1, 2], 100.0),
            ],
            "umaren": [],
            "wide": [
                _odds_entry([1, 3], WIDE_OK),   # within 10-30x
            ],
        }
        bets = optimize_bets(predictions, odds_data, _race_info(), mc_samples=100)

        types = [b["type"] for b in bets]
        assert "umatan" in types, "umatan bet expected"
        assert "wide" in types, "wide should still be selected after umatan"

    def test_umaren_and_wide_selected_alongside_honmei_umatan(self):
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = _make_predictions([(1, 90), (2, 70), (3, 60), (4, 50),
                                         (5, 40), (6, 30), (7, 20)])
        odds_data = {
            "umatan": [
                _odds_entry([1, 2], 100.0),
                _odds_entry([1, 3], 150.0),
            ],
            "umaren": [
                _odds_entry([1, 4], UMAREN_OK),
                _odds_entry([2, 3], 18.0),
            ],
            "wide": [
                _odds_entry([1, 5], WIDE_OK),
                _odds_entry([2, 4], 12.0),
            ],
        }
        bets = optimize_bets(predictions, odds_data, _race_info(), mc_samples=100)

        types_seen = {b["type"] for b in bets}
        assert "umatan" in types_seen
        # With MAX_BETS=3, umatan(2) + umaren(1) = 3, wide may not fit
        assert len(bets) <= 3


# ---------------------------------------------------------------------------
# Test 5: No ◎-anchor umatan available — falls back to non-◎ umatan
# ---------------------------------------------------------------------------

class TestNoHonmeiUmatanAvailable:
    """When no ◎-anchor umatan exists, non-◎ umatan is selected (no crash)."""

    def test_fallback_to_non_honmei_umatan(self):
        from backend.predictor.bet_optimizer import optimize_bets

        # Horse 1 = ◎, but no odds entry for 1→X; only other pairs
        predictions = _make_predictions([(1, 90), (2, 70), (3, 60), (4, 50),
                                         (5, 40), (6, 30), (7, 20)])
        odds_data = {
            "umatan": [
                _odds_entry([3, 4], 100.0),   # non-◎ only
                _odds_entry([4, 5], 150.0),   # non-◎ only
            ],
            "umaren": [],
            "wide": [],
        }
        # Must not raise
        bets = optimize_bets(predictions, odds_data, _race_info(), mc_samples=100)

        # Non-◎ umatan should be selected instead
        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) >= 1, (
            "Expected at least 1 non-◎ umatan when no ◎-anchor is available"
        )

    def test_no_crash_without_honmei_candidates(self):
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = _make_predictions([(1, 90), (2, 70), (3, 60), (4, 50),
                                         (5, 40), (6, 30), (7, 20)])
        odds_data = {
            "umatan": [
                _odds_entry([2, 3], 80.0),
            ],
            "umaren": [],
            "wide": [],
        }
        # Confirm no exception raised
        try:
            bets = optimize_bets(predictions, odds_data, _race_info(), mc_samples=100)
            assert isinstance(bets, list)
        except Exception as exc:
            pytest.fail(f"optimize_bets raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# Test 6: Empty predictions returns empty bets
# ---------------------------------------------------------------------------

class TestEmptyPredictions:
    """optimize_bets handles empty or near-empty inputs gracefully."""

    def test_empty_predictions_returns_empty_list(self):
        from backend.predictor.bet_optimizer import optimize_bets

        bets = optimize_bets([], {}, _race_info(), mc_samples=100)
        assert bets == [], f"Expected empty list, got {bets}"

    def test_empty_predictions_no_exception(self):
        from backend.predictor.bet_optimizer import optimize_bets

        try:
            result = optimize_bets([], {}, _race_info(), mc_samples=100)
            assert isinstance(result, list)
        except Exception as exc:
            pytest.fail(f"optimize_bets raised on empty predictions: {exc}")

    def test_single_horse_returns_empty(self):
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = _make_predictions([(1, 80)])
        bets = optimize_bets(predictions, {}, _race_info(head_count=1), mc_samples=100)
        assert bets == []

    def test_two_horses_returns_empty_due_to_head_count(self):
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = _make_predictions([(1, 80), (2, 60)])
        bets = optimize_bets(predictions, {}, _race_info(head_count=2), mc_samples=100)
        assert bets == []

    def test_empty_odds_data_returns_empty(self):
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = _make_predictions([(1, 90), (2, 70), (3, 60), (4, 50),
                                         (5, 40), (6, 30), (7, 20)])
        bets = optimize_bets(predictions, {}, _race_info(), mc_samples=100)
        assert bets == []


# ---------------------------------------------------------------------------
# Test 7: ◎-anchor umatan selected by highest odds within group
# ---------------------------------------------------------------------------

class TestHonmeiUmatanSortedByOdds:
    """Within ◎-anchor umatan, highest-odds pairs are selected first."""

    def test_highest_odds_honmei_umatan_selected(self):
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = _make_predictions([(1, 90), (2, 70), (3, 60), (4, 50),
                                         (5, 40), (6, 30), (7, 20)])
        # Three ◎-anchor candidates at 200x, 150x, 100x
        odds_data = {
            "umatan": [
                _odds_entry([1, 2], 100.0),   # lowest
                _odds_entry([1, 3], 150.0),   # middle
                _odds_entry([1, 4], 200.0),   # highest
            ],
            "umaren": [],
            "wide": [],
        }
        bets = optimize_bets(predictions, odds_data, _race_info(), mc_samples=100)

        honmei_umatan = [b for b in bets if b["type"] == "umatan" and b["horses"][0] == 1]
        assert len(honmei_umatan) == 2, f"Expected 2 ◎-anchor umatan, got {len(honmei_umatan)}"

        selected_odds = {b["odds"] for b in honmei_umatan}
        assert 200.0 in selected_odds, "200x ◎-anchor should be selected"
        assert 150.0 in selected_odds, "150x ◎-anchor should be selected"

    def test_lowest_odds_honmei_umatan_excluded(self):
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = _make_predictions([(1, 90), (2, 70), (3, 60), (4, 50),
                                         (5, 40), (6, 30), (7, 20)])
        odds_data = {
            "umatan": [
                _odds_entry([1, 2], 100.0),
                _odds_entry([1, 3], 150.0),
                _odds_entry([1, 4], 200.0),
            ],
            "umaren": [],
            "wide": [],
        }
        bets = optimize_bets(predictions, odds_data, _race_info(), mc_samples=100)

        honmei_umatan = [b for b in bets if b["type"] == "umatan" and b["horses"][0] == 1]
        selected_odds = {b["odds"] for b in honmei_umatan}
        assert 100.0 not in selected_odds, (
            "Lowest-odds ◎-anchor (100x) should not be selected when only 2 slots available"
        )

    def test_honmei_umatan_order_descending_odds(self):
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = _make_predictions([(1, 90), (2, 70), (3, 60), (4, 50),
                                         (5, 40), (6, 30), (7, 20)])
        odds_data = {
            "umatan": [
                _odds_entry([1, 2], 100.0),
                _odds_entry([1, 3], 200.0),
            ],
            "umaren": [],
            "wide": [],
        }
        bets = optimize_bets(predictions, odds_data, _race_info(), mc_samples=100)

        honmei_umatan = [b for b in bets if b["type"] == "umatan" and b["horses"][0] == 1]
        assert len(honmei_umatan) == 2
        # Rank 1 (lowest rank number) should have higher odds
        by_rank = sorted(honmei_umatan, key=lambda b: b["rank"])
        assert by_rank[0]["odds"] > by_rank[1]["odds"], (
            "First selected ◎-anchor umatan should have higher odds than second"
        )


# ---------------------------------------------------------------------------
# Test 8: Max 5 total bets respected
# ---------------------------------------------------------------------------

class TestMaxFiveTotalBets:
    """optimize_bets never returns more than MAX_BETS=5 bets in total."""

    def test_max_5_bets_with_six_viable_candidates(self):
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = _make_predictions([(1, 90), (2, 70), (3, 60), (4, 50),
                                         (5, 40), (6, 30), (7, 20)])
        # Provide 2 umatan + 2 umaren + 2 wide = 6 viable candidates
        odds_data = {
            "umatan": [
                _odds_entry([1, 2], 100.0),
                _odds_entry([1, 3], 150.0),
            ],
            "umaren": [
                _odds_entry([1, 4], UMAREN_OK),
                _odds_entry([2, 3], 18.0),
            ],
            "wide": [
                _odds_entry([1, 5], WIDE_OK),
                _odds_entry([2, 4], 12.0),
            ],
        }
        bets = optimize_bets(predictions, odds_data, _race_info(), mc_samples=100)

        assert len(bets) <= 5, f"Expected at most 5 bets, got {len(bets)}"

    def test_max_5_bets_with_many_candidates(self):
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = _make_predictions([(1, 90), (2, 70), (3, 60), (4, 50),
                                         (5, 40), (6, 30), (7, 20)])
        # More than 5 viable entries across types
        odds_data = {
            "umatan": [
                _odds_entry([1, 2], 100.0),
                _odds_entry([1, 3], 150.0),
                _odds_entry([2, 3], 200.0),
            ],
            "umaren": [
                _odds_entry([1, 4], UMAREN_OK),
                _odds_entry([3, 5], 25.0),
                _odds_entry([2, 6], 22.0),
            ],
            "wide": [
                _odds_entry([1, 5], WIDE_OK),
                _odds_entry([2, 4], 12.0),
                _odds_entry([3, 6], 20.0),
            ],
        }
        bets = optimize_bets(predictions, odds_data, _race_info(), mc_samples=100)

        assert len(bets) <= 5, f"Expected at most 5 bets, got {len(bets)}"

    def test_ranks_are_sequential_from_one(self):
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = _make_predictions([(1, 90), (2, 70), (3, 60), (4, 50),
                                         (5, 40), (6, 30), (7, 20)])
        odds_data = {
            "umatan": [
                _odds_entry([1, 2], 100.0),
                _odds_entry([1, 3], 150.0),
            ],
            "umaren": [
                _odds_entry([2, 3], UMAREN_OK),
            ],
            "wide": [
                _odds_entry([1, 4], WIDE_OK),
            ],
        }
        bets = optimize_bets(predictions, odds_data, _race_info(), mc_samples=100)

        for i, bet in enumerate(bets):
            assert bet["rank"] == i + 1, (
                f"Rank at position {i} should be {i + 1}, got {bet['rank']}"
            )
