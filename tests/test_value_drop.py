"""TDD tests for calc_value_drop — "Value Drop" factor.

33K race analysis: horses with good recent form (前走3着以内) but dropping in
market popularity (今走6番人気以下) have single-win recovery rates of 103-125%.

RED -> GREEN -> REFACTOR cycle.
"""
from __future__ import annotations

import pytest


def _fn():
    from backend.predictor.factors import calc_value_drop
    return calc_value_drop


# ---------------------------------------------------------------------------
# Strong value: 前走1着 + 6番人気以下
# ---------------------------------------------------------------------------

class TestValueDropStrongValue:
    def test_1st_place_6th_popularity(self):
        """前走1着 + 6番人気 → 80 (strongest value, 125% ROI in data)."""
        fn = _fn()
        result = fn([{"pos": 1}], popularity=6, odds=5.0)
        assert result == 80.0

    def test_1st_place_10th_popularity(self):
        """前走1着 + 10番人気 → 80 (still qualifies as 6番人気以下)."""
        fn = _fn()
        result = fn([{"pos": 1}], popularity=10, odds=15.0)
        assert result == 80.0

    def test_1st_place_18th_popularity(self):
        """前走1着 + 18番人気 (longest shot) → 80."""
        fn = _fn()
        result = fn([{"pos": 1}], popularity=18, odds=100.0)
        assert result == 80.0

    def test_1st_place_6th_popularity_high_odds(self):
        """前走1着 + 6番人気 + odds=20.0 → 80 (odds don't change core logic)."""
        fn = _fn()
        result = fn([{"pos": 1}], popularity=6, odds=20.0)
        assert result == 80.0


# ---------------------------------------------------------------------------
# Moderate value: 前走2-3着 + 6番人気以下
# ---------------------------------------------------------------------------

class TestValueDropModerateValue:
    def test_2nd_place_7th_popularity(self):
        """前走2着 + 7番人気 → 70."""
        fn = _fn()
        result = fn([{"pos": 2}], popularity=7, odds=8.0)
        assert result == 70.0

    def test_3rd_place_10th_popularity(self):
        """前走3着 + 10番人気 → 70."""
        fn = _fn()
        result = fn([{"pos": 3}], popularity=10, odds=12.0)
        assert result == 70.0

    def test_2nd_place_6th_popularity(self):
        """前走2着 + 6番人気 → 70 (exactly at threshold)."""
        fn = _fn()
        result = fn([{"pos": 2}], popularity=6, odds=6.5)
        assert result == 70.0

    def test_3rd_place_6th_popularity(self):
        """前走3着 + 6番人気 → 70 (exactly at threshold)."""
        fn = _fn()
        result = fn([{"pos": 3}], popularity=6, odds=7.0)
        assert result == 70.0


# ---------------------------------------------------------------------------
# Mild value: 前走4-5着 + 6番人気以下
# ---------------------------------------------------------------------------

class TestValueDropMildValue:
    def test_4th_place_8th_popularity(self):
        """前走4着 + 8番人気 → 65."""
        fn = _fn()
        result = fn([{"pos": 4}], popularity=8, odds=9.0)
        assert result == 65.0

    def test_5th_place_6th_popularity(self):
        """前走5着 + 6番人気 → 65."""
        fn = _fn()
        result = fn([{"pos": 5}], popularity=6, odds=6.5)
        assert result == 65.0

    def test_4th_place_6th_popularity(self):
        """前走4着 + 6番人気 → 65 (exactly at threshold)."""
        fn = _fn()
        result = fn([{"pos": 4}], popularity=6, odds=7.0)
        assert result == 65.0


# ---------------------------------------------------------------------------
# Slight undervaluation: 前走1-3着 + 4-5番人気
# ---------------------------------------------------------------------------

class TestValueDropSlightUndervaluation:
    def test_1st_place_4th_popularity(self):
        """前走1着 + 4番人気 → 60 (slight undervaluation)."""
        fn = _fn()
        result = fn([{"pos": 1}], popularity=4, odds=4.0)
        assert result == 60.0

    def test_2nd_place_5th_popularity(self):
        """前走2着 + 5番人気 → 60."""
        fn = _fn()
        result = fn([{"pos": 2}], popularity=5, odds=5.0)
        assert result == 60.0

    def test_3rd_place_4th_popularity(self):
        """前走3着 + 4番人気 → 60."""
        fn = _fn()
        result = fn([{"pos": 3}], popularity=4, odds=4.5)
        assert result == 60.0

    def test_3rd_place_5th_popularity(self):
        """前走3着 + 5番人気 → 60."""
        fn = _fn()
        result = fn([{"pos": 3}], popularity=5, odds=5.5)
        assert result == 60.0


# ---------------------------------------------------------------------------
# Market already reflects form: 前走1-3着 + 1-3番人気 → 45
# ---------------------------------------------------------------------------

class TestValueDropMarketReflectsForm:
    def test_1st_place_1st_popularity(self):
        """前走1着 + 1番人気 → 45 (market already prices in form)."""
        fn = _fn()
        result = fn([{"pos": 1}], popularity=1, odds=2.0)
        assert result == 45.0

    def test_3rd_place_2nd_popularity(self):
        """前走3着 + 2番人気 → 45."""
        fn = _fn()
        result = fn([{"pos": 3}], popularity=2, odds=3.0)
        assert result == 45.0

    def test_1st_place_3rd_popularity(self):
        """前走1着 + 3番人気 → 45."""
        fn = _fn()
        result = fn([{"pos": 1}], popularity=3, odds=3.5)
        assert result == 45.0

    def test_2nd_place_1st_popularity(self):
        """前走2着 + 1番人気 → 45."""
        fn = _fn()
        result = fn([{"pos": 2}], popularity=1, odds=1.8)
        assert result == 45.0

    def test_2nd_place_3rd_popularity(self):
        """前走2着 + 3番人気 → 45."""
        fn = _fn()
        result = fn([{"pos": 2}], popularity=3, odds=3.8)
        assert result == 45.0


# ---------------------------------------------------------------------------
# Neutral: 前走6着以下 → 50 (no form advantage)
# ---------------------------------------------------------------------------

class TestValueDropNeutralBadForm:
    def test_6th_place_any_popularity(self):
        """前走6着 + any popularity → 50 (no form advantage)."""
        fn = _fn()
        result = fn([{"pos": 6}], popularity=8, odds=10.0)
        assert result == 50.0

    def test_10th_place_6th_popularity(self):
        """前走10着 + 6番人気 → 50."""
        fn = _fn()
        result = fn([{"pos": 10}], popularity=6, odds=7.0)
        assert result == 50.0

    def test_6th_place_low_popularity(self):
        """前走6着 + 1番人気 → 50 (still neutral, bad form overrides)."""
        fn = _fn()
        result = fn([{"pos": 6}], popularity=1, odds=2.0)
        assert result == 50.0

    def test_7th_place_12th_popularity(self):
        """前走7着 + 12番人気 → 50."""
        fn = _fn()
        result = fn([{"pos": 7}], popularity=12, odds=20.0)
        assert result == 50.0


# ---------------------------------------------------------------------------
# Edge cases: missing/invalid data → 50 (neutral)
# ---------------------------------------------------------------------------

class TestValueDropEdgeCases:
    def test_no_past_races(self):
        """No past races → 50 (neutral, no form data)."""
        fn = _fn()
        result = fn([], popularity=6, odds=8.0)
        assert result == 50.0

    def test_empty_past_races_list(self):
        """Empty list → 50."""
        fn = _fn()
        result = fn([], popularity=10, odds=15.0)
        assert result == 50.0

    def test_popularity_zero(self):
        """popularity=0 (invalid) → 50."""
        fn = _fn()
        result = fn([{"pos": 1}], popularity=0, odds=5.0)
        assert result == 50.0

    def test_popularity_none(self):
        """popularity=None → 50."""
        fn = _fn()
        result = fn([{"pos": 1}], popularity=None, odds=5.0)
        assert result == 50.0

    def test_popularity_6_no_past_races(self):
        """popularity=6 but no past races → 50."""
        fn = _fn()
        result = fn([], popularity=6, odds=7.0)
        assert result == 50.0

    def test_past_race_missing_pos_key(self):
        """Past race dict with no 'pos' key → treated as no valid form → 50."""
        fn = _fn()
        result = fn([{"distance": 1600}], popularity=6, odds=7.0)
        assert result == 50.0

    def test_past_race_pos_zero(self):
        """pos=0 (unknown result) → 50."""
        fn = _fn()
        result = fn([{"pos": 0}], popularity=6, odds=7.0)
        assert result == 50.0


# ---------------------------------------------------------------------------
# Boundary tests
# ---------------------------------------------------------------------------

class TestValueDropBoundaries:
    def test_boundary_3rd_5th_popularity_not_value_drop(self):
        """前走3着 + 5番人気 → 60 (not 70: 5番人気 is NOT ≥6)."""
        fn = _fn()
        result = fn([{"pos": 3}], popularity=5, odds=5.5)
        assert result == 60.0

    def test_boundary_3rd_6th_popularity_is_value_drop(self):
        """前走3着 + 6番人気 → 70 (exactly at ≥6 threshold)."""
        fn = _fn()
        result = fn([{"pos": 3}], popularity=6, odds=7.0)
        assert result == 70.0

    def test_boundary_5th_5th_popularity_neutral(self):
        """前走5着 + 5番人気 → 50 (前走4-5着 needs popularity ≥6 for mild value)."""
        fn = _fn()
        result = fn([{"pos": 5}], popularity=5, odds=5.5)
        assert result == 50.0

    def test_boundary_5th_6th_popularity_mild_value(self):
        """前走5着 + 6番人気 → 65 (exactly at ≥6 threshold for mild value)."""
        fn = _fn()
        result = fn([{"pos": 5}], popularity=6, odds=6.5)
        assert result == 65.0

    def test_boundary_6th_place_is_neutral_not_mild(self):
        """前走6着 + 6番人気 → 50 (6着以下 = no form advantage, never gets mild value)."""
        fn = _fn()
        result = fn([{"pos": 6}], popularity=6, odds=7.0)
        assert result == 50.0

    def test_boundary_popularity_3_is_market_reflects_form(self):
        """前走1着 + 3番人気 → 45 (3番人気 is still in market-reflects-form zone)."""
        fn = _fn()
        result = fn([{"pos": 1}], popularity=3, odds=3.5)
        assert result == 45.0

    def test_boundary_popularity_4_is_slight_undervaluation(self):
        """前走1着 + 4番人気 → 60 (4番人気 triggers slight undervaluation)."""
        fn = _fn()
        result = fn([{"pos": 1}], popularity=4, odds=4.5)
        assert result == 60.0

    def test_multiple_past_races_uses_most_recent(self):
        """Only past_races[0] (most recent) matters for form."""
        fn = _fn()
        # Most recent is 1st, older races are bad — should be 80
        result = fn([{"pos": 1}, {"pos": 10}, {"pos": 12}], popularity=8, odds=10.0)
        assert result == 80.0

    def test_multiple_past_races_bad_most_recent(self):
        """Most recent is 6th (bad form), older was 1st — should be 50 (neutral)."""
        fn = _fn()
        result = fn([{"pos": 6}, {"pos": 1}, {"pos": 2}], popularity=8, odds=10.0)
        assert result == 50.0


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class TestValueDropReturnType:
    def test_returns_float(self):
        """Function must always return a float."""
        fn = _fn()
        result = fn([{"pos": 1}], popularity=6, odds=5.0)
        assert isinstance(result, float)

    def test_returns_float_for_neutral(self):
        """Neutral path also returns float."""
        fn = _fn()
        result = fn([], popularity=None, odds=0.0)
        assert isinstance(result, float)

    def test_score_within_0_100(self):
        """Score must always be within 0-100 range."""
        fn = _fn()
        cases = [
            ([{"pos": 1}], 6, 5.0),
            ([{"pos": 1}], 1, 2.0),
            ([], None, 0.0),
            ([{"pos": 10}], 18, 50.0),
        ]
        for past_races, pop, odds in cases:
            score = fn(past_races, popularity=pop, odds=odds)
            assert 0.0 <= score <= 100.0, f"Score {score} out of range for {past_races}, pop={pop}"
