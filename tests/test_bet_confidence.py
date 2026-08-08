"""TDD tests for evaluate_bet_confidence() — D5 ◎軸厚張り strategy.

D5 simplifies confidence to score-based only:
  "A" (勝負): ◎score >= 75 → full ◎軸展開 (up to 14 points)
  "C" (SKIP): ◎score < 75 → skip this race
"""
from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.predictor.bet_optimizer import evaluate_bet_confidence, SHOUBU_MIN_SCORE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_predictions(scores: list) -> list:
    """Build a minimal predictions list from a list of scores (horse 1, 2, ...)."""
    return [{"horseNumber": i + 1, "score": s} for i, s in enumerate(scores)]


def _make_entries(odds_by_horse: dict) -> list:
    """Build a minimal entries list mapping horseNumber → odds."""
    return [{"horseNumber": hn, "odds": odds} for hn, odds in odds_by_horse.items()]


def _make_race_info(head_count: int) -> dict:
    """Build a minimal race_info dict with the given headCount."""
    return {"headCount": head_count}


# ---------------------------------------------------------------------------
# 1. A Rank (勝負) — ◎score >= 75
# ---------------------------------------------------------------------------

class TestARank:
    """A = 勝負: ◎score >= SHOUBU_MIN_SCORE (79)."""

    def test_high_score_returns_a(self):
        """◎score=90 → 'A'."""
        predictions = _make_predictions([90, 60, 40])
        result = evaluate_bet_confidence(predictions, _make_race_info(16))
        assert result == "A"

    def test_score_exactly_at_threshold_returns_a(self):
        """◎score=79.0 (exact threshold) → 'A'."""
        predictions = _make_predictions([79.0, 60, 40])
        result = evaluate_bet_confidence(predictions, _make_race_info(12))
        assert result == "A"

    def test_score_just_above_threshold_returns_a(self):
        """◎score=79.1 → 'A'."""
        predictions = _make_predictions([79.1, 50, 30])
        result = evaluate_bet_confidence(predictions, _make_race_info(18))
        assert result == "A"

    def test_large_field_high_score_still_a(self):
        """Large field (18 heads) but ◎score=80 → 'A'."""
        scores = [80] + [40] * 17
        predictions = _make_predictions(scores)
        result = evaluate_bet_confidence(predictions, _make_race_info(18))
        assert result == "A"

    def test_entries_do_not_affect_a_rank(self):
        """Entries/odds don't matter for D5 — only ◎score counts."""
        predictions = _make_predictions([85, 50, 30])
        entries = _make_entries({1: 5.0, 2: 8.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(16), entries)
        assert result == "A"

    def test_single_horse_high_score_returns_a(self):
        """Single horse with score >= 75 → 'A'."""
        predictions = [{"horseNumber": 1, "score": 80}]
        result = evaluate_bet_confidence(predictions, _make_race_info(5))
        assert result == "A"


# ---------------------------------------------------------------------------
# 2. C Rank (SKIP) — ◎score < 75
# ---------------------------------------------------------------------------

class TestCRank:
    """C = SKIP: ◎score < SHOUBU_MIN_SCORE (78)."""

    def test_low_score_returns_c(self):
        """◎score=60 → 'C'."""
        predictions = _make_predictions([60, 55, 40])
        result = evaluate_bet_confidence(predictions, _make_race_info(14))
        assert result == "C"

    def test_score_just_below_threshold_returns_c(self):
        """◎score=77.9 → 'C'."""
        predictions = _make_predictions([77.9, 50, 30])
        result = evaluate_bet_confidence(predictions, _make_race_info(12))
        assert result == "C"

    def test_score_74_returns_c(self):
        """◎score=74 → 'C'."""
        predictions = _make_predictions([74, 60, 40])
        result = evaluate_bet_confidence(predictions, _make_race_info(10))
        assert result == "C"

    def test_small_field_low_score_still_c(self):
        """Small field (6 heads) but ◎score=50 → 'C'."""
        scores = [50, 45, 40, 35, 30, 25]
        predictions = _make_predictions(scores)
        result = evaluate_bet_confidence(predictions, _make_race_info(6))
        assert result == "C"

    def test_favourite_odds_low_but_score_below_threshold(self):
        """Even if odds suggest favourite, ◎score < 75 → 'C'."""
        predictions = _make_predictions([70, 40, 20])
        entries = _make_entries({1: 1.5, 2: 8.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(8), entries)
        assert result == "C"


# ---------------------------------------------------------------------------
# 3. Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases must not crash and must return valid grade."""

    def test_empty_predictions_returns_c(self):
        """No predictions → 'C' (early return guard)."""
        result = evaluate_bet_confidence([], {"headCount": 16}, [])
        assert result == "C"

    def test_all_scores_zero_returns_c(self):
        """All scores 0 → ◎score=0 < 75 → 'C'."""
        predictions = [{"horseNumber": i + 1, "score": 0} for i in range(8)]
        result = evaluate_bet_confidence(predictions, _make_race_info(16))
        assert result == "C"

    def test_none_entries_returns_valid_grade(self):
        """entries=None does not crash."""
        predictions = _make_predictions([80, 50, 30])
        result = evaluate_bet_confidence(predictions, _make_race_info(12), None)
        assert result in ("A", "C")

    def test_missing_head_count_returns_valid_grade(self):
        """race_info with no 'headCount' → function uses default, no crash."""
        predictions = _make_predictions([80, 50, 30])
        result = evaluate_bet_confidence(predictions, {}, None)
        assert result in ("A", "C")

    def test_result_is_always_a_or_c(self):
        """D5 invariant: output is always exactly 'A' or 'C'."""
        test_cases = [
            (_make_predictions([90, 60, 40]), _make_race_info(8)),
            (_make_predictions([50, 49, 48]), _make_race_info(18)),
            (_make_predictions([100, 1, 1]), _make_race_info(12)),
            (_make_predictions([74, 73, 72]), _make_race_info(16)),
            ([], _make_race_info(16)),
        ]
        for predictions, race_info in test_cases:
            result = evaluate_bet_confidence(predictions, race_info)
            assert result in ("A", "C"), (
                f"Unexpected grade '{result}' for predictions={predictions}"
            )

    def test_shoubu_min_score_constant_is_75(self):
        """SHOUBU_MIN_SCORE constant is 79."""
        assert SHOUBU_MIN_SCORE == 79.0
