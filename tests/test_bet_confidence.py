"""TDD tests for evaluate_bet_confidence() — D10 threshold-optimized strategy.

D10 three-level confidence:
  "A" (勝負): ◎score >= 64 AND ◎odds 2-5倍 → BUY ◎単勝
  "B" (推奨): ◎score >= 64 AND ◎odds 3-5倍 (A条件外) → 推奨
  "C" (SKIP): 条件不一致 → 見送り
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
    return [{"horseNumber": i + 1, "score": s} for i, s in enumerate(scores)]


def _make_entries(odds_by_horse: dict) -> list:
    return [{"horseNumber": hn, "odds": odds} for hn, odds in odds_by_horse.items()]


def _make_race_info(head_count: int) -> dict:
    return {"headCount": head_count}


# ---------------------------------------------------------------------------
# 1. A Rank (勝負) — ◎score >= 64 AND odds 2-5倍
# ---------------------------------------------------------------------------

class TestARank:
    """A = 勝負: ◎score >= 64 AND ◎odds in [2.0, 5.0)."""

    def test_high_score_and_good_odds_returns_a(self):
        predictions = _make_predictions([90, 60, 40])
        entries = _make_entries({1: 3.0, 2: 8.0, 3: 15.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(16), entries)
        assert result == "A"

    def test_score_at_threshold_and_odds_in_range(self):
        """◎score=64.0, odds=2.5 → 'A'."""
        predictions = _make_predictions([64.0, 50, 40])
        entries = _make_entries({1: 2.5, 2: 8.0, 3: 15.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(12), entries)
        assert result == "A"

    def test_odds_exactly_2(self):
        """◎odds=2.0 (lower bound) → 'A'."""
        predictions = _make_predictions([75, 50, 30])
        entries = _make_entries({1: 2.0, 2: 8.0, 3: 15.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(16), entries)
        assert result == "A"

    def test_odds_just_below_5(self):
        """◎odds=4.9 → 'A'."""
        predictions = _make_predictions([70, 50, 30])
        entries = _make_entries({1: 4.9, 2: 8.0, 3: 15.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(16), entries)
        assert result == "A"

    def test_odds_4_5_still_a(self):
        """◎odds=4.5, score=65 → 'A' (D10 expanded range)."""
        predictions = _make_predictions([65, 50, 30])
        entries = _make_entries({1: 4.5, 2: 8.0, 3: 15.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(16), entries)
        assert result == "A"

    def test_score_64_odds_3_returns_a(self):
        """◎score=64, odds=3.0 → 'A' (D10 lowered threshold)."""
        predictions = _make_predictions([64, 50, 40])
        entries = _make_entries({1: 3.0, 2: 8.0, 3: 15.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(16), entries)
        assert result == "A"


# ---------------------------------------------------------------------------
# 2. B Rank (推奨) — ◎score >= 64 AND odds 3-5倍 (A条件外)
# ---------------------------------------------------------------------------

class TestBRank:
    """B = 推奨: ◎score >= 64 AND odds 3-5倍, NOT already A."""

    def test_b_at_odds_3_5_score_64(self):
        """◎score=64, odds=3.5 → A (within A range 2-5)."""
        predictions = _make_predictions([64, 50, 40])
        entries = _make_entries({1: 3.5, 2: 10.0, 3: 15.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(16), entries)
        assert result == "A"  # B range (3-5) overlaps with A range (2-5)

    def test_b_requires_a_check_first(self):
        """A判定条件を満たす場合はBにならない."""
        predictions = _make_predictions([75, 62, 40])
        entries = _make_entries({1: 3.0, 2: 10.0, 3: 15.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(16), entries)
        assert result == "A"

    def test_low_score_not_b(self):
        """◎score=63 (64未満) → 'C'."""
        predictions = _make_predictions([63, 50, 40])
        entries = _make_entries({1: 4.0, 2: 10.0, 3: 15.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(16), entries)
        assert result == "C"

    def test_odds_below_3_not_b(self):
        """◎score=63, odds=2.5 → 'C' (score < 64)."""
        predictions = _make_predictions([63, 50, 40])
        entries = _make_entries({1: 2.5, 2: 5.0, 3: 15.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(16), entries)
        assert result == "C"

    def test_odds_5_plus_not_b(self):
        """◎score=64, odds=5.5 → 'C' (odds outside both A and B)."""
        predictions = _make_predictions([64, 50, 40])
        entries = _make_entries({1: 5.5, 2: 10.0, 3: 15.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(16), entries)
        assert result == "C"

    def test_odds_1_5_score_64_not_a_or_b(self):
        """◎score=64, odds=1.5 → 'C' (below A odds min)."""
        predictions = _make_predictions([64, 50, 40])
        entries = _make_entries({1: 1.5, 2: 10.0, 3: 15.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(16), entries)
        assert result == "C"


# ---------------------------------------------------------------------------
# 3. C Rank (SKIP)
# ---------------------------------------------------------------------------

class TestCRank:
    """C = SKIP."""

    def test_low_score(self):
        predictions = _make_predictions([60, 55, 40])
        entries = _make_entries({1: 3.0, 2: 5.0, 3: 10.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(14), entries)
        assert result == "C"

    def test_score_just_below_threshold(self):
        """◎score=63.9 → 'C'."""
        predictions = _make_predictions([63.9, 50, 30])
        entries = _make_entries({1: 3.0, 2: 8.0, 3: 15.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(12), entries)
        assert result == "C"

    def test_low_score_good_odds_still_c(self):
        """◎score=60, odds=3.0 → 'C'."""
        predictions = _make_predictions([60, 40, 20])
        entries = _make_entries({1: 3.0, 2: 8.0, 3: 15.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(8), entries)
        assert result == "C"

    def test_high_odds_low_score(self):
        """◎score=63, odds=6.0 → 'C'."""
        predictions = _make_predictions([63, 40, 20])
        entries = _make_entries({1: 6.0, 2: 8.0, 3: 15.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(16), entries)
        assert result == "C"


# ---------------------------------------------------------------------------
# 4. Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases must not crash and must return valid grade."""

    def test_empty_predictions_returns_c(self):
        result = evaluate_bet_confidence([], {"headCount": 16}, [])
        assert result == "C"

    def test_all_scores_zero_returns_c(self):
        predictions = [{"horseNumber": i + 1, "score": 0} for i in range(8)]
        result = evaluate_bet_confidence(predictions, _make_race_info(16))
        assert result == "C"

    def test_none_entries_returns_valid_grade(self):
        predictions = _make_predictions([80, 50, 30])
        result = evaluate_bet_confidence(predictions, _make_race_info(12), None)
        assert result in ("A", "B", "C")

    def test_missing_head_count_returns_valid_grade(self):
        predictions = _make_predictions([80, 50, 30])
        result = evaluate_bet_confidence(predictions, {}, None)
        assert result in ("A", "B", "C")

    def test_result_is_always_a_b_or_c(self):
        test_cases = [
            (_make_predictions([90, 60, 40]), _make_race_info(8), _make_entries({1: 3.0})),
            (_make_predictions([50, 49, 48]), _make_race_info(18), None),
            (_make_predictions([70, 50, 30]), _make_race_info(12), _make_entries({1: 1.2})),
            (_make_predictions([64, 60, 40]), _make_race_info(16), _make_entries({1: 10.0})),
            ([], _make_race_info(16), None),
        ]
        for predictions, race_info, entries in test_cases:
            result = evaluate_bet_confidence(predictions, race_info, entries)
            assert result in ("A", "B", "C")

    def test_shoubu_min_score_constant_is_64(self):
        assert SHOUBU_MIN_SCORE == 64.0
