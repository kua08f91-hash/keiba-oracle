"""TDD tests for evaluate_bet_confidence() — D7 EV Focus strategy.

D7 three-level confidence:
  "A" (勝負): ◎score >= 68 AND ◎odds 2-4倍 → BUY ◎単勝
  "B" (推奨): ◎score >= 68 だがodds条件外 → EV+参考表記
  "C" (SKIP): ◎score < 68 → 見送り
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
# 1. A Rank (勝負) — ◎score >= 68 AND odds 2-4倍
# ---------------------------------------------------------------------------

class TestARank:
    """A = 勝負: ◎score >= 68 AND ◎odds in [2.0, 4.0)."""

    def test_high_score_and_good_odds_returns_a(self):
        predictions = _make_predictions([90, 60, 40])
        entries = _make_entries({1: 3.0, 2: 8.0, 3: 15.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(16), entries)
        assert result == "A"

    def test_score_at_threshold_and_odds_in_range(self):
        """◎score=68.0, odds=2.5 → 'A'."""
        predictions = _make_predictions([68.0, 60, 40])
        entries = _make_entries({1: 2.5, 2: 8.0, 3: 15.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(12), entries)
        assert result == "A"

    def test_odds_exactly_2(self):
        """◎odds=2.0 (lower bound) → 'A'."""
        predictions = _make_predictions([75, 50, 30])
        entries = _make_entries({1: 2.0, 2: 8.0, 3: 15.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(16), entries)
        assert result == "A"

    def test_odds_just_below_4(self):
        """◎odds=3.9 → 'A'."""
        predictions = _make_predictions([70, 50, 30])
        entries = _make_entries({1: 3.9, 2: 8.0, 3: 15.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(16), entries)
        assert result == "A"


# ---------------------------------------------------------------------------
# 2. B Rank (推奨) — ◎score >= 68 but odds outside 2-4倍
# ---------------------------------------------------------------------------

class TestBRank:
    """B = 推奨: ◯odds>=8 AND ◯score>=60 (A判定でないレース)."""

    def test_niban_high_odds_high_score(self):
        """◯odds=10, ◯score=62 → 'B'."""
        predictions = _make_predictions([65, 62, 40])
        entries = _make_entries({1: 5.0, 2: 10.0, 3: 15.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(16), entries)
        assert result == "B"

    def test_niban_at_threshold(self):
        """◯odds=8.0, ◯score=60.0 (境界値) → 'B'."""
        predictions = _make_predictions([65, 60, 40])
        entries = _make_entries({1: 5.0, 2: 8.0, 3: 15.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(16), entries)
        assert result == "B"

    def test_a_race_not_b(self):
        """A判定条件を満たす場合はBにならない."""
        predictions = _make_predictions([75, 62, 40])
        entries = _make_entries({1: 3.0, 2: 10.0, 3: 15.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(16), entries)
        assert result == "A"

    def test_niban_low_odds_not_b(self):
        """◯odds=5.0 (8倍未満) → 'C'."""
        predictions = _make_predictions([65, 62, 40])
        entries = _make_entries({1: 5.0, 2: 5.0, 3: 15.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(16), entries)
        assert result == "C"

    def test_niban_low_score_not_b(self):
        """◯score=58 (60未満) → 'C'."""
        predictions = _make_predictions([65, 58, 40])
        entries = _make_entries({1: 5.0, 2: 10.0, 3: 15.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(16), entries)
        assert result == "C"


# ---------------------------------------------------------------------------
# 3. C Rank (SKIP) — ◎score < 68
# ---------------------------------------------------------------------------

class TestCRank:
    """C = SKIP: ◎score < 68."""

    def test_low_score(self):
        predictions = _make_predictions([60, 55, 40])
        entries = _make_entries({1: 3.0, 2: 5.0, 3: 10.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(14), entries)
        assert result == "C"

    def test_score_just_below_threshold(self):
        """◎score=67.9 → 'C'."""
        predictions = _make_predictions([67.9, 50, 30])
        entries = _make_entries({1: 3.0, 2: 8.0, 3: 15.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(12), entries)
        assert result == "C"

    def test_low_score_good_odds_still_c(self):
        """◎score=60, odds=3.0 → 'C' (score不足)."""
        predictions = _make_predictions([60, 40, 20])
        entries = _make_entries({1: 3.0, 2: 8.0, 3: 15.0})
        result = evaluate_bet_confidence(predictions, _make_race_info(8), entries)
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
        """D7 invariant: output is always 'A', 'B', or 'C'."""
        test_cases = [
            (_make_predictions([90, 60, 40]), _make_race_info(8), _make_entries({1: 3.0})),
            (_make_predictions([50, 49, 48]), _make_race_info(18), None),
            (_make_predictions([70, 50, 30]), _make_race_info(12), _make_entries({1: 1.2})),
            (_make_predictions([68, 60, 40]), _make_race_info(16), _make_entries({1: 10.0})),
            ([], _make_race_info(16), None),
        ]
        for predictions, race_info, entries in test_cases:
            result = evaluate_bet_confidence(predictions, race_info, entries)
            assert result in ("A", "B", "C"), (
                f"Unexpected grade '{result}' for predictions={predictions}"
            )

    def test_shoubu_min_score_constant_is_68(self):
        assert SHOUBU_MIN_SCORE == 68.0
