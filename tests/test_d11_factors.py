"""Tests for D11 factors: trackBias and workoutEval."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.predictor.factors import calc_track_bias, calc_workout_eval


# ---------------------------------------------------------------------------
# trackBias
# ---------------------------------------------------------------------------

class TestTrackBias:

    def test_no_same_day_results_returns_neutral(self):
        assert calc_track_bias(1, [], None, "芝") == 50.0

    def test_empty_results_returns_neutral(self):
        assert calc_track_bias(1, [], [], "芝") == 50.0

    def test_too_few_results_returns_neutral(self):
        results = [{"pos": 1, "frame": 1, "runningStyle": "逃げ", "surface": "芝"}] * 5
        assert calc_track_bias(1, [], results, "芝") == 50.0

    def test_inner_bias_favors_inner_frame(self):
        results = [
            {"pos": 1, "frame": 2, "runningStyle": "先行", "surface": "芝"},
            {"pos": 2, "frame": 1, "runningStyle": "逃げ", "surface": "芝"},
            {"pos": 3, "frame": 3, "runningStyle": "先行", "surface": "芝"},
            {"pos": 1, "frame": 2, "runningStyle": "先行", "surface": "芝"},
            {"pos": 2, "frame": 4, "runningStyle": "差し", "surface": "芝"},
            {"pos": 3, "frame": 1, "runningStyle": "先行", "surface": "芝"},
            {"pos": 4, "frame": 7, "runningStyle": "追込", "surface": "芝"},
            {"pos": 5, "frame": 6, "runningStyle": "差し", "surface": "芝"},
        ]
        inner_score = calc_track_bias(2, [], results, "芝")
        outer_score = calc_track_bias(7, [], results, "芝")
        assert inner_score > outer_score

    def test_outer_bias_favors_outer_frame(self):
        results = [
            {"pos": 1, "frame": 7, "runningStyle": "差し", "surface": "芝"},
            {"pos": 2, "frame": 6, "runningStyle": "差し", "surface": "芝"},
            {"pos": 3, "frame": 8, "runningStyle": "追込", "surface": "芝"},
            {"pos": 1, "frame": 5, "runningStyle": "先行", "surface": "芝"},
            {"pos": 2, "frame": 7, "runningStyle": "差し", "surface": "芝"},
            {"pos": 3, "frame": 6, "runningStyle": "追込", "surface": "芝"},
            {"pos": 4, "frame": 2, "runningStyle": "逃げ", "surface": "芝"},
            {"pos": 5, "frame": 1, "runningStyle": "先行", "surface": "芝"},
        ]
        outer_score = calc_track_bias(7, [], results, "芝")
        inner_score = calc_track_bias(2, [], results, "芝")
        assert outer_score > inner_score

    def test_filters_by_surface(self):
        results = [
            {"pos": 1, "frame": 1, "runningStyle": "逃げ", "surface": "ダート"},
        ] * 10
        score = calc_track_bias(1, [], results, "芝")
        assert score == 50.0

    def test_running_style_bias(self):
        past_closer = [{"runningStyle": "差し"}]
        past_front = [{"runningStyle": "逃げ"}]
        results = [
            {"pos": 1, "frame": 4, "runningStyle": "差し", "surface": "芝"},
            {"pos": 2, "frame": 5, "runningStyle": "追込", "surface": "芝"},
            {"pos": 3, "frame": 3, "runningStyle": "差し", "surface": "芝"},
            {"pos": 1, "frame": 6, "runningStyle": "差し", "surface": "芝"},
            {"pos": 2, "frame": 4, "runningStyle": "追込", "surface": "芝"},
            {"pos": 3, "frame": 5, "runningStyle": "差し", "surface": "芝"},
            {"pos": 4, "frame": 1, "runningStyle": "逃げ", "surface": "芝"},
            {"pos": 5, "frame": 2, "runningStyle": "先行", "surface": "芝"},
        ]
        closer_score = calc_track_bias(4, past_closer, results, "芝")
        front_score = calc_track_bias(4, past_front, results, "芝")
        assert closer_score > front_score

    def test_no_frame_returns_neutral(self):
        results = [{"pos": 1, "frame": 1, "runningStyle": "逃げ", "surface": "芝"}] * 10
        assert calc_track_bias(0, [], results, "芝") == 50.0


# ---------------------------------------------------------------------------
# workoutEval
# ---------------------------------------------------------------------------

class TestWorkoutEval:

    def test_no_workout_returns_neutral(self):
        assert calc_workout_eval({}) == 50.0
        assert calc_workout_eval(None) == 50.0

    def test_rank_a_high_score(self):
        score = calc_workout_eval({"rank": "A"})
        assert score >= 70.0

    def test_rank_b_above_neutral(self):
        score = calc_workout_eval({"rank": "B"})
        assert score > 50.0

    def test_rank_c_neutral(self):
        score = calc_workout_eval({"rank": "C"})
        assert score == 50.0

    def test_rank_d_below_neutral(self):
        score = calc_workout_eval({"rank": "D"})
        assert score < 50.0

    def test_strong_training_bonus(self):
        base = calc_workout_eval({"rank": "B"})
        strong = calc_workout_eval({"rank": "B", "strong": True})
        assert strong > base

    def test_strong_training_no_bonus_on_poor_rank(self):
        base = calc_workout_eval({"rank": "D"})
        strong = calc_workout_eval({"rank": "D", "strong": True})
        assert strong == base  # No bonus for poor grade

    def test_slope_course_bonus(self):
        base = calc_workout_eval({"rank": "B"})
        slope = calc_workout_eval({"rank": "B", "course": "坂路"})
        assert slope > base

    def test_no_rank_returns_neutral(self):
        assert calc_workout_eval({"comment": "動き良好"}) == 50.0

    def test_max_score_capped(self):
        score = calc_workout_eval({"rank": "A", "strong": True, "course": "坂路"})
        assert score <= 80.0

    def test_min_score_capped(self):
        score = calc_workout_eval({"rank": "E"})
        assert score >= 20.0

    def test_case_insensitive_rank(self):
        lower = calc_workout_eval({"rank": "a"})
        upper = calc_workout_eval({"rank": "A"})
        assert lower == upper
