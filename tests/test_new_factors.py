"""TDD tests for the 7 new factor calculators, enhanced _parse_past_race_td,
and mc_samples parameter in optimize_bets.

RED -> GREEN -> REFACTOR cycle verified for each function.

Coverage targets:
  - factors.py new functions: 90%+
  - _parse_past_race_td expanded fields: 80%+
  - mc_samples parameter: 100%
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock


# ===========================================================================
# HELPERS
# ===========================================================================

def _make_td(text: str, classes: list[str] | None = None):
    """Build a minimal BeautifulSoup-like mock for a <td> element."""
    td = MagicMock()
    td.get_text.return_value = text
    td.get.side_effect = lambda attr, default=None: (
        classes if attr == "class" else default
    )
    return td


# ===========================================================================
# 1. calc_same_distance_performance
# ===========================================================================

class TestCalcSameDistancePerformance:
    def _fn(self):
        from backend.predictor.factors import calc_same_distance_performance
        return calc_same_distance_performance

    # --- defaults ---
    def test_empty_races_returns_50(self):
        assert self._fn()([], 1600) == 50.0

    def test_zero_target_distance_returns_50(self):
        races = [{"distance": 1600, "pos": 1}]
        assert self._fn()(races, 0) == 50.0

    def test_no_distance_match_returns_48(self):
        # All past races at 1200, target 2000 => 800m gap > 200m allowed
        races = [{"distance": 1200, "pos": 1}, {"distance": 1200, "pos": 2}]
        assert self._fn()(races, 2000) == 48.0

    # --- scoring tiers ---
    def test_avg_pos_winner_returns_95(self):
        # avg_pos = 1.0 (<= 1.5 tier)
        races = [{"distance": 1600, "pos": 1}]
        assert self._fn()(races, 1600) == 95.0

    def test_avg_pos_1_5_boundary_returns_95(self):
        # avg = (1+2)/2 = 1.5 -> still <= 1.5
        races = [{"distance": 1600, "pos": 1}, {"distance": 1600, "pos": 2}]
        assert self._fn()(races, 1600) == 95.0

    def test_avg_pos_2_returns_80(self):
        # avg = (2+3)/2 = 2.5 -> 1.5 < avg <= 3
        races = [{"distance": 1600, "pos": 2}, {"distance": 1600, "pos": 3}]
        assert self._fn()(races, 1600) == 80.0

    def test_avg_pos_4_returns_65(self):
        # avg = (4+5)/2 = 4.5 -> 3 < avg <= 5
        races = [{"distance": 1600, "pos": 4}, {"distance": 1600, "pos": 5}]
        assert self._fn()(races, 1600) == 65.0

    def test_avg_pos_6_returns_50(self):
        # avg = (6+8)/2 = 7.0 -> 5 < avg <= 8
        races = [{"distance": 1600, "pos": 6}, {"distance": 1600, "pos": 8}]
        assert self._fn()(races, 1600) == 50.0

    def test_avg_pos_9_returns_35(self):
        # avg = 9.0 > 8
        races = [{"distance": 1600, "pos": 9}, {"distance": 1600, "pos": 9}]
        assert self._fn()(races, 1600) == 35.0

    def test_within_200m_boundary_counts(self):
        # dist = 1800, target = 2000 => gap = 200 (exactly <=200, should match)
        races = [{"distance": 1800, "pos": 1}]
        assert self._fn()(races, 2000) == 95.0

    def test_just_outside_200m_excluded(self):
        # dist = 1799, target = 2000 => gap = 201 > 200 => no match
        races = [{"distance": 1799, "pos": 1}]
        assert self._fn()(races, 2000) == 48.0

    def test_pos_zero_excluded(self):
        # pos=0 should be ignored
        races = [{"distance": 1600, "pos": 0}]
        assert self._fn()(races, 1600) == 48.0

    def test_looks_back_max_6_races(self):
        fn = self._fn()
        # 6 races at target distance with pos=1, race 7 should be ignored
        races = [{"distance": 1600, "pos": 1}] * 6 + [{"distance": 1600, "pos": 15}]
        assert fn(races, 1600) == 95.0


# ===========================================================================
# 2. calc_same_surface_performance
# ===========================================================================

class TestCalcSameSurfacePerformance:
    def _fn(self):
        from backend.predictor.factors import calc_same_surface_performance
        return calc_same_surface_performance

    def test_empty_races_returns_50(self):
        assert self._fn()([], "芝") == 50.0

    def test_empty_target_surface_returns_50(self):
        races = [{"surface": "芝", "pos": 1}]
        assert self._fn()(races, "") == 50.0

    def test_no_matching_surface_returns_40(self):
        # Past races all on ダ, target is 芝
        races = [{"surface": "ダ", "pos": 1}, {"surface": "ダ", "pos": 2}]
        assert self._fn()(races, "芝") == 40.0

    def test_avg_pos_1_turf_specialist(self):
        races = [{"surface": "芝", "pos": 1}]
        assert self._fn()(races, "芝") == 95.0

    def test_avg_pos_2_5_returns_80(self):
        races = [{"surface": "芝", "pos": 2}, {"surface": "芝", "pos": 3}]
        assert self._fn()(races, "芝") == 80.0

    def test_avg_pos_4_returns_65(self):
        races = [{"surface": "芝", "pos": 4}, {"surface": "芝", "pos": 4}]
        assert self._fn()(races, "芝") == 65.0

    def test_avg_pos_7_returns_50(self):
        races = [{"surface": "芝", "pos": 7}]
        assert self._fn()(races, "芝") == 50.0

    def test_avg_pos_9_returns_35(self):
        races = [{"surface": "ダ", "pos": 9}, {"surface": "ダ", "pos": 9}]
        assert self._fn()(races, "ダ") == 35.0

    def test_dirt_specialist_on_dirt(self):
        races = [{"surface": "ダ", "pos": 1}, {"surface": "ダ", "pos": 1}]
        assert self._fn()(races, "ダ") == 95.0

    def test_mixed_surface_only_matching_counted(self):
        # 1st on 芝, 8th on ダ: target=芝 -> only the 1st is counted
        races = [
            {"surface": "芝", "pos": 1},
            {"surface": "ダ", "pos": 8},
        ]
        assert self._fn()(races, "芝") == 95.0

    def test_pos_zero_excluded(self):
        races = [{"surface": "芝", "pos": 0}]
        assert self._fn()(races, "芝") == 40.0


# ===========================================================================
# 3. calc_same_condition_performance
# ===========================================================================

class TestCalcSameConditionPerformance:
    def _fn(self):
        from backend.predictor.factors import calc_same_condition_performance
        return calc_same_condition_performance

    def test_empty_races_returns_50(self):
        assert self._fn()([], "良") == 50.0

    def test_empty_condition_returns_50(self):
        races = [{"condition": "良", "pos": 1}]
        assert self._fn()(races, "") == 50.0

    def test_no_matching_condition_returns_42(self):
        # All races on 良, target is 重 (soft)
        races = [{"condition": "良", "pos": 1}, {"condition": "良", "pos": 1}]
        assert self._fn()(races, "重") == 42.0

    def test_avg_pos_1_returns_90(self):
        races = [{"condition": "良", "pos": 1}]
        assert self._fn()(races, "良") == 90.0

    def test_avg_pos_2_5_returns_75(self):
        races = [{"condition": "良", "pos": 2}, {"condition": "良", "pos": 3}]
        assert self._fn()(races, "良") == 75.0

    def test_avg_pos_4_returns_60(self):
        races = [{"condition": "良", "pos": 4}]
        assert self._fn()(races, "良") == 60.0

    def test_avg_pos_7_returns_48(self):
        races = [{"condition": "良", "pos": 6}, {"condition": "良", "pos": 8}]
        assert self._fn()(races, "良") == 48.0

    def test_avg_pos_9_returns_35(self):
        races = [{"condition": "良", "pos": 9}]
        assert self._fn()(races, "良") == 35.0

    def test_heavy_grouped_with_soft(self):
        # 稍重, 重, 不良 are all "not firm" -> should match each other
        races = [
            {"condition": "稍重", "pos": 1},
            {"condition": "重", "pos": 1},
            {"condition": "不良", "pos": 1},
        ]
        assert self._fn()(races, "重") == 90.0

    def test_firm_does_not_match_soft(self):
        races = [{"condition": "良", "pos": 1}]
        assert self._fn()(races, "稍重") == 42.0

    def test_soft_does_not_match_firm(self):
        races = [{"condition": "稍重", "pos": 1}]
        assert self._fn()(races, "良") == 42.0


# ===========================================================================
# 4. calc_running_style_consistency
# ===========================================================================

class TestCalcRunningStyleConsistency:
    def _fn(self):
        from backend.predictor.factors import calc_running_style_consistency
        return calc_running_style_consistency

    def test_empty_returns_50(self):
        assert self._fn()([]) == 50.0

    def test_single_race_returns_50(self):
        races = [{"runningStyle": "逃げ"}]
        assert self._fn()(races) == 50.0

    def test_no_style_data_returns_50(self):
        races = [{"pos": 1}, {"pos": 2}, {"pos": 3}]
        assert self._fn()(races) == 50.0

    def test_100_percent_consistent_returns_80(self):
        races = [{"runningStyle": "逃げ"} for _ in range(5)]
        assert self._fn()(races) == 80.0

    def test_50_percent_consistent_returns_55(self):
        # 2 same style out of 4 -> ratio = 0.5 -> 30 + 50*0.5 = 55
        races = [
            {"runningStyle": "逃げ"},
            {"runningStyle": "逃げ"},
            {"runningStyle": "差し"},
            {"runningStyle": "追込"},
        ]
        result = self._fn()(races)
        assert abs(result - 55.0) < 0.01

    def test_result_within_bounds(self):
        import itertools
        styles = ["逃げ", "先行", "差し", "追込"]
        for combo in itertools.product(styles, repeat=3):
            races = [{"runningStyle": s} for s in combo]
            score = self._fn()(races)
            assert 30.0 <= score <= 80.0

    def test_score_formula(self):
        # 3 out of 4 same -> ratio = 0.75 -> 30 + 50*0.75 = 67.5
        races = [
            {"runningStyle": "先行"},
            {"runningStyle": "先行"},
            {"runningStyle": "先行"},
            {"runningStyle": "差し"},
        ]
        result = self._fn()(races)
        assert abs(result - 67.5) < 0.01

    def test_ignores_empty_style_entries(self):
        # Only 1 valid style entry among 3 total => len(styles) < 2 => return 50
        races = [
            {"runningStyle": "逃げ"},
            {"runningStyle": ""},
            {"pos": 1},
        ]
        assert self._fn()(races) == 50.0

    def test_looks_back_max_5_races(self):
        # 5 races with 逃げ, then a 6th with 追込 (should be ignored)
        races = [{"runningStyle": "逃げ"}] * 5 + [{"runningStyle": "追込"}]
        assert self._fn()(races) == 80.0


# ===========================================================================
# 5. calc_speed_figure
# ===========================================================================

class TestCalcSpeedFigure:
    def _fn(self):
        from backend.predictor.factors import calc_speed_figure
        return calc_speed_figure

    def test_empty_races_returns_50(self):
        assert self._fn()([], 1600) == 50.0

    def test_zero_distance_returns_50(self):
        races = [{"distance": 1600, "finishTime": "1:34.2", "pos": 1}]
        assert self._fn()(races, 0) == 50.0

    def test_no_matching_distance_returns_50(self):
        # Past at 1200, target 2400 => 1200m gap > 400m
        races = [{"distance": 1200, "finishTime": "1:10.5", "pos": 1}]
        assert self._fn()(races, 2400) == 50.0

    def test_missing_time_returns_50(self):
        races = [{"distance": 1600, "finishTime": "", "pos": 1}]
        assert self._fn()(races, 1600) == 50.0

    def test_invalid_time_format_returns_50(self):
        races = [{"distance": 1600, "finishTime": "notaTime", "pos": 1}]
        assert self._fn()(races, 1600) == 50.0

    def test_fast_speed_scores_high(self):
        # 2000m in 2:00.0 = 120s -> speed = 2000/120 ≈ 16.67 m/s
        # score = 30 + (16.67-15)*25 = 30 + 41.67 ≈ 71.67
        races = [{"distance": 2000, "finishTime": "2:00.0"}]
        score = self._fn()(races, 2000)
        assert score > 65.0

    def test_very_slow_speed_clamped_at_20(self):
        # 2000m in 3:00.0 = 180s -> speed = 11.11 m/s
        # raw score = 30 + (11.11-15)*25 = 30 - 97.2 < 0 => clamp to 20
        races = [{"distance": 2000, "finishTime": "3:00.0"}]
        score = self._fn()(races, 2000)
        assert score == 20.0

    def test_score_clamped_at_95_max(self):
        # 2000m in 1:40.0 = 100s -> speed = 20 m/s -> raw = 30 + 125 = 155 => clamp to 95
        races = [{"distance": 2000, "finishTime": "1:40.0"}]
        score = self._fn()(races, 2000)
        assert score == 95.0

    def test_within_400m_counts(self):
        # target 2000, past at 1600 => 400m gap (boundary, should count)
        races = [{"distance": 1600, "finishTime": "1:36.0"}]
        # 1600m / 96s = 16.67 m/s -> score > 50
        score = self._fn()(races, 2000)
        assert score > 50.0

    def test_just_outside_400m_excluded(self):
        # target 2000, past at 1599 => 401m gap > 400m
        races = [{"distance": 1599, "finishTime": "1:36.0"}]
        score = self._fn()(races, 2000)
        assert score == 50.0

    def test_multiple_races_averaged(self):
        # Both races: 1600m in 1:36.0 = 96s -> 16.67 m/s each -> same result
        races = [
            {"distance": 1600, "finishTime": "1:36.0"},
            {"distance": 1600, "finishTime": "1:36.0"},
        ]
        single = self._fn()([{"distance": 1600, "finishTime": "1:36.0"}], 1600)
        multi = self._fn()(races, 1600)
        assert abs(single - multi) < 0.01

    def test_malformed_time_string_raises_no_exception(self):
        # "1:ab.c" causes ValueError inside time_to_sec -> graceful return
        races = [{"distance": 1600, "finishTime": "1:ab.c"}]
        score = self._fn()(races, 1600)
        assert score == 50.0

    def test_zero_seconds_time_skipped(self):
        # A time string that parses to 0 seconds should be skipped
        # "0:00.0" => 0 seconds => secs <= 0 branch
        races = [{"distance": 1600, "finishTime": "0:00.0"}]
        score = self._fn()(races, 1600)
        assert score == 50.0

    def test_benchmark_17_mps_scores_80(self):
        # According to docstring: 17 m/s -> score 80
        # speed = 17 -> score = 30 + (17-15)*25 = 30+50 = 80
        # Need distance/time giving exactly 17 m/s: e.g. 1700m / 100s = 17 m/s
        # Use 2000m at target distance ± 400m rule: set past distance = 2000
        # 2000m / (2000/17) = 17 m/s. time_str = M:SS.f
        # 2000/17 ≈ 117.647s → 1:57.6 (1m 57.6s)
        races = [{"distance": 1700, "finishTime": "1:40.0"}]  # 1700/100=17 m/s
        score = self._fn()(races, 1700)
        assert abs(score - 80.0) < 0.5


# ===========================================================================
# 6. calc_weight_carried_trend
# ===========================================================================

class TestCalcWeightCarriedTrend:
    def _fn(self):
        from backend.predictor.factors import calc_weight_carried_trend
        return calc_weight_carried_trend

    def test_empty_races_returns_50(self):
        assert self._fn()([], 57.0) == 50.0

    def test_zero_current_weight_returns_50(self):
        races = [{"weightCarried": 57.0}]
        assert self._fn()(races, 0) == 50.0

    def test_no_past_weight_data_returns_50(self):
        races = [{"weightCarried": 0}]
        assert self._fn()(races, 57.0) == 50.0

    def test_same_weight_returns_50(self):
        races = [{"weightCarried": 57.0}, {"weightCarried": 57.0}]
        assert self._fn()(races, 57.0) == 50.0

    def test_lighter_current_weight_scores_higher(self):
        # avg past = 57, current = 55 -> delta = -2 -> score = 50 - (-2)*7.5 = 65
        races = [{"weightCarried": 57.0}]
        score = self._fn()(races, 55.0)
        assert abs(score - 65.0) < 0.01

    def test_heavier_current_weight_scores_lower(self):
        # avg past = 57, current = 59 -> delta = +2 -> score = 50 - 2*7.5 = 35
        races = [{"weightCarried": 57.0}]
        score = self._fn()(races, 59.0)
        assert abs(score - 35.0) < 0.01

    def test_score_clamped_at_80_max(self):
        # avg past = 60, current = 49 -> delta = -11 -> raw = 50+82.5 = 132.5 > 80
        races = [{"weightCarried": 60.0}]
        score = self._fn()(races, 49.0)
        assert score == 80.0

    def test_score_clamped_at_25_min(self):
        # avg past = 49, current = 60 -> delta = +11 -> raw = 50-82.5 < 25
        races = [{"weightCarried": 49.0}]
        score = self._fn()(races, 60.0)
        assert score == 25.0

    def test_uses_up_to_4_past_races(self):
        # 4 valid races: avg=57. 5th should be ignored
        races = [
            {"weightCarried": 57.0},
            {"weightCarried": 57.0},
            {"weightCarried": 57.0},
            {"weightCarried": 57.0},
            {"weightCarried": 99.0},  # 5th race ignored
        ]
        score = self._fn()(races, 57.0)
        assert score == 50.0

    def test_skips_zero_weight_entries(self):
        races = [{"weightCarried": 0}, {"weightCarried": 57.0}]
        score = self._fn()(races, 57.0)
        assert score == 50.0


# ===========================================================================
# 7. calc_days_since_last_race
# ===========================================================================

class TestCalcDaysSinceLastRace:
    def _fn(self):
        from backend.predictor.factors import calc_days_since_last_race
        return calc_days_since_last_race

    def test_empty_races_returns_50(self):
        assert self._fn()([], "2026.04.15") == 50.0

    def test_missing_current_date_returns_50(self):
        races = [{"date": "2026.03.01"}]
        assert self._fn()(races, "") == 50.0

    def test_missing_last_race_date_returns_50(self):
        races = [{"pos": 1}]
        assert self._fn()(races, "2026.04.15") == 50.0

    def test_7_days_returns_50(self):
        # 0-14 days -> 50
        races = [{"date": "2026.04.08"}]
        assert self._fn()(races, "2026.04.15") == 50.0

    def test_14_days_boundary_returns_50(self):
        races = [{"date": "2026.04.01"}]
        assert self._fn()(races, "2026.04.15") == 50.0

    def test_21_days_returns_55(self):
        # 15-28 days -> 55
        races = [{"date": "2026.03.25"}]
        assert self._fn()(races, "2026.04.15") == 55.0

    def test_45_days_returns_60(self):
        # 29-60 days -> 60 (peak)
        races = [{"date": "2026.03.01"}]
        assert self._fn()(races, "2026.04.15") == 60.0

    def test_90_days_returns_55(self):
        # 61-120 days -> 55
        races = [{"date": "2026.01.15"}]
        assert self._fn()(races, "2026.04.15") == 55.0

    def test_150_days_returns_45(self):
        # 121-180 days -> 45
        races = [{"date": "2025.11.16"}]
        assert self._fn()(races, "2026.04.15") == 45.0

    def test_200_days_returns_35(self):
        # 180+ days -> 35
        races = [{"date": "2025.10.01"}]
        assert self._fn()(races, "2026.04.15") == 35.0

    def test_future_last_race_returns_50(self):
        # current < last_race => days < 0 => 50
        races = [{"date": "2026.04.20"}]
        assert self._fn()(races, "2026.04.15") == 50.0

    def test_slash_date_format_also_works(self):
        # Parser handles "/" as separator too
        races = [{"date": "2026/03/01"}]
        assert self._fn()(races, "2026/04/15") == 60.0

    def test_invalid_date_returns_50(self):
        races = [{"date": "not-a-date"}]
        assert self._fn()(races, "2026.04.15") == 50.0

    def test_uses_only_first_race(self):
        # Uses past_races[0] as last race, ignores rest
        races = [
            {"date": "2026.04.08"},   # 7 days -> 50
            {"date": "2025.10.01"},   # very old -> would give 35
        ]
        assert self._fn()(races, "2026.04.15") == 50.0

    def test_boundary_15_days_returns_55(self):
        races = [{"date": "2026.03.31"}]
        assert self._fn()(races, "2026.04.15") == 55.0

    def test_boundary_29_days_returns_60(self):
        races = [{"date": "2026.03.17"}]
        assert self._fn()(races, "2026.04.15") == 60.0

    def test_malformed_date_in_race_returns_50(self):
        # Totally invalid date string => parse_date returns None
        races = [{"date": "not-a-date"}]
        assert self._fn()(races, "2026.04.15") == 50.0

    def test_malformed_current_date_returns_50(self):
        races = [{"date": "2026.03.01"}]
        assert self._fn()(races, "bad-date") == 50.0

    def test_invalid_month_raises_no_exception_returns_50(self):
        # "2026.13.01" has valid format but month=13 raises ValueError in date()
        races = [{"date": "2026.13.01"}]
        assert self._fn()(races, "2026.04.15") == 50.0


# ===========================================================================
# 8. _parse_past_race_td  -- expanded 14-field parsing
# ===========================================================================

SAMPLE_TEXT = "2026.02.28 阪神10仁川SLダ2000 2:04.8良16頭 8番 4人 藤岡佑介 58.514-14-14-"


class TestParsePastRaceTd:
    def _fn(self):
        from backend.scraper.netkeiba import _parse_past_race_td
        return _parse_past_race_td

    def _parse(self, text: str, classes: list[str] | None = None):
        td = _make_td(text, classes or [])
        return self._fn()(td)

    def test_returns_none_for_empty_text(self):
        td = _make_td("")
        td.get.return_value = []
        assert self._fn()(td) is None

    # --- original 6 fields ---
    def test_parses_surface_dirt(self):
        result = self._parse(SAMPLE_TEXT)
        assert result["surface"] == "ダ"

    def test_parses_distance(self):
        result = self._parse(SAMPLE_TEXT)
        assert result["distance"] == 2000

    def test_parses_track(self):
        result = self._parse(SAMPLE_TEXT)
        assert result["track"] == "阪神"

    def test_parses_direction(self):
        result = self._parse(SAMPLE_TEXT)
        assert result["direction"] == "右"

    def test_parses_condition_good(self):
        result = self._parse(SAMPLE_TEXT)
        assert result["condition"] == "良"

    # --- new 8 fields ---
    def test_parses_date(self):
        result = self._parse(SAMPLE_TEXT)
        assert result["date"] == "2026.02.28"

    def test_parses_finish_time(self):
        result = self._parse(SAMPLE_TEXT)
        assert result["finishTime"] == "2:04.8"

    def test_parses_field_size(self):
        result = self._parse(SAMPLE_TEXT)
        assert result["fieldSize"] == 16

    def test_parses_post_position(self):
        result = self._parse(SAMPLE_TEXT)
        assert result["postPosition"] == 8

    def test_parses_popularity(self):
        result = self._parse(SAMPLE_TEXT)
        assert result["popularity"] == 4

    def test_parses_weight_carried(self):
        result = self._parse(SAMPLE_TEXT)
        assert result["weightCarried"] == 58.5

    def test_parses_corners(self):
        result = self._parse(SAMPLE_TEXT)
        assert result["corners"] == [14, 14, 14]

    def test_running_style_derived_as_追込(self):
        # corners avg = 14, field_size = 16 -> ratio = 14/16 = 0.875 > 0.75 -> 追込
        result = self._parse(SAMPLE_TEXT)
        assert result["runningStyle"] == "追込"

    # --- running style tiers ---
    def test_running_style_逃げ(self):
        # 16-head field, avg corner = 4 -> ratio = 4/16 = 0.25 -> 逃げ
        text = "2026.02.28 阪神10ダ2000 2:04.8良16頭 8番 4人 騎手名 58.54-4-4-4-"
        result = self._parse(text)
        assert result["runningStyle"] == "逃げ"

    def test_running_style_先行(self):
        # avg corner = 7, field_size = 16 -> ratio = 7/16 = 0.4375 -> 先行
        text = "2026.02.28 東京11ダ2000 2:04.8良16頭 8番 4人 騎手名 58.57-7-7-7-"
        result = self._parse(text)
        assert result["runningStyle"] == "先行"

    def test_running_style_差し(self):
        # avg corner = 10, field_size = 16 -> ratio = 0.625 -> 差し
        text = "2026.02.28 東京11ダ2000 2:04.8良16頭 8番 4人 騎手名 58.510-10-10-10-"
        result = self._parse(text)
        assert result["runningStyle"] == "差し"

    def test_no_corners_empty_running_style(self):
        text = "2026.02.28 阪神10ダ2000 2:04.8良16頭 8番 4人 騎手名 58.5"
        result = self._parse(text)
        assert result["corners"] == []
        assert result["runningStyle"] == ""

    def test_no_field_size_empty_running_style(self):
        # No "N頭" -> field_size = 0 -> running style not computed
        text = "2026.02.28 阪神10ダ2000 2:04.8良 8番 4人 騎手名 58.514-14-14-"
        result = self._parse(text)
        assert result["runningStyle"] == ""

    def test_ranking_class_sets_pos(self):
        td = _make_td(SAMPLE_TEXT, ["Ranking_3"])
        result = self._fn()(td)
        assert result["pos"] == 3

    def test_pos_from_text_fallback(self):
        # No ranking class, use "N着" from text
        text = "3着 " + SAMPLE_TEXT
        result = self._parse(text, [])
        assert result["pos"] == 3

    def test_turf_surface_parsed(self):
        text = "2026.03.01 東京11芝2400 2:23.8良16頭 3番 1人 ルメール 57.01-1-1-1-"
        result = self._parse(text)
        assert result["surface"] == "芝"
        assert result["direction"] == "左"

    def test_condition_稍重(self):
        text = "2026.02.28 阪神10ダ2000 2:06.3稍16頭 8番 4人 騎手名 58.514-14-14-"
        result = self._parse(text)
        assert result["condition"] == "稍重"

    def test_condition_重(self):
        text = "2026.02.28 阪神10ダ2000 2:08.0重16頭 8番 4人 騎手名 58.514-14-14-"
        result = self._parse(text)
        assert result["condition"] == "重"

    def test_output_has_all_14_keys(self):
        result = self._parse(SAMPLE_TEXT)
        expected_keys = {
            "pos", "condition", "surface", "distance", "track", "direction",
            "date", "finishTime", "fieldSize", "postPosition", "popularity",
            "weightCarried", "corners", "runningStyle",
        }
        assert expected_keys == set(result.keys())


# ===========================================================================
# 9. optimize_bets — mc_samples parameter
# ===========================================================================

class TestOptimizeBetsMcSamples:
    """Verify the new mc_samples parameter is respected by optimize_bets."""

    def _make_predictions(self):
        return [
            {
                "horseNumber": i, "frameNumber": ((i - 1) // 2) + 1,
                "score": max(10, 90 - i * 7), "isScratched": False,
            }
            for i in range(1, 9)
        ]

    def _make_race_info(self):
        return {
            "distance": 1800, "surface": "芝", "trackCondition": "良",
            "headCount": 8, "racecourseCode": "05",
        }

    def test_mc_samples_default_returns_bets(self):
        from backend.predictor.bet_optimizer import optimize_bets
        preds = self._make_predictions()
        result = optimize_bets(preds, {}, self._make_race_info())
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_mc_samples_small_still_returns_bets(self):
        from backend.predictor.bet_optimizer import optimize_bets
        preds = self._make_predictions()
        result = optimize_bets(preds, {}, self._make_race_info(), mc_samples=10)
        assert isinstance(result, list)

    def test_mc_samples_500_returns_bets(self):
        from backend.predictor.bet_optimizer import optimize_bets
        preds = self._make_predictions()
        result = optimize_bets(preds, {}, self._make_race_info(), mc_samples=500)
        assert isinstance(result, list)

    def test_mc_samples_1_returns_bets(self):
        # Extreme edge: 1 MC sample should still produce output
        from backend.predictor.bet_optimizer import optimize_bets
        preds = self._make_predictions()
        result = optimize_bets(preds, {}, self._make_race_info(), mc_samples=1)
        assert isinstance(result, list)

    def test_mc_samples_parameter_accepted(self):
        """Verify the parameter signature accepts mc_samples."""
        import inspect
        from backend.predictor import bet_optimizer
        sig = inspect.signature(bet_optimizer.optimize_bets)
        assert "mc_samples" in sig.parameters

    def test_mc_samples_default_is_MC_SAMPLES_constant(self):
        import inspect
        from backend.predictor import bet_optimizer
        sig = inspect.signature(bet_optimizer.optimize_bets)
        default = sig.parameters["mc_samples"].default
        assert default == bet_optimizer.MC_SAMPLES

    def test_mc_samples_large_still_works(self):
        from backend.predictor.bet_optimizer import optimize_bets
        preds = self._make_predictions()
        # 100 samples is fast enough for a test
        result = optimize_bets(preds, {}, self._make_race_info(), mc_samples=100)
        assert isinstance(result, list)
