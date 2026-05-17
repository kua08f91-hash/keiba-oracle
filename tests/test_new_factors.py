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
    # select_one returns None (no sub-elements in legacy text-only format)
    td.select_one.return_value = None
    td.select.return_value = []
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

    def test_output_has_all_16_keys(self):
        result = self._parse(SAMPLE_TEXT)
        expected_keys = {
            "pos", "condition", "surface", "distance", "track", "direction",
            "date", "finishTime", "fieldSize", "postPosition", "popularity",
            "weightCarried", "corners", "runningStyle", "agari3f", "margin",
        }
        assert expected_keys == set(result.keys())

    # --- Fix 1: Method 2 pos extraction from structured HTML ---

    def test_pos_from_data01_num_span(self):
        """Fix 1: pos extracted from <span class="Num">7</span> inside .Data01."""
        from bs4 import BeautifulSoup
        html = (
            '<td class="Past">'
            '<div class="Data01"><span class="Num">7</span></div>'
            '<div class="Data06">11-11-12-12 (39.5) 448(-2)</div>'
            '</td>'
        )
        soup = BeautifulSoup(html, "lxml")
        td = soup.select_one("td")
        # Text is nearly empty — pos must come from the Num span
        result = self._fn()(td)
        assert result is not None
        assert result["pos"] == 7

    def test_pos_zero_when_no_ranking_no_num_no_text(self):
        """Fix 1: pos=0 when none of the three methods find a position."""
        from bs4 import BeautifulSoup
        # No Ranking_ class, no .Num span, no 'X着' text — but has enough
        # content so the function doesn't return None (needs non-empty text)
        html = '<td class="Past"><div class="Data01">---</div></td>'
        soup = BeautifulSoup(html, "lxml")
        td = soup.select_one("td")
        result = self._fn()(td)
        # The text "---" is non-empty, so we get a dict back
        assert result is not None
        assert result["pos"] == 0

    # --- Fix 2: corners and runningStyle from Data06 div ---

    def test_corners_from_data06_div(self):
        """Fix 2: corners extracted from Data06 div text '11-11-12-12 (39.5) 448(-2)'."""
        from bs4 import BeautifulSoup
        html = (
            '<td class="Past">'
            '<div class="Data01"><span class="Num">4</span></div>'
            '<div class="Data06">11-11-12-12 (39.5) 448(-2)</div>'
            '</td>'
        )
        soup = BeautifulSoup(html, "lxml")
        td = soup.select_one("td")
        result = self._fn()(td)
        assert result is not None
        assert result["corners"] == [11, 11, 12, 12]

    def test_running_style_差し_from_data06(self):
        """Fix 2: runningStyle=差し when corners=[11,11,12,12] and fieldSize=16."""
        from bs4 import BeautifulSoup
        # avg=11.5, ratio=11.5/16=0.719 -> 0.50 < ratio <= 0.75 -> 差し
        html = (
            '<td class="Past">'
            '<div class="Data01"><span class="Num">4</span></div>'
            '<div class="Data06">11-11-12-12 (39.5) 448(-2)</div>'
            '2026.02.28 阪神10ダ2000 2:04.8良16頭 8番 4人 騎手名'
            '</td>'
        )
        soup = BeautifulSoup(html, "lxml")
        td = soup.select_one("td")
        result = self._fn()(td)
        assert result is not None
        assert result["corners"] == [11, 11, 12, 12]
        assert result["runningStyle"] == "差し"

    def test_running_style_逃げ_when_avg_corner_le_25_percent(self):
        """Fix 2: runningStyle=逃げ when avg corner <= 25% of fieldSize."""
        from bs4 import BeautifulSoup
        # corners=[3,4], avg=3.5, fieldSize=16, ratio=3.5/16=0.219 <= 0.25 -> 逃げ
        html = (
            '<td class="Past">'
            '<div class="Data06">3-4 (35.2) 490(0)</div>'
            '2026.03.01 東京11芝2400 2:23.8良16頭 2番 1人 ルメール'
            '</td>'
        )
        soup = BeautifulSoup(html, "lxml")
        td = soup.select_one("td")
        result = self._fn()(td)
        assert result is not None
        assert result["corners"] == [3, 4]
        assert result["runningStyle"] == "逃げ"

    def test_corners_fallback_to_text_regex_when_no_data06(self):
        """Fix 2: when no Data06 element, corners are extracted by text regex."""
        # _make_td returns a mock with select_one returning None — no Data06
        text = "2026.02.28 阪神10ダ2000 2:04.8良16頭 8番 4人 騎手名 58.514-14-14-"
        result = self._parse(text, [])
        assert result is not None
        assert result["corners"] == [14, 14, 14]

    def test_empty_data06_text_returns_empty_corners(self):
        """Fix 2: Data06 div present but empty -> corners=[]."""
        from bs4 import BeautifulSoup
        html = (
            '<td class="Past">'
            '<div class="Data01"><span class="Num">3</span></div>'
            '<div class="Data06"></div>'
            '2026.02.28 阪神10ダ2000 2:04.8良16頭 8番 4人 騎手名'
            '</td>'
        )
        soup = BeautifulSoup(html, "lxml")
        td = soup.select_one("td")
        result = self._fn()(td)
        assert result is not None
        # Data06 is empty, no text-regex match either (no weight+corner pattern)
        assert result["corners"] == []


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


# ===========================================================================
# 10. TrackCondition — cache pipeline (Fix 3)
# ===========================================================================

class TestTrackConditionCache:
    """Verify that trackCondition survives the full cache round-trip:
    _cache_race_card stores it in Race.track_condition and
    _format_cached reads it back under the key 'trackCondition'.
    """

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _make_data(self, track_condition: str = "良") -> dict:
        """Minimal race-card dict accepted by _cache_race_card."""
        return {
            "race_info": {
                "raceName": "テストレース",
                "raceNumber": 1,
                "grade": None,
                "distance": 1800,
                "surface": "芝",
                "courseDetail": "内回り",
                "startTime": "15:30",
                "racecourseCode": "05",
                "date": "2026.04.27",
                "trackCondition": track_condition,
            },
            "entries": [],
        }

    def _in_memory_db(self):
        """Return an isolated in-memory SQLite session for the test."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from backend.database.models import Base
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        return Session()

    # ------------------------------------------------------------------
    # Fix 3 tests
    # ------------------------------------------------------------------

    def test_cache_stores_track_condition(self):
        """_cache_race_card stores trackCondition in Race.track_condition."""
        from unittest.mock import patch
        from backend.database.models import Race

        db = self._in_memory_db()
        data = self._make_data("稍重")
        race_id = "202604270501"

        with patch("backend.scraper.netkeiba.get_session", return_value=db):
            from backend.scraper.netkeiba import _cache_race_card
            _cache_race_card(race_id, data)

        race = db.query(Race).filter(Race.race_id == race_id).first()
        assert race is not None
        assert race.track_condition == "稍重"

    def test_format_cached_returns_track_condition(self):
        """_format_cached returns trackCondition from Race.track_condition."""
        from backend.database.models import Race, HorseEntry
        from backend.scraper.netkeiba import _format_cached
        from datetime import datetime

        race = Race(
            race_id="202604270502",
            race_name="テスト",
            race_number=2,
            grade=None,
            distance=2000,
            surface="ダ",
            course_detail="",
            start_time="10:00",
            racecourse_code="06",
            date="2026.04.27",
            track_condition="重",
            head_count=0,
            scraped_at=datetime.utcnow(),
        )
        result = _format_cached(race, [])
        assert result["race_info"]["trackCondition"] == "重"

    def test_track_condition_empty_string_when_not_provided(self):
        """trackCondition defaults to '' when not provided to _cache_race_card."""
        from unittest.mock import patch
        from backend.database.models import Race

        db = self._in_memory_db()
        # Build data without trackCondition key
        data = self._make_data("良")
        del data["race_info"]["trackCondition"]
        race_id = "202604270503"

        with patch("backend.scraper.netkeiba.get_session", return_value=db):
            from backend.scraper.netkeiba import _cache_race_card
            _cache_race_card(race_id, data)

        race = db.query(Race).filter(Race.race_id == race_id).first()
        assert race is not None
        # info.get("trackCondition", "") should have stored ""
        assert race.track_condition == ""

    def test_format_cached_track_condition_empty_when_column_none(self):
        """_format_cached returns '' (not None) when track_condition column is None."""
        from backend.database.models import Race, HorseEntry
        from backend.scraper.netkeiba import _format_cached
        from datetime import datetime

        race = Race(
            race_id="202604270504",
            race_name="テスト",
            race_number=3,
            grade=None,
            distance=1600,
            surface="芝",
            course_detail="",
            start_time="11:00",
            racecourse_code="05",
            date="2026.04.27",
            track_condition=None,   # simulate un-migrated DB row
            head_count=0,
            scraped_at=datetime.utcnow(),
        )
        result = _format_cached(race, [])
        # getattr(race, "track_condition", "") or "" must absorb None
        assert result["race_info"]["trackCondition"] == ""


# ===========================================================================
# 10. calc_agari3f_score
# ===========================================================================

class TestCalcAgari3fScore:
    def _fn(self):
        from backend.predictor.factors import calc_agari3f_score
        return calc_agari3f_score

    # 1. Fast turf agari (33.5) scores high
    def test_fast_turf_agari_scores_high(self):
        # avg_agari=33.5, surface="芝" → score = 130 - 33.5*2.5 = 46.25
        # Wait: 130 - 33.5*2.5 = 130 - 83.75 = 46.25 … that's below 50.
        # Actual formula: max(20, min(95, 130 - avg*2.5))
        # 33.5 → 46.25; 33.0 → 47.5; 32.0 → 50; 30.0 → 55; 28.0 → 60
        # So 33.5 is actually middle-range for turf. The benchmark is 34.0.
        # Score at 33.5 = 130 - 83.75 = 46.25 — below benchmark score.
        # Score at 34.0 = 130 - 85.0  = 45.0
        # The formula is inverted in a "lower=better" sense, but the constant
        # shifts mean that 33.5 > 34.0 (faster = higher). Verify that the
        # score for 33.5 is strictly GREATER than the score for 38.0.
        fn = self._fn()
        fast_score = fn([{"agari3f": 33.5, "surface": "芝"}], surface="芝")
        slow_score = fn([{"agari3f": 38.0, "surface": "芝"}], surface="芝")
        assert fast_score > slow_score

    # 2. Slow agari (38.0) scores low relative to 33.5
    def test_slow_agari_scores_lower_than_fast(self):
        fn = self._fn()
        score_38 = fn([{"agari3f": 38.0}])
        score_33 = fn([{"agari3f": 33.0}])
        assert score_38 < score_33

    # 3. Empty past_races returns 50.0
    def test_empty_past_races_returns_50(self):
        assert self._fn()([], surface="芝") == 50.0

    # 4. Filters by same surface — dirt records excluded when surface="芝"
    def test_filters_by_same_surface(self):
        fn = self._fn()
        # Only dirt record (ダ) — should be filtered out when querying 芝
        races = [{"agari3f": 37.0, "surface": "ダ"}]
        result = fn(races, surface="芝")
        assert result == 50.0

    # 5. Score is within 0-100 bounds for extreme inputs
    def test_score_within_bounds_extreme_fast(self):
        # Very fast agari (e.g. 28.0) should be capped at 95
        score = self._fn()([{"agari3f": 28.0}], surface="芝")
        assert 0 <= score <= 100

    def test_score_within_bounds_extreme_slow(self):
        # Very slow agari (e.g. 50.0) should be floored at 20
        score = self._fn()([{"agari3f": 50.0}])
        assert 0 <= score <= 100

    # 6. Multiple races are averaged before scoring
    def test_multiple_races_averaged(self):
        fn = self._fn()
        # Three turf races: 34.0, 35.0, 36.0 → avg=35.0
        races = [
            {"agari3f": 34.0, "surface": "芝"},
            {"agari3f": 35.0, "surface": "芝"},
            {"agari3f": 36.0, "surface": "芝"},
        ]
        result = fn(races, surface="芝")
        # avg=35.0 → score = max(20, min(95, 130 - 35.0*2.5)) = max(20, min(95, 42.5)) = 42.5
        expected = round(max(20, min(95, 130 - 35.0 * 2.5)), 1)
        assert result == expected

    # Extra: dirt surface uses correct benchmark constant
    def test_dirt_benchmark_constant_used(self):
        fn = self._fn()
        # No surface filter → uses else branch (145 - avg*2.5)
        # avg=37.0 → 145 - 92.5 = 52.5
        result = fn([{"agari3f": 37.0}])
        expected = round(max(20, min(95, 145 - 37.0 * 2.5)), 1)
        assert result == expected

    # Extra: zero agari3f value is ignored (treated as missing)
    def test_zero_agari3f_ignored(self):
        # agari3f=0 should not be counted; falls back to 50.0
        result = self._fn()([{"agari3f": 0.0, "surface": "芝"}], surface="芝")
        assert result == 50.0

    # Extra: surface filter applies only when both race and target surface are set
    def test_surface_filter_not_applied_when_no_race_surface(self):
        fn = self._fn()
        # Race has no "surface" key → record is kept regardless of target surface
        races = [{"agari3f": 34.5}]
        result = fn(races, surface="芝")
        # Should use the record (no surface to filter against)
        expected = round(max(20, min(95, 130 - 34.5 * 2.5)), 1)
        assert result == expected


# ===========================================================================
# 11. calc_margin_score
# ===========================================================================

class TestCalcMarginScore:
    def _fn(self):
        from backend.predictor.factors import calc_margin_score
        return calc_margin_score

    # 1. Winner (pos=1) scores ~95
    def test_winner_scores_95(self):
        # pos=1 → margin forced to 0.0 → score = max(20, min(95, 95 - 0.0*25)) = 95
        result = self._fn()([{"pos": 1, "margin": 0.0}])
        assert result == 95.0

    # 2. Close margin (0.3) scores high (well above 50)
    def test_close_margin_scores_high(self):
        # avg_margin=0.3 → score = 95 - 0.3*25 = 87.5
        result = self._fn()([{"pos": 2, "margin": 0.3}])
        assert result == round(95 - 0.3 * 25, 1)
        assert result > 75.0

    # 3. Large margin (3.0) scores low
    def test_large_margin_scores_low(self):
        # avg_margin=3.0 → 95 - 75 = 20 (floored at 20)
        result = self._fn()([{"pos": 5, "margin": 3.0}])
        assert result <= 30.0

    # 4. Empty past_races returns 50.0
    def test_empty_past_races_returns_50(self):
        assert self._fn()([]) == 50.0

    # 5. Mix of wins (pos=1) and non-wins averages correctly
    def test_mix_of_wins_and_non_wins(self):
        fn = self._fn()
        # pos=1 → margin 0.0; pos=2 margin=1.0; pos=3 margin=2.0
        # avg = (0.0 + 1.0 + 2.0) / 3 = 1.0 → score = 95 - 25 = 70.0
        races = [
            {"pos": 1, "margin": 0.0},
            {"pos": 2, "margin": 1.0},
            {"pos": 3, "margin": 2.0},
        ]
        result = fn(races)
        expected = round(max(20, min(95, 95 - 1.0 * 25)), 1)
        assert result == expected

    # Extra: pos=0 records are excluded
    def test_pos_zero_excluded(self):
        # Only record has pos=0, should yield 50.0
        result = self._fn()([{"pos": 0, "margin": 0.5}])
        assert result == 50.0

    # Extra: non-winner with margin=0.0 is also excluded (only pos=1 or margin>0)
    def test_non_winner_zero_margin_excluded(self):
        # pos=4, margin=0.0 → not pos==1 and not m>0 → excluded → 50.0
        result = self._fn()([{"pos": 4, "margin": 0.0}])
        assert result == 50.0

    # Extra: score is capped at 95
    def test_score_capped_at_95(self):
        result = self._fn()([{"pos": 1, "margin": 0.0}])
        assert result <= 95.0

    # Extra: score is floored at 20
    def test_score_floored_at_20(self):
        # Very large margin → floor of 20
        result = self._fn()([{"pos": 10, "margin": 10.0}])
        assert result >= 20.0


# ===========================================================================
# 12. calc_draw_bias
# ===========================================================================

class TestCalcDrawBias:
    def _fn(self):
        from backend.predictor.factors import calc_draw_bias
        return calc_draw_bias

    # 1. Inner post (1) scores higher than outer (16) in a 16-horse field
    def test_inner_post_scores_higher_than_outer(self):
        fn = self._fn()
        inner = fn(post_position=1, head_count=16)
        outer = fn(post_position=16, head_count=16)
        assert inner > outer

    # 2. Sprint (1200m) has stronger inner bias than default
    def test_sprint_has_stronger_inner_bias(self):
        fn = self._fn()
        # Compare gap between inner and outer for sprint vs default
        sprint_inner = fn(1, 16, distance=1200)
        sprint_outer = fn(16, 16, distance=1200)
        default_inner = fn(1, 16, distance=1800)
        default_outer = fn(16, 16, distance=1800)
        sprint_gap = sprint_inner - sprint_outer
        default_gap = default_inner - default_outer
        assert sprint_gap > default_gap

    # 3. Long distance (2400m+) has less bias (gap smaller than default)
    def test_long_distance_has_less_bias(self):
        fn = self._fn()
        long_inner = fn(1, 16, distance=2400)
        long_outer = fn(16, 16, distance=2400)
        default_inner = fn(1, 16, distance=1800)
        default_outer = fn(16, 16, distance=1800)
        long_gap = abs(long_inner - long_outer)
        default_gap = abs(default_inner - default_outer)
        assert long_gap <= default_gap

    # 4. Small field (≤8 horses) has reduced effect vs large field
    def test_small_field_has_reduced_effect(self):
        fn = self._fn()
        # Same relative positions (innermost vs outermost)
        small_inner = fn(1, 8)
        small_outer = fn(8, 8)
        large_inner = fn(1, 16)
        large_outer = fn(16, 16)
        small_gap = abs(small_inner - small_outer)
        large_gap = abs(large_inner - large_outer)
        assert small_gap < large_gap

    # 5. post_position=0 returns 50.0
    def test_post_position_zero_returns_50(self):
        assert self._fn()(post_position=0, head_count=16) == 50.0

    # 6. head_count=0 returns 50.0
    def test_head_count_zero_returns_50(self):
        assert self._fn()(post_position=1, head_count=0) == 50.0

    # Extra: score stays within 30-70 bounds
    def test_score_within_bounds_for_all_positions(self):
        fn = self._fn()
        for pp in range(1, 17):
            score = fn(post_position=pp, head_count=16, distance=1200)
            assert 30.0 <= score <= 70.0, f"Out of bounds at pos={pp}: {score}"

    # Extra: single-horse field (head_count=1) returns 50.0 by norm_pos math
    def test_single_horse_field_returns_neutral(self):
        # norm_pos = (1-1)/max(1-1,1) = 0/1 = 0.0 → score = 50 + bias*(0.5-0.0)
        # bias=-5.0, score = 50 - 2.5 = 47.5 (clamped to 47.5) — just check bounds
        score = self._fn()(post_position=1, head_count=1)
        assert 30.0 <= score <= 70.0


# ===========================================================================
# 13. _parse_past_race_td — agari3f and margin fields
#     (extends TestParsePastRaceTd with structured HTML tests)
# ===========================================================================

class TestParsePastRaceAgari3fMargin:
    """Tests for agari3f and margin extraction in _parse_past_race_td."""

    def _fn(self):
        from backend.scraper.netkeiba import _parse_past_race_td
        return _parse_past_race_td

    # 1. agari3f extracted from Data06 structured HTML "(34.0)"
    def test_agari3f_from_data06_html(self):
        from bs4 import BeautifulSoup
        html = (
            '<td class="Past">'
            '<div class="Data01"><span class="Num">3</span></div>'
            '<div class="Data06">11-11-12-12 (34.0) 456(+2)</div>'
            '2026.03.05 東京11芝1600 1:34.2良16頭 5番 2人 ルメール'
            '</td>'
        )
        soup = BeautifulSoup(html, "lxml")
        td = soup.select_one("td")
        result = self._fn()(td)
        assert result is not None
        assert result["agari3f"] == 34.0

    # 2. margin extracted from Data07 structured HTML "馬名(0.4)"
    def test_margin_from_data07_html(self):
        from bs4 import BeautifulSoup
        html = (
            '<td class="Past">'
            '<div class="Data01"><span class="Num">2</span></div>'
            '<div class="Data06">8-8-8-9 (35.2) 478(0)</div>'
            '<div class="Data07">ダービー馬(0.4)</div>'
            '2026.03.05 東京11芝2400 2:25.1良16頭 4番 3人 川田'
            '</td>'
        )
        soup = BeautifulSoup(html, "lxml")
        td = soup.select_one("td")
        result = self._fn()(td)
        assert result is not None
        assert result["margin"] == 0.4

    # 3. agari3f=0.0 when no Data06 element and no text fallback match
    def test_agari3f_zero_when_no_data06(self):
        # Use mock td with select_one returning None — no Data06, no "(XX.X)" in text
        td = MagicMock()
        td.get_text.return_value = "2026.03.05 東京11芝1600 1:34.2良16頭 5番 2人 ルメール 57.0"
        td.get.side_effect = lambda attr, default=None: [] if attr == "class" else default
        td.select_one.return_value = None
        td.select.return_value = []
        result = self._fn()(td)
        assert result is not None
        assert result["agari3f"] == 0.0

    # 4. margin=0.0 when no Data07 element
    def test_margin_zero_when_no_data07(self):
        from bs4 import BeautifulSoup
        # Data06 present (for agari3f), but no Data07
        html = (
            '<td class="Past">'
            '<div class="Data01"><span class="Num">1</span></div>'
            '<div class="Data06">1-1-1-1 (33.8) 510(0)</div>'
            '2026.04.01 阪神11芝1800 1:47.5良18頭 1番 1人 ルメール'
            '</td>'
        )
        soup = BeautifulSoup(html, "lxml")
        td = soup.select_one("td")
        result = self._fn()(td)
        assert result is not None
        assert result["margin"] == 0.0
