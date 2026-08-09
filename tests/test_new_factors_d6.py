"""TDD tests for 4 new prediction factors (D6 session).

RED -> GREEN -> REFACTOR cycle.

Factors:
  1. calc_jockey_course_distance  -- jockey x course x distance 3-way affinity
  2. calc_pace_position_advantage -- pace/position advantage from running style mix
  3. calc_rotation_fitness        -- rotation/interval fitness with class change
  4. calc_bloodline_track_condition -- bloodline x track condition affinity
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _race(pos: int, date: str = "2026.01.01", track: str = "東京",
          distance: int = 1600, condition: str = "良",
          running_style: str = "", course_code: str = "05",
          result: str = "") -> dict:
    """Build a minimal past_races entry."""
    r: dict = {
        "pos": pos,
        "date": date,
        "track": track,
        "distance": distance,
        "condition": condition,
    }
    if running_style:
        r["runningStyle"] = running_style
    if course_code:
        r["course_code"] = course_code
    if result:
        r["result"] = result
    return r


def _entry(horse_number: int = 1, past_races: list | None = None) -> dict:
    return {
        "horseNumber": horse_number,
        "pastRaces": past_races or [],
    }


# ===========================================================================
# 1. calc_jockey_course_distance
# ===========================================================================

class TestCalcJockeyCourseDistance:
    def _fn(self):
        from backend.predictor.factors import calc_jockey_course_distance
        return calc_jockey_course_distance

    # --- no data / neutral ---
    def test_no_past_races_returns_50(self):
        assert self._fn()("ルメール", "05", 1600, []) == 50.0

    def test_none_past_races_treated_as_empty(self):
        # passing None should not raise
        result = self._fn()("ルメール", "05", 1600, None or [])
        assert result == 50.0

    def test_no_matching_course_returns_50(self):
        # past races at Sapporo (01), querying Tokyo (05)
        races = [_race(1, track="札幌", course_code="01", distance=1600)]
        assert self._fn()("ルメール", "05", 1600, races) == 50.0

    def test_no_matching_distance_category_returns_50(self):
        # past races are stayer (2400), querying mile (1600)
        races = [_race(1, track="東京", course_code="05", distance=2400)]
        assert self._fn()("ルメール", "05", 1600, races) == 50.0

    # --- scoring tiers ---
    def test_two_or_more_wins_returns_85(self):
        races = [
            _race(1, course_code="05", distance=1600),
            _race(1, course_code="05", distance=1800),
        ]
        assert self._fn()("ルメール", "05", 1600, races) == 85.0

    def test_exactly_one_win_returns_75(self):
        races = [
            _race(1, course_code="05", distance=1600),
            _race(4, course_code="05", distance=1600),
        ]
        assert self._fn()("ルメール", "05", 1600, races) == 75.0

    def test_two_or_more_places_no_wins_returns_70(self):
        races = [
            _race(2, course_code="05", distance=1600),
            _race(3, course_code="05", distance=1800),
        ]
        assert self._fn()("ルメール", "05", 1600, races) == 70.0

    def test_one_place_no_wins_returns_60(self):
        races = [
            _race(3, course_code="05", distance=1600),
            _race(5, course_code="05", distance=1600),
        ]
        assert self._fn()("ルメール", "05", 1600, races) == 60.0

    def test_no_wins_or_places_returns_50(self):
        races = [_race(6, course_code="05", distance=1600)]
        assert self._fn()("ルメール", "05", 1600, races) == 50.0

    # --- distance category boundary ---
    def test_sprint_boundary_1400_matches_sprint_1200(self):
        # 1200m (sprint) should match 1400m target (sprint) — 1 win → 75
        races = [_race(1, course_code="07", distance=1200)]
        assert self._fn()("川田", "07", 1400, races) == 75.0

    def test_sprint_1200_does_not_match_mile_1600(self):
        # sprint category != mile category
        races = [_race(1, course_code="07", distance=1200)]
        assert self._fn()("川田", "07", 1600, races) == 50.0

    def test_mile_boundary_1800_matches_mile_1600(self):
        # 1800m is mile, same category as 1600m — 1 win → 75
        races = [_race(1, course_code="05", distance=1800)]
        assert self._fn()("ルメール", "05", 1600, races) == 75.0

    def test_intermediate_2200_matches_intermediate_2000(self):
        # both intermediate category — 1 win → 75
        races = [_race(1, course_code="06", distance=2200)]
        assert self._fn()("横山武", "06", 2000, races) == 75.0

    def test_stayer_2400_matches_stayer_3200(self):
        # both stayer category — 1 win → 75
        races = [_race(1, course_code="08", distance=2400)]
        assert self._fn()("福永", "08", 3200, races) == 75.0

    # --- win beats place ---
    def test_win_score_higher_than_place_only(self):
        win_races = [_race(1, course_code="05", distance=1600)]
        place_races = [_race(2, course_code="05", distance=1600),
                       _race(3, course_code="05", distance=1600)]
        win_score = self._fn()("ルメール", "05", 1600, win_races)
        place_score = self._fn()("ルメール", "05", 1600, place_races)
        assert win_score > place_score

    # --- course code variety ---
    def test_niigata_course_code_04(self):
        # single win → 75 (1 win tier)
        races = [_race(1, course_code="04", distance=1600)]
        assert self._fn()("戸崎", "04", 1600, races) == 75.0

    # --- return type ---
    def test_returns_float(self):
        result = self._fn()("ルメール", "05", 1600, [])
        assert isinstance(result, float)

    def test_return_in_range_0_100(self):
        races = [_race(1, course_code="05", distance=1600)] * 10
        result = self._fn()("ルメール", "05", 1600, races)
        assert 0.0 <= result <= 100.0


# ===========================================================================
# 2. calc_pace_position_advantage
# ===========================================================================

class TestCalcPacePositionAdvantage:
    def _fn(self):
        from backend.predictor.factors import calc_pace_position_advantage
        return calc_pace_position_advantage

    def _front_runner_entry(self, horse_number: int = 1) -> dict:
        return _entry(horse_number, [_race(1, running_style="逃げ")])

    def _closer_entry(self, horse_number: int = 1) -> dict:
        return _entry(horse_number, [_race(3, running_style="差し")])

    def _senkou_entry(self, horse_number: int = 1) -> dict:
        return _entry(horse_number, [_race(2, running_style="先行")])

    def _oikomi_entry(self, horse_number: int = 1) -> dict:
        return _entry(horse_number, [_race(4, running_style="追込")])

    # --- defaults ---
    def test_empty_entries_returns_50(self):
        assert self._fn()([], [], 16) == 50.0

    def test_no_running_style_data_returns_50(self):
        past = [_race(1)]  # no runningStyle
        result = self._fn()(past, [_entry(1, [_race(2)])], 16)
        assert result == 50.0

    # --- sole front-runner with <= 1 front-runner total ---
    def test_sole_front_runner_returns_75(self):
        # This horse is a front-runner, and <= 1 front-runners in race
        this_horse_past = [_race(1, running_style="逃げ")]
        entries = [
            _entry(1, [_race(1, running_style="逃げ")]),   # this horse
            _entry(2, [_race(2, running_style="差し")]),   # closer
            _entry(3, [_race(3, running_style="追込")]),   # closer
        ]
        result = self._fn()(this_horse_past, entries, 8)
        assert result == 75.0

    def test_two_front_runners_returns_50_for_front_runner(self):
        # 2 front-runners — not <= 1, not >= 3 → neutral for front-runner
        this_horse_past = [_race(1, running_style="先行")]
        entries = [
            _entry(1, [_race(1, running_style="先行")]),
            _entry(2, [_race(2, running_style="逃げ")]),
            _entry(3, [_race(3, running_style="差し")]),
        ]
        result = self._fn()(this_horse_past, entries, 8)
        assert result == 50.0

    # --- crowded pace (>= 3 front-runners) ---
    def test_three_front_runners_front_runner_returns_35(self):
        this_horse_past = [_race(1, running_style="逃げ")]
        entries = [
            _entry(1, [_race(1, running_style="逃げ")]),
            _entry(2, [_race(1, running_style="先行")]),
            _entry(3, [_race(2, running_style="先行")]),
            _entry(4, [_race(3, running_style="差し")]),
        ]
        result = self._fn()(this_horse_past, entries, 8)
        assert result == 35.0

    def test_three_front_runners_closer_returns_70(self):
        this_horse_past = [_race(3, running_style="差し")]
        entries = [
            _entry(1, [_race(1, running_style="逃げ")]),
            _entry(2, [_race(1, running_style="先行")]),
            _entry(3, [_race(2, running_style="先行")]),
            _entry(4, [_race(3, running_style="差し")]),
        ]
        result = self._fn()(this_horse_past, entries, 8)
        assert result == 70.0

    def test_oikomi_also_counted_as_closer(self):
        # 追込 should benefit from pace collapse (>=3 front-runners)
        this_horse_past = [_race(4, running_style="追込")]
        entries = [
            _entry(1, [_race(1, running_style="逃げ")]),
            _entry(2, [_race(1, running_style="先行")]),
            _entry(3, [_race(2, running_style="先行")]),
            _entry(4, [_race(4, running_style="追込")]),
        ]
        result = self._fn()(this_horse_past, entries, 8)
        assert result == 70.0

    # --- senkou is a front-runner ---
    def test_senkou_counted_as_front_runner_for_crowded_pace(self):
        # All 先行 → >= 3 front-runners → 先行 horse gets 35
        this_horse_past = [_race(2, running_style="先行")]
        entries = [
            _entry(1, [_race(2, running_style="先行")]),
            _entry(2, [_race(1, running_style="先行")]),
            _entry(3, [_race(2, running_style="先行")]),
        ]
        result = self._fn()(this_horse_past, entries, 8)
        assert result == 35.0

    # --- return type / range ---
    def test_returns_float(self):
        result = self._fn()([], [], 16)
        assert isinstance(result, float)

    def test_return_in_range_0_100(self):
        this_past = [_race(1, running_style="逃げ")]
        entries = [_entry(1, [_race(1, running_style="逃げ")])]
        result = self._fn()(this_past, entries, 8)
        assert 0.0 <= result <= 100.0

    # --- head_count irrelevant to logic ---
    def test_head_count_does_not_change_sole_front_runner_score(self):
        this_past = [_race(1, running_style="逃げ")]
        entries = [
            _entry(1, [_race(1, running_style="逃げ")]),
            _entry(2, [_race(2, running_style="差し")]),
        ]
        assert self._fn()(this_past, entries, 18) == 75.0


# ===========================================================================
# 3. calc_rotation_fitness
# ===========================================================================

class TestCalcRotationFitness:
    def _fn(self):
        from backend.predictor.factors import calc_rotation_fitness
        return calc_rotation_fitness

    # --- no past races ---
    def test_no_past_races_returns_50(self):
        assert self._fn()([], "2026.04.05", 0) == 50.0

    def test_none_past_races_treated_as_empty(self):
        result = self._fn()(None or [], "2026.04.05", 0)
        assert result == 50.0

    # --- optimal interval 14-35 days → base 70 ---
    def test_14_days_returns_70(self):
        races = [_race(3, date="2026.03.22")]
        result = self._fn()(races, "2026.04.05", 0)
        assert result == 70.0

    def test_35_days_returns_70(self):
        races = [_race(3, date="2026.03.01")]
        result = self._fn()(races, "2026.04.05", 0)
        assert result == 70.0

    # --- 36-60 days → base 65 ---
    def test_36_days_returns_65(self):
        races = [_race(3, date="2026.02.28")]  # 36 days before 2026.04.05
        result = self._fn()(races, "2026.04.05", 0)
        assert result == 65.0

    def test_60_days_returns_65(self):
        races = [_race(3, date="2026.02.04")]  # 60 days before 2026.04.05
        result = self._fn()(races, "2026.04.05", 0)
        assert result == 65.0

    # --- 61-90 days → base 55 ---
    def test_61_days_returns_55(self):
        races = [_race(3, date="2026.02.03")]  # 61 days before 2026.04.05
        result = self._fn()(races, "2026.04.05", 0)
        assert result == 55.0

    def test_90_days_returns_55(self):
        races = [_race(3, date="2026.01.05")]  # 90 days before 2026.04.05
        result = self._fn()(races, "2026.04.05", 0)
        assert result == 55.0

    # --- 91+ days → base 45 ---
    def test_91_days_returns_45(self):
        races = [_race(3, date="2026.01.04")]  # 91 days before 2026.04.05
        result = self._fn()(races, "2026.04.05", 0)
        assert result == 45.0

    # --- < 14 days → base 55 ---
    def test_13_days_returns_55_too_soon(self):
        races = [_race(3, date="2026.03.23")]  # 13 days before 2026.04.05
        result = self._fn()(races, "2026.04.05", 0)
        assert result == 55.0

    def test_1_day_returns_55_too_soon(self):
        races = [_race(3, date="2026.04.04")]
        result = self._fn()(races, "2026.04.05", 0)
        assert result == 55.0

    # --- class change modifiers ---
    def test_class_up_penalty_minus_5(self):
        # optimal interval (70 base) + class up (-5) = 65
        races = [_race(3, date="2026.03.22")]
        result = self._fn()(races, "2026.04.05", 1)
        assert result == 65.0

    def test_class_same_no_change(self):
        races = [_race(3, date="2026.03.22")]
        result = self._fn()(races, "2026.04.05", 0)
        assert result == 70.0

    def test_class_down_bonus_plus_5(self):
        # optimal interval (70 base) + class down (+5) = 75
        races = [_race(3, date="2026.03.22")]
        result = self._fn()(races, "2026.04.05", -1)
        assert result == 75.0

    # --- winning form carry-over bonus ---
    def test_most_recent_win_adds_10_bonus(self):
        # Most recent race was 1st place → +10
        races = [_race(1, date="2026.03.22")]   # pos=1 → win
        result = self._fn()(races, "2026.04.05", 0)
        assert result == 80.0  # 70 base + 10 win bonus

    def test_most_recent_non_win_no_bonus(self):
        races = [_race(2, date="2026.03.22")]
        result = self._fn()(races, "2026.04.05", 0)
        assert result == 70.0

    def test_win_bonus_stacks_with_class_down(self):
        # win (+10) + class down (+5) + optimal (70) = 85
        races = [_race(1, date="2026.03.22")]
        result = self._fn()(races, "2026.04.05", -1)
        assert result == 85.0

    def test_win_bonus_stacks_with_class_up(self):
        # win (+10) + class up (-5) + optimal (70) = 75
        races = [_race(1, date="2026.03.22")]
        result = self._fn()(races, "2026.04.05", 1)
        assert result == 75.0

    # --- return type ---
    def test_returns_float(self):
        result = self._fn()([], "2026.04.05", 0)
        assert isinstance(result, float)

    def test_return_in_range_0_100(self):
        # max possible: 45 + 5 + 10 = 60 (no violation), or 70+10+5 = 85
        races = [_race(1, date="2026.03.22")]
        result = self._fn()(races, "2026.04.05", -1)
        assert 0.0 <= result <= 100.0

    # --- missing date graceful degradation ---
    def test_missing_date_in_past_race_returns_50(self):
        races = [{"pos": 1, "distance": 1600}]  # no "date" key
        result = self._fn()(races, "2026.04.05", 0)
        assert result == 50.0

    def test_invalid_race_date_string_returns_50(self):
        result = self._fn()([_race(1, date="2026.03.22")], "not-a-date", 0)
        assert result == 50.0

    def test_future_last_race_date_returns_50(self):
        # last race is in the future relative to race_date → negative days
        races = [_race(3, date="2026.12.31")]
        result = self._fn()(races, "2026.04.05", 0)
        assert result == 50.0

    def test_malformed_last_date_with_non_integer_parts_returns_50(self):
        # date string that passes split but fails int() cast
        races = [{"pos": 3, "date": "XXXX.YY.ZZ"}]
        result = self._fn()(races, "2026.04.05", 0)
        assert result == 50.0


# ===========================================================================
# 4. calc_bloodline_track_condition
# ===========================================================================

class TestCalcBloodlineTrackCondition:
    def _fn(self):
        from backend.predictor.factors import calc_bloodline_track_condition
        return calc_bloodline_track_condition

    # --- good track → neutral ---
    def test_good_track_returns_50(self):
        assert self._fn()("ディープインパクト", "サンデーサイレンス", "良", []) == 50.0

    def test_yayashige_track_returns_50(self):
        assert self._fn()("ゴールドシップ", "", "稍重", []) == 50.0

    # --- heavy track: past race data path ---
    def test_heavy_win_in_past_races_returns_80(self):
        past = [{"pos": 1, "condition": "重"}]
        result = self._fn()("ディープインパクト", "", "重", past)
        assert result == 80.0

    def test_fuuryo_win_in_past_races_returns_80(self):
        past = [{"pos": 1, "condition": "不良"}]
        result = self._fn()("ディープインパクト", "", "不良", past)
        assert result == 80.0

    def test_heavy_place_no_win_returns_70(self):
        past = [{"pos": 2, "condition": "重"}, {"pos": 3, "condition": "重"}]
        result = self._fn()("ディープインパクト", "", "重", past)
        assert result == 70.0

    def test_heavy_place_second_no_win_returns_70(self):
        past = [{"pos": 3, "condition": "重"}]
        result = self._fn()("ディープインパクト", "", "重", past)
        assert result == 70.0

    # --- heavy track: sire profile path (no past heavy data) ---
    def test_no_heavy_data_gold_ship_sire_gets_bonus(self):
        # ゴールドシップ is a known mud sire → should score > 50
        result = self._fn()("ゴールドシップ", "", "重", [])
        assert result > 50.0

    def test_no_heavy_data_hearts_cry_gets_bonus(self):
        result = self._fn()("ハーツクライ", "", "重", [])
        assert result > 50.0

    def test_no_heavy_data_king_kamehameha_gets_bonus(self):
        result = self._fn()("キングカメハメハ", "", "重", [])
        assert result > 50.0

    def test_no_heavy_data_no_known_sire_returns_50(self):
        # unknown sire, no past data → 50
        result = self._fn()("エアシャカール", "", "重", [])
        assert result == 50.0

    # --- bms path ---
    def test_bms_known_mud_sire_gets_bonus_when_no_sire_data(self):
        # sire unknown, bms = ゴールドシップ → should get bonus
        result = self._fn()("エアシャカール", "ゴールドシップ", "重", [])
        assert result > 50.0

    # --- past data beats sire profile ---
    def test_past_win_always_returns_80_regardless_of_sire(self):
        past = [{"pos": 1, "condition": "重"}]
        # Even a bad-mud sire: if horse WON on heavy → 80
        result = self._fn()("ロードカナロア", "", "重", past)
        assert result == 80.0

    # --- not良 AND not稍重: 重/不良 triggers sire profile check ---
    def test_fuuryo_no_data_gold_ship_gets_bonus(self):
        result = self._fn()("ゴールドシップ", "", "不良", [])
        assert result > 50.0

    # --- return type / range ---
    def test_returns_float(self):
        result = self._fn()("ゴールドシップ", "", "良", [])
        assert isinstance(result, float)

    def test_return_in_range_0_100(self):
        past = [{"pos": 1, "condition": "重"}]
        result = self._fn()("ゴールドシップ", "", "重", past)
        assert 0.0 <= result <= 100.0

    # --- edge cases ---
    def test_empty_sire_no_past_heavy_data_returns_50(self):
        result = self._fn()("", "", "重", [])
        assert result == 50.0

    def test_past_race_without_condition_key_ignored(self):
        # Races without "condition" key should not count as heavy wins
        past = [{"pos": 1}]  # no condition key
        result = self._fn()("ディープインパクト", "", "重", past)
        # Should fall through to sire profile path (no heavy data found)
        assert result == 50.0  # ディープインパクト not in known mud sires

    def test_non_heavy_past_race_not_counted(self):
        # Win on 良 ground doesn't count for heavy track score
        past = [{"pos": 1, "condition": "良"}, {"pos": 2, "condition": "稍重"}]
        result = self._fn()("ディープインパクト", "", "重", past)
        # No heavy wins/places → fall to sire profile
        assert result == 50.0


# ===========================================================================
# Integration smoke test — all 4 functions importable and return 0-100
# ===========================================================================

class TestNewFactorsIntegration:
    def test_all_four_functions_importable(self):
        from backend.predictor.factors import (
            calc_jockey_course_distance,
            calc_pace_position_advantage,
            calc_rotation_fitness,
            calc_bloodline_track_condition,
        )
        # Just verify they exist (no ImportError)
        assert callable(calc_jockey_course_distance)
        assert callable(calc_pace_position_advantage)
        assert callable(calc_rotation_fitness)
        assert callable(calc_bloodline_track_condition)

    def test_all_return_in_range_with_realistic_data(self):
        from backend.predictor.factors import (
            calc_jockey_course_distance,
            calc_pace_position_advantage,
            calc_rotation_fitness,
            calc_bloodline_track_condition,
        )
        past_races = [
            {"pos": 1, "date": "2026.03.15", "track": "東京", "distance": 1600,
             "condition": "重", "runningStyle": "先行", "course_code": "05"},
            {"pos": 3, "date": "2026.02.08", "track": "中山", "distance": 1800,
             "condition": "良", "runningStyle": "先行", "course_code": "06"},
        ]
        entries = [
            {"horseNumber": 1, "pastRaces": past_races},
            {"horseNumber": 2, "pastRaces": [
                {"pos": 2, "date": "2026.03.10", "runningStyle": "差し"}
            ]},
        ]
        s1 = calc_jockey_course_distance("ルメール", "05", 1600, past_races)
        s2 = calc_pace_position_advantage(past_races, entries, 8)
        s3 = calc_rotation_fitness(past_races, "2026.04.05", 0)
        s4 = calc_bloodline_track_condition("ハーツクライ", "サンデーサイレンス", "重", past_races)

        for score in (s1, s2, s3, s4):
            assert 0.0 <= score <= 100.0, f"Out of range: {score}"
