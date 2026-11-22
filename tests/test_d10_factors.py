"""Tests for D10 new factors: paceScenario, cornerMove, bmsAptitude, classChange, fieldLevel, jockeyChange."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.predictor.factors import (
    calc_pace_scenario, calc_corner_move, calc_bms_aptitude,
    calc_class_change, calc_field_level, calc_jockey_change,
)


def _make_entries_with_styles(styles):
    entries = []
    for i, s in enumerate(styles):
        entries.append({
            "horseNumber": i + 1,
            "pastRaces": [{"runningStyle": s, "pos": 3}] if s else [],
        })
    return entries


class TestPaceScenario:
    def test_closer_in_fast_pace(self):
        entries = _make_entries_with_styles(["逃げ", "先行", "先行", "先行", "差し", "差し"])
        past = [{"runningStyle": "差し"}]
        score = calc_pace_scenario(past, entries)
        assert score > 60

    def test_front_runner_in_slow_pace(self):
        entries = _make_entries_with_styles(["逃げ", "差し", "差し", "差し", "追込", "追込"])
        past = [{"runningStyle": "逃げ"}]
        score = calc_pace_scenario(past, entries)
        assert score > 65

    def test_front_runner_in_fast_pace(self):
        entries = _make_entries_with_styles(["逃げ", "逃げ", "先行", "先行", "先行", "差し"])
        past = [{"runningStyle": "逃げ"}]
        score = calc_pace_scenario(past, entries)
        assert score < 45

    def test_empty_returns_neutral(self):
        assert calc_pace_scenario([], [], "") == 50.0

    def test_no_style_returns_neutral(self):
        entries = _make_entries_with_styles(["先行", "差し"])
        past = [{"pos": 3}]
        assert calc_pace_scenario(past, entries) == 50.0


class TestCornerMove:
    def test_strong_closer(self):
        past = [{"corners": [10, 7]}, {"corners": [8, 5]}, {"corners": [12, 8]}]
        assert calc_corner_move(past) > 65

    def test_fades(self):
        past = [{"corners": [3, 6]}, {"corners": [2, 5]}]
        assert calc_corner_move(past) < 45

    def test_holds_position(self):
        past = [{"corners": [5, 5]}, {"corners": [4, 4]}]
        score = calc_corner_move(past)
        assert 45 <= score <= 55

    def test_empty(self):
        assert calc_corner_move([]) == 50.0

    def test_no_corners(self):
        past = [{"pos": 1}]
        assert calc_corner_move(past) == 50.0


class TestBmsAptitude:
    def test_turf_bms_on_turf(self):
        assert calc_bms_aptitude("ディープインパクト", "芝", 2000) > 55

    def test_dirt_bms_on_dirt(self):
        assert calc_bms_aptitude("ゴールドアリュール", "ダ", 1800) > 55

    def test_dirt_bms_on_turf(self):
        assert calc_bms_aptitude("ゴールドアリュール", "芝", 2000) < 50

    def test_stayer_bms_on_long(self):
        assert calc_bms_aptitude("ステイゴールド", "芝", 2400) > 60

    def test_sprint_bms_on_sprint(self):
        assert calc_bms_aptitude("サクラバクシンオー", "芝", 1200) > 55

    def test_unknown_bms(self):
        assert calc_bms_aptitude("UnknownSire", "芝", 2000) == 50.0

    def test_empty_bms(self):
        assert calc_bms_aptitude("", "芝", 2000) == 50.0


class TestClassChange:
    def test_empty_returns_neutral(self):
        assert calc_class_change([]) == 50.0

    def test_won_last_race_same_class(self):
        past = [{"pos": 1, "fieldSize": 16, "margin": 0.2}]
        score = calc_class_change(past, "")
        assert score >= 55

    def test_poor_last_race(self):
        past = [{"pos": 10, "fieldSize": 16, "margin": 3.0}]
        score = calc_class_change(past, "")
        assert score < 50


class TestFieldLevel:
    def test_fastest_in_field(self):
        my_past = [{"finishTime": "1:33.0", "distance": 1600}]
        entries = [
            {"pastRaces": [{"finishTime": "1:35.0", "distance": 1600}]},
            {"pastRaces": [{"finishTime": "1:36.0", "distance": 1600}]},
            {"pastRaces": [{"finishTime": "1:34.5", "distance": 1600}]},
        ]
        score = calc_field_level(my_past, entries, 1600)
        assert score > 60

    def test_slowest_in_field(self):
        my_past = [{"finishTime": "1:38.0", "distance": 1600}]
        entries = [
            {"pastRaces": [{"finishTime": "1:34.0", "distance": 1600}]},
            {"pastRaces": [{"finishTime": "1:33.5", "distance": 1600}]},
            {"pastRaces": [{"finishTime": "1:34.5", "distance": 1600}]},
        ]
        score = calc_field_level(my_past, entries, 1600)
        assert score < 40

    def test_empty(self):
        assert calc_field_level([], [], 1600) == 50.0


class TestJockeyChange:
    def test_top_jockey_on_poor_runner(self):
        past = [{"pos": 8, "popularity": 5}]
        score = calc_jockey_change(past, "ルメール")
        assert score > 60

    def test_weak_jockey_after_good_result(self):
        past = [{"pos": 1, "popularity": 1}]
        score = calc_jockey_change(past, "水口")
        assert score < 45

    def test_mid_jockey(self):
        past = [{"pos": 5, "popularity": 5}]
        score = calc_jockey_change(past, "丸山")
        assert 45 <= score <= 55

    def test_empty(self):
        assert calc_jockey_change([], "ルメール") == 50.0
