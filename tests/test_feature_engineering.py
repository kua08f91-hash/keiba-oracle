"""TDD tests for feature engineering module.

Covers: column definitions, race context extraction, horse feature extraction,
parsing utilities, feature vector ordering.
"""
from __future__ import annotations

import pytest


class TestColumnDefinitions:
    def test_analytical_columns_no_market(self):
        from backend.predictor.feature_engineering import ANALYTICAL_COLUMNS
        for col in ANALYTICAL_COLUMNS:
            assert "market" not in col.lower()
            assert "odds" not in col.lower()
            assert "popularity" not in col.lower()
            assert "implied" not in col.lower()

    def test_all_columns_includes_analytical(self):
        from backend.predictor.feature_engineering import ANALYTICAL_COLUMNS, ALL_COLUMNS
        for col in ANALYTICAL_COLUMNS:
            assert col in ALL_COLUMNS

    def test_market_columns_separate(self):
        from backend.predictor.feature_engineering import MARKET_COLUMNS, ANALYTICAL_COLUMNS
        for col in MARKET_COLUMNS:
            assert col not in ANALYTICAL_COLUMNS

    def test_no_duplicate_columns(self):
        from backend.predictor.feature_engineering import ALL_COLUMNS
        assert len(ALL_COLUMNS) == len(set(ALL_COLUMNS))

    def test_column_count(self):
        from backend.predictor.feature_engineering import ANALYTICAL_COLUMNS, MARKET_COLUMNS, ALL_COLUMNS
        assert len(ALL_COLUMNS) == len(ANALYTICAL_COLUMNS) + len(MARKET_COLUMNS)


class TestParseHorseWeight:
    def test_normal_format(self):
        from backend.predictor.feature_engineering import _parse_horse_weight
        kg, change = _parse_horse_weight("486(+2)")
        assert kg == 486.0
        assert change == 2.0

    def test_negative_change(self):
        from backend.predictor.feature_engineering import _parse_horse_weight
        kg, change = _parse_horse_weight("468(-8)")
        assert kg == 468.0
        assert change == -8.0

    def test_zero_change(self):
        from backend.predictor.feature_engineering import _parse_horse_weight
        kg, change = _parse_horse_weight("490(0)")
        assert kg == 490.0
        assert change == 0.0

    def test_weight_only(self):
        from backend.predictor.feature_engineering import _parse_horse_weight
        kg, change = _parse_horse_weight("500")
        assert kg == 500.0
        assert change == 0.0

    def test_empty_string(self):
        from backend.predictor.feature_engineering import _parse_horse_weight
        kg, change = _parse_horse_weight("")
        assert kg == 0.0
        assert change == 0.0


class TestDistanceCategory:
    def test_sprint(self):
        from backend.predictor.feature_engineering import _distance_category
        assert _distance_category(1200) == 0

    def test_mile(self):
        from backend.predictor.feature_engineering import _distance_category
        assert _distance_category(1600) == 1

    def test_intermediate(self):
        from backend.predictor.feature_engineering import _distance_category
        assert _distance_category(2000) == 2

    def test_stayer(self):
        from backend.predictor.feature_engineering import _distance_category
        assert _distance_category(2500) == 3


class TestExtractRaceContext:
    def test_basic_fields(self, sample_race_info, sample_entries):
        from backend.predictor.feature_engineering import extract_race_context
        ctx = extract_race_context(sample_race_info, sample_entries)
        assert ctx["field_size"] == 4  # 5 entries minus 1 scratched
        assert ctx["is_turf"] == 1.0
        assert ctx["distance_raw"] == 2500.0
        assert ctx["distance_category"] == 3.0  # stayer
        assert ctx["condition_severity"] == 0.8  # 重
        assert ctx["racecourse_code"] == 6.0  # 中山

    def test_dirt_surface(self):
        from backend.predictor.feature_engineering import extract_race_context
        info = {"surface": "ダート", "distance": 1800, "trackCondition": "良", "racecourseCode": "09"}
        ctx = extract_race_context(info, [])
        assert ctx["is_turf"] == 0.0


class TestExtractHorseFeatures:
    def test_returns_tuple(self, sample_race_info, sample_entries):
        from backend.predictor.feature_engineering import extract_race_context, extract_horse_features
        ctx = extract_race_context(sample_race_info, sample_entries)
        active = [e for e in sample_entries if not e.get("isScratched")]
        weights = [e["weightCarried"] for e in active]
        odds = [e.get("odds") for e in active]
        result = extract_horse_features(active[0], sample_race_info, ctx, weights, odds)
        assert isinstance(result, tuple)
        assert len(result) == 2
        feat_dict, factors = result
        assert isinstance(feat_dict, dict)
        assert isinstance(factors, dict)

    def test_feature_dict_has_all_columns(self, sample_race_info, sample_entries):
        from backend.predictor.feature_engineering import (
            extract_race_context, extract_horse_features, ALL_COLUMNS)
        ctx = extract_race_context(sample_race_info, sample_entries)
        active = [e for e in sample_entries if not e.get("isScratched")]
        weights = [e["weightCarried"] for e in active]
        odds = [e.get("odds") for e in active]
        feat_dict, _ = extract_horse_features(active[0], sample_race_info, ctx, weights, odds)
        for col in ALL_COLUMNS:
            assert col in feat_dict, f"Missing column: {col}"

    def test_factors_dict_has_13_keys(self, sample_race_info, sample_entries):
        from backend.predictor.feature_engineering import extract_race_context, extract_horse_features
        ctx = extract_race_context(sample_race_info, sample_entries)
        active = [e for e in sample_entries if not e.get("isScratched")]
        weights = [e["weightCarried"] for e in active]
        odds = [e.get("odds") for e in active]
        _, factors = extract_horse_features(active[0], sample_race_info, ctx, weights, odds)
        expected = ["marketScore", "pastPerformance", "jockeyAbility", "courseAffinity",
                    "distanceAptitude", "trainerAbility", "trackCondition", "trackDirection",
                    "trackSpecific", "ageAndSex", "weightCarried", "horseWeightChange", "formTrend"]
        for key in expected:
            assert key in factors

    def test_debut_detection(self, sample_race_info):
        from backend.predictor.feature_engineering import extract_race_context, extract_horse_features
        entry = {
            "horseNumber": 1, "frameNumber": 1, "horseName": "新馬",
            "sireName": "", "age": "牡2", "weightCarried": 55.0,
            "jockeyName": "川田", "trainerName": "友道",
            "horseWeight": "470(0)", "odds": 3.0, "popularity": 1,
            "pastRaces": [], "isScratched": False,
        }
        ctx = extract_race_context(sample_race_info, [entry])
        feat, _ = extract_horse_features(entry, sample_race_info, ctx, [55.0], [3.0])
        assert feat["is_debut"] == 1.0
        assert feat["past_race_count"] == 0.0

    def test_horse_weight_parsed(self, sample_race_info, sample_entries):
        from backend.predictor.feature_engineering import extract_race_context, extract_horse_features
        ctx = extract_race_context(sample_race_info, sample_entries)
        active = [e for e in sample_entries if not e.get("isScratched")]
        feat, _ = extract_horse_features(active[0], sample_race_info, ctx,
                                          [e["weightCarried"] for e in active],
                                          [e.get("odds") for e in active])
        assert feat["horse_weight_kg"] == 486.0
        assert feat["horse_weight_change_kg"] == 2.0


class TestFeaturesToVector:
    def test_correct_length(self, sample_race_info, sample_entries):
        from backend.predictor.feature_engineering import (
            extract_race_context, extract_horse_features, features_to_vector, ALL_COLUMNS)
        ctx = extract_race_context(sample_race_info, sample_entries)
        active = [e for e in sample_entries if not e.get("isScratched")]
        feat, _ = extract_horse_features(active[0], sample_race_info, ctx,
                                          [e["weightCarried"] for e in active],
                                          [e.get("odds") for e in active])
        vec = features_to_vector(feat)
        assert len(vec) == len(ALL_COLUMNS)

    def test_analytical_only(self, sample_race_info, sample_entries):
        from backend.predictor.feature_engineering import (
            extract_race_context, extract_horse_features, features_to_vector, ANALYTICAL_COLUMNS)
        ctx = extract_race_context(sample_race_info, sample_entries)
        active = [e for e in sample_entries if not e.get("isScratched")]
        feat, _ = extract_horse_features(active[0], sample_race_info, ctx,
                                          [e["weightCarried"] for e in active],
                                          [e.get("odds") for e in active])
        vec = features_to_vector(feat, ANALYTICAL_COLUMNS)
        assert len(vec) == len(ANALYTICAL_COLUMNS)

    def test_all_values_numeric(self, sample_race_info, sample_entries):
        from backend.predictor.feature_engineering import (
            extract_race_context, extract_horse_features, features_to_vector)
        ctx = extract_race_context(sample_race_info, sample_entries)
        active = [e for e in sample_entries if not e.get("isScratched")]
        feat, _ = extract_horse_features(active[0], sample_race_info, ctx,
                                          [e["weightCarried"] for e in active],
                                          [e.get("odds") for e in active])
        vec = features_to_vector(feat)
        for i, v in enumerate(vec):
            assert isinstance(v, (int, float)), f"Non-numeric at index {i}: {type(v)}"
