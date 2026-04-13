"""TDD tests for scoring engines (v5 WeightedScoringModel + v7 MLScoringModel).

RED → GREEN → REFACTOR cycle applied to:
- Individual factor calculations (12+ factors)
- v5 linear scoring model
- v7 ML dual-model scoring
- Score normalization and mark assignment
"""
from __future__ import annotations

import pytest


# =====================================================================
# 1. FACTOR CALCULATION TESTS
# =====================================================================
class TestCalcMarketScore:
    def test_top_favorite_scores_high(self):
        from backend.predictor.factors import calc_market_score
        score = calc_market_score(odds=2.0, popularity=1, head_count=16)
        assert score >= 90

    def test_low_popularity_scores_low(self):
        from backend.predictor.factors import calc_market_score
        score = calc_market_score(odds=100.0, popularity=16, head_count=16)
        assert score < 50

    def test_unknown_returns_default(self):
        from backend.predictor.factors import calc_market_score
        score = calc_market_score(odds=None, popularity=None, head_count=16)
        assert score == 45.0

    def test_score_within_bounds(self):
        from backend.predictor.factors import calc_market_score
        for pop in range(1, 19):
            score = calc_market_score(odds=None, popularity=pop, head_count=18)
            assert 0 <= score <= 100


class TestCalcPastPerformance:
    def test_all_wins_scores_high(self):
        from backend.predictor.factors import calc_past_performance
        races = [{"pos": 1}, {"pos": 1}, {"pos": 1}]
        score = calc_past_performance(races)
        assert score >= 95

    def test_all_bad_finishes(self):
        from backend.predictor.factors import calc_past_performance
        races = [{"pos": 15}, {"pos": 14}, {"pos": 16}]
        score = calc_past_performance(races)
        assert score < 30

    def test_empty_returns_default(self):
        from backend.predictor.factors import calc_past_performance
        assert calc_past_performance([]) == 45.0

    def test_recent_weighted_more(self):
        from backend.predictor.factors import calc_past_performance
        improving = calc_past_performance([{"pos": 1}, {"pos": 5}, {"pos": 10}])
        declining = calc_past_performance([{"pos": 10}, {"pos": 5}, {"pos": 1}])
        assert improving > declining


class TestCalcJockeyAbility:
    def test_top_jockey_high(self):
        from backend.predictor.factors import calc_jockey_ability
        assert calc_jockey_ability("ルメール") == 96.0

    def test_unknown_jockey_default(self):
        from backend.predictor.factors import calc_jockey_ability
        assert calc_jockey_ability("不明騎手") == 55.0

    def test_partial_match(self):
        from backend.predictor.factors import calc_jockey_ability
        assert calc_jockey_ability("Ｃ.ルメール") == 96.0


class TestCalcTrainerAbility:
    def test_top_trainer(self):
        from backend.predictor.factors import calc_trainer_ability
        assert calc_trainer_ability("矢作") == 92.0

    def test_unknown(self):
        from backend.predictor.factors import calc_trainer_ability
        assert calc_trainer_ability("不明") == 55.0


class TestCalcAgeAndSex:
    def test_peak_age(self):
        from backend.predictor.factors import calc_age_and_sex
        score = calc_age_and_sex("牡4")
        assert score == 92.0

    def test_female_penalty(self):
        from backend.predictor.factors import calc_age_and_sex
        male = calc_age_and_sex("牡4")
        female = calc_age_and_sex("牝4")
        assert female < male

    def test_old_horse(self):
        from backend.predictor.factors import calc_age_and_sex
        assert calc_age_and_sex("牡9") < 50

    def test_empty_returns_default(self):
        from backend.predictor.factors import calc_age_and_sex
        assert calc_age_and_sex("") == 50.0


class TestCalcFormTrend:
    def test_improving_form(self):
        from backend.predictor.factors import calc_form_trend
        races = [{"pos": 1}, {"pos": 3}, {"pos": 5}, {"pos": 8}]
        score = calc_form_trend(races)
        assert score >= 70

    def test_declining_form(self):
        from backend.predictor.factors import calc_form_trend
        races = [{"pos": 10}, {"pos": 5}, {"pos": 3}, {"pos": 1}]
        score = calc_form_trend(races)
        assert score <= 40

    def test_insufficient_data(self):
        from backend.predictor.factors import calc_form_trend
        assert calc_form_trend([{"pos": 1}]) == 50.0
        assert calc_form_trend([]) == 50.0


class TestCalcTrackCondition:
    def test_good_ground_neutral(self):
        from backend.predictor.factors import calc_track_condition_affinity
        assert calc_track_condition_affinity("ゴールドシップ", "良") == 50.0

    def test_heavy_ground_specialist(self):
        from backend.predictor.factors import calc_track_condition_affinity
        score = calc_track_condition_affinity("ゴールドシップ", "重")
        assert score > 70

    def test_heavy_ground_weak(self):
        from backend.predictor.factors import calc_track_condition_affinity
        score = calc_track_condition_affinity("ロードカナロア", "重")
        assert score < 45

    def test_bms_blending(self):
        from backend.predictor.factors import calc_track_condition_affinity
        sire_only = calc_track_condition_affinity("ゴールドシップ", "重", "")
        with_bms = calc_track_condition_affinity("ゴールドシップ", "重", "サンデーサイレンス")
        assert sire_only != with_bms  # BMS changes the result


class TestCalcTrackDirection:
    def test_winner_same_direction(self):
        from backend.predictor.factors import calc_track_direction
        races = [{"direction": "右", "pos": 1, "distance": 2000}]
        score = calc_track_direction(races, "右回り", 2000)
        assert score > 70

    def test_no_direction_data(self):
        from backend.predictor.factors import calc_track_direction
        assert calc_track_direction([], "右", 2000) == 50.0

    def test_distance_relevance(self):
        from backend.predictor.factors import calc_track_direction
        close = calc_track_direction(
            [{"direction": "右", "pos": 1, "distance": 2000}], "右回り", 2000)
        far = calc_track_direction(
            [{"direction": "右", "pos": 1, "distance": 2000}], "右回り", 1200)
        assert close >= far


class TestCalcHorseWeightChange:
    def test_stable_weight(self):
        from backend.predictor.factors import calc_horse_weight_change
        assert calc_horse_weight_change("486(+2)") == 80.0

    def test_large_change(self):
        from backend.predictor.factors import calc_horse_weight_change
        assert calc_horse_weight_change("500(+14)") == 25.0

    def test_no_data(self):
        from backend.predictor.factors import calc_horse_weight_change
        assert calc_horse_weight_change("") == 50.0


class TestCalcWeightCarried:
    def test_lightest_scores_highest(self):
        from backend.predictor.factors import calc_weight_carried
        weights = [54.0, 56.0, 58.0]
        assert calc_weight_carried(54.0, weights) > calc_weight_carried(58.0, weights)

    def test_equal_weights(self):
        from backend.predictor.factors import calc_weight_carried
        assert calc_weight_carried(57.0, [57.0, 57.0]) == 50.0


# =====================================================================
# 2. v5 WEIGHTED SCORING MODEL TESTS
# =====================================================================
class TestWeightedScoringModel:
    def test_predict_returns_list(self, sample_race_info, sample_entries):
        from backend.predictor.scoring import WeightedScoringModel
        model = WeightedScoringModel()
        preds = model.predict(sample_race_info, sample_entries)
        assert isinstance(preds, list)
        assert len(preds) == len(sample_entries)

    def test_scores_in_range(self, sample_race_info, sample_entries):
        from backend.predictor.scoring import WeightedScoringModel
        model = WeightedScoringModel()
        preds = model.predict(sample_race_info, sample_entries)
        for p in preds:
            assert 0 <= p["score"] <= 100

    def test_scratched_horse_score_zero(self, sample_race_info, sample_entries):
        from backend.predictor.scoring import WeightedScoringModel
        model = WeightedScoringModel()
        preds = model.predict(sample_race_info, sample_entries)
        scratched = [p for p in preds if p["horseNumber"] == 5]
        assert scratched[0]["score"] == 0
        assert scratched[0]["mark"] == ""

    def test_marks_assigned_correctly(self, sample_race_info, sample_entries):
        from backend.predictor.scoring import WeightedScoringModel
        model = WeightedScoringModel()
        preds = model.predict(sample_race_info, sample_entries)
        marks = [p["mark"] for p in preds if p["mark"]]
        assert "◎" in marks
        assert "◯" in marks

    def test_factors_dict_complete(self, sample_race_info, sample_entries):
        from backend.predictor.scoring import WeightedScoringModel, ALL_FACTOR_KEYS
        model = WeightedScoringModel()
        preds = model.predict(sample_race_info, sample_entries)
        active = [p for p in preds if p["score"] > 0]
        for p in active:
            for key in ALL_FACTOR_KEYS:
                assert key in p["factors"], f"Missing factor: {key}"

    def test_weights_sum_to_one(self):
        from backend.predictor.scoring import ANALYTICAL_WEIGHTS
        total = sum(ANALYTICAL_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01


# =====================================================================
# 3. v7 ML SCORING MODEL TESTS
# =====================================================================
class TestMLScoringModel:
    def test_loads_or_fallback(self):
        from backend.predictor.ml_scoring import MLScoringModel
        model = MLScoringModel()
        # Should either have ML model or fallback
        assert model._model_combined is not None or model._fallback is not None

    def test_predict_returns_valid(self, sample_race_info, sample_entries):
        from backend.predictor.ml_scoring import MLScoringModel
        model = MLScoringModel()
        preds = model.predict(sample_race_info, sample_entries)
        assert isinstance(preds, list)
        assert len(preds) == len(sample_entries)

    def test_scores_ordered(self, sample_race_info, sample_entries):
        from backend.predictor.ml_scoring import MLScoringModel
        model = MLScoringModel()
        preds = model.predict(sample_race_info, sample_entries)
        active = sorted([p for p in preds if p["score"] > 0],
                        key=lambda x: -x["score"])
        assert active[0]["mark"] == "◎"

    def test_value_edge_in_factors(self, sample_race_info, sample_entries):
        from backend.predictor.ml_scoring import MLScoringModel
        model = MLScoringModel()
        if model._model_combined is None:
            pytest.skip("No ML model available")
        preds = model.predict(sample_race_info, sample_entries)
        active = [p for p in preds if p["score"] > 0]
        # v7 should include valueEdge
        assert "valueEdge" in active[0]["factors"]

    def test_fallback_on_few_entries(self, sample_race_info):
        from backend.predictor.ml_scoring import MLScoringModel
        model = MLScoringModel()
        few_entries = [
            {"horseNumber": 1, "horseName": "A", "age": "牡3",
             "weightCarried": 55, "jockeyName": "ルメール", "trainerName": "矢作",
             "odds": 2.0, "popularity": 1, "horseWeight": "480(0)",
             "sireName": "", "pastRaces": [], "isScratched": False},
        ]
        preds = model.predict(sample_race_info, few_entries)
        assert isinstance(preds, list)
