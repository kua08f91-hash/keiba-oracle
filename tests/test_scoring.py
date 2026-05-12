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
        preds = model.predict(sample_race_info, sample_entries)
        active = [p for p in preds if p["score"] > 0]
        # v5 primary: factors include analytical keys (no valueEdge)
        assert "pastPerformance" in active[0]["factors"]
        assert "jockeyAbility" in active[0]["factors"]

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


# =====================================================================
# 4. MARKET WEIGHT BLEND TESTS  (market_weight increased 0.147 → 0.40)
# =====================================================================
class TestMarketWeightBlend:
    """Verify the market_weight=0.40 blend behaviour introduced in the latest
    optimized_weights.json update.

    Blend formula (scoring.py WeightedScoringModel.predict):
        final_score = analytical * (1 - market_weight) + market * market_weight
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _minimal_entry(
        self,
        horse_number: int = 1,
        odds: float = 2.0,
        popularity: int = 1,
    ) -> dict:
        """Return the smallest valid entry dict that the scoring engine accepts."""
        return {
            "horseNumber": horse_number,
            "horseName": f"TestHorse{horse_number}",
            "age": "牡4",
            "weightCarried": 57.0,
            "jockeyName": "ルメール",
            "trainerName": "矢作",
            "sireName": "ゴールドシップ",
            "broodmareSire": "",
            "horseWeight": "486(0)",
            "odds": odds,
            "popularity": popularity,
            "isScratched": False,
            "pastRaces": [{"pos": 1, "track": "中山"}],
        }

    def _minimal_race(self) -> dict:
        return {
            "distance": 2000,
            "surface": "芝",
            "courseDetail": "右回り",
            "trackCondition": "良",
            "racecourseCode": "06",
            "date": "20260427",
        }

    # ------------------------------------------------------------------
    # Test 1: market score has exactly 40% influence when mkt_weight=0.40
    # ------------------------------------------------------------------
    def test_market_weight_040_gives_40pct_market_influence(self):
        """With market_weight=0.40, the market component contributes exactly 40%
        of the blended score and the analytical component contributes 60%."""
        from backend.predictor.scoring import WeightedScoringModel
        from backend.predictor.factors import calc_market_score

        model = WeightedScoringModel(market_weight=0.40)
        assert model._market_weight == 0.40
        assert abs(model._analytical_weight - 0.60) < 1e-9

        race = self._minimal_race()
        entry = self._minimal_entry(horse_number=1, odds=2.0, popularity=1)
        preds = model.predict(race, [entry])

        # Reconstruct expected score from raw components
        head_count = 1
        market = calc_market_score(2.0, 1, head_count)

        # Analytical score: sum weights * factor_scores for the non-market keys.
        # Because the model stores _weights we must compute it the same way.
        # The simplest check: final ≈ analytical*0.60 + market*0.40.
        # We extract market from the factors dict and verify the split.
        p = preds[0]
        reported_market = p["factors"]["marketScore"]  # rounded to 1dp
        # If market contributes 40%, then:
        # final = analytical*0.60 + market*0.40
        # → analytical = (final - market*0.40) / 0.60
        final = p["score"]
        implied_analytical = (final - reported_market * 0.40) / 0.60
        # Reconstruct analytical independently
        analytical_recomputed = sum(
            p["factors"].get(k, 0) * model._weights.get(k, 0)
            for k in model._weights
        )
        # Allow for the rounding applied to factors (1 dp) and score (2 dp)
        assert abs(implied_analytical - analytical_recomputed) < 1.0

    # ------------------------------------------------------------------
    # Test 2: popular horse ranks higher with mkt_weight=0.40 vs 0.147
    # ------------------------------------------------------------------
    def test_popular_horse_score_higher_with_larger_market_weight(self):
        """A horse that is the race favourite (low odds) gets a higher absolute
        score when market_weight=0.40 than when market_weight=0.147, because
        the high market score contributes more to the blend."""
        from backend.predictor.scoring import WeightedScoringModel

        race = self._minimal_race()
        # Two identical horses except horse 1 is the clear favourite
        favourite = self._minimal_entry(horse_number=1, odds=2.0, popularity=1)
        rival = self._minimal_entry(horse_number=2, odds=20.0, popularity=4)

        model_low = WeightedScoringModel(market_weight=0.147)
        model_high = WeightedScoringModel(market_weight=0.40)

        preds_low = {p["horseNumber"]: p["score"]
                     for p in model_low.predict(race, [favourite, rival])}
        preds_high = {p["horseNumber"]: p["score"]
                      for p in model_high.predict(race, [favourite, rival])}

        # The favourite's score should be strictly higher when market weight is
        # larger because its market_score >> 50 (the neutral default).
        assert preds_high[1] > preds_low[1], (
            "Favourite score should increase with higher market_weight"
        )

    # ------------------------------------------------------------------
    # Test 3: integration — real weights file has market_weight=0.40
    # ------------------------------------------------------------------
    def test_optimized_weights_file_has_market_weight_040(self):
        """Integration test: data/optimized_weights.json must contain
        market_weight=0.40 (the value changed from 0.147 in the latest run)."""
        import json
        import os

        weights_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "optimized_weights.json",
        )
        assert os.path.exists(weights_path), (
            f"optimized_weights.json not found at {weights_path}"
        )
        with open(weights_path) as f:
            data = json.load(f)

        assert "market_weight" in data, "market_weight key missing from weights file"
        assert data["market_weight"] == 0.40, (
            f"Expected market_weight=0.40, got {data['market_weight']}"
        )

    # ------------------------------------------------------------------
    # Test 4: WeightedScoringModel uses the injected market_weight value
    # ------------------------------------------------------------------
    def test_injected_market_weight_is_stored_and_used(self):
        """Constructor parameter market_weight must be stored on the instance
        and must affect _analytical_weight so both sum to 1.0."""
        from backend.predictor.scoring import WeightedScoringModel

        for mw in (0.0, 0.147, 0.40, 0.75, 1.0):
            model = WeightedScoringModel(market_weight=mw)
            assert model._market_weight == mw, (
                f"Expected _market_weight={mw}, got {model._market_weight}"
            )
            assert abs(model._analytical_weight - (1.0 - mw)) < 1e-9, (
                f"_analytical_weight should be {1 - mw}, got {model._analytical_weight}"
            )

    # ------------------------------------------------------------------
    # Test 5: score ordering changes — low-AI but popular horse ranks higher
    #         with mkt=0.40 than with mkt=0.147
    # ------------------------------------------------------------------
    def test_score_ordering_changes_with_higher_market_weight(self):
        """A horse with weak analytical factors but very low odds can overtake
        a horse with stronger analytical factors when market_weight is raised
        from 0.147 to 0.40."""
        from backend.predictor.scoring import WeightedScoringModel

        race = self._minimal_race()

        # Horse A: analytically strong, but an outsider (high odds)
        horse_a = {
            "horseNumber": 1,
            "horseName": "AnalyticalStar",
            "age": "牡4",           # peak age → good ageAndSex score
            "weightCarried": 54.0,  # lightest → good weightCarried score
            "jockeyName": "ルメール",   # top jockey
            "trainerName": "矢作",     # top trainer
            "sireName": "ゴールドシップ",
            "broodmareSire": "サンデーサイレンス",
            "horseWeight": "486(0)",
            "odds": 50.0,           # long shot — low market score
            "popularity": 10,
            "isScratched": False,
            "pastRaces": [{"pos": 1, "track": "中山"}, {"pos": 1, "track": "中山"}],
        }

        # Horse B: poor analytical profile but heavy favourite (very low odds)
        horse_b = {
            "horseNumber": 2,
            "horseName": "MarketFav",
            "age": "牡9",           # old → penalty on ageAndSex
            "weightCarried": 60.0,  # heaviest → penalty on weightCarried
            "jockeyName": "不明騎手",
            "trainerName": "不明",
            "sireName": "",
            "broodmareSire": "",
            "horseWeight": "500(+14)",  # large weight change → penalty
            "odds": 1.5,            # overwhelming favourite — high market score
            "popularity": 1,
            "isScratched": False,
            "pastRaces": [{"pos": 12, "track": "東京"}, {"pos": 11, "track": "東京"}],
        }

        model_low = WeightedScoringModel(market_weight=0.147)
        model_high = WeightedScoringModel(market_weight=0.40)

        def score_map(preds):
            return {p["horseNumber"]: p["score"] for p in preds}

        scores_low = score_map(model_low.predict(race, [horse_a, horse_b]))
        scores_high = score_map(model_high.predict(race, [horse_a, horse_b]))

        # At low market_weight the analytical star should lead
        assert scores_low[1] > scores_low[2], (
            "With mkt_weight=0.147 the analytically-strong horse should score higher"
        )
        # At high market_weight the market favourite should catch up or overtake
        assert scores_high[2] > scores_low[2], (
            "Market favourite score must increase when market_weight rises"
        )
        # The gap between horses must shrink (or reverse) at higher market weight
        gap_low = scores_low[1] - scores_low[2]
        gap_high = scores_high[1] - scores_high[2]
        assert gap_high < gap_low, (
            "Gap between analytical star and market favourite should narrow "
            "when market_weight increases"
        )

    # ------------------------------------------------------------------
    # Test 6: market_weight=0 → pure analytical score
    # ------------------------------------------------------------------
    def test_market_weight_zero_gives_pure_analytical_score(self):
        """When market_weight=0, the market component is entirely excluded.
        Changing the odds should not change the final score at all."""
        from backend.predictor.scoring import WeightedScoringModel

        race = self._minimal_race()
        model = WeightedScoringModel(market_weight=0.0)

        entry_fav = self._minimal_entry(horse_number=1, odds=1.5, popularity=1)
        entry_outsider = self._minimal_entry(horse_number=1, odds=99.9, popularity=16)

        # Same analytical profile → scores must be identical regardless of odds
        score_fav = model.predict(race, [entry_fav])[0]["score"]
        score_outsider = model.predict(race, [entry_outsider])[0]["score"]
        assert score_fav == score_outsider, (
            "market_weight=0 should produce identical scores regardless of odds"
        )

    # ------------------------------------------------------------------
    # Test 7: market_weight=1.0 → pure market score
    # ------------------------------------------------------------------
    def test_market_weight_one_gives_pure_market_score(self):
        """When market_weight=1.0, the final score equals the market score.
        Changing all analytical inputs while keeping odds/popularity identical
        must not affect the final score."""
        from backend.predictor.scoring import WeightedScoringModel
        from backend.predictor.factors import calc_market_score

        race = self._minimal_race()
        model = WeightedScoringModel(market_weight=1.0)

        # Two horses with completely different analytical profiles but same odds
        horse_peak = {
            "horseNumber": 1, "horseName": "Peak",
            "age": "牡4", "weightCarried": 54.0,
            "jockeyName": "ルメール", "trainerName": "矢作",
            "sireName": "ゴールドシップ", "broodmareSire": "",
            "horseWeight": "486(0)", "odds": 3.0, "popularity": 1,
            "isScratched": False,
            "pastRaces": [{"pos": 1}, {"pos": 1}, {"pos": 1}],
        }
        horse_poor = {
            "horseNumber": 1, "horseName": "Poor",
            "age": "牡9", "weightCarried": 60.0,
            "jockeyName": "不明騎手", "trainerName": "不明",
            "sireName": "", "broodmareSire": "",
            "horseWeight": "500(+16)", "odds": 3.0, "popularity": 1,
            "isScratched": False,
            "pastRaces": [{"pos": 15}, {"pos": 14}, {"pos": 16}],
        }

        score_peak = model.predict(race, [horse_peak])[0]["score"]
        score_poor = model.predict(race, [horse_poor])[0]["score"]

        # Both share the same odds/popularity → same market score → same final
        assert score_peak == score_poor, (
            "market_weight=1.0 should produce equal scores for horses with "
            "identical odds/popularity but different analytical profiles"
        )

        # Also verify the score equals the raw market score (rounded to 2dp)
        raw_market = calc_market_score(3.0, 1, 1)
        assert score_peak == round(raw_market, 2), (
            f"With market_weight=1.0 score should equal market score "
            f"({raw_market}), got {score_peak}"
        )
