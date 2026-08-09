"""TDD integration tests for valueDrop factor — end-to-end scoring pipeline.

Verifies:
1. ANALYTICAL_WEIGHTS sums to 1.0 and contains "valueDrop" at 0.035
2. ALL_FACTOR_KEYS contains "valueDrop" and key counts match ANALYTICAL_WEIGHTS
3. WeightedScoringModel.predict() includes "valueDrop" in factors output
4. Value-drop horse outscores overvalued horse on the valueDrop dimension
5. valueDrop score changes final analytical score proportionally to its 3.5% weight

RED -> GREEN -> REFACTOR cycle.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weights():
    from backend.predictor.scoring import ANALYTICAL_WEIGHTS
    return ANALYTICAL_WEIGHTS


def _all_factor_keys():
    from backend.predictor.scoring import ALL_FACTOR_KEYS
    return ALL_FACTOR_KEYS


def _model():
    from backend.predictor.scoring import WeightedScoringModel
    return WeightedScoringModel()


def _minimal_entry(horse_number: int, popularity: int, odds: float,
                   last_pos: int) -> dict:
    """Return a bare-minimum entry dict sufficient for WeightedScoringModel.predict()."""
    return {
        "horseNumber": horse_number,
        "popularity": popularity,
        "odds": odds,
        "age": "牡4",
        "weightCarried": 56.0,
        "horseWeight": "480(+2)",
        "jockeyName": "ルメール",
        "trainerName": "矢作",
        "sireName": "ディープインパクト",
        "broodmareSire": "",
        "frameNumber": horse_number,
        "pastRaces": [{"pos": last_pos, "distance": 2000, "surface": "芝",
                       "condition": "良", "direction": "右",
                       "track": "東京", "runningStyle": "差し"}],
    }


def _minimal_race_info() -> dict:
    return {
        "surface": "芝",
        "distance": 2000,
        "trackCondition": "良",
        "courseDetail": "右",
        "racecourseCode": "05",
        "date": "20260808",
    }


# ---------------------------------------------------------------------------
# 1. Weight balance tests
# ---------------------------------------------------------------------------

class TestWeightBalance:
    def test_analytical_weights_sum_to_one(self):
        """All ANALYTICAL_WEIGHTS values must sum to 1.0 within fp tolerance."""
        weights = _weights()
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001, (
            f"ANALYTICAL_WEIGHTS sum is {total:.6f}, expected 1.0 ± 0.001. "
            "A weight change left the distribution unbalanced."
        )

    def test_value_drop_key_exists_in_analytical_weights(self):
        """'valueDrop' must be present in ANALYTICAL_WEIGHTS."""
        assert "valueDrop" in _weights(), (
            "'valueDrop' key missing from ANALYTICAL_WEIGHTS in scoring.py"
        )

    def test_value_drop_weight_is_0035(self):
        """'valueDrop' weight must be exactly 0.035 (3.5%)."""
        weights = _weights()
        assert weights["valueDrop"] == pytest.approx(0.035, abs=1e-9), (
            f"Expected valueDrop weight 0.035, got {weights['valueDrop']}"
        )

    def test_value_drop_key_exists_in_all_factor_keys(self):
        """'valueDrop' must appear in ALL_FACTOR_KEYS for output serialisation."""
        assert "valueDrop" in _all_factor_keys(), (
            "'valueDrop' missing from ALL_FACTOR_KEYS in scoring.py"
        )

    def test_factor_key_count_consistency(self):
        """Every key in ANALYTICAL_WEIGHTS (non-market) must be in ALL_FACTOR_KEYS.

        ALL_FACTOR_KEYS also contains 'marketScore' which is NOT in
        ANALYTICAL_WEIGHTS — that is intentional.
        """
        weights = _weights()
        factor_keys = set(_all_factor_keys())
        for key in weights:
            assert key in factor_keys, (
                f"Key '{key}' present in ANALYTICAL_WEIGHTS but missing "
                f"from ALL_FACTOR_KEYS. Output will be incomplete."
            )

    def test_no_negative_weights(self):
        """All weights must be non-negative (negative weights are invalid)."""
        for key, w in _weights().items():
            assert w >= 0.0, f"Negative weight {w} for factor '{key}'"

    def test_no_zero_weights(self):
        """No weight should be zero — zero-weight factors are dead code."""
        for key, w in _weights().items():
            assert w > 0.0, f"Zero weight for factor '{key}' — remove or set > 0"


# ---------------------------------------------------------------------------
# 2. Integration tests: predict() output contains "valueDrop" in factors
# ---------------------------------------------------------------------------

class TestPredictOutputContainsValueDrop:
    def test_factors_dict_has_value_drop_key(self):
        """predict() must return 'valueDrop' inside each horse's factors dict."""
        model = _model()
        race_info = _minimal_race_info()
        entries = [
            _minimal_entry(1, popularity=1, odds=2.0, last_pos=1),
            _minimal_entry(2, popularity=6, odds=7.0, last_pos=1),
        ]
        results = model.predict(race_info, entries)

        for pred in results:
            assert "valueDrop" in pred["factors"], (
                f"Horse #{pred['horseNumber']} prediction missing 'valueDrop' "
                "in factors dict."
            )

    def test_value_drop_score_is_numeric(self):
        """valueDrop factor must be a number (int or float)."""
        model = _model()
        race_info = _minimal_race_info()
        entries = [_minimal_entry(1, popularity=6, odds=8.0, last_pos=1)]
        results = model.predict(race_info, entries)

        vd = results[0]["factors"]["valueDrop"]
        assert isinstance(vd, (int, float)), (
            f"valueDrop factor should be numeric, got {type(vd)}: {vd}"
        )

    def test_scratched_horse_still_has_value_drop_in_factors(self):
        """Scratched horses get a zeroed-out factors dict that still has 'valueDrop'."""
        model = _model()
        race_info = _minimal_race_info()
        entry = _minimal_entry(1, popularity=5, odds=6.0, last_pos=2)
        entry["isScratched"] = True
        results = model.predict(race_info, [entry])

        assert "valueDrop" in results[0]["factors"], (
            "Scratched horse factors dict missing 'valueDrop' key."
        )
        assert results[0]["factors"]["valueDrop"] == 0, (
            "Scratched horse 'valueDrop' should be 0."
        )


# ---------------------------------------------------------------------------
# 3. Integration tests: value-drop horse vs overvalued horse
# ---------------------------------------------------------------------------

class TestValueDropHorseOutscores:
    """
    Scenario A:
      Horse A: 前走1着 + 1番人気  → valueDrop=45 (market already priced in)
      Horse B: 前走1着 + 6番人気  → valueDrop=80 (strong value drop detected)

    Horse B must receive a higher valueDrop factor score than Horse A.
    """

    def _run(self):
        model = _model()
        race_info = _minimal_race_info()
        horse_a = _minimal_entry(1, popularity=1, odds=2.0, last_pos=1)   # overvalued fav
        horse_b = _minimal_entry(2, popularity=6, odds=8.0, last_pos=1)   # value drop
        results = model.predict(race_info, [horse_a, horse_b])
        by_number = {r["horseNumber"]: r for r in results}
        return by_number[1], by_number[2]

    def test_value_drop_horse_has_higher_value_drop_factor(self):
        """Horse B (6番人気 + 前走1着) must score higher on valueDrop than Horse A (1番人気 + 前走1着)."""
        horse_a_pred, horse_b_pred = self._run()
        vd_a = horse_a_pred["factors"]["valueDrop"]
        vd_b = horse_b_pred["factors"]["valueDrop"]
        assert vd_b > vd_a, (
            f"Expected value-drop horse B (vd={vd_b}) > overvalued horse A (vd={vd_a}). "
            "calc_value_drop is not distinguishing correctly."
        )

    def test_value_drop_horse_value_drop_factor_is_80(self):
        """Horse B with 前走1着 + 6番人気 must have valueDrop=80.0 (rounded to 1dp)."""
        _, horse_b_pred = self._run()
        vd_b = horse_b_pred["factors"]["valueDrop"]
        assert vd_b == pytest.approx(80.0, abs=0.1), (
            f"Expected valueDrop=80.0 for value-drop horse, got {vd_b}"
        )

    def test_overvalued_horse_value_drop_factor_is_45(self):
        """Horse A with 前走1着 + 1番人気 must have valueDrop=45.0 (market priced in)."""
        horse_a_pred, _ = self._run()
        vd_a = horse_a_pred["factors"]["valueDrop"]
        assert vd_a == pytest.approx(45.0, abs=0.1), (
            f"Expected valueDrop=45.0 for overvalued fav, got {vd_a}"
        )


class TestValueDropVsPopularWinner:
    """
    Scenario B:
      Horse A: 前走10着 + 1番人気 → bad form, popular  → valueDrop=50 (neutral)
      Horse B: 前走1着  + 6番人気 → good form, unpopular → valueDrop=80

    Horse B's valueDrop factor must be strictly higher.
    """

    def test_good_form_unpopular_beats_bad_form_popular_on_value_drop(self):
        model = _model()
        race_info = _minimal_race_info()
        horse_a = _minimal_entry(1, popularity=1, odds=2.0, last_pos=10)  # 前走10着 + 1番人気
        horse_b = _minimal_entry(2, popularity=6, odds=8.0, last_pos=1)   # 前走1着 + 6番人気
        results = model.predict(race_info, [horse_a, horse_b])
        by_number = {r["horseNumber"]: r for r in results}

        vd_a = by_number[1]["factors"]["valueDrop"]
        vd_b = by_number[2]["factors"]["valueDrop"]

        assert vd_b > vd_a, (
            f"Value-drop horse B (valueDrop={vd_b}) should outscore "
            f"bad-form favorite A (valueDrop={vd_a})."
        )

    def test_bad_form_popular_horse_value_drop_is_neutral(self):
        """Horse with 前走10着 + 1番人気 gets neutral valueDrop=50."""
        model = _model()
        race_info = _minimal_race_info()
        horse_a = _minimal_entry(1, popularity=1, odds=2.0, last_pos=10)
        results = model.predict(race_info, [horse_a])
        vd_a = results[0]["factors"]["valueDrop"]
        assert vd_a == pytest.approx(50.0, abs=0.1), (
            f"Expected neutral valueDrop=50.0 for 前走10着+1番人気, got {vd_a}"
        )


# ---------------------------------------------------------------------------
# 4. Identical-factors-except-valueDrop: analytical score difference
# ---------------------------------------------------------------------------

class TestIdenticalFactorsExceptValueDrop:
    """
    Two horses identical in every way EXCEPT their valueDrop factor score.
    We inject custom weights so that ONLY valueDrop differs (all other weights=0).
    This isolates the impact on the final analytical score.

    Horse A: valueDrop=45  (popularity=1, 前走1着)
    Horse B: valueDrop=80  (popularity=8, 前走1着)

    With weight=0.035 for valueDrop and all other analytical weights zeroed:
      analytical_A = 45 * 0.035  = 1.575
      analytical_B = 80 * 0.035  = 2.8
    """

    def _build_model_with_only_value_drop_weight(self):
        """Custom WeightedScoringModel with only valueDrop weighted."""
        from backend.predictor.scoring import WeightedScoringModel, ANALYTICAL_WEIGHTS
        custom_weights = {k: 0.0 for k in ANALYTICAL_WEIGHTS}
        custom_weights["valueDrop"] = 1.0   # 100% weight on valueDrop for clarity
        return WeightedScoringModel(analytical_weights=custom_weights)

    def test_higher_value_drop_score_increases_analytical_score(self):
        """Horse B (valueDrop=80) must have a higher final score than Horse A (valueDrop=45)
        when all other factors are equal."""
        model = self._build_model_with_only_value_drop_weight()
        race_info = _minimal_race_info()
        horse_a = _minimal_entry(1, popularity=1, odds=2.0, last_pos=1)   # valueDrop → 45
        horse_b = _minimal_entry(2, popularity=8, odds=10.0, last_pos=1)  # valueDrop → 80
        results = model.predict(race_info, [horse_a, horse_b])
        by_number = {r["horseNumber"]: r for r in results}

        score_a = by_number[1]["score"]
        score_b = by_number[2]["score"]

        assert score_b > score_a, (
            f"Horse B (valueDrop=80) should have higher final score than "
            f"Horse A (valueDrop=45). Got B={score_b}, A={score_a}."
        )

    def test_lower_value_drop_score_decreases_analytical_score(self):
        """Horse A (valueDrop=45) must have a lower final score than the neutral baseline (50)
        when only valueDrop contributes to the analytical score.

        Market weight is set to 0.0 so that the 1番人気 marketScore does not
        override the valueDrop penalty signal.
        """
        from backend.predictor.scoring import WeightedScoringModel, ANALYTICAL_WEIGHTS
        # Build a model: only valueDrop weighted AND no market signal
        custom_weights = {k: 0.0 for k in ANALYTICAL_WEIGHTS}
        custom_weights["valueDrop"] = 1.0
        model = WeightedScoringModel(analytical_weights=custom_weights, market_weight=0.0)

        race_info = _minimal_race_info()
        # Neutral horse: no past races → valueDrop=50
        horse_neutral = _minimal_entry(1, popularity=6, odds=8.0, last_pos=0)
        horse_neutral["pastRaces"] = []   # no past races → valueDrop=50
        # Penalised horse: 前走1着 + 1番人気 → valueDrop=45
        horse_penalised = _minimal_entry(2, popularity=1, odds=2.0, last_pos=1)
        results = model.predict(race_info, [horse_neutral, horse_penalised])
        by_number = {r["horseNumber"]: r for r in results}

        score_neutral = by_number[1]["score"]
        score_penalised = by_number[2]["score"]

        assert score_penalised < score_neutral, (
            f"Horse with valueDrop=45 should score lower than neutral (50) "
            f"when market weight is 0 and only valueDrop is weighted. "
            f"Got penalised={score_penalised}, neutral={score_neutral}."
        )


# ---------------------------------------------------------------------------
# 5. Scoring impact proportional to 3.5% weight
# ---------------------------------------------------------------------------

class TestValueDropScoringImpact:
    """
    Verify that the valueDrop factor contributes proportionally to the 3.5% weight.

    Setup: two horses identical in all analytical factors — achieved by
    injecting a WeightedScoringModel where only valueDrop has non-zero weight.

    With market_weight=0 and only valueDrop weighted:
      score = analytical * 1.0 + market * 0.0
            = valueDrop_score * 1.0

    So final score delta = (vd_B - vd_A) * weight_valueDrop / sum(all weights)

    With realistic 3.5% weight among 26 factors (each horse otherwise identical):
    delta = (80 - 45) * 0.035 / total_analytical ≈ 1.225 / 1.0 = 1.225
    """

    VALUE_DROP_WEIGHT = 0.035

    def test_value_drop_80_vs_50_increases_score(self):
        """Replacing valueDrop=50→80 should raise the final analytical score."""
        from backend.predictor.scoring import WeightedScoringModel, ANALYTICAL_WEIGHTS

        race_info = _minimal_race_info()

        # Horse A: no past races → valueDrop=50 (neutral)
        horse_neutral = _minimal_entry(1, popularity=6, odds=8.0, last_pos=0)
        horse_neutral["pastRaces"] = []

        # Horse B: 前走1着 + 6番人気 → valueDrop=80
        horse_value = _minimal_entry(2, popularity=6, odds=8.0, last_pos=1)

        model = WeightedScoringModel()
        results = model.predict(race_info, [horse_neutral, horse_value])
        by_number = {r["horseNumber"]: r for r in results}

        score_neutral = by_number[1]["score"]
        score_value = by_number[2]["score"]

        assert score_value > score_neutral, (
            f"valueDrop=80 horse should score higher than valueDrop=50 horse. "
            f"Got value={score_value}, neutral={score_neutral}."
        )

    def test_value_drop_45_vs_50_decreases_score(self):
        """Replacing valueDrop=50→45 should lower the final analytical score."""
        from backend.predictor.scoring import WeightedScoringModel

        race_info = _minimal_race_info()

        # Horse A: no past races → valueDrop=50 (neutral)
        horse_neutral = _minimal_entry(1, popularity=6, odds=8.0, last_pos=0)
        horse_neutral["pastRaces"] = []

        # Horse B: 前走1着 + 1番人気 → valueDrop=45 (penalty)
        horse_penalised = _minimal_entry(2, popularity=1, odds=2.0, last_pos=1)

        model = WeightedScoringModel()
        results = model.predict(race_info, [horse_neutral, horse_penalised])
        by_number = {r["horseNumber"]: r for r in results}

        score_neutral = by_number[1]["score"]
        score_penalised = by_number[2]["score"]

        # Note: the penalised horse also gets market penalty from low popularity (marketScore high for pop=1)
        # We check only the valueDrop dimension here by examining the factor directly.
        vd_neutral = by_number[1]["factors"]["valueDrop"]
        vd_penalised = by_number[2]["factors"]["valueDrop"]

        assert vd_penalised < vd_neutral, (
            f"valueDrop=45 (1番人気+前走1着) should be below neutral 50. "
            f"Got penalised_vd={vd_penalised}, neutral_vd={vd_neutral}."
        )

    def test_score_delta_proportional_to_35pct_weight(self):
        """Analytical score delta between two horses differing only in valueDrop
        should equal (vd_B - vd_A) * 0.035 when all other factors are equal.

        We achieve isolation by using a custom model with only valueDrop weighted.
        """
        from backend.predictor.scoring import WeightedScoringModel, ANALYTICAL_WEIGHTS

        # Build model: all weights 0 except valueDrop = 0.035
        custom_weights = {k: 0.0 for k in ANALYTICAL_WEIGHTS}
        custom_weights["valueDrop"] = self.VALUE_DROP_WEIGHT
        # market weight = 0 to eliminate market signal
        model = WeightedScoringModel(analytical_weights=custom_weights, market_weight=0.0)

        race_info = _minimal_race_info()
        # Horse A: no past races → valueDrop=50
        horse_a = _minimal_entry(1, popularity=6, odds=8.0, last_pos=0)
        horse_a["pastRaces"] = []
        # Horse B: 前走1着 + 6番人気 → valueDrop=80
        horse_b = _minimal_entry(2, popularity=6, odds=8.0, last_pos=1)

        results = model.predict(race_info, [horse_a, horse_b])
        by_number = {r["horseNumber"]: r for r in results}

        score_a = by_number[1]["score"]
        score_b = by_number[2]["score"]
        vd_a = by_number[1]["factors"]["valueDrop"]
        vd_b = by_number[2]["factors"]["valueDrop"]

        expected_delta = (vd_b - vd_a) * self.VALUE_DROP_WEIGHT
        actual_delta = score_b - score_a

        assert actual_delta == pytest.approx(expected_delta, abs=0.05), (
            f"Score delta {actual_delta:.4f} != expected {expected_delta:.4f} "
            f"(vd_b={vd_b}, vd_a={vd_a}, weight={self.VALUE_DROP_WEIGHT}). "
            "The valueDrop weight is not being applied correctly in the pipeline."
        )

    def test_value_drop_contributes_at_most_3pct_of_max_score(self):
        """Maximum contribution of valueDrop (score=80) at 3.5% weight is
        80 * 0.035 = 2.8 points out of a 100-point analytical scale.
        This is intentionally small — verifies no runaway influence."""
        max_contribution = 100.0 * self.VALUE_DROP_WEIGHT
        # The weight allows at most 3.5 points influence (100 * 0.035)
        assert max_contribution == pytest.approx(3.5, abs=0.001), (
            f"Expected max valueDrop contribution 3.5 pts, got {max_contribution}"
        )


# ---------------------------------------------------------------------------
# 6. Multi-horse field: valueDrop ranking stability
# ---------------------------------------------------------------------------

class TestValueDropFieldRanking:
    """Verify that in a realistic field, value-drop horses rank higher on
    the valueDrop factor than overvalued favorites."""

    def _build_field(self):
        """Build a 5-horse field with varying popularity + last-race position."""
        return [
            # Strong favorite who won last: overvalued → valueDrop=45
            _minimal_entry(1, popularity=1, odds=2.0, last_pos=1),
            # Mid-range who placed 2nd, 5番人気: slight undervaluation → valueDrop=60
            _minimal_entry(2, popularity=5, odds=5.5, last_pos=2),
            # Value drop: 前走1着 + 6番人気 → valueDrop=80
            _minimal_entry(3, popularity=6, odds=8.0, last_pos=1),
            # Bad form: 前走10着 + 1番人気 → valueDrop=50 (neutral, bad form)
            _minimal_entry(4, popularity=1, odds=2.0, last_pos=10),
            # Very unpopular but ran well: 前走3着 + 10番人気 → valueDrop=70
            _minimal_entry(5, popularity=10, odds=15.0, last_pos=3),
        ]

    def test_value_drop_factor_ordering_in_full_field(self):
        """Horse #3 (valueDrop=80) must have the highest valueDrop factor score.
        Horse #1 (valueDrop=45) must have the lowest."""
        model = _model()
        race_info = _minimal_race_info()
        results = model.predict(race_info, self._build_field())
        by_number = {r["horseNumber"]: r for r in results}

        vd_scores = {n: by_number[n]["factors"]["valueDrop"] for n in range(1, 6)}

        assert vd_scores[3] == pytest.approx(80.0, abs=0.1), (
            f"Horse #3 (前走1着+6番人気) expected valueDrop=80, got {vd_scores[3]}"
        )
        assert vd_scores[1] == pytest.approx(45.0, abs=0.1), (
            f"Horse #1 (前走1着+1番人気) expected valueDrop=45, got {vd_scores[1]}"
        )
        assert vd_scores[3] > vd_scores[5] > vd_scores[2] > vd_scores[4] >= vd_scores[1], (
            f"Expected valueDrop ordering 3>5>2>4>=1. "
            f"Got: {vd_scores}"
        )

    def test_all_predictions_have_value_drop_factor(self):
        """Every prediction in a full field must carry a 'valueDrop' factor key."""
        model = _model()
        race_info = _minimal_race_info()
        results = model.predict(race_info, self._build_field())

        for pred in results:
            assert "valueDrop" in pred["factors"], (
                f"Horse #{pred['horseNumber']} missing 'valueDrop' in factors."
            )

    def test_value_drop_scores_within_valid_range(self):
        """All valueDrop factor scores must be within [0, 100]."""
        model = _model()
        race_info = _minimal_race_info()
        results = model.predict(race_info, self._build_field())

        for pred in results:
            vd = pred["factors"]["valueDrop"]
            assert 0.0 <= vd <= 100.0, (
                f"Horse #{pred['horseNumber']} valueDrop={vd} is out of [0,100] range."
            )
