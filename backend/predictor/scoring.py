"""Weighted scoring prediction engine v5.

Strategy: Analytical-factor-driven prediction with minimal market signal.
Market score (odds/popularity) is only 15% — a light confirmation signal.
85% of the prediction comes from data-driven analytical factors (12 factors).

v5 improvements over v4:
- New factor: formTrend (3%) — detects improving/declining form
- Enhanced trackDirection: distance-weighted scoring (not just direction match)
- Enhanced trackCondition: uses broodmare sire (BMS) data for better estimation
- Bet optimizer: race-pattern-based temperature adjustment

Base weights from 1,107-race historical optimization (2023-2024), with
trackDirection/trackCondition each reduced by 1.5% to fund formTrend.
"""
from __future__ import annotations
from .model import PredictionModel
from .factors import (
    calc_market_score,
    calc_course_affinity,
    calc_distance_aptitude,
    calc_age_and_sex,
    calc_weight_carried,
    calc_jockey_ability,
    calc_trainer_ability,
    calc_horse_weight_change,
    calc_past_performance,
    calc_track_condition_affinity,
    calc_track_direction,
    calc_track_specific,
    calc_form_trend,
    calc_same_distance_performance,
    calc_same_surface_performance,
    calc_same_condition_performance,
    calc_running_style_consistency,
    calc_speed_figure,
    calc_weight_carried_trend,
    calc_days_since_last_race,
    calc_agari3f_score,
    calc_margin_score,
    calc_pace_predict,
    calc_draw_bias,
    calc_jockey_course_distance,
    calc_pace_position_advantage,
    calc_rotation_fitness,
    calc_bloodline_track_condition,
)

# Analytical factor weights (non-market factors, must sum to ~1.0)
# Optimized via 1,107-race historical data (2023-2024)
# Constrained: max 30% per factor to prevent overfitting
ANALYTICAL_WEIGHTS = {
    "trackDirection": 0.0527,  # Reduced from 0.1027 to fund D6 new factors
    "trackCondition": 0.1017,  # Reduced from 0.1217 to fund D6 new factors
    "trackSpecific": 0.0476,
    "jockeyAbility": 0.0552,  # Reduced from 0.0952 to fund D6 new factors
    "sameDistance": 0.0635,
    "sameSurface": 0.0635,
    "sameCondition": 0.0476,
    "pastPerformance": 0.0476,
    "speedFigure": 0.0476,
    "runningStyle": 0.0370,
    "daysSinceLast": 0.0212,
    "weightCarriedTrend": 0.0212,
    "formTrend": 0.0370,
    "ageAndSex": 0.0370,
    "weightCarried": 0.0265,
    "horseWeightChange": 0.0265,
    "trainerAbility": 0.0265,
    "courseAffinity": 0.0265,
    "distanceAptitude": 0.0212,
    "agari3f": 0.0212,
    "marginScore": 0.0212,
    "drawBias": 0.0400,
    # D6 new factors
    "jockeyCourseDistance": 0.04,
    "pacePositionAdvantage": 0.03,
    "rotationFitness": 0.02,
    "bloodlineTrackCondition": 0.02,
}
# Defaults (22 keys, sum=1.0). When optimized_weights.json is loaded,
# its values replace these entirely.

# Final score blend: 85% analytical + 15% market
MARKET_WEIGHT = 0.15
ANALYTICAL_WEIGHT = 0.85

# Mark assignment
MARK_MAP = {
    0: "◎",
    1: "◯",
    2: "▲",
    3: "▲",
    4: "△",
    5: "△",
}

# All factor keys for output
ALL_FACTOR_KEYS = ["marketScore", "pastPerformance", "jockeyAbility",
                   "courseAffinity", "distanceAptitude", "trainerAbility",
                   "trackCondition", "trackDirection", "trackSpecific",
                   "ageAndSex", "weightCarried", "horseWeightChange",
                   "formTrend", "sameDistance", "sameSurface", "sameCondition",
                   "speedFigure", "runningStyle", "daysSinceLast", "weightCarriedTrend",
                   "agari3f", "marginScore", "drawBias",
                   "jockeyCourseDistance", "pacePositionAdvantage",
                   "rotationFitness", "bloodlineTrackCondition"]


class WeightedScoringModel(PredictionModel):
    """Analytical-factor-driven prediction with market confirmation."""

    def __init__(self, analytical_weights=None, market_weight=None):
        """Optionally inject custom weights (avoids module-global mutation)."""
        self._weights = analytical_weights or dict(ANALYTICAL_WEIGHTS)
        self._market_weight = market_weight if market_weight is not None else MARKET_WEIGHT
        self._analytical_weight = 1.0 - self._market_weight

    def predict(self, race_info: dict, entries: list[dict], external_adjustments: dict = None) -> list[dict]:
        """Predict scores for all entries.

        Args:
            external_adjustments: Optional dict {horseName: adjustment_score} from
                text analysis. Applied as post-hoc score modifier (-10 to +10).
        """
        surface = race_info.get("surface", "芝")
        distance = race_info.get("distance", 2000)
        head_count = len([e for e in entries if not e.get("isScratched")])
        all_weights = [e.get("weightCarried", 0) for e in entries if not e.get("isScratched")]
        track_condition = race_info.get("trackCondition", "")
        course_detail = race_info.get("courseDetail", "")
        racecourse_code = race_info.get("racecourseCode", "")
        race_date = race_info.get("date", "")  # YYYYMMDD → convert to YYYY.MM.DD later

        # Race-level pace prediction (shared across all entries)
        pace_score = calc_pace_predict(entries)

        raw_data = []
        for entry in entries:
            if entry.get("isScratched"):
                raw_data.append(None)
                continue

            sire = entry.get("sireName", "")
            bms = entry.get("broodmareSire", "")
            jockey = entry.get("jockeyName", "")
            trainer = entry.get("trainerName", "")
            age_str = entry.get("age", "")
            weight = entry.get("weightCarried", 0)
            odds = entry.get("odds")
            popularity = entry.get("popularity")
            horse_weight = entry.get("horseWeight", "")
            past_races = entry.get("pastRaces", [])

            # Normalize race_date to YYYY.MM.DD for date calc
            race_date_norm = ""
            if race_date and len(race_date) == 8 and race_date.isdigit():
                race_date_norm = f"{race_date[:4]}.{race_date[4:6]}.{race_date[6:]}"

            # Derive class_change for rotation_fitness from entry data
            class_change = entry.get("classChange", 0)

            factors = {
                "marketScore": calc_market_score(odds, popularity, head_count),
                "pastPerformance": calc_past_performance(past_races),
                "jockeyAbility": calc_jockey_ability(jockey),
                "courseAffinity": calc_course_affinity(sire, surface),
                "distanceAptitude": calc_distance_aptitude(sire, distance),
                "trainerAbility": calc_trainer_ability(trainer),
                "trackCondition": calc_track_condition_affinity(sire, track_condition, bms),
                "trackDirection": calc_track_direction(past_races, course_detail, distance),
                "trackSpecific": calc_track_specific(past_races, racecourse_code),
                "ageAndSex": calc_age_and_sex(age_str),
                "weightCarried": calc_weight_carried(weight, all_weights),
                "horseWeightChange": calc_horse_weight_change(horse_weight),
                "formTrend": calc_form_trend(past_races),
                "sameDistance": calc_same_distance_performance(past_races, distance),
                "sameSurface": calc_same_surface_performance(past_races, surface),
                "sameCondition": calc_same_condition_performance(past_races, track_condition),
                "speedFigure": calc_speed_figure(past_races, distance),
                "runningStyle": calc_running_style_consistency(past_races),
                "daysSinceLast": calc_days_since_last_race(past_races, race_date_norm),
                "weightCarriedTrend": calc_weight_carried_trend(past_races, weight),
                "agari3f": calc_agari3f_score(past_races, surface),
                "marginScore": calc_margin_score(past_races),
                "drawBias": calc_draw_bias(
                    entry.get("frameNumber", entry.get("horseNumber", 0)),
                    head_count, surface, distance, course_detail,
                    course_code=racecourse_code,
                    track_condition=track_condition,
                ),
                # D6 new factors
                "jockeyCourseDistance": calc_jockey_course_distance(
                    jockey, racecourse_code, distance, past_races
                ),
                "pacePositionAdvantage": calc_pace_position_advantage(
                    past_races, entries, head_count
                ),
                "rotationFitness": calc_rotation_fitness(
                    past_races, race_date_norm, class_change
                ),
                "bloodlineTrackCondition": calc_bloodline_track_condition(
                    sire, bms, track_condition, past_races
                ),
            }

            # Analytical score (non-market factors only)
            analytical = sum(
                factors[k] * self._weights.get(k, 0)
                for k in self._weights
            )

            market = factors["marketScore"]

            raw_data.append({
                "horseNumber": entry["horseNumber"],
                "factors": factors,
                "analytical": analytical,
                "market": market,
            })

        # Build predictions (zip to avoid index misalignment)
        predictions = []
        for entry, d in zip(entries, raw_data):
            if entry.get("isScratched") or d is None:
                predictions.append({
                    "horseNumber": entry["horseNumber"],
                    "score": 0,
                    "mark": "",
                    "factors": {k: 0 for k in ALL_FACTOR_KEYS},
                })
                continue

            # Final blended score
            final_score = (
                d["analytical"] * self._analytical_weight +
                d["market"] * self._market_weight
            )

            predictions.append({
                "horseNumber": d["horseNumber"],
                "score": round(final_score, 2),
                "mark": "",
                "factors": {k: round(v, 1) for k, v in d["factors"].items()},
            })

        # Apply external adjustments (from text analysis)
        if external_adjustments:
            for pred in predictions:
                horse_name = pred.get("horseName", "")
                adj = external_adjustments.get(horse_name, 0)
                if adj:
                    pred["score"] = max(0, min(100, pred["score"] + adj))

        # Sort and assign marks
        active = [p for p in predictions if p["score"] > 0]
        active.sort(key=lambda p: p["score"], reverse=True)

        for i, pred in enumerate(active):
            pred["mark"] = MARK_MAP.get(i, "")

        return predictions
