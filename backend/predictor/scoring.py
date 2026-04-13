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
)

# Analytical factor weights (non-market factors, must sum to ~1.0)
# Optimized via 1,107-race historical data (2023-2024)
# Constrained: max 30% per factor to prevent overfitting
ANALYTICAL_WEIGHTS = {
    "trackDirection": 0.2822,       # 28.2% — 右回り/左回り適性（距離考慮版）
    "trackCondition": 0.2786,       # 27.9% — 馬場状態適性（BMS対応版）
    "jockeyAbility": 0.1541,        # 15.4% — 騎手能力
    "trackSpecific": 0.0815,        # 8.2%  — コース別実績
    "pastPerformance": 0.0687,      # 6.9%  — 過去成績
    "formTrend": 0.0300,            # 3.0%  — 調子トレンド（新規）
    "ageAndSex": 0.0314,            # 3.1%  — 年齢・性別
    "weightCarried": 0.0307,        # 3.1%  — 斤量
    "courseAffinity": 0.0113,        # 1.1%  — コース適性（血統）
    "horseWeightChange": 0.0110,    # 1.1%  — 馬体重変動
    "trainerAbility": 0.0102,       # 1.0%  — 調教師能力
    "distanceAptitude": 0.0102,     # 1.0%  — 距離適性（血統）
}

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
                   "formTrend"]


class WeightedScoringModel(PredictionModel):
    """Analytical-factor-driven prediction with market confirmation."""

    def predict(self, race_info: dict, entries: list[dict]) -> list[dict]:
        surface = race_info.get("surface", "芝")
        distance = race_info.get("distance", 2000)
        head_count = len([e for e in entries if not e.get("isScratched")])
        all_weights = [e.get("weightCarried", 0) for e in entries if not e.get("isScratched")]
        track_condition = race_info.get("trackCondition", "")
        course_detail = race_info.get("courseDetail", "")
        racecourse_code = race_info.get("racecourseCode", "")

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
            }

            # Analytical score (non-market factors only)
            analytical = sum(
                factors[k] * ANALYTICAL_WEIGHTS[k]
                for k in ANALYTICAL_WEIGHTS
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

            # Final blended score: 85% analytical + 15% market
            final_score = (
                d["analytical"] * ANALYTICAL_WEIGHT +
                d["market"] * MARKET_WEIGHT
            )

            predictions.append({
                "horseNumber": d["horseNumber"],
                "score": round(final_score, 2),
                "mark": "",
                "factors": {k: round(v, 1) for k, v in d["factors"].items()},
            })

        # Sort and assign marks
        active = [p for p in predictions if p["score"] > 0]
        active.sort(key=lambda p: p["score"], reverse=True)

        for i, pred in enumerate(active):
            pred["mark"] = MARK_MAP.get(i, "")

        return predictions
