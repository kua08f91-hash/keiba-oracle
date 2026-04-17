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
)

# Analytical factor weights (non-market factors, must sum to ~1.0)
# Optimized via 1,107-race historical data (2023-2024)
# Constrained: max 30% per factor to prevent overfitting
ANALYTICAL_WEIGHTS = {
    # Core track/condition (32%)
    "trackDirection": 0.1300,       # 13.0% — 右回り/左回り適性
    "trackCondition": 0.1300,       # 13.0% — 馬場状態適性（BMS対応）
    "trackSpecific": 0.0500,        # 5.0%  — コース別実績
    "jockeyAbility": 0.1000,        # 10.0% — 騎手能力
    # Condition matching (20%)
    "sameDistance": 0.0700,         # 7.0%  — 同距離実績
    "sameSurface": 0.0700,          # 7.0%  — 同馬場種実績
    "sameCondition": 0.0500,        # 5.0%  — 同馬場状態実績
    "pastPerformance": 0.0500,      # 5.0%  — 過去成績
    # New — from enhanced scraping (13%)
    "speedFigure": 0.0500,          # 5.0%  — 上がり/タイム指数（新規）
    "runningStyle": 0.0400,         # 4.0%  — 脚質一貫性（新規）
    "daysSinceLast": 0.0200,        # 2.0%  — 休養明け（新規）
    "weightCarriedTrend": 0.0200,   # 2.0%  — 斤量トレンド（新規）
    # Supporting factors (15%)
    "formTrend": 0.0400,            # 4.0%  — 調子トレンド
    "ageAndSex": 0.0400,            # 4.0%  — 年齢・性別
    "weightCarried": 0.0300,        # 3.0%  — 斤量絶対値
    "horseWeightChange": 0.0300,    # 3.0%  — 馬体重変動
    "trainerAbility": 0.0300,       # 3.0%  — 調教師能力
    # Pedigree (5%)
    "courseAffinity": 0.0300,       # 3.0%  — コース適性（血統）
    "distanceAptitude": 0.0200,     # 2.0%  — 距離適性（血統）
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
                   "formTrend", "sameDistance", "sameSurface", "sameCondition",
                   "speedFigure", "runningStyle", "daysSinceLast", "weightCarriedTrend"]


class WeightedScoringModel(PredictionModel):
    """Analytical-factor-driven prediction with market confirmation."""

    def __init__(self, analytical_weights=None, market_weight=None):
        """Optionally inject custom weights (avoids module-global mutation)."""
        self._weights = analytical_weights or dict(ANALYTICAL_WEIGHTS)
        self._market_weight = market_weight if market_weight is not None else MARKET_WEIGHT
        self._analytical_weight = 1.0 - self._market_weight

    def predict(self, race_info: dict, entries: list[dict]) -> list[dict]:
        surface = race_info.get("surface", "芝")
        distance = race_info.get("distance", 2000)
        head_count = len([e for e in entries if not e.get("isScratched")])
        all_weights = [e.get("weightCarried", 0) for e in entries if not e.get("isScratched")]
        track_condition = race_info.get("trackCondition", "")
        course_detail = race_info.get("courseDetail", "")
        racecourse_code = race_info.get("racecourseCode", "")
        race_date = race_info.get("date", "")  # YYYYMMDD → convert to YYYY.MM.DD later

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

        # Sort and assign marks
        active = [p for p in predictions if p["score"] > 0]
        active.sort(key=lambda p: p["score"], reverse=True)

        for i, pred in enumerate(active):
            pred["mark"] = MARK_MAP.get(i, "")

        return predictions
