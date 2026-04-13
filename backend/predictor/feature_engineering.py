"""Shared feature extraction for ML model training and inference.

v7: Market-independent analytical features + value detection.
Two feature sets:
  - ANALYTICAL_COLUMNS: Market-free features for independent horse evaluation
  - ALL_COLUMNS: Full features including market for combined model
"""
from __future__ import annotations

import hashlib
import re
import math

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
    RACECOURSE_CODE_MAP,
)

# --- Feature column definitions ---

# Analytical features (NO market/odds information)
ANALYTICAL_COLUMNS = [
    # Core factor scores (excluding marketScore)
    "f_past_performance",
    "f_jockey_ability",
    "f_course_affinity",
    "f_distance_aptitude",
    "f_trainer_ability",
    "f_track_condition",
    "f_track_direction",
    "f_track_specific",
    "f_age_and_sex",
    "f_weight_carried",
    "f_horse_weight_change",
    "f_form_trend",
    # Race context
    "field_size",
    "is_turf",
    "distance_raw",
    "distance_category",
    "condition_severity",
    "racecourse_code",
    # Horse physical
    "horse_weight_kg",
    "horse_weight_change_kg",
    "weight_per_carry",
    # Past race aggregates (richer)
    "past_mean_pos",
    "past_best_pos",
    "past_worst_pos",
    "past_win_rate",
    "past_place_rate",
    "past_pos_variance",
    "past_race_count",
    "past_same_track_best",
    "past_recent_trend",
    # Frame/position bias
    "frame_number",
    "horse_number_ratio",
    # Weight carried relative
    "carry_vs_field_avg",
    "carry_vs_field_min",
    # Jockey-trainer-course interactions
    "jockey_course_hash",
    "trainer_course_hash",
    "jockey_surface_hash",
    # Debut/experience
    "is_debut",
    "is_prior_winner",
]

# Market features (added on top of analytical for full model)
MARKET_COLUMNS = [
    "f_market_score",
    "odds_raw",
    "odds_rank",
    "popularity_rank",
    "odds_vs_field_avg",
    "implied_win_prob",
]

# Combined
ALL_COLUMNS = ANALYTICAL_COLUMNS + MARKET_COLUMNS

# Backward compat
FEATURE_COLUMNS = ALL_COLUMNS


def _stable_hash(s: str, mod: int = 997) -> float:
    """Deterministic hash that is consistent across Python processes."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod / mod


def _distance_category(distance: int) -> int:
    if distance <= 1400:
        return 0
    elif distance <= 1800:
        return 1
    elif distance <= 2200:
        return 2
    return 3


def _condition_severity(condition: str) -> float:
    return {"良": 0.0, "稍重": 0.5, "重": 0.8, "不良": 1.0}.get(condition, 0.0)


def _parse_horse_weight(hw_str: str) -> tuple:
    """Parse '468(+4)' -> (468.0, 4.0). Returns (0,0) on failure."""
    if not hw_str:
        return 0.0, 0.0
    m = re.match(r"(\d+)\(([+-]?\d+)\)", hw_str.replace(" ", ""))
    if m:
        return float(m.group(1)), float(m.group(2))
    m2 = re.match(r"(\d+)", hw_str)
    if m2:
        return float(m2.group(1)), 0.0
    return 0.0, 0.0


def extract_race_context(race_info: dict, entries: list) -> dict:
    """Extract race-level context features."""
    active = [e for e in entries if not e.get("isScratched")]
    surface = race_info.get("surface", "芝")
    distance = race_info.get("distance", 2000)
    condition = race_info.get("trackCondition", "")
    code = race_info.get("racecourseCode", "00")

    return {
        "field_size": len(active),
        "is_turf": 1.0 if surface == "芝" else 0.0,
        "distance_raw": float(distance),
        "distance_category": float(_distance_category(distance)),
        "condition_severity": _condition_severity(condition),
        "racecourse_code": float(int(code) if code.isdigit() else 0),
        "surface": surface,
    }


def extract_horse_features(
    entry: dict,
    race_info: dict,
    race_context: dict,
    all_weights: list,
    all_odds: list,
) -> tuple:
    """Extract features for a single horse.

    Returns:
        (feature_dict, factor_scores_dict)
    """
    surface = race_info.get("surface", "芝")
    distance = race_info.get("distance", 2000)
    head_count = int(race_context["field_size"]) or 14
    track_condition = race_info.get("trackCondition", "")
    course_detail = race_info.get("courseDetail", "")
    racecourse_code = race_info.get("racecourseCode", "")

    sire = entry.get("sireName", "")
    bms = entry.get("broodmareSire", "")
    jockey = entry.get("jockeyName", "")
    trainer = entry.get("trainerName", "")
    age_str = entry.get("age", "")
    weight = entry.get("weightCarried", 0)
    odds = entry.get("odds")
    popularity = entry.get("popularity")
    horse_weight_str = entry.get("horseWeight", "")
    past_races = entry.get("pastRaces", [])
    frame_number = entry.get("frameNumber", 0)
    horse_number = entry.get("horseNumber", 0)

    # --- Factor scores (for UI display) ---
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
        "horseWeightChange": calc_horse_weight_change(horse_weight_str),
        "formTrend": calc_form_trend(past_races),
    }

    # --- Horse weight parsing ---
    hw_kg, hw_change = _parse_horse_weight(horse_weight_str)

    # --- Past race aggregates (richer) ---
    positions = [r.get("pos", 0) for r in past_races[:5] if r.get("pos", 0) > 0]
    past_race_count = float(len(positions))

    if positions:
        past_mean_pos = sum(positions) / len(positions)
        past_best_pos = float(min(positions))
        past_worst_pos = float(max(positions))
        past_win_rate = sum(1 for p in positions if p == 1) / len(positions)
        past_place_rate = sum(1 for p in positions if p <= 3) / len(positions)
        if len(positions) > 1:
            past_pos_variance = sum((p - past_mean_pos) ** 2 for p in positions) / len(positions)
        else:
            past_pos_variance = 0.0
        # Recent trend: compare first half vs second half positions
        if len(positions) >= 3:
            recent = positions[:len(positions) // 2 + 1]
            older = positions[len(positions) // 2:]
            past_recent_trend = (sum(older) / len(older)) - (sum(recent) / len(recent))
        else:
            past_recent_trend = 0.0
    else:
        past_mean_pos = 10.0
        past_best_pos = 10.0
        past_worst_pos = 18.0
        past_win_rate = 0.0
        past_place_rate = 0.0
        past_pos_variance = 0.0
        past_recent_trend = 0.0

    # Same track performance
    track_name = RACECOURSE_CODE_MAP.get(racecourse_code, "")
    same_track_positions = [
        r.get("pos", 0) for r in past_races[:5]
        if r.get("pos", 0) > 0 and r.get("track") == track_name
    ]
    past_same_track_best = float(min(same_track_positions)) if same_track_positions else 10.0

    # --- Weight carried relative ---
    valid_weights = [w for w in all_weights if w > 0]
    if valid_weights and weight > 0:
        carry_avg = sum(valid_weights) / len(valid_weights)
        carry_min = min(valid_weights)
        carry_vs_field_avg = weight - carry_avg
        carry_vs_field_min = weight - carry_min
    else:
        carry_vs_field_avg = 0.0
        carry_vs_field_min = 0.0

    # Weight per carry ratio (heavier horse can carry more)
    weight_per_carry = hw_kg / weight if weight > 0 and hw_kg > 0 else 0.0

    # --- Frame/position features ---
    horse_number_ratio = horse_number / head_count if head_count > 0 else 0.5

    # --- Debut / class up detection ---
    is_debut = 1.0 if past_race_count == 0 else 0.0
    # Class up heuristic: if best past position is 1 (won before), likely stepping up
    is_prior_winner = 1.0 if past_race_count > 0 and past_best_pos == 1.0 else 0.0

    # --- Interaction hashes (larger space for less collision) ---
    jockey_course_hash = _stable_hash(jockey + racecourse_code)
    trainer_course_hash = _stable_hash(trainer + racecourse_code)
    jockey_surface_hash = _stable_hash(jockey + surface)

    # --- Market features ---
    odds_raw = float(odds) if odds is not None and odds > 0 else 30.0
    valid_odds = [o for o in all_odds if o is not None and o > 0]
    if valid_odds:
        sorted_odds = sorted(valid_odds)
        if odds_raw in sorted_odds:
            odds_rank = (sorted_odds.index(odds_raw) + 1) / len(sorted_odds)
        else:
            odds_rank = 0.5
        odds_avg = sum(valid_odds) / len(valid_odds)
        odds_vs_field_avg = math.log(odds_raw / odds_avg) if odds_avg > 0 else 0.0
    else:
        odds_rank = 0.5
        odds_vs_field_avg = 0.0

    popularity_rank = (popularity / head_count) if popularity is not None and head_count > 0 else 0.5
    implied_win_prob = 1.0 / odds_raw if odds_raw > 0 else 0.05

    # --- Build feature dict ---
    feature_dict = {
        # Analytical
        "f_past_performance": factors["pastPerformance"],
        "f_jockey_ability": factors["jockeyAbility"],
        "f_course_affinity": factors["courseAffinity"],
        "f_distance_aptitude": factors["distanceAptitude"],
        "f_trainer_ability": factors["trainerAbility"],
        "f_track_condition": factors["trackCondition"],
        "f_track_direction": factors["trackDirection"],
        "f_track_specific": factors["trackSpecific"],
        "f_age_and_sex": factors["ageAndSex"],
        "f_weight_carried": factors["weightCarried"],
        "f_horse_weight_change": factors["horseWeightChange"],
        "f_form_trend": factors["formTrend"],
        "field_size": float(head_count),
        "is_turf": race_context["is_turf"],
        "distance_raw": race_context["distance_raw"],
        "distance_category": race_context["distance_category"],
        "condition_severity": race_context["condition_severity"],
        "racecourse_code": race_context["racecourse_code"],
        "horse_weight_kg": hw_kg,
        "horse_weight_change_kg": hw_change,
        "weight_per_carry": weight_per_carry,
        "past_mean_pos": past_mean_pos,
        "past_best_pos": past_best_pos,
        "past_worst_pos": past_worst_pos,
        "past_win_rate": past_win_rate,
        "past_place_rate": past_place_rate,
        "past_pos_variance": past_pos_variance,
        "past_race_count": past_race_count,
        "past_same_track_best": past_same_track_best,
        "past_recent_trend": past_recent_trend,
        "frame_number": float(frame_number),
        "horse_number_ratio": horse_number_ratio,
        "carry_vs_field_avg": carry_vs_field_avg,
        "carry_vs_field_min": carry_vs_field_min,
        "jockey_course_hash": jockey_course_hash,
        "trainer_course_hash": trainer_course_hash,
        "jockey_surface_hash": jockey_surface_hash,
        "is_debut": is_debut,
        "is_prior_winner": is_prior_winner,
        # Market
        "f_market_score": factors["marketScore"],
        "odds_raw": odds_raw,
        "odds_rank": odds_rank,
        "popularity_rank": popularity_rank,
        "odds_vs_field_avg": odds_vs_field_avg,
        "implied_win_prob": implied_win_prob,
    }

    return feature_dict, factors


def features_to_vector(feature_dict: dict, columns: list = None) -> list:
    """Convert feature dict to ordered list matching specified columns."""
    cols = columns or ALL_COLUMNS
    return [feature_dict.get(col, 0.0) for col in cols]
