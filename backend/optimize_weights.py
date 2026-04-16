"""Optimize v5 analytical factor weights using historical race results.

Uses scipy differential_evolution to find the 12 factor weights + market
blend ratio that maximize prediction accuracy on historical data.

Usage:
    /usr/bin/python3 -m backend.optimize_weights

Output:
    data/optimized_weights.json — loaded by MLScoringModel on startup
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.optimize import differential_evolution
import numpy as np

from backend.database.db import init_db
from backend.predictor.scoring import ANALYTICAL_WEIGHTS, MARKET_WEIGHT
from backend.predictor.factors import (
    calc_market_score, calc_course_affinity, calc_distance_aptitude,
    calc_age_and_sex, calc_weight_carried, calc_jockey_ability,
    calc_trainer_ability, calc_horse_weight_change, calc_past_performance,
    calc_track_condition_affinity, calc_track_direction, calc_track_specific,
    calc_form_trend, calc_same_distance_performance,
    calc_same_surface_performance, calc_same_condition_performance,
    calc_running_style_consistency, calc_speed_figure,
    calc_weight_carried_trend, calc_days_since_last_race,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
HISTORY_PATH = os.path.join(DATA_DIR, "historical_races.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "optimized_weights.json")

FACTOR_KEYS = list(ANALYTICAL_WEIGHTS.keys())


def load_races():
    """Load historical races with results."""
    if not os.path.exists(HISTORY_PATH):
        print(f"No historical data at {HISTORY_PATH}")
        return []
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        races = json.load(f)
    valid = [r for r in races if r.get("results") and r.get("entries") and len(r["entries"]) >= 5]
    return valid


def compute_factors_for_race(race):
    """Compute all 12 factor scores for each entry in a race."""
    entries = race["entries"]
    race_info = race.get("race_info", {})
    surface = race_info.get("surface", "芝")
    distance = race_info.get("distance", 2000)
    track_condition = race_info.get("trackCondition", "")
    course_detail = race_info.get("courseDetail", "")
    racecourse_code = race_info.get("racecourseCode", "")

    active = [e for e in entries if not e.get("isScratched")]
    all_weights = [e.get("weightCarried", 0) for e in active]

    factor_data = []
    for entry in active:
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

        race_date_norm = ""
        rd = race_info.get("date", "")
        if rd and len(rd) == 8 and rd.isdigit():
            race_date_norm = f"{rd[:4]}.{rd[4:6]}.{rd[6:]}"

        factors = {
            "trackDirection": calc_track_direction(past_races, course_detail, distance),
            "trackCondition": calc_track_condition_affinity(sire, track_condition, bms),
            "trackSpecific": calc_track_specific(past_races, racecourse_code),
            "jockeyAbility": calc_jockey_ability(jockey),
            "sameDistance": calc_same_distance_performance(past_races, distance),
            "sameSurface": calc_same_surface_performance(past_races, surface),
            "sameCondition": calc_same_condition_performance(past_races, track_condition),
            "pastPerformance": calc_past_performance(past_races),
            "speedFigure": calc_speed_figure(past_races, distance),
            "runningStyle": calc_running_style_consistency(past_races),
            "daysSinceLast": calc_days_since_last_race(past_races, race_date_norm),
            "weightCarriedTrend": calc_weight_carried_trend(past_races, weight),
            "formTrend": calc_form_trend(past_races),
            "ageAndSex": calc_age_and_sex(age_str),
            "weightCarried": calc_weight_carried(weight, all_weights),
            "horseWeightChange": calc_horse_weight_change(horse_weight),
            "trainerAbility": calc_trainer_ability(trainer),
            "courseAffinity": calc_course_affinity(sire, surface),
            "distanceAptitude": calc_distance_aptitude(sire, distance),
        }
        market = calc_market_score(odds, popularity, len(active))

        factor_data.append({
            "horseNumber": entry.get("horseNumber", 0),
            "factors": factors,
            "market": market,
        })

    return factor_data


def score_with_weights(factor_data, weights_vec):
    """Score horses using given weight vector. [12 factor weights, market_weight]"""
    n = len(FACTOR_KEYS)
    factor_weights = {k: w for k, w in zip(FACTOR_KEYS, weights_vec[:n])}
    market_w = weights_vec[n]
    analytical_w = 1.0 - market_w

    scores = []
    for fd in factor_data:
        analytical = sum(fd["factors"][k] * factor_weights[k] for k in FACTOR_KEYS)
        score = analytical * analytical_w + fd["market"] * market_w
        scores.append((fd["horseNumber"], score))

    return sorted(scores, key=lambda x: -x[1])


def evaluate_weights(weights_vec, races_data):
    """Evaluate weights by top-1 + top-3 prediction accuracy."""
    top1_hits = 0
    top3_hits = 0
    total = 0

    for race_factors, winners in races_data:
        if not race_factors or not winners:
            continue

        ranked = score_with_weights(race_factors, weights_vec)
        if len(ranked) < 3:
            continue

        top3_predicted = [h for h, _ in ranked[:3]]
        winner = winners[0]

        total += 1
        if top3_predicted[0] == winner:
            top1_hits += 1
        if winner in top3_predicted:
            top3_hits += 1

    if total == 0:
        return 0.0

    # 40% top-1 + 60% top-3 (top-3 matters more for ワイド/3連系)
    return 0.4 * (top1_hits / total) + 0.6 * (top3_hits / total)


def objective(weights_vec, races_data):
    """Minimization objective (negate for maximization)."""
    n = len(FACTOR_KEYS)
    factor_w = np.abs(weights_vec[:n])
    total_fw = factor_w.sum()
    if total_fw > 0:
        factor_w = factor_w / total_fw
    market_w = np.clip(weights_vec[n], 0.0, 0.30)
    normalized = np.concatenate([factor_w, [market_w]])
    return -evaluate_weights(normalized, races_data)


def main():
    init_db()
    print("=" * 60, flush=True)
    print("  v5 Weight Optimizer", flush=True)
    print("=" * 60, flush=True)

    print("\nLoading historical races...", flush=True)
    races = load_races()
    print(f"  {len(races)} races with results", flush=True)

    if len(races) < 50:
        print("Not enough data (need >= 50 races).")
        return

    print("Computing factors...", flush=True)
    races_data = []
    for race in races:
        factors = compute_factors_for_race(race)
        results_data = race.get("results", [])
        winners = []
        if isinstance(results_data, dict) and results_data:
            # Dict format: {horse_number_str: finish_position}
            sorted_pairs = sorted(results_data.items(), key=lambda x: x[1])
            winners = [int(hn) for hn, pos in sorted_pairs[:3]]
        elif isinstance(results_data, list) and results_data:
            if isinstance(results_data[0], dict):
                winners = [r.get("horseNumber", 0) for r in
                          sorted(results_data, key=lambda r: r.get("finishPosition", 99))[:3]]
            else:
                winners = results_data[:3]
        if factors and winners:
            races_data.append((factors, winners))

    print(f"  {len(races_data)} races ready for optimization", flush=True)

    x0 = np.array([ANALYTICAL_WEIGHTS[k] for k in FACTOR_KEYS] + [MARKET_WEIGHT])
    current_score = -objective(x0, races_data)
    print(f"\nCurrent weights score: {current_score:.4f}", flush=True)

    print("\nOptimizing (differential evolution)...", flush=True)
    n = len(FACTOR_KEYS)
    bounds = [(0.0, 0.30)] * n + [(0.0, 0.30)]

    result = differential_evolution(
        objective,
        bounds,
        args=(races_data,),
        seed=42,
        maxiter=200,
        tol=1e-6,
        polish=True,
    )

    opt_factors = np.abs(result.x[:n])
    opt_factors = opt_factors / opt_factors.sum()
    opt_market = np.clip(result.x[n], 0.0, 0.30)
    opt_score = -result.fun

    print(f"\nOptimized score: {opt_score:.4f} (was {current_score:.4f}, Δ{opt_score-current_score:+.4f})", flush=True)

    opt_weights = {k: round(float(v), 4) for k, v in zip(FACTOR_KEYS, opt_factors)}
    output = {
        "version": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "analytical_weights": opt_weights,
        "market_weight": round(float(opt_market), 4),
        "score_before": round(current_score, 4),
        "score_after": round(opt_score, 4),
        "improvement": round(opt_score - current_score, 4),
        "n_races": len(races_data),
        "optimized_at": datetime.now().isoformat(),
    }

    print(f"\n  --- Optimized Weights ---", flush=True)
    for k, v in opt_weights.items():
        prev = ANALYTICAL_WEIGHTS[k]
        delta = v - prev
        marker = "▲" if delta > 0.01 else ("▼" if delta < -0.01 else " ")
        print(f"  {marker} {k:20s}: {prev:.4f} → {v:.4f} ({delta:+.4f})", flush=True)
    print(f"  {'─'*50}", flush=True)
    print(f"    market_weight:      {MARKET_WEIGHT:.4f} → {opt_market:.4f}", flush=True)

    if opt_score > current_score:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved to {OUTPUT_PATH}", flush=True)
    else:
        print(f"\n  No improvement — weights NOT saved", flush=True)

    print(f"\n{'='*60}", flush=True)


if __name__ == "__main__":
    main()
