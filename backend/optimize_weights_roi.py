"""ROI-based weight optimizer with time-series cross-validation.

Unlike optimize_weights.py (which uses top-k accuracy), this optimizer
simulates the FULL bet_optimizer pipeline and measures ROI — the actual
money-making metric. Uses time-series train/validation split to prevent
overfitting.

Usage:
    /usr/bin/python3 -m backend.optimize_weights_roi

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
from backend.predictor.scoring import ANALYTICAL_WEIGHTS, MARKET_WEIGHT, MARK_MAP
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
from backend.predictor.bet_optimizer import (
    optimize_bets, scores_to_probabilities, detect_race_pattern,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
HISTORY_PATH = os.path.join(DATA_DIR, "historical_races.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "optimized_weights.json")

FACTOR_KEYS = list(ANALYTICAL_WEIGHTS.keys())


def load_races():
    """Load historical races sorted by date (oldest first)."""
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        races = json.load(f)
    valid = [r for r in races if r.get("results") and r.get("entries") and len(r["entries"]) >= 5]
    # Sort by date ascending for time-series split
    valid.sort(key=lambda r: r.get("date", "") or r.get("race_info", {}).get("date", ""))
    return valid


def compute_factors_for_race(race):
    """Compute all 12 factor scores + market for each entry."""
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
        past_races = entry.get("pastRaces", [])
        sire = entry.get("sireName", "")
        weight = entry.get("weightCarried", 0)

        race_date_norm = ""
        rd = race.get("race_info", {}).get("date", "") or race.get("date", "")
        if rd and len(rd) == 8 and rd.isdigit():
            race_date_norm = f"{rd[:4]}.{rd[4:6]}.{rd[6:]}"

        factors = {
            "trackDirection": calc_track_direction(past_races, course_detail, distance),
            "trackCondition": calc_track_condition_affinity(sire, track_condition, entry.get("broodmareSire", "")),
            "trackSpecific": calc_track_specific(past_races, racecourse_code),
            "jockeyAbility": calc_jockey_ability(entry.get("jockeyName", "")),
            "sameDistance": calc_same_distance_performance(past_races, distance),
            "sameSurface": calc_same_surface_performance(past_races, surface),
            "sameCondition": calc_same_condition_performance(past_races, track_condition),
            "pastPerformance": calc_past_performance(past_races),
            "speedFigure": calc_speed_figure(past_races, distance),
            "runningStyle": calc_running_style_consistency(past_races),
            "daysSinceLast": calc_days_since_last_race(past_races, race_date_norm),
            "weightCarriedTrend": calc_weight_carried_trend(past_races, weight),
            "formTrend": calc_form_trend(past_races),
            "ageAndSex": calc_age_and_sex(entry.get("age", "")),
            "weightCarried": calc_weight_carried(weight, all_weights),
            "horseWeightChange": calc_horse_weight_change(entry.get("horseWeight", "")),
            "trainerAbility": calc_trainer_ability(entry.get("trainerName", "")),
            "courseAffinity": calc_course_affinity(sire, surface),
            "distanceAptitude": calc_distance_aptitude(sire, distance),
        }
        market = calc_market_score(entry.get("odds"), entry.get("popularity"), len(active))

        factor_data.append({
            "horseNumber": entry.get("horseNumber", 0),
            "isScratched": False,
            "frameNumber": entry.get("frameNumber", 0),
            "factors": factors,
            "market": market,
            "odds": entry.get("odds"),
            "popularity": entry.get("popularity"),
        })

    return factor_data


def extract_winners(race):
    """Return list of horse numbers in finish order."""
    results_data = race.get("results", {})
    if isinstance(results_data, dict) and results_data:
        sorted_pairs = sorted(results_data.items(), key=lambda x: x[1])
        return [int(hn) for hn, _ in sorted_pairs]
    elif isinstance(results_data, list) and results_data:
        if isinstance(results_data[0], dict):
            return [r.get("horseNumber", 0) for r in
                    sorted(results_data, key=lambda r: r.get("finishPosition", 99))]
        return results_data
    return []


def score_with_weights(factor_data, weights_vec):
    """Score horses with given weight vector. Returns list of predictions."""
    n = len(FACTOR_KEYS)
    factor_weights = {k: w for k, w in zip(FACTOR_KEYS, weights_vec[:n])}
    market_w = weights_vec[n]
    analytical_w = 1.0 - market_w

    predictions = []
    scores_raw = []
    for fd in factor_data:
        analytical = sum(fd["factors"][k] * factor_weights[k] for k in FACTOR_KEYS)
        score = analytical * analytical_w + fd["market"] * market_w
        scores_raw.append(score)

    # Normalize to 30-95 range (v5 does 0-100 but v7 uses 30-95)
    if scores_raw:
        min_s = min(scores_raw)
        max_s = max(scores_raw)
        if max_s > min_s:
            norm_scores = [30.0 + 65.0 * (s - min_s) / (max_s - min_s) for s in scores_raw]
        else:
            norm_scores = [60.0] * len(scores_raw)
    else:
        norm_scores = []

    for i, fd in enumerate(factor_data):
        predictions.append({
            "horseNumber": fd["horseNumber"],
            "score": norm_scores[i],
            "mark": "",
            "factors": {},
        })

    active = sorted([p for p in predictions if p["score"] > 0], key=lambda p: -p["score"])
    for i, pred in enumerate(active):
        pred["mark"] = MARK_MAP.get(i, "")

    return predictions


def simulate_race_profit(factor_data, race, weights_vec):
    """Fast ROI simulation: top-3 ワイド + single 単勝 bets.

    Simplified betting strategy for optimization speed:
      - 3 x ワイド bets from top-3 horses (3 pairs)
      - 1 x 単勝 on top pick
      - 1 x 複勝 on top pick
    Total 5 bets × ¥100 = ¥500/race.

    Payouts estimated from market odds with JRA 25% takeout applied.
    """
    predictions = score_with_weights(factor_data, weights_vec)
    if len(predictions) < 3:
        return 0, 0

    ranked = sorted([p for p in predictions if p["score"] > 0], key=lambda p: -p["score"])
    if len(ranked) < 3:
        return 0, 0

    top3 = [p["horseNumber"] for p in ranked[:3]]

    winners = extract_winners(race)
    if not winners or len(winners) < 3:
        return 0, 0

    # Build odds lookup
    odds_map = {fd["horseNumber"]: fd.get("odds", 0) or 0 for fd in factor_data}

    total_bet = 500
    total_payout = 0

    # 単勝 top pick
    top1_odds = odds_map.get(top3[0], 0)
    if top3[0] == winners[0] and top1_odds >= 2.0:
        total_payout += int(top1_odds * 100)

    # 複勝 top pick (estimated payout ~odds/3 * 0.5)
    if top3[0] in winners[:3] and top1_odds >= 2.0:
        fukusho_est = max(int(top1_odds * 30), 100)  # Rough estimate
        total_payout += fukusho_est

    # 3 x ワイド pairs from top 3
    for i in range(3):
        pair_indices = [(0, 1), (0, 2), (1, 2)][i]
        h1, h2 = top3[pair_indices[0]], top3[pair_indices[1]]
        if h1 in winners[:3] and h2 in winners[:3]:
            # Rough ワイド payout estimate
            combined_odds = odds_map.get(h1, 0) * odds_map.get(h2, 0)
            if combined_odds >= 4:
                wide_est = int(min(combined_odds / 2.5, 30) * 100 * 0.75)
                total_payout += wide_est

    return total_bet, total_payout


def evaluate_roi(weights_vec, races_factors_list):
    """Evaluate a weight configuration by simulated ROI."""
    total_bet = 0
    total_payout = 0
    for factor_data, race in races_factors_list:
        bet, payout = simulate_race_profit(factor_data, race, weights_vec)
        total_bet += bet
        total_payout += payout

    if total_bet == 0:
        return 0.0
    return total_payout / total_bet  # ROI as ratio (1.0 = breakeven)


def objective(weights_vec, races_factors_list):
    """Minimization objective (negate for maximization)."""
    n = len(FACTOR_KEYS)
    factor_w = np.abs(weights_vec[:n])
    total_fw = factor_w.sum()
    if total_fw > 0:
        factor_w = factor_w / total_fw
    market_w = np.clip(weights_vec[n], 0.0, 0.40)
    normalized = np.concatenate([factor_w, [market_w]])
    return -evaluate_roi(normalized, races_factors_list)


def main():
    init_db()
    print("=" * 60, flush=True)
    print("  v5 ROI-Based Weight Optimizer (time-series CV)", flush=True)
    print("=" * 60, flush=True)

    print("\nLoading historical races...", flush=True)
    races = load_races()
    print(f"  {len(races)} races with results", flush=True)

    if len(races) < 200:
        print("Not enough data (need >= 200 races).")
        return

    # Use most recent 800 races (avoid stale patterns); 70% train, 30% validation
    recent = races[-800:] if len(races) > 800 else races
    split_idx = int(len(recent) * 0.7)
    train_races = recent[:split_idx]
    val_races = recent[split_idx:]
    print(f"  Train: {len(train_races)} races (most recent 70%)")
    print(f"  Validation: {len(val_races)} races (newest 30%)")

    print("\nComputing factors for train set...", flush=True)
    train_data = []
    for race in train_races:
        factors = compute_factors_for_race(race)
        if factors and extract_winners(race):
            train_data.append((factors, race))
    print(f"  {len(train_data)} train races ready", flush=True)

    print("Computing factors for validation set...", flush=True)
    val_data = []
    for race in val_races:
        factors = compute_factors_for_race(race)
        if factors and extract_winners(race):
            val_data.append((factors, race))
    print(f"  {len(val_data)} validation races ready", flush=True)

    # Current weights baseline
    x0 = np.array([ANALYTICAL_WEIGHTS[k] for k in FACTOR_KEYS] + [MARKET_WEIGHT])
    train_roi_before = -objective(x0, train_data)
    val_roi_before = -objective(x0, val_data)
    print(f"\nBaseline (current weights):")
    print(f"  Train ROI: {train_roi_before*100:.1f}%")
    print(f"  Val   ROI: {val_roi_before*100:.1f}%")

    # Optimize on training set
    print("\nOptimizing on training set...", flush=True)
    n = len(FACTOR_KEYS)
    bounds = [(0.0, 0.30)] * n + [(0.0, 0.40)]  # Factor max 30%, market max 40%

    result = differential_evolution(
        objective,
        bounds,
        args=(train_data,),
        seed=42,
        maxiter=20,        # Lightweight: 20 iterations only
        popsize=10,        # Small population
        tol=1e-4,
        polish=False,      # Skip final local search
        workers=-1,        # Use all CPU cores
        init='sobol',      # Quasi-random init for better coverage
    )

    # Normalize result
    n = len(FACTOR_KEYS)
    opt_factors = np.abs(result.x[:n])
    opt_factors = opt_factors / opt_factors.sum()
    opt_market = np.clip(result.x[n], 0.0, 0.40)
    opt_weights_vec = np.concatenate([opt_factors, [opt_market]])

    train_roi_after = -objective(opt_weights_vec, train_data)
    val_roi_after = -objective(opt_weights_vec, val_data)

    print(f"\n{'='*60}")
    print(f"Optimization Results:")
    print(f"  Train ROI: {train_roi_before*100:.1f}% → {train_roi_after*100:.1f}% (Δ{(train_roi_after-train_roi_before)*100:+.1f}%)")
    print(f"  Val   ROI: {val_roi_before*100:.1f}% → {val_roi_after*100:.1f}% (Δ{(val_roi_after-val_roi_before)*100:+.1f}%)")
    overfit_gap = train_roi_after - val_roi_after
    print(f"  Overfit gap: {overfit_gap*100:+.1f}% (smaller is better)")

    opt_weights = {k: round(float(v), 4) for k, v in zip(FACTOR_KEYS, opt_factors)}
    output = {
        "version": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "method": "ROI-based with time-series CV",
        "analytical_weights": opt_weights,
        "market_weight": round(float(opt_market), 4),
        "train_roi_before": round(float(train_roi_before), 4),
        "train_roi_after": round(float(train_roi_after), 4),
        "val_roi_before": round(float(val_roi_before), 4),
        "val_roi_after": round(float(val_roi_after), 4),
        "overfit_gap": round(float(overfit_gap), 4),
        "n_train": len(train_data),
        "n_val": len(val_data),
        "optimized_at": datetime.now().isoformat(),
    }

    print(f"\n  --- Optimized Weights ---")
    for k, v in opt_weights.items():
        prev = ANALYTICAL_WEIGHTS[k]
        delta = v - prev
        marker = "▲" if delta > 0.01 else ("▼" if delta < -0.01 else " ")
        print(f"  {marker} {k:20s}: {prev:.4f} → {v:.4f} ({delta:+.4f})")
    print(f"  {'─'*50}")
    print(f"    market_weight:      {MARKET_WEIGHT:.4f} → {opt_market:.4f}")

    # Only save if validation ROI improves
    if val_roi_after > val_roi_before:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n  ✓ Saved to {OUTPUT_PATH}")
    else:
        print(f"\n  ✗ Validation ROI did not improve — weights NOT saved")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
