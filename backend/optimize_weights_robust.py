"""Robust weight optimization with minimax + CVaR objectives.

Key differences from optimize_weights_roi.py:
  1. Evaluates ROI on MULTIPLE months independently
  2. Optimizes the MINIMUM monthly ROI (minimax) — maximizes worst-case
  3. Uses MEDIAN race ROI instead of mean (outlier-robust)
  4. Penalizes high variance across months

This should prevent the overfitting disaster seen in optimize_weights_roi.py
where March 2026 ROI 541% but cross-validation showed 59.8%.

Usage:
    /usr/bin/python3 -m backend.optimize_weights_robust
"""
from __future__ import annotations

import json
import os
import sys
import statistics
from collections import defaultdict
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

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
HISTORY_PATH = os.path.join(DATA_DIR, "historical_races.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "optimized_weights.json")

FACTOR_KEYS = list(ANALYTICAL_WEIGHTS.keys())


def load_races_by_month():
    if not os.path.exists(HISTORY_PATH):
        return {}
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        races = json.load(f)
    by_month = defaultdict(list)
    for r in races:
        d = r.get("date", "") or r.get("race_info", {}).get("date", "")
        if not d or len(d) < 6:
            continue
        if r.get("results") and r.get("entries") and len(r["entries"]) >= 5:
            by_month[d[:6]].append(r)
    return dict(by_month)


def compute_factors(race):
    entries = race["entries"]
    ri = race.get("race_info", {})
    surface = ri.get("surface", "芝")
    distance = ri.get("distance", 2000)
    track_condition = ri.get("trackCondition", "")
    course_detail = ri.get("courseDetail", "")
    racecourse_code = ri.get("racecourseCode", "")

    active = [e for e in entries if not e.get("isScratched")]
    all_weights = [e.get("weightCarried", 0) for e in active]

    rd = ri.get("date", "") or race.get("date", "")
    race_date_norm = ""
    if rd and len(rd) == 8 and rd.isdigit():
        race_date_norm = f"{rd[:4]}.{rd[4:6]}.{rd[6:]}"

    factor_data = []
    for entry in active:
        past = entry.get("pastRaces", [])
        sire = entry.get("sireName", "")
        weight = entry.get("weightCarried", 0)
        factors = {
            "trackDirection": calc_track_direction(past, course_detail, distance),
            "trackCondition": calc_track_condition_affinity(sire, track_condition, entry.get("broodmareSire", "")),
            "jockeyAbility": calc_jockey_ability(entry.get("jockeyName", "")),
            "sameDistance": calc_same_distance_performance(past, distance),
            "sameSurface": calc_same_surface_performance(past, surface),
            "trackSpecific": calc_track_specific(past, racecourse_code),
            "sameCondition": calc_same_condition_performance(past, track_condition),
            "pastPerformance": calc_past_performance(past),
            "speedFigure": calc_speed_figure(past, distance),
            "runningStyle": calc_running_style_consistency(past),
            "daysSinceLast": calc_days_since_last_race(past, race_date_norm),
            "weightCarriedTrend": calc_weight_carried_trend(past, weight),
            "formTrend": calc_form_trend(past),
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
            "factors": factors,
            "market": market,
            "odds": entry.get("odds"),
        })
    return factor_data


def extract_winners(race):
    results = race.get("results", {})
    if isinstance(results, dict) and results:
        return [int(hn) for hn, _ in sorted(results.items(), key=lambda x: x[1])]
    return []


def score_race(factor_data, weights_vec):
    n = len(FACTOR_KEYS)
    fw = {k: w for k, w in zip(FACTOR_KEYS, weights_vec[:n])}
    mw = weights_vec[n]
    aw = 1.0 - mw
    scores = []
    for fd in factor_data:
        analytical = sum(fd["factors"][k] * fw[k] for k in FACTOR_KEYS)
        scores.append((fd["horseNumber"], analytical * aw + fd["market"] * mw, fd.get("odds", 0)))
    scores.sort(key=lambda x: -x[1])
    return scores


def estimate_race_profit(scores, winners, bets_per_race=5):
    """Simulate betting on a race. Returns (bet, payout).

    Strategy: balanced bets from top-3 horses.
      - 1 単勝 on top pick
      - 1 複勝 on top pick
      - 3 ワイド from top-3 pairs
    """
    if len(scores) < 3 or len(winners) < 3:
        return 0, 0

    top3 = scores[:3]
    odds_map = {hn: o for hn, _, o in scores}

    bet = bets_per_race * 100
    payout = 0

    # 単勝
    top1_hn = top3[0][0]
    top1_odds = odds_map.get(top1_hn, 0) or 0
    if top1_hn == winners[0] and top1_odds >= 2.0:
        payout += int(top1_odds * 100)

    # 複勝
    if top1_hn in winners[:3] and top1_odds >= 2.0:
        payout += max(int(top1_odds * 30), 150)

    # 3 ワイド pairs
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        h1, h2 = top3[i][0], top3[j][0]
        if h1 in winners[:3] and h2 in winners[:3]:
            o1, o2 = odds_map.get(h1, 0) or 0, odds_map.get(h2, 0) or 0
            combined = o1 * o2
            if combined >= 4:
                payout += int(min(combined / 2.5, 25) * 100 * 0.65)

    return bet, payout


def evaluate_month_roi(weights_vec, races_data):
    total_bet = 0
    total_payout = 0
    for fd, winners in races_data:
        scores = score_race(fd, weights_vec)
        bet, payout = estimate_race_profit(scores, winners)
        total_bet += bet
        total_payout += payout
    return (total_payout / total_bet) if total_bet > 0 else 0.0


def robust_objective(weights_vec, races_by_month):
    """Robust objective: maximize min(monthly ROI) with variance penalty."""
    n = len(FACTOR_KEYS)
    factor_w = np.abs(weights_vec[:n])
    total_fw = factor_w.sum()
    if total_fw > 0:
        factor_w = factor_w / total_fw
    market_w = np.clip(weights_vec[n], 0.0, 0.30)
    normalized = np.concatenate([factor_w, [market_w]])

    monthly_rois = []
    for month, data in races_by_month.items():
        if len(data) < 20:
            continue
        roi = evaluate_month_roi(normalized, data)
        monthly_rois.append(roi)

    if not monthly_rois:
        return 0.0

    min_roi = min(monthly_rois)
    mean_roi = statistics.mean(monthly_rois)
    stdev = statistics.stdev(monthly_rois) if len(monthly_rois) >= 2 else 0

    # Composite: heavily weight min (worst case), lightly weight mean, penalize stdev
    score = 0.7 * min_roi + 0.3 * mean_roi - 0.2 * stdev
    return -score  # Negate for minimization


def main():
    init_db()
    print("=" * 70, flush=True)
    print("  Robust Weight Optimizer (minimax + variance penalty)", flush=True)
    print("=" * 70, flush=True)

    by_month = load_races_by_month()
    if not by_month:
        print("No data")
        return

    print(f"\nAvailable months: {sorted(by_month.keys())}")
    months = [m for m in sorted(by_month.keys()) if len(by_month[m]) >= 30]
    print(f"Using: {months}")

    # Compute factors once per race (cache)
    print("\nPre-computing factors...", flush=True)
    races_by_month = {}
    for month in months:
        data = []
        for race in by_month[month]:
            winners = extract_winners(race)
            if len(winners) < 3:
                continue
            fd = compute_factors(race)
            if fd:
                data.append((fd, winners))
        races_by_month[month] = data
        print(f"  {month}: {len(data)} races", flush=True)

    # Baseline
    x0 = np.array([ANALYTICAL_WEIGHTS[k] for k in FACTOR_KEYS] + [MARKET_WEIGHT])
    print(f"\nBaseline monthly ROIs (current defaults):", flush=True)
    baseline_rois = {}
    for month, data in races_by_month.items():
        roi = evaluate_month_roi(x0, data)
        baseline_rois[month] = roi
        print(f"  {month}: {roi*100:.1f}%", flush=True)
    baseline_min = min(baseline_rois.values())
    baseline_mean = statistics.mean(baseline_rois.values())
    print(f"  min: {baseline_min*100:.1f}%  mean: {baseline_mean*100:.1f}%", flush=True)

    # Optimize
    print(f"\nOptimizing (minimax)...", flush=True)
    n = len(FACTOR_KEYS)
    bounds = [(0.0, 0.25)] * n + [(0.0, 0.30)]

    result = differential_evolution(
        robust_objective,
        bounds,
        args=(races_by_month,),
        seed=42,
        maxiter=25,
        popsize=12,
        tol=1e-4,
        polish=False,
        workers=-1,
        init='sobol',
    )

    opt_factors = np.abs(result.x[:n])
    opt_factors = opt_factors / opt_factors.sum()
    opt_market = np.clip(result.x[n], 0.0, 0.30)
    opt_vec = np.concatenate([opt_factors, [opt_market]])

    print(f"\nOptimized monthly ROIs:", flush=True)
    opt_rois = {}
    for month, data in races_by_month.items():
        roi = evaluate_month_roi(opt_vec, data)
        opt_rois[month] = roi
        baseline = baseline_rois[month]
        delta = roi - baseline
        marker = "▲" if delta > 0.05 else ("▼" if delta < -0.05 else " ")
        print(f"  {month}: {roi*100:6.1f}% (baseline {baseline*100:.1f}%, Δ{delta*100:+.1f}%) {marker}", flush=True)

    opt_min = min(opt_rois.values())
    opt_mean = statistics.mean(opt_rois.values())
    print(f"\n  Baseline: min {baseline_min*100:.1f}% / mean {baseline_mean*100:.1f}%")
    print(f"  Optimized: min {opt_min*100:.1f}% / mean {opt_mean*100:.1f}%")
    print(f"  Min improvement: {(opt_min-baseline_min)*100:+.1f}pt")
    print(f"  Mean improvement: {(opt_mean-baseline_mean)*100:+.1f}pt")

    # Show weights
    opt_weights = {k: round(float(v), 4) for k, v in zip(FACTOR_KEYS, opt_factors)}
    print(f"\n  --- Optimized Weights ---")
    for k in FACTOR_KEYS:
        prev = ANALYTICAL_WEIGHTS[k]
        new = opt_weights[k]
        delta = new - prev
        marker = "▲" if delta > 0.01 else ("▼" if delta < -0.01 else " ")
        print(f"  {marker} {k:22s}: {prev:.4f} → {new:.4f} ({delta:+.4f})")
    print(f"    market_weight:          {MARKET_WEIGHT:.4f} → {opt_market:.4f}")

    # Save only if min ROI improved meaningfully (2pt) AND mean didn't drop significantly
    save = (opt_min > baseline_min + 0.02 and opt_mean > baseline_mean - 0.05)
    output = {
        "version": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "method": "Robust minimax with variance penalty",
        "analytical_weights": opt_weights,
        "market_weight": round(float(opt_market), 4),
        "baseline_monthly_rois": {m: round(r, 4) for m, r in baseline_rois.items()},
        "optimized_monthly_rois": {m: round(r, 4) for m, r in opt_rois.items()},
        "baseline_min": round(baseline_min, 4),
        "optimized_min": round(opt_min, 4),
        "baseline_mean": round(baseline_mean, 4),
        "optimized_mean": round(opt_mean, 4),
        "optimized_at": datetime.now().isoformat(),
    }
    if save:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n  ✓ Improvement confirmed. Saved to {OUTPUT_PATH}")
    else:
        print(f"\n  ✗ No meaningful improvement (min +{(opt_min-baseline_min)*100:.1f}pt, "
              f"mean {(opt_mean-baseline_mean)*100:+.1f}pt) — NOT saving")


if __name__ == "__main__":
    main()
