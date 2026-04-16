"""Real-pipeline weight optimizer using the actual bet_optimizer.

This resolves the over-fitting problem seen in previous optimizers by using
the EXACT same bet_optimizer pipeline during optimization that is used in
production simulation. Key design choices:

  1. **Real bet_optimizer**: uses optimize_bets() with all 8 bet types,
     TYPE_LIMITS, MIN_ODDS_BY_TYPE, confidence gates — exactly as production.
  2. **Reduced MC samples (500)**: during optimization only. Validation and
     production use 5000. Trade-off: ~15% noise in ROI estimate, 10x speedup.
  3. **Real payout haircuts**: matches cross_validate.py's haircut factors
     per bet type (validated against real March 2026 payouts).
  4. **Train/Hold-out split**: optimize on months A, validate on held-out
     months B. Save ONLY if held-out ROI improves.
  5. **Minimax across training months**: objective = min monthly ROI (robust).
  6. **Parallelization**: differential_evolution workers=-1.

Usage:
    /usr/bin/python3 -m backend.optimize_weights_real

Output:
    data/optimized_weights.json — saved only if validated improvement
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
from backend.predictor.scoring import (
    ANALYTICAL_WEIGHTS, MARKET_WEIGHT, MARK_MAP, ALL_FACTOR_KEYS,
    WeightedScoringModel,
)
from backend.predictor.bet_optimizer import optimize_bets

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
HISTORY_PATH = os.path.join(DATA_DIR, "historical_races.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "optimized_weights.json")

FACTOR_KEYS = list(ANALYTICAL_WEIGHTS.keys())

# Realistic payout haircuts (matched against March 2026 real payouts)
PAYOUT_HAIRCUTS = {
    "tansho": 1.00,
    "fukusho": 0.35,
    "wakuren": 0.65,
    "umaren": 0.70,
    "umatan": 0.70,
    "wide": 0.60,
    "sanrenpuku": 0.65,
    "sanrentan": 0.70,
}


def load_races_by_month(max_per_month: int = None):
    """Load races grouped by month. Optionally subsample to speed up optimization."""
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

    # Deterministic subsample (always same races picked)
    if max_per_month:
        import random as _r
        rng = _r.Random(42)
        for m in list(by_month.keys()):
            if len(by_month[m]) > max_per_month:
                by_month[m] = rng.sample(by_month[m], max_per_month)
    return dict(by_month)


def extract_winners(race):
    results = race.get("results", {})
    if isinstance(results, dict) and results:
        return [int(hn) for hn, _ in sorted(results.items(), key=lambda x: x[1])]
    return []


def check_bet_hit(bet, winners):
    if len(winners) < 3:
        return False
    horses = bet["horses"]
    bt = bet["type"]
    if bt == "tansho":
        return horses[0] == winners[0]
    elif bt == "fukusho":
        return horses[0] in winners[:3]
    elif bt == "umaren":
        return set(horses) == set(winners[:2])
    elif bt == "wide":
        return set(horses).issubset(set(winners[:3]))
    elif bt == "sanrenpuku":
        return set(horses) == set(winners[:3])
    elif bt == "umatan":
        return horses == winners[:2]
    elif bt == "sanrentan":
        return horses == winners[:3]
    elif bt == "wakuren":
        return False  # No frame info in historical data
    return False


def evaluate_weights_on_month(weights_vec, races_data, mc_samples=500):
    """Evaluate weights on one month using REAL bet_optimizer pipeline.

    Returns (total_bet, total_payout, hit_count, bet_count).
    """
    # Inject weights into scoring module
    from backend.predictor import scoring
    n = len(FACTOR_KEYS)
    factor_w = np.abs(weights_vec[:n])
    total_fw = factor_w.sum()
    if total_fw > 0:
        factor_w = factor_w / total_fw
    market_w = float(np.clip(weights_vec[n], 0.0, 0.30))

    # Save originals
    orig_aw = dict(scoring.ANALYTICAL_WEIGHTS)
    orig_mw = scoring.MARKET_WEIGHT
    orig_sw = scoring.ANALYTICAL_WEIGHT

    try:
        for k, w in zip(FACTOR_KEYS, factor_w):
            scoring.ANALYTICAL_WEIGHTS[k] = float(w)
        scoring.MARKET_WEIGHT = market_w
        scoring.ANALYTICAL_WEIGHT = 1.0 - market_w

        predictor = WeightedScoringModel()

        total_bet = 0
        total_payout = 0

        for race in races_data:
            race_info = race.get("race_info", {})
            entries = race.get("entries", [])
            if len(entries) < 5:
                continue

            try:
                predictions = predictor.predict(race_info, entries)
            except Exception:
                continue

            # Build odds data (tansho from entries)
            odds_data = {"tansho": []}
            for e in entries:
                if e.get("isScratched") or not e.get("odds"):
                    continue
                odds_data["tansho"].append({
                    "horses": [e["horseNumber"]],
                    "odds": e["odds"],
                    "payout": int(e["odds"] * 100),
                })

            try:
                bets = optimize_bets(
                    predictions, odds_data, race_info,
                    entries=entries, mc_samples=mc_samples
                )
            except Exception:
                continue

            if not bets:
                continue

            winners = extract_winners(race)
            if len(winners) < 3:
                continue

            for bet in bets:
                total_bet += 100
                if check_bet_hit(bet, winners):
                    odds = bet.get("odds", 0) or 0
                    if odds > 0:
                        haircut = PAYOUT_HAIRCUTS.get(bet["type"], 0.70)
                        total_payout += int(odds * 100 * haircut)

        return total_bet, total_payout
    finally:
        # Restore originals
        for k, v in orig_aw.items():
            scoring.ANALYTICAL_WEIGHTS[k] = v
        scoring.MARKET_WEIGHT = orig_mw
        scoring.ANALYTICAL_WEIGHT = orig_sw


def robust_objective(weights_vec, train_months_data, mc_samples=500):
    """Minimax across training months with variance penalty."""
    monthly_rois = []
    for month, races in train_months_data.items():
        if len(races) < 20:
            continue
        bet, payout = evaluate_weights_on_month(weights_vec, races, mc_samples)
        if bet == 0:
            continue
        monthly_rois.append(payout / bet)

    if not monthly_rois:
        return 0.0

    min_roi = min(monthly_rois)
    mean_roi = statistics.mean(monthly_rois)
    stdev = statistics.stdev(monthly_rois) if len(monthly_rois) >= 2 else 0

    # 60% min, 30% mean, -10% stdev
    score = 0.6 * min_roi + 0.3 * mean_roi - 0.1 * stdev
    return -score


def main():
    init_db()
    print("=" * 70, flush=True)
    print("  Real-Pipeline Weight Optimizer (using actual bet_optimizer)", flush=True)
    print("=" * 70, flush=True)

    # Subsample to 80 races per month during optimization for speed
    # Full baseline/validation uses all races
    by_month_full = load_races_by_month()
    by_month = load_races_by_month(max_per_month=80)
    if not by_month:
        print("No data")
        return

    available = sorted([m for m in by_month.keys() if len(by_month[m]) >= 30])
    print(f"\nAvailable months: {available}")

    # Train/Hold-out split: oldest months for train, newest for hold-out
    if len(available) < 3:
        print("Need at least 3 months")
        return

    train_months = available[:-2]  # All but last 2
    holdout_months = available[-2:]  # Last 2 (most recent)
    print(f"\nTrain: {train_months}")
    print(f"Hold-out: {holdout_months}")

    # Optimization uses subsampled data; validation uses full data
    train_data_opt = {m: by_month[m] for m in train_months}
    train_data_full = {m: by_month_full[m] for m in train_months}
    holdout_data_full = {m: by_month_full[m] for m in holdout_months}

    print(f"\nTrain races (opt subsample): {sum(len(v) for v in train_data_opt.values())}")
    print(f"Train races (full): {sum(len(v) for v in train_data_full.values())}")
    print(f"Hold-out races (full): {sum(len(v) for v in holdout_data_full.values())}")

    # Baseline evaluation (with full MC 5000)
    print(f"\n{'─'*70}")
    print(f"Baseline (current defaults, MC=5000):", flush=True)
    x0 = np.array([ANALYTICAL_WEIGHTS[k] for k in FACTOR_KEYS] + [MARKET_WEIGHT])

    baseline_train = {}
    for m, races in train_data_full.items():
        bet, payout = evaluate_weights_on_month(x0, races, mc_samples=5000)
        roi = payout / bet if bet > 0 else 0
        baseline_train[m] = roi
        print(f"  {m}: ROI {roi*100:6.1f}%  (¥{bet:,} → ¥{payout:,})", flush=True)

    baseline_holdout = {}
    print(f"  Hold-out:")
    for m, races in holdout_data_full.items():
        bet, payout = evaluate_weights_on_month(x0, races, mc_samples=5000)
        roi = payout / bet if bet > 0 else 0
        baseline_holdout[m] = roi
        print(f"  {m}: ROI {roi*100:6.1f}%  (¥{bet:,} → ¥{payout:,})", flush=True)

    baseline_train_min = min(baseline_train.values())
    baseline_train_mean = statistics.mean(baseline_train.values())
    baseline_ho_min = min(baseline_holdout.values())
    baseline_ho_mean = statistics.mean(baseline_holdout.values())
    print(f"\n  Train   min {baseline_train_min*100:.1f}% / mean {baseline_train_mean*100:.1f}%")
    print(f"  Hold-out min {baseline_ho_min*100:.1f}% / mean {baseline_ho_mean*100:.1f}%")

    # Optimize (using subsampled train data + MC=300 for speed)
    print(f"\n{'─'*70}")
    print(f"Optimizing (80 races/month × MC=300, workers=-1)...", flush=True)
    n = len(FACTOR_KEYS)
    bounds = [(0.0, 0.25)] * n + [(0.0, 0.30)]

    result = differential_evolution(
        robust_objective,
        bounds,
        args=(train_data_opt, 300),
        seed=42,
        maxiter=10,
        popsize=8,
        tol=1e-3,
        polish=False,
        workers=-1,
        init='sobol',
    )

    # Extract optimized weights
    opt_factors = np.abs(result.x[:n])
    opt_factors = opt_factors / opt_factors.sum()
    opt_market = np.clip(result.x[n], 0.0, 0.30)
    opt_vec = np.concatenate([opt_factors, [opt_market]])

    # Re-evaluate with full MC=5000 on both train and hold-out
    print(f"\n{'─'*70}")
    print(f"Optimized weights evaluation (MC=5000):", flush=True)

    opt_train = {}
    print(f"  Train:")
    for m, races in train_data_full.items():
        bet, payout = evaluate_weights_on_month(opt_vec, races, mc_samples=5000)
        roi = payout / bet if bet > 0 else 0
        opt_train[m] = roi
        delta = (roi - baseline_train[m]) * 100
        marker = "▲" if delta > 2 else ("▼" if delta < -2 else " ")
        print(f"  {m}: ROI {roi*100:6.1f}%  (Δ{delta:+5.1f}pt) {marker}", flush=True)

    opt_holdout = {}
    print(f"  Hold-out (unseen during optimization):")
    for m, races in holdout_data_full.items():
        bet, payout = evaluate_weights_on_month(opt_vec, races, mc_samples=5000)
        roi = payout / bet if bet > 0 else 0
        opt_holdout[m] = roi
        delta = (roi - baseline_holdout[m]) * 100
        marker = "▲" if delta > 2 else ("▼" if delta < -2 else " ")
        print(f"  {m}: ROI {roi*100:6.1f}%  (Δ{delta:+5.1f}pt) {marker}", flush=True)

    opt_train_min = min(opt_train.values())
    opt_train_mean = statistics.mean(opt_train.values())
    opt_ho_min = min(opt_holdout.values())
    opt_ho_mean = statistics.mean(opt_holdout.values())

    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"{'='*70}")
    print(f"  Train:     min {baseline_train_min*100:.1f}% → {opt_train_min*100:.1f}%  "
          f"| mean {baseline_train_mean*100:.1f}% → {opt_train_mean*100:.1f}%")
    print(f"  Hold-out:  min {baseline_ho_min*100:.1f}% → {opt_ho_min*100:.1f}%  "
          f"| mean {baseline_ho_mean*100:.1f}% → {opt_ho_mean*100:.1f}%")

    # Decision criteria (latest-month-priority):
    #   - Latest hold-out month ROI improves by >= 5pt
    #   - Hold-out min ROI doesn't worsen by more than 2pt
    # Rationale: production environment prioritizes most recent data (time-series drift)
    holdout_mean_delta = (opt_ho_mean - baseline_ho_mean) * 100
    holdout_min_delta = (opt_ho_min - baseline_ho_min) * 100
    latest_month = sorted(opt_holdout.keys())[-1]
    latest_delta = (opt_holdout[latest_month] - baseline_holdout[latest_month]) * 100

    print(f"\n  Latest month ({latest_month}) Δ: {latest_delta:+.1f}pt")
    print(f"  Hold-out min  Δ: {holdout_min_delta:+.1f}pt")
    print(f"  Hold-out mean Δ: {holdout_mean_delta:+.1f}pt")

    opt_weights = {k: round(float(v), 4) for k, v in zip(FACTOR_KEYS, opt_factors)}

    # Always show weights (for inspection)
    print(f"\n  --- Optimized Weights ---")
    for k, v in sorted(opt_weights.items(), key=lambda x: -x[1]):
        prev = ANALYTICAL_WEIGHTS[k]
        delta = v - prev
        marker = "▲" if delta > 0.01 else ("▼" if delta < -0.01 else " ")
        print(f"  {marker} {k:22s}: {prev:.4f} → {v:.4f} ({delta:+.4f})")
    print(f"    market_weight:          {MARKET_WEIGHT:.4f} → {opt_market:.4f}")

    # Save decision
    should_save = latest_delta >= 5.0 and holdout_min_delta >= -2.0
    if should_save:
        output = {
            "version": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "method": "Real-pipeline minimax with latest-month-priority validation",
            "analytical_weights": opt_weights,
            "market_weight": round(float(opt_market), 4),
            "baseline_train": {m: round(r, 4) for m, r in baseline_train.items()},
            "baseline_holdout": {m: round(r, 4) for m, r in baseline_holdout.items()},
            "optimized_train": {m: round(r, 4) for m, r in opt_train.items()},
            "optimized_holdout": {m: round(r, 4) for m, r in opt_holdout.items()},
            "latest_month": latest_month,
            "latest_delta": round(latest_delta, 2),
            "holdout_min_delta": round(holdout_min_delta, 2),
            "holdout_mean_delta": round(holdout_mean_delta, 2),
            "optimized_at": datetime.now().isoformat(),
        }
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n  ✓ Latest-month criteria met. Saved to {OUTPUT_PATH}")
    else:
        print(f"\n  ✗ Criteria not met (need latest ≥ +5pt AND min ≥ -2pt). NOT saving.")


if __name__ == "__main__":
    main()
