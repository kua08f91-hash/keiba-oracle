"""Unified train → calibrate → evaluate pipeline.

Loads cached data, trains LTR model with walk-forward split,
calibrates probabilities, and evaluates with EV selector.

Usage:
    /usr/bin/python3 -m backtest.train_and_evaluate
"""
from __future__ import annotations

import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.predictor.ltr_model import LTRModel, extract_features
from backend.predictor.calibration import (
    IsotonicCalibrator, TemperatureCalibrator, compute_ece, compute_log_loss,
)
from backend.predictor.scoring import WeightedScoringModel, ALL_FACTOR_KEYS
from backend.predictor.bet_optimizer import (
    optimize_bets, scores_to_probabilities,
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

TYPE_JP_MAP = {
    "tansho": ["単勝"], "fukusho": ["複勝"], "wakuren": ["枠連"],
    "umaren": ["馬連"], "umatan": ["馬単"], "wide": ["ワイド"],
    "sanrenpuku": ["三連複", "3連複"], "sanrentan": ["三連単", "3連単"],
}
TYPE_LABELS = {
    "tansho": "単勝", "fukusho": "複勝", "umaren": "馬連",
    "wide": "ワイド", "umatan": "馬単", "sanrenpuku": "3連複", "sanrentan": "3連単",
}


def load_all_cached() -> list:
    """Load all cached race data."""
    all_races = []
    for f in sorted(glob.glob(os.path.join(CACHE_DIR, "hist_*.pkl"))):
        with open(f, "rb") as fh:
            races = pickle.load(fh)
        all_races.extend(races)
        print(f"  {os.path.basename(f)}: {len(races)}R")

    # Also load april/may caches
    for f in ["cached_april_races.pkl", "cached_516_517_v2.pkl"]:
        path = os.path.join(CACHE_DIR, f)
        if os.path.exists(path):
            with open(path, "rb") as fh:
                races = pickle.load(fh)
            all_races.extend(races)
            print(f"  {f}: {len(races)}R")

    return all_races


def check_hit(bt, horses, positions, payouts):
    hit = False
    for jp_key in TYPE_JP_MAP.get(bt, []):
        if hit:
            break
        for e in payouts.get(jp_key, []):
            if bt in ("umatan", "sanrentan"):
                if e["nums"] == horses:
                    return True, e["amount"]
            elif bt in ("tansho", "fukusho"):
                if horses[0] in e["nums"]:
                    return True, e["amount"]
            else:
                if set(e["nums"]) == set(horses):
                    return True, e["amount"]
    # Also check position-based
    if bt == "tansho":
        hit = positions.get(horses[0], 99) == 1
    elif bt == "fukusho":
        hit = positions.get(horses[0], 99) <= 3
    elif bt == "umaren":
        hit = all(positions.get(h, 99) <= 2 for h in horses)
    elif bt == "wide":
        hit = all(positions.get(h, 99) <= 3 for h in horses)
    elif bt == "sanrenpuku":
        hit = all(positions.get(h, 99) <= 3 for h in horses)
    elif bt == "sanrentan":
        hit = len(horses) == 3 and [positions.get(h, 99) for h in horses] == [1, 2, 3]
    elif bt == "umatan":
        hit = len(horses) == 2 and positions.get(horses[0], 99) == 1 and positions.get(horses[1], 99) == 2
    return hit, 0


def evaluate_model(races, model, label, bet_amount=500):
    """Evaluate a model's predictions + S8 bets."""
    total_inv = 0
    total_ret = 0
    honmei_win = 0
    honmei_top3 = 0
    n = 0

    for rd in races:
        preds = model.predict(rd["info"], rd["entries"])
        if len(preds) < 3:
            continue
        n += 1

        sorted_p = sorted(preds, key=lambda p: -p["score"])
        pos = rd["positions"].get(sorted_p[0]["horseNumber"], 99)
        if pos == 1:
            honmei_win += 1
        if pos <= 3:
            honmei_top3 += 1

        bets = optimize_bets(preds, rd["odds_data"], rd["info"], entries=rd["entries"])
        for bet in bets:
            total_inv += bet_amount
            hit, payout = check_hit(bet["type"], bet["horses"], rd["positions"], rd["payouts"])
            if hit:
                total_ret += payout * (bet_amount // 100)

    roi = total_ret / total_inv * 100 if total_inv > 0 else 0
    print(f"  {label:35s} ROI:{roi:6.1f}% ◎win:{honmei_win}/{n}({honmei_win/n*100:.0f}%) ◎top3:{honmei_top3}/{n}({honmei_top3/n*100:.0f}%)")
    return roi


def main():
    print("Loading cached data...")
    all_races = load_all_cached()
    print(f"Total: {len(all_races)}R\n")

    if len(all_races) < 100:
        print("Not enough data. Run collect_fast.py first.")
        return

    # Sort by date
    all_races.sort(key=lambda r: r.get("date", ""))

    # Walk-forward split: 70% train, 30% test
    split = int(len(all_races) * 0.7)
    train_races = all_races[:split]
    test_races = all_races[split:]
    print(f"Train: {len(train_races)}R, Test: {len(test_races)}R\n")

    # === Baseline: v5 optimized ===
    print("=" * 60)
    print("  Model Evaluation")
    print("=" * 60)

    import json
    weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "optimized_weights.json")
    opt_w, opt_mw = None, None
    if os.path.exists(weights_path):
        with open(weights_path) as f:
            wd = json.load(f)
        opt_w = wd.get("analytical_weights")
        opt_mw = wd.get("market_weight")

    v5 = WeightedScoringModel(analytical_weights=opt_w, market_weight=opt_mw)

    print("\n--- Test Set ---")
    evaluate_model(test_races, v5, "v5 Optimized (current)")
    evaluate_model(train_races, v5, "v5 Optimized (train)")

    # === LTR Model ===
    print("\n--- LTR Training ---")
    ltr = LTRModel()

    # Split train further for early stopping
    val_split = int(len(train_races) * 0.8)
    ltr_train = train_races[:val_split]
    ltr_val = train_races[val_split:]
    print(f"LTR train: {len(ltr_train)}R, val: {len(ltr_val)}R")

    ltr.train(ltr_train, ltr_val)

    print("\n--- LTR Evaluation ---")
    evaluate_model(test_races, ltr, "LTR (test)")
    evaluate_model(train_races, ltr, "LTR (train)")

    # Feature importance
    print("\n--- Feature Importance (Top 15) ---")
    for name, imp in ltr.feature_importance(15):
        print(f"  {name:30s} {imp:.0f}")

    # === Calibration ===
    print("\n--- Calibration ---")
    # Collect predicted probs and actual outcomes from validation set
    pred_probs = []
    actuals = []
    for rd in ltr_val:
        preds = ltr.predict(rd["info"], rd["entries"])
        if len(preds) < 3:
            continue
        total_score = sum(p["score"] for p in preds)
        for p in preds:
            prob = p["score"] / total_score if total_score > 0 else 1.0 / len(preds)
            actual = 1 if rd["positions"].get(p["horseNumber"], 99) == 1 else 0
            pred_probs.append(prob)
            actuals.append(actual)

    pred_probs = np.array(pred_probs)
    actuals = np.array(actuals)

    print(f"  Samples: {len(pred_probs)}")
    print(f"  Before calibration: ECE={compute_ece(pred_probs, actuals):.4f}, LogLoss={compute_log_loss(pred_probs, actuals):.4f}")

    iso_cal = IsotonicCalibrator()
    iso_cal.fit(pred_probs, actuals)
    calibrated = iso_cal.transform(pred_probs)
    print(f"  After isotonic:     ECE={compute_ece(calibrated, actuals):.4f}, LogLoss={compute_log_loss(calibrated, actuals):.4f}")

    temp_cal = TemperatureCalibrator()
    temp_cal.fit(pred_probs, actuals)
    temp_calibrated = temp_cal.transform(pred_probs)
    print(f"  After temperature:  ECE={compute_ece(temp_calibrated, actuals):.4f}, LogLoss={compute_log_loss(temp_calibrated, actuals):.4f}")

    # Save best models
    ltr.save()
    iso_cal.save()
    print("\nModels saved to data/ltr_model.pkl and data/calibrator.pkl")


if __name__ == "__main__":
    main()
