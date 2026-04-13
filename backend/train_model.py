"""Train dual ML models: analytical (market-free) + combined.

The analytical model predicts win probability WITHOUT knowing odds,
enabling detection of "value" horses where AI disagrees with the market.

Usage:
    cd "/Users/atsushi.furutani/Claude Code/jra-prediction-app"
    /usr/bin/python3 -m backend.train_model
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

from backend.predictor.feature_engineering import (
    ANALYTICAL_COLUMNS,
    ALL_COLUMNS,
    FEATURE_COLUMNS,
    extract_race_context,
    extract_horse_features,
    features_to_vector,
)

DATA_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "historical_races.json"
)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "predictor", "trained_model.pkl"
)


def load_historical_data() -> list:
    print(f"Loading data from {DATA_FILE}...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        races = json.load(f)
    valid = [r for r in races if r.get("results") and len(r.get("entries", [])) >= 3]
    print(f"  Loaded {len(valid)} valid races (of {len(races)} total)")
    return valid


def build_dataset(races: list) -> tuple:
    """Extract features and labels. Returns full feature vectors."""
    X_rows = []
    y_labels = []
    meta = []

    for race in races:
        race_info = race["race_info"]
        entries = race["entries"]
        results = race["results"]
        race_id = race.get("race_id", "")
        date = race.get("date", "")

        active = [e for e in entries if not e.get("isScratched")]
        if len(active) < 3:
            continue

        context = extract_race_context(race_info, entries)
        all_weights = [e.get("weightCarried", 0) for e in active]
        all_odds = [e.get("odds") for e in active]

        for entry in active:
            hn = entry["horseNumber"]
            feat_dict, _ = extract_horse_features(
                entry, race_info, context, all_weights, all_odds
            )
            # Store full feature vector
            feature_vec = features_to_vector(feat_dict, ALL_COLUMNS)

            finish = results.get(str(hn), 99)
            label = 1 if finish == 1 else 0

            X_rows.append(feature_vec)
            y_labels.append(label)
            meta.append({
                "race_id": race_id,
                "horse_number": hn,
                "date": date,
                "finish": finish,
                "odds": entry.get("odds", 30.0),
                "popularity": entry.get("popularity", 99),
            })

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_labels, dtype=np.int32)
    print(f"  Dataset: {len(y)} entries, {sum(y)} winners ({sum(y)/len(y)*100:.1f}%)")
    print(f"  All features: {X.shape[1]}, Analytical: {len(ANALYTICAL_COLUMNS)}")
    return X, y, meta


def time_based_split(X, y, meta, test_months: int = 3):
    dates = sorted(set(m["date"] for m in meta))
    unique_months = sorted(set(d[:6] for d in dates))
    cutoff_idx = max(0, len(unique_months) - test_months)
    cutoff_month = unique_months[cutoff_idx]

    print(f"  Date range: {dates[0]} - {dates[-1]}")
    print(f"  Split at: {cutoff_month}")

    train_mask = np.array([m["date"][:6] < cutoff_month for m in meta])
    test_mask = ~train_mask

    return (
        X[train_mask], y[train_mask], [m for m, t in zip(meta, train_mask) if t],
        X[test_mask], y[test_mask], [m for m, t in zip(meta, test_mask) if t],
    )


def get_column_indices(columns: list, target_columns: list) -> list:
    """Get indices of target_columns within columns."""
    return [columns.index(c) for c in target_columns]


def train_model(X, y, name: str = "model"):
    print(f"\nTraining {name} ({X.shape[1]} features)...")
    model = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        max_depth=6,
        l2_regularization=1.0,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=50,
        verbose=0,
    )
    model.fit(X, y)
    print(f"  Stopped at iteration: {model.n_iter_}")
    return model


def evaluate_racing(model, X, y, meta, label: str = ""):
    """Race-level evaluation."""
    probs = model.predict_proba(X)[:, 1]

    auc = roc_auc_score(y, probs)
    print(f"\n--- {label} (AUC={auc:.4f}) ---")

    race_groups = {}
    for i, m in enumerate(meta):
        rid = m["race_id"]
        if rid not in race_groups:
            race_groups[rid] = []
        race_groups[rid].append({
            "hn": m["horse_number"], "prob": probs[i], "finish": m["finish"],
            "odds": m.get("odds", 30), "pop": m.get("popularity", 99),
        })

    total = tansho = umaren = wide = sanrenpuku = w_top3 = 0
    # Value analysis: when AI disagrees with market
    value_bets = 0
    value_wins = 0
    pop1_wins = 0

    for rid, horses in race_groups.items():
        if len(horses) < 3:
            continue
        actual = {h["hn"]: h["finish"] for h in horses}
        actual_1st = [hn for hn, f in actual.items() if f == 1]
        if not actual_1st:
            continue
        total += 1
        winner = actual_1st[0]

        ranked = sorted(horses, key=lambda h: -h["prob"])
        ai_top1 = ranked[0]["hn"]
        ai_top2 = {ranked[0]["hn"], ranked[1]["hn"]}
        ai_top3 = {h["hn"] for h in ranked[:3]}
        actual_top2 = {hn for hn, f in actual.items() if f <= 2}
        actual_top3 = {hn for hn, f in actual.items() if f <= 3}

        if ai_top1 == winner:
            tansho += 1
        if ai_top2 == actual_top2:
            umaren += 1
        if len(ai_top3 & actual_top3) >= 2:
            wide += 1
        if ai_top3 == actual_top3:
            sanrenpuku += 1
        if winner in ai_top3:
            w_top3 += 1

        # Pop1 == winner check
        pop_sorted = sorted(horses, key=lambda h: h.get("pop") or 99)
        if pop_sorted[0]["hn"] == winner:
            pop1_wins += 1

        # Value: AI top-1 != pop top-1 (AI disagrees with market)
        if ai_top1 != pop_sorted[0]["hn"]:
            value_bets += 1
            if ai_top1 == winner:
                value_wins += 1

    if total == 0:
        return {}

    print(f"  Races: {total}")
    print(f"  単勝: {tansho}/{total} = {tansho/total*100:.1f}%")
    print(f"  馬連: {umaren}/{total} = {umaren/total*100:.1f}%")
    print(f"  ワイド: {wide}/{total} = {wide/total*100:.1f}%")
    print(f"  3連複: {sanrenpuku}/{total} = {sanrenpuku/total*100:.1f}%")
    print(f"  Winner in top-3: {w_top3}/{total} = {w_top3/total*100:.1f}%")
    print(f"  1番人気勝率: {pop1_wins}/{total} = {pop1_wins/total*100:.1f}%")
    print(f"  AI独自選択(!=1番人気): {value_bets} races")
    if value_bets > 0:
        print(f"  →そのうち的中: {value_wins}/{value_bets} = {value_wins/value_bets*100:.1f}%")

    return {
        "auc": auc, "tansho": tansho / total, "umaren": umaren / total,
        "wide": wide / total, "sanrenpuku": sanrenpuku / total,
        "total_races": total,
    }


def main():
    print("=" * 60)
    print("KEIBA ORACLE - Dual Model Training v7")
    print("=" * 60)

    races = load_historical_data()
    print("\nExtracting features...")
    X_all, y, meta = build_dataset(races)

    # Column indices
    analytical_idx = get_column_indices(ALL_COLUMNS, ANALYTICAL_COLUMNS)
    X_analytical = X_all[:, analytical_idx]

    # Split
    print("\nSplitting data...")
    X_tr, y_tr, m_tr, X_te, y_te, m_te = time_based_split(X_all, y, meta, test_months=4)
    X_tr_a = X_tr[:, analytical_idx]
    X_te_a = X_te[:, analytical_idx]

    # Train ANALYTICAL model (no market data)
    model_a = train_model(X_tr_a, y_tr, "Analytical (market-free)")
    print("\n>>> Analytical Model - Test Set:")
    evaluate_racing(model_a, X_te_a, y_te, m_te, "Analytical TEST")
    print("\n>>> Analytical Model - Train Set:")
    evaluate_racing(model_a, X_tr_a, y_tr, m_tr, "Analytical TRAIN")

    # Train COMBINED model (with market)
    model_c = train_model(X_tr, y_tr, "Combined (with market)")
    print("\n>>> Combined Model - Test Set:")
    evaluate_racing(model_c, X_te, y_te, m_te, "Combined TEST")

    # Final: retrain on ALL data
    print("\n" + "=" * 60)
    print("Final training on ALL data...")
    model_a_final = train_model(X_analytical, y, "Final Analytical")
    model_c_final = train_model(X_all, y, "Final Combined")

    print("\n>>> Final Analytical - Full Dataset:")
    evaluate_racing(model_a_final, X_analytical, y, meta, "Analytical FULL")
    print("\n>>> Final Combined - Full Dataset:")
    metrics = evaluate_racing(model_c_final, X_all, y, meta, "Combined FULL")

    # Save dual model bundle
    bundle = {
        "model_analytical": model_a_final,
        "model_combined": model_c_final,
        "analytical_columns": ANALYTICAL_COLUMNS,
        "all_columns": ALL_COLUMNS,
        "version": "v7.0",
        "test_metrics": metrics,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nDual model saved to {MODEL_PATH}")
    print(f"  File size: {os.path.getsize(MODEL_PATH) / 1024:.0f} KB")
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
