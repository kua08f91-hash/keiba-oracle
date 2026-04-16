"""Analyze time-series drift: why 2026-04 performance differs from 2023 months.

Compares factor value distributions, bet type hit patterns, and odds
distributions between historical and recent periods to identify drift causes.

Usage:
    /usr/bin/python3 -m backend.analyze_drift
"""
from __future__ import annotations

import json
import os
import sys
import statistics
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database.db import init_db
from backend.predictor.ml_scoring import MLScoringModel
from backend.predictor.bet_optimizer import optimize_bets

HISTORY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "historical_races.json"
)


def extract_winners(race):
    results = race.get("results", {})
    if isinstance(results, dict) and results:
        return [int(hn) for hn, _ in sorted(results.items(), key=lambda x: x[1])]
    elif isinstance(results, list):
        if results and isinstance(results[0], dict):
            return [r.get("horseNumber", 0) for r in
                    sorted(results, key=lambda r: r.get("finishPosition", 99))]
        return results
    return []


def analyze_period(races, predictor, label):
    """Gather statistics for a period."""
    factor_values = defaultdict(list)
    popularity_winner = []
    odds_winner = []
    field_sizes = []
    pastraces_per_horse = []
    pos_distribution = Counter()

    # Running factors through predictor
    for race in races:
        entries = race.get("entries", [])
        if not entries:
            continue
        winners = extract_winners(race)
        if not winners:
            continue

        active = [e for e in entries if not e.get("isScratched")]
        field_sizes.append(len(active))

        # Count past race data per entry
        for e in active:
            past = e.get("pastRaces", [])
            pastraces_per_horse.append(len(past))

        # Winner's popularity and odds
        for e in active:
            hn = e.get("horseNumber")
            if hn == winners[0]:
                if e.get("popularity"):
                    popularity_winner.append(e["popularity"])
                if e.get("odds"):
                    odds_winner.append(e["odds"])
                break

        # Get predictions (factor scores)
        try:
            preds = predictor.predict(race.get("race_info", {}), entries)
            for p in preds:
                if p.get("score", 0) > 0:
                    for k, v in p.get("factors", {}).items():
                        if isinstance(v, (int, float)):
                            factor_values[k].append(v)
                    # Check where winner was ranked
                    pass
        except Exception:
            pass

    print(f"\n{'='*70}")
    print(f"  {label} ({len(races)} races)")
    print(f"{'='*70}")
    print(f"  Average field size: {statistics.mean(field_sizes):.1f}")
    print(f"  Avg pastRaces per horse: {statistics.mean(pastraces_per_horse):.1f}")

    if popularity_winner:
        print(f"  Winner popularity — mean: {statistics.mean(popularity_winner):.2f}, "
              f"median: {statistics.median(popularity_winner)}")
        from collections import Counter as C
        pop_dist = C(popularity_winner)
        print(f"  Winner from 1人気: {pop_dist.get(1,0)} ({pop_dist.get(1,0)/len(popularity_winner)*100:.1f}%)")
        print(f"  Winner from 1-3人気: {sum(pop_dist.get(p,0) for p in range(1,4))} "
              f"({sum(pop_dist.get(p,0) for p in range(1,4))/len(popularity_winner)*100:.1f}%)")

    if odds_winner:
        print(f"  Winner odds — mean: {statistics.mean(odds_winner):.2f}, "
              f"median: {statistics.median(odds_winner):.2f}")

    print(f"\n  Factor value distributions (mean ± stdev):")
    for k in sorted(factor_values.keys()):
        vals = factor_values[k]
        if len(vals) >= 10:
            mean = statistics.mean(vals)
            stdev = statistics.stdev(vals) if len(vals) >= 2 else 0
            print(f"    {k:22s}: {mean:6.2f} ± {stdev:5.2f} (n={len(vals)})")

    return {
        "n_races": len(races),
        "avg_field": statistics.mean(field_sizes) if field_sizes else 0,
        "avg_pastraces": statistics.mean(pastraces_per_horse) if pastraces_per_horse else 0,
        "pop_winner": popularity_winner,
        "odds_winner": odds_winner,
        "factors": {k: statistics.mean(v) for k, v in factor_values.items() if len(v) >= 10},
    }


def main():
    init_db()
    print("=" * 70, flush=True)
    print("  Time-Series Drift Analysis", flush=True)
    print("  Why 2026-04 (ROI 64.9%) vs 2023 (ROI 106-177%)?", flush=True)
    print("=" * 70, flush=True)

    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        all_races = json.load(f)

    # Group by year
    by_period = {"2023 (Apr-Jul)": [], "2026-04": []}
    for race in all_races:
        date = race.get("date", "") or race.get("race_info", {}).get("date", "")
        if not date or len(date) < 6:
            continue
        if not race.get("results") or not race.get("entries"):
            continue
        ym = date[:6]
        if ym in ("202304", "202305", "202306", "202307"):
            by_period["2023 (Apr-Jul)"].append(race)
        elif ym == "202604":
            by_period["2026-04"].append(race)

    predictor = MLScoringModel()
    stats = {}
    for period, races in by_period.items():
        if len(races) >= 30:
            stats[period] = analyze_period(races, predictor, period)

    # Comparative analysis
    if len(stats) == 2:
        print(f"\n{'='*70}")
        print(f"  COMPARATIVE ANALYSIS (2023 vs 2026-04)")
        print(f"{'='*70}")

        s_old = stats["2023 (Apr-Jul)"]
        s_new = stats["2026-04"]

        # Winner popularity
        pop_old = Counter(s_old["pop_winner"])
        pop_new = Counter(s_new["pop_winner"])
        if s_old["pop_winner"] and s_new["pop_winner"]:
            old_fav_rate = pop_old.get(1, 0) / len(s_old["pop_winner"]) * 100
            new_fav_rate = pop_new.get(1, 0) / len(s_new["pop_winner"]) * 100
            print(f"\n  1番人気勝率:")
            print(f"    2023: {old_fav_rate:.1f}%")
            print(f"    2026: {new_fav_rate:.1f}%")
            print(f"    Δ: {new_fav_rate - old_fav_rate:+.1f}pt")

        # Odds of winner
        if s_old["odds_winner"] and s_new["odds_winner"]:
            old_med = statistics.median(s_old["odds_winner"])
            new_med = statistics.median(s_new["odds_winner"])
            print(f"\n  Winner odds (median):")
            print(f"    2023: {old_med:.2f}x")
            print(f"    2026: {new_med:.2f}x")
            print(f"    Δ: {new_med - old_med:+.2f}x")

        # Data quality
        print(f"\n  Avg pastRaces per horse:")
        print(f"    2023: {s_old['avg_pastraces']:.1f}")
        print(f"    2026: {s_new['avg_pastraces']:.1f}")

        # Factor-by-factor comparison
        print(f"\n  Factor mean shifts (2026 - 2023):")
        common = set(s_old["factors"].keys()) & set(s_new["factors"].keys())
        shifts = []
        for k in sorted(common):
            old_v = s_old["factors"][k]
            new_v = s_new["factors"][k]
            delta = new_v - old_v
            shifts.append((abs(delta), k, old_v, new_v, delta))

        shifts.sort(reverse=True)
        for _, k, old_v, new_v, delta in shifts:
            marker = "▲" if delta > 2 else ("▼" if delta < -2 else " ")
            print(f"    {marker} {k:22s}: {old_v:5.1f} → {new_v:5.1f} ({delta:+5.2f})")


if __name__ == "__main__":
    main()
