"""Cross-validate weight configuration across multiple time periods.

Uses historical_races.json (which has results) to evaluate ROI for each month.
Uses estimated ワイド payouts and actual-result-based hit checking.

Usage:
    /usr/bin/python3 -m backend.cross_validate
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database.db import init_db
from backend.predictor.ml_scoring import MLScoringModel
from backend.predictor.bet_optimizer import optimize_bets

HISTORY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "historical_races.json"
)


def extract_winners(race):
    """Return horse numbers in finish order (1st, 2nd, 3rd, ...)."""
    results = race.get("results", {})
    if isinstance(results, dict) and results:
        return [int(hn) for hn, _ in sorted(results.items(), key=lambda x: x[1])]
    elif isinstance(results, list):
        if results and isinstance(results[0], dict):
            return [r.get("horseNumber", 0) for r in
                    sorted(results, key=lambda r: r.get("finishPosition", 99))]
        return results
    return []


def check_bet_hit(bet, winners):
    """Check if bet hits based on actual finish order."""
    if len(winners) < 3:
        return False
    horses = bet["horses"]
    bt = bet["type"]
    if bt == "tansho":
        return horses[0] == winners[0]
    elif bt == "fukusho":
        return horses[0] in winners[:3]
    elif bt in ("umaren", "wide", "wakuren", "sanrenpuku"):
        if bt == "umaren":
            return set(horses) == set(winners[:2])
        elif bt == "wide":
            return set(horses).issubset(set(winners[:3]))
        elif bt == "wakuren":
            return False  # No frame info in historical data
        elif bt == "sanrenpuku":
            return set(horses) == set(winners[:3])
    elif bt == "umatan":
        return horses == winners[:2]
    elif bt == "sanrentan":
        return horses == winners[:3]
    return False


def estimate_payout_from_odds(bet):
    """Estimate actual payout from bet odds with takeout haircut."""
    odds = bet.get("odds", 0)
    if not odds or odds <= 0:
        return 0
    bt = bet["type"]
    # Apply realistic haircut (actual payouts tend to be lower than theoretical)
    haircut = {
        "tansho": 1.0,
        "fukusho": 0.35,    # 複勝 is ~1/3 of win odds
        "umaren": 0.70,
        "umatan": 0.70,
        "wide": 0.60,
        "sanrenpuku": 0.65,
        "sanrentan": 0.70,
        "wakuren": 0.65,
    }.get(bt, 0.70)
    return int(odds * 100 * haircut)


def simulate_month(races):
    """Simulate betting on all races in a month."""
    predictor = MLScoringModel()
    total_bet = 0
    total_payout = 0
    hit_count = 0
    bet_count = 0

    for race in races:
        race_info = race.get("race_info", {})
        entries = race.get("entries", [])
        if len(entries) < 5:
            continue

        try:
            predictions = predictor.predict(race_info, entries)
        except Exception:
            continue

        # Build estimated odds data
        odds_data = {"tansho": [], "fukusho": [], "umaren": [], "umatan": [],
                     "wide": [], "sanrenpuku": [], "sanrentan": []}
        for e in entries:
            if e.get("isScratched") or not e.get("odds"):
                continue
            odds_data["tansho"].append({
                "horses": [e["horseNumber"]],
                "odds": e["odds"],
                "payout": int(e["odds"] * 100),
            })

        try:
            bets = optimize_bets(predictions, odds_data, race_info, entries=entries)
        except Exception:
            continue

        if not bets:
            continue

        winners = extract_winners(race)
        if len(winners) < 3:
            continue

        for bet in bets:
            total_bet += 100
            bet_count += 1
            if check_bet_hit(bet, winners):
                payout = estimate_payout_from_odds(bet)
                total_payout += payout
                hit_count += 1

    return total_bet, total_payout, hit_count, bet_count


def main():
    init_db()

    print("=" * 70, flush=True)
    print("  Cross-Validation: Multiple Months", flush=True)
    print("=" * 70, flush=True)

    if not os.path.exists(HISTORY_PATH):
        print(f"No historical data at {HISTORY_PATH}")
        return

    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        all_races = json.load(f)

    # Group by month
    by_month = defaultdict(list)
    for race in all_races:
        date = race.get("date", "") or race.get("race_info", {}).get("date", "")
        if not date or len(date) < 6:
            continue
        if race.get("results") and race.get("entries"):
            ym = date[:6]
            by_month[ym].append(race)

    results = {}
    for ym in sorted(by_month.keys()):
        races = by_month[ym]
        if len(races) < 30:
            continue

        print(f"\n{ym}: {len(races)} races", flush=True)
        bet, payout, hits, total_bets = simulate_month(races)
        if bet == 0:
            print(f"  No bets placed", flush=True)
            continue
        roi = payout / bet * 100
        hit_rate = hits / total_bets * 100 if total_bets else 0
        results[ym] = {
            "races": len(races),
            "bet": bet,
            "payout": payout,
            "roi": roi,
            "hit_rate": hit_rate,
            "hits": hits,
            "total_bets": total_bets,
        }
        sign = "+" if payout >= bet else ""
        print(f"  投資: ¥{bet:,}  払戻: ¥{payout:,}  収支: {sign}¥{payout-bet:,}", flush=True)
        print(f"  ROI: {roi:.1f}%  的中率: {hit_rate:.1f}% ({hits}/{total_bets})", flush=True)

    # Summary
    print(f"\n{'='*70}")
    print(f"  Summary across months")
    print(f"{'='*70}")
    total_bet = sum(r["bet"] for r in results.values())
    total_payout = sum(r["payout"] for r in results.values())
    total_hits = sum(r["hits"] for r in results.values())
    total_bets_count = sum(r["total_bets"] for r in results.values())
    overall_roi = total_payout / total_bet * 100 if total_bet else 0

    print(f"  Total invested: ¥{total_bet:,}")
    print(f"  Total returned: ¥{total_payout:,}")
    print(f"  Overall ROI: {overall_roi:.1f}%")
    print(f"  Overall hit rate: {total_hits/total_bets_count*100:.1f}%")

    print(f"\n  Per-month ROI:")
    for ym in sorted(results.keys()):
        r = results[ym]
        stability = "✓" if 80 <= r["roi"] <= 200 else ("△" if 60 <= r["roi"] < 80 or 200 < r["roi"] <= 400 else "✗")
        print(f"  {ym}: ROI {r['roi']:6.1f}%  的中率 {r['hit_rate']:5.1f}%  {stability}")

    # Stability analysis
    monthly_rois = [r["roi"] for r in results.values()]
    if monthly_rois:
        import statistics
        mean = statistics.mean(monthly_rois)
        stdev = statistics.stdev(monthly_rois) if len(monthly_rois) >= 2 else 0
        print(f"\n  Stability (monthly ROI variance):")
        print(f"    Mean: {mean:.1f}%")
        print(f"    StdDev: {stdev:.1f}%")
        print(f"    CV (coefficient of variation): {stdev/mean*100:.1f}% (lower = more stable)")


if __name__ == "__main__":
    main()
