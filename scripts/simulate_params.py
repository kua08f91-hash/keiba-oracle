"""Phase 2: Fast parameter tuning against cached race data.

Tests multiple deflation coefficient sets and weight adjustments.
No API calls — runs in seconds.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.predictor.bet_optimizer import (
    scores_to_probabilities, monte_carlo_finish, estimate_hit_probabilities,
    generate_candidates, find_odds_for_bet, implied_fair_odds,
    detect_race_pattern, MIN_EV_THRESHOLD, MC_SAMPLES, JRA_TAKEOUT,
    MIN_ODDS_BY_TYPE,
)

CACHE_FILE = os.path.join(os.path.dirname(__file__), "cached_races_mar_apr.json")

TYPE_MAP = {
    "tansho": "単勝", "fukusho": "複勝", "wakuren": "枠連",
    "umaren": "馬連", "umatan": "馬単", "wide": "ワイド",
    "sanrenpuku": "三連複", "sanrentan": "三連単",
}


def check_bet_hit(bet, payouts):
    label = TYPE_MAP.get(bet["type"])
    if not label or label not in payouts:
        return False, 0
    horses = bet["horses"]
    for entry in payouts[label]:
        pnums = entry["nums"]
        pamt = entry["amount"]
        if bet["type"] == "tansho" and len(horses) == 1 and horses[0] in pnums:
            return True, pamt
        elif bet["type"] == "fukusho" and len(horses) == 1 and horses[0] == pnums[0]:
            return True, pamt
        elif bet["type"] in ("umaren", "wide", "wakuren", "sanrenpuku") and set(horses) == set(pnums):
            return True, pamt
        elif bet["type"] in ("umatan", "sanrentan") and horses == pnums:
            return True, pamt
    return False, 0


def run_optimizer(preds, entries, info, odds_data, deflation, shrinkage, min_ev_thr):
    """Run optimizer with custom parameters."""
    head_count = info.get("headCount", 16)
    if head_count < 3:
        return []

    probs = scores_to_probabilities(preds, head_count)
    if len(probs) < 3:
        return []

    pattern = detect_race_pattern(probs)
    temp_mult = {"本命堅軸": 0.85, "混戦模様": 1.15, "2強対決": 0.92}.get(pattern, 1.0)
    if temp_mult != 1.0:
        probs = scores_to_probabilities(preds, head_count, temp_adjust=temp_mult)

    candidates = generate_candidates(probs, top_n=min(7, len(probs)), entries=entries)

    rng = random.Random(42)
    finishes = monte_carlo_finish(probs, MC_SAMPLES, rng=rng)
    candidates = estimate_hit_probabilities(finishes, candidates)

    # Apply deflation
    for c in candidates:
        c["hitProb"] *= deflation.get(c["type"], 1.0)

    # EV calculation
    real_count = 0
    for c in candidates:
        oi = find_odds_for_bet(c, odds_data)
        c["_oi"] = oi
        if oi:
            real_count += 1
    has_real = real_count > 5

    for c in candidates:
        odds_info = c.pop("_oi", None)
        ai_hit = c["hitProb"]
        if odds_info:
            c["odds"] = odds_info["odds"]
            c["payout"] = odds_info["payout"]
            if "oddsMin" in odds_info:
                c["oddsMin"] = odds_info["oddsMin"]
            if "oddsMax" in odds_info:
                c["oddsMax"] = odds_info["oddsMax"]
            base_ev = ai_hit * odds_info["odds"] - 1.0
            market_hit = 1.0 / odds_info["odds"] if odds_info["odds"] > 0 else ai_hit
            edge = ai_hit - market_hit
            edge_bonus = max(-0.15, min(0.30, edge * 0.20))
            c["ev"] = base_ev + edge_bonus
        else:
            est_odds = implied_fair_odds(ai_hit) * shrinkage
            c["odds"] = round(est_odds, 1)
            c["payout"] = int(est_odds * 100)
            if has_real:
                c["ev"] = ai_hit * est_odds - 1.0 - 0.10
            else:
                bonus = {"sanrentan": 0.25, "umatan": 0.10, "wide": 0.05,
                         "umaren": 0.0, "fukusho": -0.03, "sanrenpuku": -0.05, "wakuren": -0.03}.get(c["type"], 0)
                c["ev"] = ai_hit * est_odds - 1.0 + bonus

    for c in candidates:
        c.pop("_frame_map", None)

    # Diversify with anchor
    TYPE_LIMITS = {"wide": 2, "umaren": 2, "sanrentan": 1, "umatan": 1,
                   "sanrenpuku": 1, "tansho": 1, "fukusho": 1, "wakuren": 1}
    TYPE_BONUS = {"umaren": 0.20, "wide": 0.18, "sanrentan": 0.05, "umatan": 0.03,
                  "tansho": 0.02, "sanrenpuku": 0.01, "fukusho": 0.0, "wakuren": 0.0}

    max_bets = 5
    top_evs = sorted([c["ev"] for c in candidates if c.get("ev") is not None], reverse=True)
    if top_evs:
        if top_evs[0] < -0.35:
            max_bets = min(max_bets, 3)
        elif top_evs[0] < -0.20:
            max_bets = min(max_bets, 4)

    viable = [c for c in candidates
              if c["ev"] > min_ev_thr and c["hitProb"] > 0.001
              and c.get("odds", 0) >= MIN_ODDS_BY_TYPE.get(c["type"], 2.0)]

    for c in viable:
        c["_sort_ev"] = c["ev"] + TYPE_BONUS.get(c["type"], 0)
    viable.sort(key=lambda x: -x["_sort_ev"])

    selected = []
    selected_ids = set()
    type_counts = {}

    def _pick(bet):
        selected.append(bet)
        selected_ids.add(id(bet))
        type_counts[bet["type"]] = type_counts.get(bet["type"], 0) + 1

    # Anchor from ALL candidates
    ranked = sorted(probs.items(), key=lambda x: -x[1])
    top1 = ranked[0][0] if ranked else None

    if top1 is not None:
        honmei_u = [c for c in candidates if c["type"] == "umaren" and top1 in c["horses"]
                    and c["hitProb"] > 0.001 and c.get("odds", 0) >= 2.0]
        if honmei_u:
            _pick(max(honmei_u, key=lambda c: c.get("ev", -99)))
        honmei_w = [c for c in candidates if c["type"] == "wide" and top1 in c["horses"]
                    and c["hitProb"] > 0.001 and c.get("odds", 0) >= 2.5 and id(c) not in selected_ids]
        if honmei_w:
            _pick(max(honmei_w, key=lambda c: c.get("ev", -99)))

    if not any(b["type"] == "wide" for b in selected):
        for bet in viable:
            if id(bet) not in selected_ids and bet["type"] == "wide":
                _pick(bet)
                break

    for bet in viable:
        if len(selected) >= max_bets:
            break
        if id(bet) in selected_ids:
            continue
        if type_counts.get(bet["type"], 0) >= TYPE_LIMITS.get(bet["type"], 1):
            continue
        _pick(bet)

    if len(selected) < max_bets:
        for bet in viable:
            if len(selected) >= max_bets:
                break
            if id(bet) in selected_ids:
                continue
            if type_counts.get(bet["type"], 0) >= TYPE_LIMITS.get(bet["type"], 1):
                continue
            _pick(bet)

    for c in viable:
        c.pop("_sort_ev", None)
    for i, bet in enumerate(selected):
        bet["rank"] = i + 1

    return selected


def simulate(races, deflation, shrinkage, min_ev_thr):
    """Run simulation over all races with given parameters."""
    total_bet = 0
    total_ret = 0
    total_hits = 0
    total_bets = 0
    month_stats = {}
    type_stats = {}

    for rd in races:
        info = rd["info"]
        preds = rd["predictions"]
        entries = rd["entries"]
        odds_data = rd["odds_data"]
        payouts = rd["payouts"]
        month = rd["date"][:6]

        if month not in month_stats:
            month_stats[month] = {"bet": 0, "ret": 0, "hits": 0, "races": 0}

        bets = run_optimizer(preds, entries, info, odds_data, deflation, shrinkage, min_ev_thr)

        race_bet = len(bets) * 100
        race_ret = 0

        for bet in bets:
            total_bets += 1
            hit, amt = check_bet_hit(bet, payouts)
            bt = bet["type"]
            if bt not in type_stats:
                type_stats[bt] = {"hits": 0, "bets": 0, "inv": 0, "ret": 0}
            type_stats[bt]["bets"] += 1
            type_stats[bt]["inv"] += 100
            if hit:
                race_ret += amt
                total_hits += 1
                type_stats[bt]["hits"] += 1
                type_stats[bt]["ret"] += amt

        total_bet += race_bet
        total_ret += race_ret
        month_stats[month]["bet"] += race_bet
        month_stats[month]["ret"] += race_ret
        month_stats[month]["hits"] += (1 if race_ret > race_bet else 0)
        month_stats[month]["races"] += 1

    roi = total_ret / total_bet * 100 if total_bet > 0 else 0
    return {
        "roi": roi, "bet": total_bet, "ret": total_ret,
        "hits": total_hits, "bets": total_bets,
        "months": month_stats, "types": type_stats,
    }


def main():
    if not os.path.exists(CACHE_FILE):
        print(f"Cache file not found: {CACHE_FILE}")
        print("Run cache_race_data.py first.")
        return

    with open(CACHE_FILE, "r") as f:
        cached = json.load(f)

    races = list(cached.values())
    # Split: 3月=tuning, 4月=validation
    mar = [r for r in races if r["date"].startswith("202603")]
    apr = [r for r in races if r["date"].startswith("202604")]
    print(f"Loaded {len(races)} races: 3月={len(mar)}R, 4月={len(apr)}R\n")

    # ─── Parameter Grid ───

    # Deflation coefficient sets to test
    deflation_sets = {
        "v10_current": {
            "sanrenpuku": 0.15, "sanrentan": 0.19, "umatan": 0.25,
            "umaren": 0.40, "wide": 0.36, "tansho": 0.60, "fukusho": 0.70, "wakuren": 0.40,
        },
        "mild": {
            "sanrenpuku": 0.25, "sanrentan": 0.30, "umatan": 0.35,
            "umaren": 0.55, "wide": 0.50, "tansho": 0.75, "fukusho": 0.80, "wakuren": 0.50,
        },
        "aggressive": {
            "sanrenpuku": 0.10, "sanrentan": 0.12, "umatan": 0.18,
            "umaren": 0.30, "wide": 0.28, "tansho": 0.50, "fukusho": 0.60, "wakuren": 0.30,
        },
        "no_deflation": {
            "sanrenpuku": 1.0, "sanrentan": 1.0, "umatan": 1.0,
            "umaren": 1.0, "wide": 1.0, "tansho": 1.0, "fukusho": 1.0, "wakuren": 1.0,
        },
        "umaren_wide_boost": {
            "sanrenpuku": 0.10, "sanrentan": 0.12, "umatan": 0.18,
            "umaren": 0.55, "wide": 0.50, "tansho": 0.60, "fukusho": 0.70, "wakuren": 0.35,
        },
        "balanced": {
            "sanrenpuku": 0.20, "sanrentan": 0.22, "umatan": 0.30,
            "umaren": 0.50, "wide": 0.45, "tansho": 0.65, "fukusho": 0.75, "wakuren": 0.45,
        },
    }

    # Shrinkage values
    shrinkage_vals = [0.60, 0.75, 0.90, 1.0]

    # MIN_EV_THRESHOLD values
    ev_thresholds = [-0.60, -0.45, -0.30]

    print(f"Testing {len(deflation_sets)} x {len(shrinkage_vals)} x {len(ev_thresholds)} = {len(deflation_sets)*len(shrinkage_vals)*len(ev_thresholds)} combinations\n")

    header = f"{'Config':40s} | {'3月ROI':>7s} {'3月pts':>6s} | {'4月ROI':>7s} {'4月pts':>6s} | {'Total':>7s} {'Min':>5s}"
    print(header)
    print("-" * len(header))

    results = []

    for defl_name, deflation in deflation_sets.items():
        for shrinkage in shrinkage_vals:
            for ev_thr in ev_thresholds:
                label = f"{defl_name} sh={shrinkage} ev={ev_thr}"

                r_mar = simulate(mar, deflation, shrinkage, ev_thr)
                r_apr = simulate(apr, deflation, shrinkage, ev_thr)

                total_roi = (r_mar["ret"] + r_apr["ret"]) / (r_mar["bet"] + r_apr["bet"]) * 100 if (r_mar["bet"] + r_apr["bet"]) > 0 else 0
                min_roi = min(r_mar["roi"], r_apr["roi"])

                avg_bets_mar = r_mar["bets"] / len(mar) if mar else 0
                avg_bets_apr = r_apr["bets"] / len(apr) if apr else 0

                print(f"{label:40s} | {r_mar['roi']:6.1f}% {avg_bets_mar:5.1f}p | {r_apr['roi']:6.1f}% {avg_bets_apr:5.1f}p | {total_roi:6.1f}% {min_roi:5.1f}")

                results.append({
                    "label": label,
                    "defl": defl_name, "shrinkage": shrinkage, "ev_thr": ev_thr,
                    "mar_roi": r_mar["roi"], "apr_roi": r_apr["roi"],
                    "total_roi": total_roi, "min_roi": min_roi,
                    "mar_hits": r_mar["hits"], "apr_hits": r_apr["hits"],
                    "mar_bets": r_mar["bets"], "apr_bets": r_apr["bets"],
                    "types_mar": r_mar["types"], "types_apr": r_apr["types"],
                })

    # Sort by min(3月,4月) ROI to find most stable
    results.sort(key=lambda x: -x["min_roi"])

    print(f"\n{'='*80}")
    print("  Top 10 by stability (min of 3月/4月 ROI):")
    print(f"{'='*80}")
    for i, r in enumerate(results[:10]):
        print(f"  {i+1:2d}. {r['label']:40s} 3月:{r['mar_roi']:6.1f}% 4月:{r['apr_roi']:6.1f}% Total:{r['total_roi']:6.1f}% Min:{r['min_roi']:5.1f}")

    # Sort by total ROI
    results.sort(key=lambda x: -x["total_roi"])
    print(f"\n  Top 10 by total ROI:")
    for i, r in enumerate(results[:10]):
        print(f"  {i+1:2d}. {r['label']:40s} 3月:{r['mar_roi']:6.1f}% 4月:{r['apr_roi']:6.1f}% Total:{r['total_roi']:6.1f}% Min:{r['min_roi']:5.1f}")

    # Show type breakdown for top 3
    print(f"\n{'='*80}")
    print("  券種別 (Top 3 stable configs):")
    print(f"{'='*80}")
    results.sort(key=lambda x: -x["min_roi"])
    for r in results[:3]:
        print(f"\n  {r['label']}")
        for bt in ["umaren", "wide", "tansho", "fukusho", "sanrentan", "umatan", "sanrenpuku"]:
            tm = r["types_mar"].get(bt, {"hits": 0, "bets": 0, "inv": 0, "ret": 0})
            ta = r["types_apr"].get(bt, {"hits": 0, "bets": 0, "inv": 0, "ret": 0})
            total_inv = tm["inv"] + ta["inv"]
            total_ret = tm["ret"] + ta["ret"]
            total_h = tm["hits"] + ta["hits"]
            total_b = tm["bets"] + ta["bets"]
            if total_b > 0:
                roi = total_ret / total_inv * 100
                print(f"    {bt:12s} {total_h:3d}/{total_b:4d} ({total_h/total_b*100:5.1f}%) ROI:{roi:6.1f}%")


if __name__ == "__main__":
    main()
