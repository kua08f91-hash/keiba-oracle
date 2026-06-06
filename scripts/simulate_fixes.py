"""Simulate fix combinations on 5/3 data with train/val split to avoid overfitting.

Split: first 18 races = Set A, last 18 races = Set B.
Each fix is toggled on/off, all 2^5=32 combinations tested.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from itertools import combinations, permutations, product

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database.db import init_db
from backend.scraper.odds import estimate_from_entries, fetch_live_combination_odds
from backend.predictor.bet_optimizer import (
    scores_to_probabilities, monte_carlo_finish, estimate_hit_probabilities,
    generate_candidates, find_odds_for_bet, implied_fair_odds,
    detect_race_pattern, MIN_EV_THRESHOLD, MC_SAMPLES, JRA_TAKEOUT,
    MIN_ODDS_BY_TYPE,
)

API_H = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# ─── Fix definitions ───
# A: MC hitProb deflation by bet type
HITPROB_DEFLATION = {
    "sanrenpuku": 0.15,
    "sanrentan": 0.19,
    "umatan": 0.25,
    "umaren": 0.40,
    "wide": 0.36,
    "tansho": 0.60,
    "fukusho": 0.70,
    "wakuren": 0.40,
}

# B: Dead factor weight redistribution (handled by score recalculation - skip for now,
#    just flag as "would need scoring change")

# C: Force umaren anchor (bypass viable filter for ◎ umaren)

# D: Apply ESTIMATED_ODDS_SHRINKAGE = 0.75

# E: Cap base_ev at 2.0


def fetch_result_and_payouts(race_id):
    """Fetch finish positions and payouts for a completed race."""
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    r = requests.get(url, headers=API_H, timeout=10)
    r.encoding = r.apparent_encoding or "UTF-8"
    soup = BeautifulSoup(r.text, "lxml")

    positions = {}
    for row in soup.select(".HorseList"):
        tds = row.select("td")
        if len(tds) < 3:
            continue
        try:
            pos = int(tds[0].get_text(strip=True))
            hn = int(tds[2].get_text(strip=True))
            positions[hn] = pos
        except (ValueError, IndexError):
            pass

    payouts = {}
    pay_back = soup.select_one(".Result_Pay_Back")
    if pay_back:
        for tr in pay_back.select("tr"):
            th = tr.select_one("th")
            if not th:
                continue
            label = th.get_text(strip=True)
            result_td = tr.select_one("td.Result")
            payout_td = tr.select_one("td.Payout")
            if not result_td or not payout_td:
                continue
            uls = result_td.select("ul")
            combos = []
            if uls:
                for ul in uls:
                    nums = [int(s.get_text(strip=True)) for s in ul.select("span") if s.get_text(strip=True).isdigit()]
                    if nums:
                        combos.append(nums)
            else:
                nums = [int(s.get_text(strip=True)) for s in result_td.select("span") if s.get_text(strip=True).isdigit()]
                if nums:
                    combos = [nums]
            pt = payout_td.decode_contents().replace("<br/>", "|").replace("<br>", "|")
            pt = re.sub(r"<[^>]+>", "", pt)
            amounts = []
            for part in pt.split("|"):
                m = re.search(r"([\d,]+)円", part)
                if m:
                    amounts.append(int(m.group(1).replace(",", "")))
            if combos and amounts:
                for i, combo in enumerate(combos):
                    amt = amounts[i] if i < len(amounts) else amounts[-1]
                    payouts.setdefault(label, []).append({"combo": combo, "payout": amt})

    return positions, payouts


TYPE_JP = {
    "tansho": "単勝", "fukusho": "複勝", "wakuren": "枠連",
    "umaren": "馬連", "umatan": "馬単", "wide": "ワイド",
    "sanrenpuku": "三連複", "sanrentan": "三連単",
}


def check_hit(bt, horses, positions, entries, payouts):
    """Check if a bet hit and return payout."""
    hit = False
    payout = 0
    jp = TYPE_JP.get(bt, bt)

    if bt == "tansho":
        hit = positions.get(horses[0], 99) == 1
    elif bt == "fukusho":
        hit = positions.get(horses[0], 99) <= 3
    elif bt == "umaren":
        hit = all(positions.get(h, 99) <= 2 for h in horses)
    elif bt == "umatan":
        hit = len(horses) == 2 and positions.get(horses[0], 99) == 1 and positions.get(horses[1], 99) == 2
    elif bt == "wide":
        hit = all(positions.get(h, 99) <= 3 for h in horses)
    elif bt == "sanrenpuku":
        hit = all(positions.get(h, 99) <= 3 for h in horses)
    elif bt == "sanrentan":
        hit = len(horses) == 3 and [positions.get(h, 99) for h in horses] == [1, 2, 3]
    elif bt == "wakuren":
        f_map = {e["horseNumber"]: e.get("frameNumber", 0) for e in entries if not e.get("isScratched")}
        top2 = [h for h, p in sorted(positions.items(), key=lambda x: x[1])[:2]]
        frames = [f_map.get(h, 0) for h in top2]
        hit = set(horses) == set(frames) and all(f > 0 for f in frames)

    if hit:
        for e in payouts.get(jp, []):
            if bt in ("umatan", "sanrentan"):
                if e["combo"] == horses:
                    payout = e["payout"]
            else:
                if set(e["combo"]) == set(horses):
                    payout = e["payout"]

    return hit, payout


def run_optimizer_with_fixes(
    preds, entries, info, odds_data, probs,
    fix_a=False, fix_c=False, fix_d=False, fix_e=False,
):
    """Run bet selection with configurable fixes."""
    head_count = info.get("headCount", 16)
    if head_count < 3:
        return []

    pattern = detect_race_pattern(probs)
    temp_multiplier = {
        "本命堅軸": 0.85, "混戦模様": 1.15, "2強対決": 0.92,
        "標準配置": 1.0, "少頭数": 1.0,
    }.get(pattern, 1.0)

    if temp_multiplier != 1.0:
        probs = scores_to_probabilities(preds, head_count, temp_adjust=temp_multiplier)

    candidates = generate_candidates(probs, top_n=min(7, len(probs)), entries=entries)

    rng = random.Random(42)
    finishes = monte_carlo_finish(probs, MC_SAMPLES, rng=rng)
    candidates = estimate_hit_probabilities(finishes, candidates)

    # Fix A: deflate hitProb by bet type
    if fix_a:
        for c in candidates:
            factor = HITPROB_DEFLATION.get(c["type"], 1.0)
            c["hitProb"] *= factor

    # Calculate EV
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
            c["hasRealOdds"] = True
            if "oddsMin" in odds_info:
                c["oddsMin"] = odds_info["oddsMin"]
            if "oddsMax" in odds_info:
                c["oddsMax"] = odds_info["oddsMax"]
            base_ev = ai_hit * odds_info["odds"] - 1.0
            # Fix E: cap base_ev
            if fix_e:
                base_ev = min(base_ev, 2.0)
            market_implied_hit = 1.0 / odds_info["odds"] if odds_info["odds"] > 0 else ai_hit
            edge = ai_hit - market_implied_hit
            edge_bonus = max(-0.15, min(0.30, edge * 0.20))
            c["ev"] = base_ev + edge_bonus
        else:
            est_odds = implied_fair_odds(ai_hit)
            # Fix D: apply shrinkage to estimated odds
            if fix_d:
                est_odds *= 0.75
            c["odds"] = round(est_odds, 1)
            c["payout"] = int(est_odds * 100)
            c["hasRealOdds"] = False
            if has_real:
                c["ev"] = ai_hit * est_odds - 1.0 - 0.10
            else:
                type_roi_bonus = {
                    "sanrentan": 0.25, "umatan": 0.10, "wide": 0.05,
                    "tansho": 0.0, "fukusho": -0.03, "umaren": 0.0,
                    "sanrenpuku": -0.05, "wakuren": -0.03,
                }
                bonus = type_roi_bonus.get(c["type"], 0)
                c["ev"] = ai_hit * est_odds - 1.0 + bonus
            # Fix E on estimated too
            if fix_e:
                c["ev"] = min(c["ev"], 2.0)

    for c in candidates:
        c.pop("_frame_map", None)

    # Selection with configurable anchor
    TYPE_LIMITS = {
        "wide": 2, "umaren": 2, "sanrentan": 1, "umatan": 1,
        "sanrenpuku": 1, "tansho": 1, "fukusho": 1, "wakuren": 1,
    }
    TYPE_BONUS = {
        "umaren": 0.20, "wide": 0.18, "sanrentan": 0.05, "umatan": 0.03,
        "tansho": 0.02, "sanrenpuku": 0.01, "fukusho": 0.0, "wakuren": 0.0,
    }

    max_bets = 5
    top_evs = sorted([c["ev"] for c in candidates if c.get("ev") is not None], reverse=True)
    if top_evs:
        best_ev = top_evs[0]
        if best_ev < -0.35:
            max_bets = min(max_bets, 3)
        elif best_ev < -0.20:
            max_bets = min(max_bets, 4)

    # Filter viable
    viable = []
    for c in candidates:
        min_odds = MIN_ODDS_BY_TYPE.get(c["type"], 2.0)
        if c["ev"] > MIN_EV_THRESHOLD and c["hitProb"] > 0.001 and c.get("odds", 0) >= min_odds:
            viable.append(c)

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

    # Identify top horse
    ranked = sorted(probs.items(), key=lambda x: -x[1])
    top1 = ranked[0][0] if ranked else None

    # Fix C: force umaren anchor from ALL candidates (not just viable)
    if fix_c and top1 is not None:
        honmei_umarens = [c for c in candidates if c["type"] == "umaren" and top1 in c["horses"]
                          and c["hitProb"] > 0.001 and c.get("odds", 0) >= 2.0]
        if honmei_umarens:
            best_u = max(honmei_umarens, key=lambda c: c.get("ev", -99))
            _pick(best_u)
        # Also force wide anchor
        honmei_wides = [c for c in candidates if c["type"] == "wide" and top1 in c["horses"]
                        and c["hitProb"] > 0.001 and c.get("odds", 0) >= 2.5]
        if honmei_wides:
            best_w = max(honmei_wides, key=lambda c: c.get("ev", -99))
            if id(best_w) not in selected_ids:
                _pick(best_w)
    else:
        # Default v9 anchor from viable only
        if top1 is not None:
            honmei_u = [c for c in viable if c["type"] == "umaren" and top1 in c["horses"]]
            if honmei_u:
                _pick(max(honmei_u, key=lambda c: c.get("ev", -99)))
            honmei_w = [c for c in viable if c["type"] == "wide" and top1 in c["horses"] and id(c) not in selected_ids]
            if honmei_w:
                _pick(max(honmei_w, key=lambda c: c.get("ev", -99)))
        if not any(b["type"] == "wide" for b in selected):
            for bet in viable:
                if id(bet) not in selected_ids and bet["type"] == "wide":
                    _pick(bet)
                    break

    # Phase 2: fill
    for bet in viable:
        if len(selected) >= max_bets:
            break
        if id(bet) in selected_ids:
            continue
        t = bet["type"]
        if type_counts.get(t, 0) >= TYPE_LIMITS.get(t, 1):
            continue
        _pick(bet)

    # Overflow
    if len(selected) < max_bets:
        for bet in viable:
            if len(selected) >= max_bets:
                break
            if id(bet) in selected_ids:
                continue
            t = bet["type"]
            if type_counts.get(t, 0) >= TYPE_LIMITS.get(t, 1):
                continue
            _pick(bet)

    for c in viable:
        c.pop("_sort_ev", None)
    for i, bet in enumerate(selected):
        bet["rank"] = i + 1

    return selected


def main():
    init_db()

    with open("docs/data/predictions.json", "r") as f:
        data = json.load(f)

    # Collect all 5/3 races
    races = []
    for day in data.get("predictions", []):
        if day["date"] != "20260503":
            continue
        for course in day["courses"]:
            for race in course["races"]:
                races.append((course["name"], race))

    if not races:
        print("No 5/3 data found")
        return

    print(f"Loading results for {len(races)} races...")

    # Pre-fetch results and odds for all races
    race_data = []
    for i, (cname, race) in enumerate(races):
        rid = race["raceId"]
        entries = race["entries"]
        info = race["raceInfo"]
        preds = race["predictions"]

        time.sleep(2)
        positions, payouts = fetch_result_and_payouts(rid)

        time.sleep(1)
        od = estimate_from_entries(entries) or {}
        live_od = fetch_live_combination_odds(rid, include_win_place=True)
        if live_od:
            od.update(live_od)

        probs = scores_to_probabilities(preds, info.get("headCount", 16))

        race_data.append({
            "name": f"{cname}{race['raceNumber']:2d}R",
            "rid": rid,
            "entries": entries,
            "info": info,
            "preds": preds,
            "probs": probs,
            "odds_data": od,
            "positions": positions,
            "payouts": payouts,
        })
        print(f"  [{i+1}/{len(races)}] {cname}{race['raceNumber']:2d}R loaded")

    # Split: Set A (first 18), Set B (last 18)
    set_a = race_data[:18]
    set_b = race_data[18:]

    # All fix combinations (A, C, D, E) — B skipped (requires scoring change)
    fixes = ["A:hitProb deflation", "C:force ◎anchor", "D:odds shrinkage", "E:EV cap"]
    fix_keys = ["a", "c", "d", "e"]

    print(f"\n{'='*80}")
    print(f"  Fix Combination Simulation (5/3, {len(races)}R)")
    print(f"  Set A (tune): {len(set_a)}R | Set B (validate): {len(set_b)}R")
    print(f"{'='*80}\n")

    header = f"{'Fixes':30s} | {'SetA ROI':>8s} {'A hits':>6s} {'A ret':>8s} | {'SetB ROI':>8s} {'B hits':>6s} {'B ret':>8s} | {'Total ROI':>9s}"
    print(header)
    print("-" * len(header))

    results = []

    for combo in product([False, True], repeat=4):
        fix_a, fix_c, fix_d, fix_e = combo
        label_parts = []
        if fix_a: label_parts.append("A")
        if fix_c: label_parts.append("C")
        if fix_d: label_parts.append("D")
        if fix_e: label_parts.append("E")
        label = "+".join(label_parts) if label_parts else "baseline(v9)"

        set_results = {}
        for sname, sdata in [("A", set_a), ("B", set_b)]:
            invest = 0
            ret = 0
            hits = 0
            total_bets = 0

            for rd in sdata:
                bets = run_optimizer_with_fixes(
                    rd["preds"], rd["entries"], rd["info"], rd["odds_data"], rd["probs"],
                    fix_a=fix_a, fix_c=fix_c, fix_d=fix_d, fix_e=fix_e,
                )
                for bet in bets:
                    total_bets += 1
                    invest += 500
                    hit, payout = check_hit(
                        bet["type"], bet["horses"],
                        rd["positions"], rd["entries"], rd["payouts"],
                    )
                    if hit:
                        hits += 1
                        ret += payout * 5

            roi = ret / invest * 100 if invest > 0 else 0
            set_results[sname] = {"invest": invest, "return": ret, "hits": hits, "bets": total_bets, "roi": roi}

        total_inv = set_results["A"]["invest"] + set_results["B"]["invest"]
        total_ret = set_results["A"]["return"] + set_results["B"]["return"]
        total_roi = total_ret / total_inv * 100 if total_inv > 0 else 0

        a = set_results["A"]
        b = set_results["B"]
        print(f"{label:30s} | {a['roi']:7.1f}% {a['hits']:2d}/{a['bets']:3d} {a['return']:7,}円 | {b['roi']:7.1f}% {b['hits']:2d}/{b['bets']:3d} {b['return']:7,}円 | {total_roi:8.1f}%")

        results.append({
            "label": label,
            "fixes": {"A": fix_a, "C": fix_c, "D": fix_d, "E": fix_e},
            "set_a": set_results["A"],
            "set_b": set_results["B"],
            "total_roi": total_roi,
        })

    # Sort by total ROI
    results.sort(key=lambda x: -x["total_roi"])

    print(f"\n{'='*80}")
    print("  Top 5 combinations (by total ROI):")
    print(f"{'='*80}")
    for i, r in enumerate(results[:5]):
        a, b = r["set_a"], r["set_b"]
        stability = "stable" if min(a["roi"], b["roi"]) > 50 else "UNSTABLE"
        print(f"  {i+1}. {r['label']:30s} Total:{r['total_roi']:6.1f}%  A:{a['roi']:6.1f}%  B:{b['roi']:6.1f}%  [{stability}]")

    print(f"\n  Bottom 3:")
    for r in results[-3:]:
        a, b = r["set_a"], r["set_b"]
        print(f"     {r['label']:30s} Total:{r['total_roi']:6.1f}%  A:{a['roi']:6.1f}%  B:{b['roi']:6.1f}%")


if __name__ == "__main__":
    main()
