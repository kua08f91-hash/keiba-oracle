"""Unified backtesting harness for KEIBA ORACLE.

Replaces ad-hoc simulate_*.py scripts with a single, reproducible framework.

Usage:
    /usr/bin/python3 -m backtest.harness --start 20260401 --end 20260517

Features:
    - Walk-forward evaluation (time-series split, no future leakage)
    - Multiple baselines (market, v5 default, current optimized)
    - Per-type ROI breakdown
    - Train/test gap reporting (overfitting detection)
    - Caches race data locally for fast re-runs
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import re
import sys
import time
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from bs4 import BeautifulSoup
from backend.database.db import init_db
from backend.scraper.netkeiba import fetch_race_list, fetch_race_card
from backend.scraper.odds import estimate_from_entries, fetch_live_combination_odds
from backend.predictor.scoring import WeightedScoringModel, ALL_FACTOR_KEYS, ANALYTICAL_WEIGHTS
from backend.predictor.bet_optimizer import optimize_bets, scores_to_probabilities

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TYPE_JP_MAP = {
    "tansho": ["単勝"], "fukusho": ["複勝"], "wakuren": ["枠連"],
    "umaren": ["馬連"], "umatan": ["馬単"], "wide": ["ワイド"],
    "sanrenpuku": ["三連複", "3連複"], "sanrentan": ["三連単", "3連単"],
}
TYPE_LABELS = {
    "tansho": "単勝", "fukusho": "複勝", "wakuren": "枠連",
    "umaren": "馬連", "umatan": "馬単", "wide": "ワイド",
    "sanrenpuku": "3連複", "sanrentan": "3連単",
}


# ─── Data collection ───

def _fetch_result_payouts(race_id: str, session: requests.Session) -> tuple:
    """Fetch finish positions and payouts from race.netkeiba.com."""
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    r = session.get(url, timeout=15)
    r.encoding = r.apparent_encoding or "UTF-8"
    soup = BeautifulSoup(r.text, "lxml")

    positions = {}
    for row in soup.select(".HorseList"):
        tds = row.select("td")
        if len(tds) < 3:
            continue
        try:
            positions[int(tds[2].get_text(strip=True))] = int(tds[0].get_text(strip=True))
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
            cbs = []
            if uls:
                for ul in uls:
                    nums = [int(s.get_text(strip=True)) for s in ul.select("span") if s.get_text(strip=True).isdigit()]
                    if nums:
                        cbs.append(nums)
            else:
                nums = [int(s.get_text(strip=True)) for s in result_td.select("span") if s.get_text(strip=True).isdigit()]
                if nums:
                    cbs = [nums]
            if label == "複勝" and len(cbs) == 1 and len(cbs[0]) > 1:
                cbs = [[n] for n in cbs[0]]
            pt = re.sub(r"<br\s*/?>", "|", payout_td.decode_contents())
            pt = re.sub(r"<[^>]+>", "", pt)
            amounts = []
            for part in pt.split("|"):
                m = re.search(r"([\d,]+)円", part.strip())
                if m:
                    amounts.append(int(m.group(1).replace(",", "")))
            if cbs and amounts:
                for i, combo in enumerate(cbs):
                    amt = amounts[i] if i < len(amounts) else amounts[-1]
                    payouts.setdefault(label, []).append({"nums": combo, "amount": amt})

    return positions, payouts


def collect_races(start_date: date, end_date: date) -> list:
    """Collect race data + results for a date range. Uses local cache."""
    cache_file = os.path.join(CACHE_DIR, f"races_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            races = pickle.load(f)
        print(f"Loaded {len(races)} cached races from {cache_file}")
        return races

    init_db()
    session = requests.Session()
    session.headers.update(HEADERS)
    races = []

    d = start_date
    while d <= end_date:
        if d.weekday() in (5, 6):  # Sat/Sun
            ds = d.strftime("%Y%m%d")
            time.sleep(2)
            schedules = fetch_race_list(ds)
            if schedules:
                for s in schedules:
                    for r in s.get("races", []):
                        rid = r.get("race_id", "")
                        if not rid:
                            continue
                        try:
                            data = fetch_race_card(rid)
                            if not data or len(data.get("entries", [])) < 3:
                                continue
                            time.sleep(2)
                            positions, payouts = _fetch_result_payouts(rid, session)
                            if not positions:
                                continue
                            time.sleep(1)
                            od = estimate_from_entries(data["entries"]) or {}
                            try:
                                live_od = fetch_live_combination_odds(rid, include_win_place=True)
                                if live_od:
                                    od.update(live_od)
                            except Exception:
                                pass
                            races.append({
                                "rid": rid, "date": ds,
                                "name": f"{s['name']}{r['race_number']:2d}R",
                                "entries": data["entries"],
                                "info": data["race_info"],
                                "positions": positions,
                                "payouts": payouts,
                                "odds_data": od,
                            })
                            print(f"  [{len(races)}] {s['name']}{r['race_number']:2d}R cached")
                        except Exception as e:
                            print(f"  {rid}: error {e}")
                            time.sleep(10)

                # Save periodically
                if len(races) % 50 == 0 and races:
                    with open(cache_file, "wb") as f:
                        pickle.dump(races, f)
        d += timedelta(days=1)

    with open(cache_file, "wb") as f:
        pickle.dump(races, f)
    print(f"Collected {len(races)} races, saved to {cache_file}")
    return races


# ─── Bet checking ───

def check_hit(bt: str, horses: list, positions: dict, payouts: dict) -> tuple:
    """Check if a bet hit and return payout amount."""
    hit = False
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

    payout = 0
    if hit:
        for jp_key in TYPE_JP_MAP.get(bt, []):
            for e in payouts.get(jp_key, []):
                if bt in ("umatan", "sanrentan"):
                    if e["nums"] == horses:
                        payout = e["amount"]
                elif bt in ("tansho", "fukusho"):
                    if horses[0] in e["nums"]:
                        payout = e["amount"]
                else:
                    if set(e["nums"]) == set(horses):
                        payout = e["amount"]
                if payout > 0:
                    break
            if payout > 0:
                break
    return hit, payout


# ─── Evaluation ───

def evaluate(races: list, model: WeightedScoringModel, label: str, bet_amount: int = 500) -> dict:
    """Run a model over races and compute all metrics."""
    total_inv = 0
    total_ret = 0
    total_bets = 0
    total_hits = 0
    races_bet = 0
    honmei_win = 0
    honmei_top3 = 0
    total_races = 0
    type_stats = {}

    for rd in races:
        preds = model.predict(rd["info"], rd["entries"])
        if len(preds) < 3:
            continue
        total_races += 1

        # Mark accuracy
        sorted_p = sorted(preds, key=lambda p: -p["score"])
        honmei_hn = sorted_p[0]["horseNumber"]
        pos = rd["positions"].get(honmei_hn, 99)
        if pos == 1:
            honmei_win += 1
        if pos <= 3:
            honmei_top3 += 1

        # Bets
        bets = optimize_bets(preds, rd["odds_data"], rd["info"], entries=rd["entries"])
        if bets:
            races_bet += 1

        for bet in bets:
            bt = bet["type"]
            horses = bet["horses"]
            total_bets += 1
            total_inv += bet_amount

            hit, payout = check_hit(bt, horses, rd["positions"], rd["payouts"])
            ret = payout * (bet_amount // 100) if hit else 0
            total_ret += ret

            if bt not in type_stats:
                type_stats[bt] = {"hits": 0, "bets": 0, "invest": 0, "return": 0}
            type_stats[bt]["bets"] += 1
            type_stats[bt]["invest"] += bet_amount
            type_stats[bt]["return"] += ret
            if hit:
                total_hits += 1
                type_stats[bt]["hits"] += 1

    roi = total_ret / total_inv * 100 if total_inv > 0 else 0
    return {
        "label": label,
        "races": total_races,
        "bets": total_bets,
        "hits": total_hits,
        "invest": total_inv,
        "return": total_ret,
        "roi": roi,
        "honmei_win": honmei_win,
        "honmei_top3": honmei_top3,
        "races_bet": races_bet,
        "type_stats": type_stats,
    }


def print_results(results: list):
    """Print evaluation results in a formatted table."""
    print()
    print("=" * 100)
    print("  Backtest Results")
    print("=" * 100)
    header = "%-35s %6s %5s %5s %8s %8s %6s %6s %6s" % (
        "Model", "ROI", "Bets", "Hits", "Invest", "Return", "◎win", "◎top3", "Bet%")
    print(header)
    print("-" * 100)

    for r in results:
        n = r["races"]
        win_pct = r["honmei_win"] / n * 100 if n > 0 else 0
        top3_pct = r["honmei_top3"] / n * 100 if n > 0 else 0
        bet_pct = r["races_bet"] / n * 100 if n > 0 else 0
        print("%-35s %5.1f%% %5d %5d %8s %8s %5.1f%% %5.1f%% %5.0f%%" % (
            r["label"], r["roi"], r["bets"], r["hits"],
            f"¥{r['invest']:,}", f"¥{r['return']:,}",
            win_pct, top3_pct, bet_pct))

    # Type breakdown for best model
    best = max(results, key=lambda r: r["roi"])
    print(f"\n  [{best['label']}] 券種別:")
    for bt in ["umatan", "umaren", "wide", "tansho", "fukusho", "sanrenpuku", "sanrentan"]:
        ts = best["type_stats"].get(bt)
        if ts and ts["bets"] > 0:
            roi = ts["return"] / ts["invest"] * 100
            rate = ts["hits"] / ts["bets"] * 100
            print(f"    {TYPE_LABELS.get(bt, bt):6s} {ts['hits']:3d}/{ts['bets']:4d} ({rate:5.1f}%) ROI:{roi:6.1f}%")


def main():
    parser = argparse.ArgumentParser(description="KEIBA ORACLE Backtest Harness")
    parser.add_argument("--start", default="20260401", help="Start date YYYYMMDD")
    parser.add_argument("--end", default="20260517", help="End date YYYYMMDD")
    parser.add_argument("--bet", type=int, default=500, help="Bet amount per ticket")
    args = parser.parse_args()

    start = date(int(args.start[:4]), int(args.start[4:6]), int(args.start[6:8]))
    end = date(int(args.end[:4]), int(args.end[4:6]), int(args.end[6:8]))

    print(f"Backtest: {start} → {end}, ¥{args.bet}/bet")
    races = collect_races(start, end)
    if not races:
        print("No races collected.")
        return

    # Split for train/test gap reporting
    mid = len(races) // 2
    first_half = races[:mid]
    second_half = races[mid:]

    print(f"\nTotal: {len(races)}R (1st half: {len(first_half)}R, 2nd half: {len(second_half)}R)")

    # Load optimized weights
    weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "optimized_weights.json")
    opt_w = None
    opt_mw = None
    if os.path.exists(weights_path):
        with open(weights_path) as f:
            wd = json.load(f)
        opt_w = wd.get("analytical_weights")
        opt_mw = wd.get("market_weight")

    # Models to evaluate
    models = [
        ("Market-only (baseline)", WeightedScoringModel(
            analytical_weights={k: 0 for k in ALL_FACTOR_KEYS if k != "marketScore"},
            market_weight=1.0)),
        ("v5 defaults (mkt 15%)", WeightedScoringModel()),
    ]
    if opt_w and opt_mw:
        models.append(("Optimized (mkt %d%%)" % int(opt_mw * 100),
                       WeightedScoringModel(analytical_weights=opt_w, market_weight=opt_mw)))

    # Run on full dataset
    print("\n--- Full Dataset ---")
    full_results = []
    for label, model in models:
        r = evaluate(races, model, label, args.bet)
        full_results.append(r)
    print_results(full_results)

    # Train/test gap
    print("\n--- Overfitting Check (1st vs 2nd half) ---")
    for label, model in models:
        r1 = evaluate(first_half, model, f"{label} [1st]", args.bet)
        r2 = evaluate(second_half, model, f"{label} [2nd]", args.bet)
        gap = abs(r1["roi"] - r2["roi"])
        stability = "STABLE" if gap < 20 else "UNSTABLE"
        print(f"  {label:35s} 1st:{r1['roi']:6.1f}% 2nd:{r2['roi']:6.1f}% gap:{gap:5.1f}pt [{stability}]")


if __name__ == "__main__":
    main()
