"""Simulate all March 2026 races with v7 dual model.

Scans all JRA race dates in March 2026, runs AI predictions on each race,
compares against actual payouts, and computes comprehensive ROI metrics.

Usage:
    cd "/Users/atsushi.furutani/Claude Code/jra-prediction-app"
    /usr/bin/python3 -m backend.simulate_march
"""
from __future__ import annotations

import re
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from bs4 import BeautifulSoup
from backend.predictor.bet_optimizer import optimize_bets, detect_race_pattern, scores_to_probabilities

BACKEND = "http://localhost:8000"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
COURSE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}


def get_march_race_dates():
    """Get all JRA race dates in March 2026."""
    dates = []
    # Scan all weekends + holidays in March
    from datetime import date, timedelta
    d = date(2026, 3, 1)
    end = date(2026, 3, 31)
    while d <= end:
        if d.weekday() in (5, 6, 0, 4):  # Sat, Sun, Mon, Fri
            dates.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return dates


def fetch_race_ids(date_str):
    """Fetch all race IDs for a given date from netkeiba."""
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = resp.apparent_encoding or "UTF-8"
        all_ids = re.findall(r"race_id=(\d+)", resp.text)
        race_ids = []
        for rid in all_ids:
            if len(rid) >= 12 and rid[4:6] in COURSE_MAP and rid not in race_ids:
                race_ids.append(rid)
        return race_ids
    except Exception as e:
        print(f"  [WARN] fetch_race_ids({date_str}): {e}")
        return []


def fetch_payouts(race_id):
    """Fetch payout data from db.netkeiba.com."""
    url = f"https://db.netkeiba.com/race/{race_id}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = resp.apparent_encoding or "UTF-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        payouts = {}
        for table in soup.select("table.pay_table_01"):
            for row in table.select("tr"):
                th = row.select_one("th")
                tds = row.select("td")
                if not th or len(tds) < 2:
                    continue
                label = th.get_text(strip=True)
                combos = tds[0].get_text("|", strip=True).split("|")
                amounts = tds[1].get_text("|", strip=True).split("|")
                entries = []
                for combo, amount in zip(combos, amounts):
                    try:
                        amt = int(amount.replace(",", ""))
                    except ValueError:
                        continue
                    nums = [int(n) for n in re.findall(r"\d+", combo)]
                    if nums:
                        entries.append({"nums": nums, "amount": amt, "ordered": "\u2192" in combo or "→" in combo})
                if entries:
                    payouts[label] = entries
        return payouts
    except Exception as e:
        return {}


def check_bet_hit(bet, payouts):
    """Check if a bet hits and return payout amount per 100 yen."""
    type_map = {
        "tansho": "単勝", "fukusho": "複勝", "umaren": "馬連",
        "wide": "ワイド", "sanrenpuku": "三連複", "sanrentan": "三連単",
    }
    label = type_map.get(bet["type"])
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
        elif bet["type"] in ("umaren", "wide", "sanrenpuku") and set(horses) == set(pnums):
            return True, pamt
        elif bet["type"] == "sanrentan" and horses == pnums:
            return True, pamt
    return False, 0


def main():
    print("=" * 70)
    print("  KEIBA ORACLE v7 - 2026年3月 全レースシミュレーション")
    print("  各レース AI Top5買い目 x 100円 = 500円/レース")
    print("=" * 70)

    # Collect all race dates
    march_dates = get_march_race_dates()
    print(f"\nScanning {len(march_dates)} candidate dates in March 2026...")

    all_race_ids = {}  # date -> [race_ids]
    for ds in march_dates:
        time.sleep(1)
        ids = fetch_race_ids(ds)
        if ids:
            all_race_ids[ds] = ids
            print(f"  {ds}: {len(ids)} races")

    total_races_count = sum(len(v) for v in all_race_ids.values())
    print(f"\nTotal: {total_races_count} races across {len(all_race_ids)} days")

    # Process each race
    all_results = []
    type_stats = {}  # type -> {bets, hits, invested, returned}

    for ds in sorted(all_race_ids.keys()):
        race_ids = all_race_ids[ds]
        day_label = f"{int(ds[4:6])}/{int(ds[6:8])}"
        print(f"\n{'─'*70}")
        print(f"  {day_label} ({len(race_ids)} races)")
        print(f"{'─'*70}")

        for race_id in race_ids:
            course = COURSE_MAP.get(race_id[4:6], "??")
            rnum = int(race_id[10:12])

            # Get AI predictions
            time.sleep(0.5)
            try:
                r = requests.get(f"{BACKEND}/api/racecard/{race_id}", timeout=30)
                if r.status_code != 200:
                    continue
                data = r.json()
            except Exception:
                continue

            predictions = data.get("predictions", [])
            entries = data.get("entries", [])
            race_info = data.get("raceInfo", {})
            if len(predictions) < 3:
                continue

            # Get odds for optimizer
            from backend.scraper.odds import estimate_from_entries
            odds_data = estimate_from_entries(entries) or {}
            try:
                from backend.scraper.odds import fetch_combination_odds
                real = fetch_combination_odds(race_id)
                if real:
                    for k, el in real.items():
                        if k in odds_data:
                            rhs = [frozenset(e["horses"]) for e in el]
                            odds_data[k] = el + [e for e in odds_data[k] if frozenset(e["horses"]) not in rhs]
                        else:
                            odds_data[k] = el
            except Exception:
                pass

            # Run optimizer
            try:
                optimized = optimize_bets(predictions, odds_data, race_info)
            except Exception:
                continue
            if not optimized:
                continue

            # Get payouts
            time.sleep(0.5)
            payouts = fetch_payouts(race_id)
            if not payouts:
                continue

            # Check each bet
            race_bet = len(optimized) * 100
            race_payout = 0
            race_hits = []

            for bet in optimized:
                hit, amount = check_bet_hit(bet, payouts)
                bt = bet["type"]
                if bt not in type_stats:
                    type_stats[bt] = {"bets": 0, "hits": 0, "invested": 0, "returned": 0, "label": bet.get("typeLabel", bt)}
                type_stats[bt]["bets"] += 1
                type_stats[bt]["invested"] += 100
                if hit:
                    race_payout += amount
                    race_hits.append(bet.get("typeLabel", bt))
                    type_stats[bt]["hits"] += 1
                    type_stats[bt]["returned"] += amount

            profit = race_payout - race_bet
            probs = scores_to_probabilities(predictions, race_info.get("headCount", 16))
            pattern = detect_race_pattern(probs)

            mark = "+" if profit > 0 else (" " if profit == 0 else "-")
            hit_str = ",".join(race_hits) if race_hits else "---"
            print(f"  {mark} {course}{rnum:2d}R [{pattern:4s}] 投資¥{race_bet} 払戻¥{race_payout:>6,} 収支{profit:>+7,} ({hit_str})")

            all_results.append({
                "date": ds, "course": course, "rnum": rnum,
                "bet": race_bet, "payout": race_payout, "profit": profit,
                "hits": race_hits, "pattern": pattern, "race_id": race_id,
            })

    # ============================================================
    # SUMMARY
    # ============================================================
    total_bet = sum(r["bet"] for r in all_results)
    total_payout = sum(r["payout"] for r in all_results)
    total_profit = total_payout - total_bet
    total_bets_n = sum(len(r["hits"]) for r in all_results) + sum(1 for r in all_results for _ in range(5 - len(r["hits"])) if True)
    total_hits_n = sum(len(r["hits"]) for r in all_results)
    win_races = sum(1 for r in all_results if r["profit"] > 0)
    lose_races = sum(1 for r in all_results if r["profit"] < 0)
    even_races = sum(1 for r in all_results if r["profit"] == 0)
    roi = total_payout / total_bet * 100 if total_bet > 0 else 0
    n = len(all_results)

    print(f"\n{'='*70}")
    print(f"  2026年3月 全レース検証結果 (v7 Dual Model)")
    print(f"{'='*70}")
    print(f"  対象レース数:   {n}")
    print(f"  総投資額:       ¥{total_bet:,}")
    print(f"  総払戻額:       ¥{total_payout:,}")
    print(f"  総収支:         {'+' if total_profit >= 0 else ''}¥{total_profit:,}")
    print(f"  回収率 (ROI):   {roi:.1f}%")
    total_bets_actual = sum(len(optimized) for _ in [0])  # approximate
    hit_rate = total_hits_n / (n * 5) * 100 if n > 0 else 0
    print(f"  的中率:         {total_hits_n}/{n*5} ({hit_rate:.1f}%)")
    print(f"  レース勝率:     {win_races}勝 {lose_races}敗 {even_races}分 ({win_races}/{n} = {win_races/n*100:.1f}%)")

    # Per-date summary
    print(f"\n  --- 日別成績 ---")
    for ds in sorted(all_race_ids.keys()):
        dr = [r for r in all_results if r["date"] == ds]
        if not dr:
            continue
        d_bet = sum(r["bet"] for r in dr)
        d_pay = sum(r["payout"] for r in dr)
        d_profit = d_pay - d_bet
        d_wins = sum(1 for r in dr if r["profit"] > 0)
        d_roi = d_pay / d_bet * 100 if d_bet > 0 else 0
        day_label = f"{int(ds[4:6])}/{int(ds[6:8])}"
        print(f"  {day_label}: {len(dr):2d}R 投資¥{d_bet:>6,} 払戻¥{d_pay:>7,} 収支{d_profit:>+7,} ROI {d_roi:5.1f}% ({d_wins}勝)")

    # Per-course summary
    print(f"\n  --- 競馬場別成績 ---")
    for cname in ["中山", "阪神", "中京", "東京", "京都", "小倉", "福島", "新潟", "札幌", "函館"]:
        cr = [r for r in all_results if r["course"] == cname]
        if not cr:
            continue
        c_bet = sum(r["bet"] for r in cr)
        c_pay = sum(r["payout"] for r in cr)
        c_profit = c_pay - c_bet
        c_roi = c_pay / c_bet * 100 if c_bet > 0 else 0
        c_wins = sum(1 for r in cr if r["profit"] > 0)
        print(f"  {cname}: {len(cr):2d}R 投資¥{c_bet:>6,} 払戻¥{c_pay:>7,} 収支{c_profit:>+7,} ROI {c_roi:5.1f}% ({c_wins}勝)")

    # Per-bet-type summary
    print(f"\n  --- 券種別成績 ---")
    for bt in ["tansho", "fukusho", "umaren", "wide", "sanrenpuku", "sanrentan"]:
        if bt not in type_stats:
            continue
        s = type_stats[bt]
        s_roi = s["returned"] / s["invested"] * 100 if s["invested"] > 0 else 0
        s_hit = s["hits"] / s["bets"] * 100 if s["bets"] > 0 else 0
        print(f"  {s['label']:4s}: {s['bets']:3d}回 的中{s['hits']:3d} ({s_hit:5.1f}%) 投資¥{s['invested']:>6,} 払戻¥{s['returned']:>7,} ROI {s_roi:5.1f}%")

    # Pattern analysis
    print(f"\n  --- パターン別成績 ---")
    pat_stats = {}
    for r in all_results:
        p = r["pattern"]
        if p not in pat_stats:
            pat_stats[p] = {"count": 0, "bet": 0, "payout": 0, "wins": 0}
        pat_stats[p]["count"] += 1
        pat_stats[p]["bet"] += r["bet"]
        pat_stats[p]["payout"] += r["payout"]
        if r["profit"] > 0:
            pat_stats[p]["wins"] += 1
    for p, s in sorted(pat_stats.items(), key=lambda x: -x[1]["payout"]):
        p_roi = s["payout"] / s["bet"] * 100 if s["bet"] > 0 else 0
        print(f"  {p:6s}: {s['count']:2d}R ROI {p_roi:5.1f}% ({s['wins']}勝)")

    # Top 5 best races
    print(f"\n  --- Top5 高額払戻レース ---")
    top = sorted(all_results, key=lambda x: -x["payout"])[:5]
    for r in top:
        if r["payout"] > 0:
            print(f"  {int(r['date'][4:6])}/{int(r['date'][6:8])} {r['course']}{r['rnum']:2d}R: ¥{r['payout']:>7,} ({','.join(r['hits'])})")

    print(f"\n{'='*70}")
    print(f"  検証完了")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
