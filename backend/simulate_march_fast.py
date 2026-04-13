"""Fast March 2026 simulation — direct module calls, no HTTP.

Scrapes race data and payouts directly, runs AI predictions in-process.
Much faster than going through the backend API.
"""
from __future__ import annotations

import re
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from bs4 import BeautifulSoup
from backend.scraper.netkeiba import fetch_race_list, fetch_race_card
from backend.scraper.odds import estimate_from_entries, fetch_combination_odds
from backend.predictor.ml_scoring import MLScoringModel
from backend.predictor.bet_optimizer import optimize_bets, detect_race_pattern, scores_to_probabilities
from backend.database.db import init_db

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
COURSE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}

predictor = MLScoringModel()


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
                        entries.append({"nums": nums, "amount": amt})
                if entries:
                    payouts[label] = entries
        return payouts
    except Exception:
        return {}


def check_bet_hit(bet, payouts):
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


def get_march_race_ids():
    """Get all race IDs for March 2026 weekends."""
    from datetime import date, timedelta
    results = {}  # date_str -> [race_ids]
    d = date(2026, 3, 1)
    end = date(2026, 3, 31)
    while d <= end:
        if d.weekday() in (5, 6):  # Sat, Sun only
            ds = d.strftime("%Y%m%d")
            time.sleep(2)
            schedules = fetch_race_list(ds)
            if schedules:
                ids = []
                for s in schedules:
                    for r in s.get("races", []):
                        rid = r.get("race_id", "")
                        if rid and rid not in ids:
                            ids.append(rid)
                if ids:
                    results[ds] = ids
                    print(f"  {int(ds[4:6])}/{int(ds[6:8])}: {len(ids)} races ({', '.join(s['name'] for s in schedules)})")
        d += timedelta(days=1)
    return results


def main():
    init_db()

    print("=" * 70)
    print("  KEIBA ORACLE v7 - 2026年3月 全レース検証 (高速版)")
    print("  各レース AI Top5買い目 x ¥100 = ¥500/レース")
    print("=" * 70)

    print("\nCollecting March 2026 race dates...")
    all_race_ids = get_march_race_ids()
    total_races = sum(len(v) for v in all_race_ids.values())
    print(f"\nTotal: {total_races} races across {len(all_race_ids)} days\n")

    all_results = []
    type_stats = {}
    processed = 0

    for ds in sorted(all_race_ids.keys()):
        race_ids = all_race_ids[ds]
        day_label = f"{int(ds[4:6])}/{int(ds[6:8])}"
        print(f"{'─'*70}")
        print(f"  {day_label} ({len(race_ids)} races)")
        print(f"{'─'*70}")

        for race_id in race_ids:
            course = COURSE_MAP.get(race_id[4:6], "??")
            rnum = int(race_id[10:12])
            processed += 1

            # Get race card (cached after first fetch), with retry
            data = None
            for attempt in range(3):
                try:
                    data = fetch_race_card(race_id)
                    if data:
                        break
                    time.sleep(5)
                except Exception as e:
                    if attempt < 2:
                        time.sleep(10)
                    else:
                        print(f"  - {course}{rnum:2d}R: error {e}")
            if not data:
                print(f"  - {course}{rnum:2d}R: no data (skipped)")
                continue

            entries = data.get("entries", [])
            race_info = data.get("race_info", {})
            if len(entries) < 3:
                continue

            # AI predictions
            try:
                predictions = predictor.predict(race_info, entries)
            except Exception:
                continue
            if len(predictions) < 3:
                continue

            # Odds for optimizer
            odds_data = estimate_from_entries(entries) or {}
            try:
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

            # Optimizer
            try:
                optimized = optimize_bets(predictions, odds_data, race_info)
            except Exception:
                continue
            if not optimized:
                continue

            # Payouts with retry
            payouts = None
            for attempt in range(3):
                payouts = fetch_payouts(race_id)
                if payouts:
                    break
                time.sleep(5)
            if not payouts:
                print(f"  - {course}{rnum:2d}R: no payouts (skipped)")
                continue

            # Check bets
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
            print(f"  {mark} {course}{rnum:2d}R [{pattern:4s}] ¥{race_bet}→¥{race_payout:>6,} {profit:>+7,} ({hit_str})")

            all_results.append({
                "date": ds, "course": course, "rnum": rnum,
                "bet": race_bet, "payout": race_payout, "profit": profit,
                "hits": race_hits, "pattern": pattern,
            })

    # ============================================================
    # SUMMARY
    # ============================================================
    n = len(all_results)
    if n == 0:
        print("\nNo results to summarize.")
        return

    total_bet = sum(r["bet"] for r in all_results)
    total_payout = sum(r["payout"] for r in all_results)
    total_profit = total_payout - total_bet
    total_hits = sum(len(r["hits"]) for r in all_results)
    total_bets_n = n * 5
    win_races = sum(1 for r in all_results if r["profit"] > 0)
    lose_races = sum(1 for r in all_results if r["profit"] < 0)
    even_races = sum(1 for r in all_results if r["profit"] == 0)
    roi = total_payout / total_bet * 100 if total_bet > 0 else 0

    print(f"\n{'='*70}")
    print(f"  2026年3月 全レース検証結果 (KEIBA ORACLE v7)")
    print(f"{'='*70}")
    print(f"  対象レース数:   {n}")
    print(f"  総購入点数:     {total_bets_n}点 (各レース5点)")
    print(f"  総投資額:       ¥{total_bet:,}")
    print(f"  総払戻額:       ¥{total_payout:,}")
    print(f"  総収支:         {'+' if total_profit >= 0 else ''}¥{total_profit:,}")
    print(f"  回収率 (ROI):   {roi:.1f}%")
    print(f"  的中率:         {total_hits}/{total_bets_n} ({total_hits/total_bets_n*100:.1f}%)")
    print(f"  レース勝率:     {win_races}勝 {lose_races}敗 {even_races}分 ({win_races/n*100:.1f}%)")

    # Per-date
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
        print(f"  {int(ds[4:6])}/{int(ds[6:8])}: {len(dr):2d}R 投資¥{d_bet:>6,} 払戻¥{d_pay:>8,} 収支{d_profit:>+8,} ROI {d_roi:6.1f}% ({d_wins}勝)")

    # Per-course
    print(f"\n  --- 競馬場別成績 ---")
    for cname in sorted(set(r["course"] for r in all_results)):
        cr = [r for r in all_results if r["course"] == cname]
        c_bet = sum(r["bet"] for r in cr)
        c_pay = sum(r["payout"] for r in cr)
        c_roi = c_pay / c_bet * 100 if c_bet > 0 else 0
        c_wins = sum(1 for r in cr if r["profit"] > 0)
        print(f"  {cname}: {len(cr):3d}R 投資¥{c_bet:>7,} 払戻¥{c_pay:>8,} 収支{c_pay-c_bet:>+8,} ROI {c_roi:6.1f}% ({c_wins}勝)")

    # Per-bet-type
    print(f"\n  --- 券種別成績 ---")
    for bt in ["tansho", "fukusho", "umaren", "wide", "sanrenpuku", "sanrentan"]:
        if bt not in type_stats:
            continue
        s = type_stats[bt]
        s_roi = s["returned"] / s["invested"] * 100 if s["invested"] > 0 else 0
        s_hit = s["hits"] / s["bets"] * 100 if s["bets"] > 0 else 0
        print(f"  {s['label']:4s}: {s['bets']:4d}点 的中{s['hits']:3d} ({s_hit:5.1f}%) 投資¥{s['invested']:>7,} 払戻¥{s['returned']:>8,} ROI {s_roi:6.1f}%")

    # Pattern
    print(f"\n  --- パターン別成績 ---")
    pat = {}
    for r in all_results:
        p = r["pattern"]
        if p not in pat:
            pat[p] = {"n": 0, "bet": 0, "pay": 0, "wins": 0}
        pat[p]["n"] += 1
        pat[p]["bet"] += r["bet"]
        pat[p]["pay"] += r["payout"]
        if r["profit"] > 0:
            pat[p]["wins"] += 1
    for p, s in sorted(pat.items(), key=lambda x: -x[1]["pay"]):
        p_roi = s["pay"] / s["bet"] * 100 if s["bet"] > 0 else 0
        print(f"  {p:6s}: {s['n']:3d}R ROI {p_roi:6.1f}% ({s['wins']}勝)")

    # Top 5
    print(f"\n  --- Top5 高額払戻 ---")
    for r in sorted(all_results, key=lambda x: -x["payout"])[:5]:
        if r["payout"] > 0:
            print(f"  {int(r['date'][4:6])}/{int(r['date'][6:8])} {r['course']}{r['rnum']:2d}R: ¥{r['payout']:>7,} ({','.join(r['hits'])})")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
