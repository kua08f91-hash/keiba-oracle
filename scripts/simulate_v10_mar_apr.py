"""v10 simulation: March + April 2026 full race backtest.

Uses the same logic as simulate_march_fast.py but covers both months.
"""
from __future__ import annotations

import re
import sys
import os
import time
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from bs4 import BeautifulSoup
from backend.scraper.netkeiba import fetch_race_list, fetch_race_card
from backend.scraper.odds import estimate_from_entries, fetch_combination_odds
from backend.predictor.ml_scoring import MLScoringModel
from backend.predictor.bet_optimizer import optimize_bets, detect_race_pattern, scores_to_probabilities
from backend.database.db import init_db

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
}
COURSE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}

predictor = MLScoringModel()
_session = requests.Session()
_session.headers.update(HEADERS)


def fetch_payouts(race_id):
    url = f"https://db.netkeiba.com/race/{race_id}/"
    try:
        resp = _session.get(url, timeout=20)
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
        "tansho": "単勝", "fukusho": "複勝", "wakuren": "枠連",
        "umaren": "馬連", "umatan": "馬単", "wide": "ワイド",
        "sanrenpuku": "三連複", "sanrentan": "三連単",
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
        elif bet["type"] in ("umaren", "wide", "wakuren", "sanrenpuku") and set(horses) == set(pnums):
            return True, pamt
        elif bet["type"] in ("umatan", "sanrentan") and horses == pnums:
            return True, pamt
    return False, 0


def get_race_ids(start_date, end_date):
    results = {}
    d = start_date
    while d <= end_date:
        if d.weekday() in (5, 6):
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
                    print(f"  {int(ds[4:6])}/{int(ds[6:8])}: {len(ids)} races ({', '.join(s['name'] for s in schedules)})", flush=True)
        d += timedelta(days=1)
    return results


def main():
    init_db()

    print("=" * 70)
    print("  KEIBA ORACLE v10 - 2026年3-4月 全レース検証")
    print("  各レース AI Top5買い目 x ¥100 = ¥500/レース")
    print("=" * 70)

    print("\nCollecting race dates...")
    all_race_ids = get_race_ids(date(2026, 3, 1), date(2026, 4, 30))
    total_races = sum(len(v) for v in all_race_ids.values())
    print(f"\nTotal: {total_races} races across {len(all_race_ids)} days\n")

    all_results = []
    type_stats = {}
    month_stats = {}
    processed = 0

    for ds in sorted(all_race_ids.keys()):
        race_ids = all_race_ids[ds]
        day_label = f"{int(ds[4:6])}/{int(ds[6:8])}"
        month = ds[:6]
        if month not in month_stats:
            month_stats[month] = {"bet": 0, "payout": 0, "races": 0, "hits": 0}

        print(f"{'─'*70}")
        print(f"  {day_label} ({len(race_ids)} races)")
        print(f"{'─'*70}")

        for race_id in race_ids:
            course = COURSE_MAP.get(race_id[4:6], "??")
            rnum = int(race_id[10:12])
            processed += 1

            data = None
            try:
                data = fetch_race_card(race_id)
            except Exception as e:
                print(f"  - {course}{rnum:2d}R: error {e}", flush=True)
            if not data:
                print(f"  - {course}{rnum:2d}R: no data", flush=True)
                continue

            entries = data.get("entries", [])
            race_info = data.get("race_info", {})
            if len(entries) < 3:
                continue

            try:
                predictions = predictor.predict(race_info, entries)
            except Exception:
                continue
            if len(predictions) < 3:
                continue

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

            try:
                optimized = optimize_bets(predictions, odds_data, race_info, entries=entries)
            except Exception:
                continue
            if not optimized:
                continue

            payouts = None
            for attempt in range(3):
                payouts = fetch_payouts(race_id)
                if payouts:
                    break
                time.sleep(5 * (attempt + 1))
            if not payouts:
                print(f"  - {course}{rnum:2d}R: no payouts", flush=True)
                continue

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
            mark = "+" if profit > 0 else (" " if profit == 0 else "-")
            hit_str = ",".join(race_hits) if race_hits else "---"
            print(f"  {mark} {course}{rnum:2d}R ¥{race_bet}→¥{race_payout:>6,} {profit:>+7,} ({hit_str})", flush=True)

            time.sleep(3)

            month_stats[month]["bet"] += race_bet
            month_stats[month]["payout"] += race_payout
            month_stats[month]["races"] += 1
            month_stats[month]["hits"] += len(race_hits)

            all_results.append({
                "date": ds, "course": course, "rnum": rnum,
                "bet": race_bet, "payout": race_payout, "profit": profit, "hits": race_hits,
            })

    # ─── SUMMARY ───
    n = len(all_results)
    if n == 0:
        print("\nNo results.")
        return

    total_bet = sum(r["bet"] for r in all_results)
    total_payout = sum(r["payout"] for r in all_results)
    total_profit = total_payout - total_bet
    total_hits = sum(len(r["hits"]) for r in all_results)
    total_bets_n = sum(len(r["hits"]) for r in all_results) + sum(
        (r["bet"] // 100 - len(r["hits"])) for r in all_results
    )
    win_races = sum(1 for r in all_results if r["profit"] > 0)

    print(f"\n{'='*70}")
    print(f"  v10 Simulation Results: 2026年3-4月")
    print(f"{'='*70}")
    print(f"  レース数: {n}")
    print(f"  投資: ¥{total_bet:,}")
    print(f"  回収: ¥{total_payout:,}")
    print(f"  収支: ¥{total_profit:+,}")
    print(f"  ROI: {total_payout/total_bet*100:.1f}%")
    print(f"  的中: {total_hits}/{n*5} ({total_hits/(n*5)*100:.1f}%)")
    print(f"  勝ちレース: {win_races}/{n} ({win_races/n*100:.1f}%)")

    print(f"\n  月別:")
    for m in sorted(month_stats.keys()):
        ms = month_stats[m]
        if ms["bet"] > 0:
            roi = ms["payout"] / ms["bet"] * 100
            print(f"    {m[:4]}/{int(m[4:6])}月: {ms['races']}R 投資:¥{ms['bet']:,} 回収:¥{ms['payout']:,} ROI:{roi:.1f}% 的中:{ms['hits']}")

    print(f"\n  券種別:")
    for bt in ["umaren", "wide", "sanrentan", "umatan", "sanrenpuku", "tansho", "fukusho", "wakuren"]:
        ts = type_stats.get(bt)
        if ts and ts["bets"] > 0:
            roi = ts["returned"] / ts["invested"] * 100
            print(f"    {ts['label']:6s}: {ts['hits']:3d}/{ts['bets']:4d} ({ts['hits']/ts['bets']*100:5.1f}%)  投資:¥{ts['invested']:>7,}  回収:¥{ts['returned']:>8,}  ROI:{roi:6.1f}%")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
