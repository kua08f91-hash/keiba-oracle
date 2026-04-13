"""Simulate betting using dynamic EV-optimized strategy.

For each race:
1. Get AI predictions from backend
2. Get actual payouts from db.netkeiba.com
3. Run the dynamic bet optimizer (same logic as production)
4. Calculate profit/loss for ¥100 bets on each optimized pick

Output: per-race detail and total P&L summary.
"""
import requests
import re
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from bs4 import BeautifulSoup
from backend.predictor.bet_optimizer import optimize_bets, detect_race_pattern, scores_to_probabilities

BACKEND = "http://localhost:8000"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def fetch_payouts(race_id):
    """Fetch payout data from db.netkeiba.com."""
    url = f"https://db.netkeiba.com/race/{race_id}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")

        payouts = {}
        for table in soup.select("table.pay_table_01"):
            for row in table.select("tr"):
                th = row.select_one("th")
                tds = row.select("td")
                if not th or len(tds) < 2:
                    continue

                label = th.get_text(strip=True)
                combos_text = tds[0].get_text("|", strip=True)
                amounts_text = tds[1].get_text("|", strip=True)

                combos = combos_text.split("|")
                amounts = amounts_text.split("|")

                entries = []
                for combo, amount in zip(combos, amounts):
                    try:
                        amt = int(amount.replace(",", ""))
                    except:
                        continue
                    nums = [int(n) for n in re.findall(r"\d+", combo)]
                    if nums:
                        entries.append({"nums": nums, "amount": amt, "ordered": "→" in combo})

                if entries:
                    payouts[label] = entries

        return payouts
    except Exception as e:
        print(f"  [WARN] Failed to fetch payouts for {race_id}: {e}")
        return {}


def fetch_odds_for_optimizer(race_id, entries):
    """Fetch estimated odds for ALL combinations (for optimizer input).

    Uses estimate_from_entries for broad coverage,
    merged with real payouts when available.
    """
    from backend.scraper.odds import estimate_from_entries, fetch_combination_odds

    odds_data = estimate_from_entries(entries) or {}

    # Merge real payouts if available
    real_payouts = fetch_combination_odds(race_id)
    if real_payouts:
        for key, entries_list in real_payouts.items():
            if key in odds_data:
                real_horses_sets = [frozenset(e["horses"]) for e in entries_list]
                filtered = [e for e in odds_data[key]
                           if frozenset(e["horses"]) not in real_horses_sets]
                odds_data[key] = entries_list + filtered
            else:
                odds_data[key] = entries_list

    return odds_data


def check_bet_hit(bet, payouts):
    """Check if a bet hits and return payout amount (per ¥100).

    Returns (hit: bool, payout_amount: int)
    """
    # Map optimizer type to payout label
    type_map = {
        "tansho": "単勝",
        "fukusho": "複勝",
        "umaren": "馬連",
        "wide": "ワイド",
        "sanrenpuku": "三連複",
        "sanrentan": "三連単",
    }
    payout_label = type_map.get(bet["type"])
    if not payout_label or payout_label not in payouts:
        return False, 0

    horses = bet["horses"]

    for entry in payouts[payout_label]:
        payout_nums = entry["nums"]
        payout_amount = entry["amount"]

        if bet["type"] == "tansho":
            if len(horses) == 1 and horses[0] in payout_nums:
                return True, payout_amount

        elif bet["type"] == "fukusho":
            if len(horses) == 1 and horses[0] == payout_nums[0]:
                return True, payout_amount

        elif bet["type"] == "umaren":
            if set(horses) == set(payout_nums):
                return True, payout_amount

        elif bet["type"] == "wide":
            if set(horses) == set(payout_nums):
                return True, payout_amount

        elif bet["type"] == "sanrenpuku":
            if set(horses) == set(payout_nums):
                return True, payout_amount

        elif bet["type"] == "sanrentan":
            if horses == payout_nums:
                return True, payout_amount

    return False, 0


def main():
    date_str = "20260329"
    print(f"{'='*70}")
    print(f"  EV最適化 買い目シミュレーション: 3/29(日)")
    print(f"  各レース 動的Top5 × ¥100 = ¥500/レース")
    print(f"{'='*70}")

    # Get race list
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.encoding = "UTF-8"
    all_ids = re.findall(r"race_id=(\d+)", resp.text)
    race_ids = []
    for rid in all_ids:
        if len(rid) >= 12 and rid[4:6] in ["01","02","03","04","05","06","07","08","09","10"]:
            if rid not in race_ids:
                race_ids.append(rid)

    print(f"Total races: {len(race_ids)}\n")

    total_bet = 0
    total_payout = 0
    total_hits = 0
    total_bets_count = 0
    race_results = []

    course_map = {"06": "中山", "09": "阪神", "07": "中京"}

    for race_id in race_ids:
        course = course_map.get(race_id[4:6], race_id[4:6])
        rnum = int(race_id[10:12])

        # Get AI predictions
        time.sleep(1)
        try:
            api_resp = requests.get(f"{BACKEND}/api/racecard/{race_id}", timeout=30)
            data = api_resp.json()
        except:
            print(f"  ✗ {course}{rnum:2d}R: API error")
            continue

        predictions = data.get("predictions", [])
        entries = data.get("entries", [])
        race_info = data.get("raceInfo", {})

        if len(predictions) < 3:
            print(f"  ✗ {course}{rnum:2d}R: Not enough predictions")
            continue

        # Get estimated odds for optimizer (covers all combinations)
        odds_data = fetch_odds_for_optimizer(race_id, entries)

        # Run dynamic optimizer (same as production)
        optimized = optimize_bets(predictions, odds_data, race_info)
        if not optimized:
            print(f"  ✗ {course}{rnum:2d}R: Optimizer returned no bets")
            continue

        # Get actual payouts
        time.sleep(1)
        payouts = fetch_payouts(race_id)
        if not payouts:
            print(f"  ✗ {course}{rnum:2d}R: No payout data")
            continue

        # Detect pattern
        probs = scores_to_probabilities(predictions, race_info.get("headCount", 16))
        pattern = detect_race_pattern(probs)

        # Check each bet
        race_bet = len(optimized) * 100
        race_payout = 0
        race_hits = []

        bet_types_used = " + ".join(b["typeLabel"] for b in optimized)
        print(f"\n  {course}{rnum:2d}R [{pattern}] ({bet_types_used}) 投資: ¥{race_bet}")
        for i, bet in enumerate(optimized):
            hit, amount = check_bet_hit(bet, payouts)
            if bet["ordered"]:
                horse_str = " → ".join(str(h) for h in bet["horses"])
            else:
                horse_str = " - ".join(str(h) for h in bet["horses"])

            ev_str = f"EV{bet.get('ev', 0):+.2f}" if "ev" in bet else ""
            prob_str = f"P={bet.get('hitProb', 0)*100:.1f}%" if "hitProb" in bet else ""

            if hit:
                race_payout += amount
                race_hits.append(bet["typeLabel"])
                print(f"    {i+1}. {bet['typeLabel']:4s} [{horse_str}] {ev_str} {prob_str} → ✅ 的中! ¥{amount:,}")
            else:
                print(f"    {i+1}. {bet['typeLabel']:4s} [{horse_str}] {ev_str} {prob_str} → ✗")

        profit = race_payout - race_bet
        mark = "🎯" if profit > 0 else ("±" if profit == 0 else "")
        print(f"    収支: ¥{race_bet} → ¥{race_payout:,} = {'+' if profit >= 0 else ''}{profit:,}円 {mark}")

        total_bet += race_bet
        total_payout += race_payout
        total_hits += len(race_hits)
        total_bets_count += len(optimized)
        race_results.append({
            "course": course,
            "rnum": rnum,
            "bet": race_bet,
            "payout": race_payout,
            "profit": profit,
            "hits": race_hits,
            "pattern": pattern,
        })

    # Summary
    total_profit = total_payout - total_bet
    hit_rate = total_hits / total_bets_count * 100 if total_bets_count > 0 else 0
    roi = total_payout / total_bet * 100 if total_bet > 0 else 0
    win_races = sum(1 for r in race_results if r["profit"] > 0)
    lose_races = sum(1 for r in race_results if r["profit"] < 0)
    even_races = sum(1 for r in race_results if r["profit"] == 0)

    print(f"\n{'='*70}")
    print(f"  TOTAL SUMMARY: 3/29(日) 全{len(race_results)}レース (EV最適化)")
    print(f"{'='*70}")
    print(f"  総投資額:     ¥{total_bet:,}")
    print(f"  総払戻額:     ¥{total_payout:,}")
    print(f"  総収支:       {'+' if total_profit >= 0 else ''}¥{total_profit:,}")
    print(f"  回収率:       {roi:.1f}%")
    print(f"  的中率:       {total_hits}/{total_bets_count} ({hit_rate:.1f}%)")
    print(f"  レース勝敗:   {win_races}勝 {lose_races}敗 {even_races}分 (勝率{win_races/len(race_results)*100:.1f}%)")
    print()

    # Per-course summary
    for course_name in ["中山", "阪神", "中京"]:
        cr = [r for r in race_results if r["course"] == course_name]
        if not cr:
            continue
        c_bet = sum(r["bet"] for r in cr)
        c_pay = sum(r["payout"] for r in cr)
        c_profit = c_pay - c_bet
        c_wins = sum(1 for r in cr if r["profit"] > 0)
        c_hits = sum(len(r["hits"]) for r in cr)
        c_bets = sum(5 for _ in cr)
        print(f"  {course_name}: 投資¥{c_bet:,} 払戻¥{c_pay:,} 収支{'+' if c_profit >= 0 else ''}{c_profit:,}円 回収率{c_pay/c_bet*100:.1f}% 的中{c_hits}/{c_bets} ({c_wins}勝{len(cr)-c_wins}敗)")

    # Pattern analysis
    print(f"\n  レースパターン別:")
    patterns = {}
    for r in race_results:
        p = r["pattern"]
        if p not in patterns:
            patterns[p] = {"count": 0, "bet": 0, "payout": 0, "hits": 0}
        patterns[p]["count"] += 1
        patterns[p]["bet"] += r["bet"]
        patterns[p]["payout"] += r["payout"]
        patterns[p]["hits"] += len(r["hits"])
    for p, d in sorted(patterns.items(), key=lambda x: -x[1]["payout"]):
        p_roi = d["payout"] / d["bet"] * 100 if d["bet"] > 0 else 0
        print(f"    {p}: {d['count']}レース 回収率{p_roi:.1f}% 的中{d['hits']}回")

    # Top winning races
    print(f"\n  🎯 Top 3 高額払戻レース:")
    top_races = sorted(race_results, key=lambda x: -x["payout"])
    for r in top_races[:3]:
        if r["payout"] > 0:
            print(f"    {r['course']}{r['rnum']:2d}R [{r['pattern']}]: ¥{r['payout']:,} ({', '.join(r['hits'])})")

    # By bet type
    print(f"\n  券種別的中:")
    type_hits = {}
    for r in race_results:
        for h in r["hits"]:
            type_hits[h] = type_hits.get(h, 0) + 1
    for t in ["単勝", "複勝", "馬連", "ワイド", "3連複", "3連単"]:
        count = type_hits.get(t, 0)
        print(f"    {t}: {count}回的中")


if __name__ == "__main__":
    main()
