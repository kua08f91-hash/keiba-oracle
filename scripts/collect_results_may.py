"""Collect race results for 5/2 and 5/3 (2026) and compare against predictions.

Usage:
    /usr/bin/python3 scripts/collect_results_may.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Optional, Dict, List, Tuple

import requests
from bs4 import BeautifulSoup

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}

BET_AMOUNT = 500  # yen per bet


def fetch_race_list_for_date(date_str: str) -> List[Dict]:
    """Fetch race list for a date, return list of {race_id, race_number, course_name}."""
    from backend.scraper.netkeiba import fetch_race_list
    schedules = fetch_race_list(date_str)
    races = []
    for schedule in schedules:
        course_name = schedule.get("course_name", "")
        for race in schedule.get("races", []):
            rid = race.get("race_id", "")
            rnum = race.get("race_number", 0)
            rname = race.get("race_name", "")
            if rid:
                races.append({
                    "race_id": rid,
                    "race_number": rnum,
                    "race_name": rname,
                    "course_name": course_name,
                })
    return races


def fetch_result_from_netkeiba(race_id: str) -> Optional[Tuple[List[Dict], Dict]]:
    """Fetch race result from race.netkeiba.com result page.

    Returns (results_list, payouts_dict) where:
      results_list: [{position, horse_number, horse_name, odds}, ...]
      payouts_dict: {bet_type: [{horses: [...], payout: int}, ...]}
    """
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = resp.apparent_encoding or "UTF-8"
        if resp.status_code != 200:
            print(f"    [WARN] HTTP {resp.status_code} for {race_id}")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # race.netkeiba.com layout:
        # td[0]=着順 td[1]=枠番 td[2]=馬番 td[3]=馬名 ... td[10]=単勝オッズ
        result_table = soup.select_one("table.RaceTable01")
        if not result_table:
            return None

        results = []
        rows = result_table.select("tr")
        for row in rows:
            tds = row.select("td")
            if len(tds) < 4:
                continue

            pos_text = tds[0].get_text(strip=True)
            pos_match = re.match(r"(\d+)", pos_text)
            if not pos_match:
                continue
            position = int(pos_match.group(1))

            # td[2] = 馬番 (class Num Txt_C)
            try:
                horse_number = int(tds[2].get_text(strip=True))
            except (ValueError, IndexError):
                continue

            # td[3] = 馬名 (class Horse_Info)
            horse_name = tds[3].get_text(strip=True) if len(tds) > 3 else f"馬番{horse_number}"

            # td[10] = 単勝オッズ (class Odds Txt_R)
            odds = None
            if len(tds) > 10:
                try:
                    odds = float(tds[10].get_text(strip=True))
                except (ValueError, IndexError):
                    pass

            results.append({
                "position": position,
                "horse_number": horse_number,
                "horse_name": horse_name,
                "odds": odds,
            })

        if not results:
            return None

        # Parse payout tables
        payouts = parse_payouts(soup)

        return sorted(results, key=lambda x: x["position"]), payouts

    except Exception as e:
        print(f"    [ERROR] fetch_result race.netkeiba {race_id}: {e}")
        return None


def parse_payouts(soup: BeautifulSoup) -> Dict:
    """Parse payout tables from race result page.

    Returns dict like:
      {"tansho": [{"horses": [13], "payout": 250}],
       "fukusho": [{"horses": [13], "payout": 120}, ...],
       "umaren": [...], "wide": [...], "umatan": [...],
       "sanrenpuku": [...], "sanrentan": [...]}
    """
    label_to_key = {
        "単勝": "tansho",
        "複勝": "fukusho",
        "枠連": "wakuren",
        "馬連": "umaren",
        "ワイド": "wide",
        "馬単": "umatan",
        "3連複": "sanrenpuku",
        "3連単": "sanrentan",
    }
    payouts = {}
    tables = soup.select("table.Payout_Detail_Table")
    for table in tables:
        rows = table.select("tr")
        for row in rows:
            cells = row.select("th, td")
            if len(cells) < 3:
                continue
            label = cells[0].get_text(strip=True)
            key = label_to_key.get(label)
            if not key:
                continue

            result_text = cells[1].get_text(strip=True)
            payout_text = cells[2].get_text(strip=True)

            # Parse horse numbers - they are concatenated (e.g. "313" for 3-13, "1334" for 13-3-4)
            # Parse payouts - can be multiple (e.g. "120円210円200円")
            payout_values = re.findall(r"([\d,]+)円", payout_text)
            payout_ints = [int(p.replace(",", "")) for p in payout_values]

            # For simple types with single combo
            if key in ("tansho", "umaren", "umatan"):
                # Horse numbers separated - parse from Result cell
                horse_nums = parse_result_horse_numbers(result_text, 1 if key == "tansho" else 2)
                if horse_nums and payout_ints:
                    payouts[key] = [{"horses": horse_nums, "payout": payout_ints[0]}]
            elif key == "fukusho":
                # Multiple horses, each with own payout
                # Result like "1334" = horses 13, 3, 4 with payouts 120, 210, 200
                horse_nums = parse_result_horse_numbers(result_text, 3)
                if horse_nums and payout_ints:
                    entries = []
                    for i, hn in enumerate(horse_nums):
                        p = payout_ints[i] if i < len(payout_ints) else 0
                        entries.append({"horses": [hn], "payout": p})
                    payouts[key] = entries
            elif key == "wide":
                # Multiple combos, e.g. "31341334" = [3,13] [4,13] [3,4] with 3 payouts
                horse_combos = parse_result_horse_numbers_multi(result_text, 2, len(payout_ints))
                entries = []
                for i, combo in enumerate(horse_combos):
                    p = payout_ints[i] if i < len(payout_ints) else 0
                    entries.append({"horses": combo, "payout": p})
                payouts[key] = entries
            elif key == "sanrenpuku":
                horse_nums = parse_result_horse_numbers(result_text, 3)
                if horse_nums and payout_ints:
                    payouts[key] = [{"horses": horse_nums, "payout": payout_ints[0]}]
            elif key == "sanrentan":
                horse_nums = parse_result_horse_numbers(result_text, 3)
                if horse_nums and payout_ints:
                    payouts[key] = [{"horses": horse_nums, "payout": payout_ints[0]}]

    return payouts


def parse_result_horse_numbers(text: str, count: int) -> List[int]:
    """Parse concatenated horse numbers from result text.

    E.g. "313" with count=2 -> [3, 13] or [31, 3] - need to try combinations.
    "1334" with count=3 -> [13, 3, 4]
    """
    # Try to find all numbers using smart splitting
    # The numbers are 1-18 range
    nums = []
    remaining = text
    for i in range(count):
        if not remaining:
            break
        # Try 2-digit first if it makes a valid horse number
        if len(remaining) >= 2:
            two_digit = int(remaining[:2])
            one_digit = int(remaining[:1])
            remaining_after_two = remaining[2:]
            remaining_after_one = remaining[1:]

            if i == count - 1:
                # Last number - take what's left
                try:
                    nums.append(int(remaining))
                    remaining = ""
                except ValueError:
                    break
            elif 1 <= two_digit <= 18:
                # Check if taking 2 digits leaves enough for remaining numbers
                digits_left = len(remaining_after_two)
                numbers_left = count - i - 1
                if digits_left >= numbers_left:
                    nums.append(two_digit)
                    remaining = remaining_after_two
                else:
                    nums.append(one_digit)
                    remaining = remaining_after_one
            elif 1 <= one_digit <= 18:
                nums.append(one_digit)
                remaining = remaining_after_one
            else:
                break
        elif len(remaining) >= 1:
            try:
                nums.append(int(remaining))
                remaining = ""
            except ValueError:
                break

    return nums if len(nums) == count else []


def parse_result_horse_numbers_multi(text: str, per_combo: int, num_combos: int) -> List[List[int]]:
    """Parse multiple combos of horse numbers from concatenated text.

    E.g. "31341334" with per_combo=2, num_combos=3 -> [[3,13],[4,13],[3,4]]
    """
    total_nums = per_combo * num_combos
    all_nums = parse_result_horse_numbers(text, total_nums)
    if len(all_nums) != total_nums:
        return []
    combos = []
    for i in range(num_combos):
        combo = all_nums[i * per_combo:(i + 1) * per_combo]
        combos.append(combo)
    return combos



def load_predictions_for_date(date_str: str) -> Dict[str, Dict]:
    """Load predictions from predictions.json for a specific date.

    Returns {race_id: {predictions: [...], bets: [...], ...}}
    """
    pred_file = os.path.join(BASE_DIR, "docs", "data", "predictions.json")
    if not os.path.exists(pred_file):
        return {}

    with open(pred_file, "r") as f:
        data = json.load(f)

    result = {}
    for pred_set in data.get("predictions", []):
        if pred_set.get("date") != date_str:
            continue
        for course in pred_set.get("courses", []):
            for race in course.get("races", []):
                rid = race.get("raceId", "")
                if rid:
                    result[rid] = {
                        "raceName": race.get("raceName", ""),
                        "raceNumber": race.get("raceNumber", 0),
                        "courseName": course.get("name", ""),
                        "predictions": race.get("predictions", []),
                        "bets": race.get("bets", []),
                        "longshot": race.get("longshot"),
                        "pattern": race.get("pattern", ""),
                    }
    return result


def check_bet_hit(bet: Dict, results: List[Dict], payouts: Dict) -> Tuple[bool, float]:
    """Check if a bet hit and return (is_hit, payout_for_bet_amount).

    Uses actual payouts from result page when available, otherwise estimates from odds.
    Payouts from netkeiba are per 100 yen, so we multiply by BET_AMOUNT/100.
    """
    pos_by_hn = {}
    for r in results:
        pos_by_hn[r["horse_number"]] = r["position"]

    bet_type = bet.get("type", "")
    horses = bet.get("horses", [])
    odds = bet.get("odds", 0)

    # Check all horses finished
    for h in horses:
        if h not in pos_by_hn:
            return False, 0

    hit = False
    if bet_type == "tansho":
        hit = pos_by_hn[horses[0]] == 1
    elif bet_type == "fukusho":
        hit = pos_by_hn[horses[0]] <= 3
    elif bet_type == "umaren":
        positions = sorted([pos_by_hn[h] for h in horses])
        hit = positions == [1, 2]
    elif bet_type == "umatan":
        hit = pos_by_hn[horses[0]] == 1 and pos_by_hn[horses[1]] == 2
    elif bet_type == "wide":
        positions = [pos_by_hn[h] for h in horses]
        hit = all(p <= 3 for p in positions)
    elif bet_type == "sanrenpuku":
        positions = sorted([pos_by_hn[h] for h in horses])
        hit = positions == [1, 2, 3]
    elif bet_type == "sanrentan":
        hit = len(horses) >= 3 and all(pos_by_hn[horses[i]] == i + 1 for i in range(3))

    if not hit:
        return False, 0

    # Look up actual payout from result page
    actual_payout = find_actual_payout(bet_type, horses, payouts)
    if actual_payout:
        # netkeiba payouts are per 100 yen
        return True, actual_payout * (BET_AMOUNT / 100)
    else:
        # Fallback to predicted odds
        return True, odds * BET_AMOUNT


def find_actual_payout(bet_type: str, horses: List[int], payouts: Dict) -> Optional[int]:
    """Look up actual payout from parsed payout data."""
    entries = payouts.get(bet_type, [])
    if not entries:
        return None

    bet_set = sorted(horses)
    for entry in entries:
        entry_set = sorted(entry.get("horses", []))
        if bet_type in ("umatan", "sanrentan"):
            # Ordered - exact match
            if entry.get("horses", []) == horses:
                return entry.get("payout")
        else:
            # Unordered - set match
            if entry_set == bet_set:
                return entry.get("payout")

    # For wide, check if horses are a subset
    if bet_type == "wide":
        for entry in entries:
            if sorted(entry.get("horses", [])) == bet_set:
                return entry.get("payout")

    return None


def analyze_predictions(pred_data: Dict, results: List[Dict], payouts: Dict) -> Dict:
    """Analyze prediction accuracy for a single race."""
    pos_by_hn = {}
    for r in results:
        pos_by_hn[r["horse_number"]] = r["position"]

    predictions = pred_data.get("predictions", [])

    # Find marked horses
    honmei = None  # ◎
    taikou = None  # ◯
    tanana = None  # ▲
    renka = None   # △

    for p in predictions:
        mark = p.get("mark", "")
        hn = p.get("horseNumber")
        if mark == "◎":
            honmei = hn
        elif mark == "◯":
            taikou = hn
        elif mark == "▲":
            tanana = hn
        elif mark == "△":
            renka = hn

    analysis = {
        "honmei_hn": honmei,
        "taikou_hn": taikou,
        "tanana_hn": tanana,
        "renka_hn": renka,
    }

    # ◎ results
    if honmei and honmei in pos_by_hn:
        analysis["honmei_pos"] = pos_by_hn[honmei]
        analysis["honmei_win"] = pos_by_hn[honmei] == 1
        analysis["honmei_rentai"] = pos_by_hn[honmei] <= 2
        analysis["honmei_fukusho"] = pos_by_hn[honmei] <= 3
    else:
        analysis["honmei_pos"] = None
        analysis["honmei_win"] = False
        analysis["honmei_rentai"] = False
        analysis["honmei_fukusho"] = False

    # ◯ results
    if taikou and taikou in pos_by_hn:
        analysis["taikou_pos"] = pos_by_hn[taikou]
        analysis["taikou_rentai"] = pos_by_hn[taikou] <= 2
        analysis["taikou_fukusho"] = pos_by_hn[taikou] <= 3
    else:
        analysis["taikou_pos"] = None
        analysis["taikou_rentai"] = False
        analysis["taikou_fukusho"] = False

    # ▲ results
    if tanana and tanana in pos_by_hn:
        analysis["tanana_pos"] = pos_by_hn[tanana]
        analysis["tanana_fukusho"] = pos_by_hn[tanana] <= 3
    else:
        analysis["tanana_pos"] = None
        analysis["tanana_fukusho"] = False

    # △ results
    if renka and renka in pos_by_hn:
        analysis["renka_pos"] = pos_by_hn[renka]
        analysis["renka_fukusho"] = pos_by_hn[renka] <= 3
    else:
        analysis["renka_pos"] = None
        analysis["renka_fukusho"] = False

    # Top 3 prediction accuracy
    top3_marks = [honmei, taikou, tanana]
    top3_in_top3 = sum(1 for h in top3_marks if h and h in pos_by_hn and pos_by_hn[h] <= 3)
    analysis["top3_in_top3"] = top3_in_top3

    # Bet analysis
    bets = pred_data.get("bets", [])
    bet_results = []
    for bet in bets:
        hit, payout = check_bet_hit(bet, results, payouts)
        bet_results.append({
            "type": bet.get("typeLabel", bet.get("type", "")),
            "horses": bet.get("horses", []),
            "odds": bet.get("odds", 0),
            "hit": hit,
            "payout": payout,
            "investment": BET_AMOUNT,
        })
    analysis["bet_results"] = bet_results

    # Longshot
    longshot = pred_data.get("longshot")
    if longshot:
        hit, payout = check_bet_hit(longshot, results, payouts)
        analysis["longshot_result"] = {
            "type": longshot.get("typeLabel", longshot.get("type", "")),
            "horses": longshot.get("horses", []),
            "odds": longshot.get("odds", 0),
            "hit": hit,
            "payout": payout,
        }

    return analysis


def print_race_detail(race_id: str, course_name: str, race_number: int,
                      race_name: str, results: List[Dict], analysis: Optional[Dict]):
    """Print detailed results for one race."""
    print(f"\n{'='*70}")
    print(f"  {course_name} R{race_number} {race_name} (ID: {race_id})")
    print(f"{'='*70}")

    # Show top 5 finishers
    print("  着順  馬番  馬名                      オッズ")
    print("  " + "-" * 55)
    for r in results[:5]:
        odds_str = f"{r['odds']:.1f}" if r.get("odds") else "---"
        print(f"  {r['position']:>3}着  {r['horse_number']:>2}番  {r['horse_name']:<24s}  {odds_str}")

    if not analysis:
        print("  ※ 予想データなし")
        return

    # Show marks vs results
    print()
    mark_info = []
    for mark_name, mark_char, hn_key, pos_key in [
        ("本命", "◎", "honmei_hn", "honmei_pos"),
        ("対抗", "◯", "taikou_hn", "taikou_pos"),
        ("単穴", "▲", "tanana_hn", "tanana_pos"),
        ("連下", "△", "renka_hn", "renka_pos"),
    ]:
        hn = analysis.get(hn_key)
        pos = analysis.get(pos_key)
        if hn:
            # Find horse name
            hname = ""
            for r in results:
                if r["horse_number"] == hn:
                    hname = r["horse_name"]
                    break
            pos_str = f"{pos}着" if pos else "出走取消"
            result_mark = ""
            if pos == 1:
                result_mark = " ★的中!"
            elif pos and pos <= 2:
                result_mark = " (連対)"
            elif pos and pos <= 3:
                result_mark = " (複勝圏)"
            print(f"  {mark_char}{mark_name}: {hn:>2}番 {hname:<20s} → {pos_str}{result_mark}")

    # Show top3 marks in top3
    print(f"  上位3印(◎◯▲)の3着以内入り: {analysis['top3_in_top3']}/3")

    # Show bet results
    bet_results = analysis.get("bet_results", [])
    if bet_results:
        print()
        print("  【買い目結果】")
        total_invest = 0
        total_return = 0
        for br in bet_results:
            total_invest += br["investment"]
            total_return += br["payout"]
            horses_str = "-".join(str(h) for h in br["horses"])
            hit_mark = "◎的中!" if br["hit"] else "×"
            payout_str = f"¥{br['payout']:,.0f}" if br["hit"] else ""
            print(f"    {br['type']:<6s} [{horses_str}] odds={br['odds']:.1f}  {hit_mark} {payout_str}")

        # Longshot
        ls = analysis.get("longshot_result")
        if ls:
            hit_mark = "◎的中!" if ls["hit"] else "×"
            payout_str = f"¥{ls['payout']:,.0f}" if ls["hit"] else ""
            horses_str = "-".join(str(h) for h in ls["horses"])
            print(f"    [穴] {ls['type']:<6s} [{horses_str}] odds={ls['odds']:.1f}  {hit_mark} {payout_str}")

        print(f"  投資: ¥{total_invest:,} / 回収: ¥{total_return:,.0f} / 収支: ¥{total_return - total_invest:+,.0f}")


def main():
    print("=" * 70)
    print("  JRA予想検証: 2026年5月2日(土) & 5月3日(日)")
    print("  投資単位: ¥{:,}/買い目".format(BET_AMOUNT))
    print("=" * 70)

    # -- Phase 1: Load predictions for 5/3 --
    pred_0503 = load_predictions_for_date("20260503")
    print(f"\n5/3 予想データ: {len(pred_0503)} レース読み込み済み")

    # -- Phase 2: Get race IDs for 5/2 --
    print("\n5/2 (20260502) のレース一覧を取得中...")
    time.sleep(1)
    races_0502 = fetch_race_list_for_date("20260502")
    print(f"  → {len(races_0502)} レース")

    # -- Phase 3: Get race IDs for 5/3 (from predictions) --
    races_0503 = []
    for rid, pdata in pred_0503.items():
        races_0503.append({
            "race_id": rid,
            "race_number": pdata["raceNumber"],
            "race_name": pdata["raceName"],
            "course_name": pdata["courseName"],
        })
    print(f"5/3: {len(races_0503)} レース (予想データより)")

    # -- Phase 4: Collect results --
    all_results = {}  # race_id -> {results, analysis, ...}

    for date_label, races, predictions in [
        ("5/2(土)", races_0502, {}),
        ("5/3(日)", races_0503, pred_0503),
    ]:
        print(f"\n{'#'*70}")
        print(f"#  {date_label} レース結果")
        print(f"{'#'*70}")

        for race in sorted(races, key=lambda x: (x["course_name"], x["race_number"])):
            rid = race["race_id"]
            print(f"\n  取得中: {race['course_name']} R{race['race_number']} {race['race_name']} ({rid})...")
            time.sleep(3)

            fetch_result = fetch_result_from_netkeiba(rid)
            if not fetch_result:
                print(f"    → 結果取得失敗 (まだ未確定の可能性)")
                continue

            results, payouts = fetch_result

            pred_data = predictions.get(rid)
            analysis = analyze_predictions(pred_data, results, payouts) if pred_data else None

            all_results[rid] = {
                "date": date_label,
                "course_name": race["course_name"],
                "race_number": race["race_number"],
                "race_name": race["race_name"],
                "results": results,
                "analysis": analysis,
                "has_prediction": pred_data is not None,
            }

            print_race_detail(
                rid,
                race["course_name"],
                race["race_number"],
                race["race_name"],
                results,
                analysis,
            )

    # -- Phase 5: Summary Statistics --
    print("\n\n")
    print("=" * 70)
    print("  総合集計")
    print("=" * 70)

    # Separate by date
    for date_label in ["5/2(土)", "5/3(日)"]:
        date_races = {k: v for k, v in all_results.items() if v["date"] == date_label}
        pred_races = {k: v for k, v in date_races.items() if v["has_prediction"]}

        print(f"\n--- {date_label} ---")
        print(f"  結果取得: {len(date_races)} レース")

        if not pred_races:
            print(f"  予想データなし")

            # Show top finishers summary
            winners = []
            for rid, rd in date_races.items():
                winner = rd["results"][0] if rd["results"] else None
                if winner:
                    odds_str = f"{winner['odds']:.1f}" if winner.get("odds") else "---"
                    winners.append(f"    {rd['course_name']}R{rd['race_number']}: "
                                   f"{winner['horse_number']}番 {winner['horse_name']} ({odds_str}倍)")
            if winners:
                print("  1着一覧:")
                for w in winners:
                    print(w)
            continue

        # ◎ statistics
        honmei_total = 0
        honmei_win = 0
        honmei_rentai = 0
        honmei_fukusho = 0

        # ◯ statistics
        taikou_total = 0
        taikou_rentai = 0
        taikou_fukusho = 0

        # Top3 marks stats
        top3_total_marks = 0
        top3_in_top3_count = 0

        # Bet stats
        total_bets = 0
        total_hits = 0
        total_investment = 0
        total_return = 0.0

        # Per type
        type_stats = {}

        for rid, rd in pred_races.items():
            a = rd["analysis"]
            if not a:
                continue

            if a.get("honmei_hn"):
                honmei_total += 1
                if a["honmei_win"]:
                    honmei_win += 1
                if a["honmei_rentai"]:
                    honmei_rentai += 1
                if a["honmei_fukusho"]:
                    honmei_fukusho += 1

            if a.get("taikou_hn"):
                taikou_total += 1
                if a["taikou_rentai"]:
                    taikou_rentai += 1
                if a["taikou_fukusho"]:
                    taikou_fukusho += 1

            top3_total_marks += 3
            top3_in_top3_count += a.get("top3_in_top3", 0)

            for br in a.get("bet_results", []):
                total_bets += 1
                total_investment += br["investment"]
                if br["hit"]:
                    total_hits += 1
                    total_return += br["payout"]

                bt = br["type"]
                if bt not in type_stats:
                    type_stats[bt] = {"count": 0, "hits": 0, "invest": 0, "return": 0.0}
                type_stats[bt]["count"] += 1
                type_stats[bt]["invest"] += br["investment"]
                if br["hit"]:
                    type_stats[bt]["hits"] += 1
                    type_stats[bt]["return"] += br["payout"]

        print(f"\n  予想精度:")
        if honmei_total > 0:
            print(f"    ◎(本命) 勝率:   {honmei_win}/{honmei_total} = {honmei_win/honmei_total*100:.1f}%")
            print(f"    ◎(本命) 連対率: {honmei_rentai}/{honmei_total} = {honmei_rentai/honmei_total*100:.1f}%")
            print(f"    ◎(本命) 複勝率: {honmei_fukusho}/{honmei_total} = {honmei_fukusho/honmei_total*100:.1f}%")
        if taikou_total > 0:
            print(f"    ◯(対抗) 連対率: {taikou_rentai}/{taikou_total} = {taikou_rentai/taikou_total*100:.1f}%")
            print(f"    ◯(対抗) 複勝率: {taikou_fukusho}/{taikou_total} = {taikou_fukusho/taikou_total*100:.1f}%")
        if top3_total_marks > 0:
            print(f"    上位3印 3着内率: {top3_in_top3_count}/{top3_total_marks} = {top3_in_top3_count/top3_total_marks*100:.1f}%")

        print(f"\n  買い目成績:")
        print(f"    総買い目数:  {total_bets}")
        print(f"    的中数:      {total_hits}")
        if total_bets > 0:
            print(f"    的中率:      {total_hits/total_bets*100:.1f}%")
        print(f"    総投資額:    ¥{total_investment:,}")
        print(f"    総回収額:    ¥{total_return:,.0f}")
        if total_investment > 0:
            roi = total_return / total_investment * 100
            print(f"    回収率:      {roi:.1f}%")
            print(f"    収支:        ¥{total_return - total_investment:+,.0f}")

        if type_stats:
            print(f"\n  券種別成績:")
            for bt, ts in sorted(type_stats.items()):
                hit_rate = ts["hits"] / ts["count"] * 100 if ts["count"] > 0 else 0
                roi = ts["return"] / ts["invest"] * 100 if ts["invest"] > 0 else 0
                print(f"    {bt:<6s}: {ts['hits']}/{ts['count']} ({hit_rate:.0f}%) "
                      f"投資¥{ts['invest']:,} 回収¥{ts['return']:,.0f} ROI {roi:.0f}%")

    # Overall across both days
    all_pred_races = {k: v for k, v in all_results.items() if v["has_prediction"] and v.get("analysis")}
    if all_pred_races:
        print(f"\n\n{'='*70}")
        print(f"  全日程 総合成績 ({len(all_pred_races)} レース)")
        print(f"{'='*70}")

        total_bets_all = 0
        total_hits_all = 0
        total_invest_all = 0
        total_return_all = 0.0
        honmei_total_all = 0
        honmei_win_all = 0
        honmei_rentai_all = 0

        for rid, rd in all_pred_races.items():
            a = rd["analysis"]
            if a.get("honmei_hn"):
                honmei_total_all += 1
                if a["honmei_win"]:
                    honmei_win_all += 1
                if a["honmei_rentai"]:
                    honmei_rentai_all += 1

            for br in a.get("bet_results", []):
                total_bets_all += 1
                total_invest_all += br["investment"]
                if br["hit"]:
                    total_hits_all += 1
                    total_return_all += br["payout"]

        if honmei_total_all > 0:
            print(f"  ◎勝率: {honmei_win_all}/{honmei_total_all} = {honmei_win_all/honmei_total_all*100:.1f}%")
            print(f"  ◎連対率: {honmei_rentai_all}/{honmei_total_all} = {honmei_rentai_all/honmei_total_all*100:.1f}%")
        print(f"  買い目: {total_hits_all}/{total_bets_all} 的中")
        print(f"  投資: ¥{total_invest_all:,} / 回収: ¥{total_return_all:,.0f}")
        if total_invest_all > 0:
            print(f"  回収率: {total_return_all/total_invest_all*100:.1f}%")
            print(f"  収支: ¥{total_return_all - total_invest_all:+,.0f}")


if __name__ == "__main__":
    main()
