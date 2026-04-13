"""Full validation script: Check AI prediction accuracy against actual results.

Tests all races on 3/28(Sat) and 3/29(Sun) at 中山・阪神・中京.
"""
import requests
import time
import re
from bs4 import BeautifulSoup

API_BASE = "http://localhost:8000/api"
NETKEIBA_BASE = "https://race.netkeiba.com/race"

DATES = ["20260328", "20260329"]
DATE_LABELS = {"20260328": "3/28(土)", "20260329": "3/29(日)"}

def fetch_actual_results(race_id: str) -> dict:
    """Fetch actual race results from netkeiba result page."""
    url = f"{NETKEIBA_BASE}/result.html?race_id={race_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")

        results = {}
        table = soup.select_one("table.RaceTable01")
        if not table:
            return {}

        rows = table.select("tr.HorseList")
        for rank_idx, row in enumerate(rows):
            tds = row.select("td")
            if len(tds) < 3:
                continue

            # Finish position
            finish_pos = rank_idx + 1
            pos_text = tds[0].get_text(strip=True)
            try:
                finish_pos = int(pos_text)
            except:
                continue  # Skip scratched/DNF

            # Horse number
            try:
                horse_num = int(tds[2].get_text(strip=True))
            except:
                continue

            results[horse_num] = finish_pos

        return results
    except Exception as e:
        print(f"  [ERROR] Failed to fetch results for {race_id}: {e}")
        return {}


def check_predictions(predictions, actual_results):
    """Check prediction accuracy against actual results."""
    if not actual_results:
        return None

    # Get ranked predictions (non-zero score)
    ranked = sorted([p for p in predictions if p["score"] > 0], key=lambda x: -x["score"])
    if not ranked:
        return None

    # Get actual top 3
    sorted_actual = sorted(actual_results.items(), key=lambda x: x[1])
    winner = sorted_actual[0][0] if sorted_actual else None
    top2 = [h for h, _ in sorted_actual[:2]]
    top3 = [h for h, _ in sorted_actual[:3]]

    # AI predictions
    ai_top1 = ranked[0]["horseNumber"] if len(ranked) > 0 else None
    ai_top2 = [r["horseNumber"] for r in ranked[:2]]
    ai_top3 = [r["horseNumber"] for r in ranked[:3]]
    ai_top5 = [r["horseNumber"] for r in ranked[:5]]
    ai_top6 = [r["horseNumber"] for r in ranked[:6]]  # ◎◯▲▲△△

    # Marks
    marks = {}
    for r in ranked[:6]:
        marks[r["horseNumber"]] = r.get("mark", "")

    result = {
        "winner": winner,
        "actual_top3": top3,
        "ai_top1": ai_top1,
        "ai_top6": ai_top6,
        "marks": marks,
        # 単勝: AI ◎ = actual winner
        "tansho_hit": ai_top1 == winner,
        # 複勝: any of AI top 3 in actual top 3
        "fukusho_hit": any(h in top3 for h in ai_top3),
        # 馬連: AI top2 both in actual top2
        "umaren_hit": set(ai_top2) == set(top2),
        # ワイド: any 2 of AI top3 in actual top3
        "wide_hit": len(set(ai_top3) & set(top3)) >= 2,
        # 3連複: AI top3 = actual top3 (any order)
        "sanrenpuku_hit": set(ai_top3) == set(top3),
        # 3連単: AI top3 = actual top3 (same order)
        "sanrentan_hit": ai_top3 == top3,
        # Broader checks
        "winner_in_top3": winner in [r["horseNumber"] for r in ranked[:3]],
        "winner_in_top5": winner in ai_top5,
        "winner_in_top6": winner in ai_top6,
        # How many of actual top3 are in AI top6 (marked horses)
        "top3_in_marked": len(set(top3) & set(ai_top6)),
    }
    return result


def main():
    all_results = {}
    grand_totals = {
        "races": 0, "tansho": 0, "fukusho": 0, "umaren": 0,
        "wide": 0, "sanrenpuku": 0, "sanrentan": 0,
        "winner_in_top3": 0, "winner_in_top5": 0, "winner_in_top6": 0,
        "top3_coverage": 0,
    }

    for date in DATES:
        print(f"\n{'='*70}")
        print(f"  {DATE_LABELS[date]} 検証開始")
        print(f"{'='*70}")

        # Fetch race list
        resp = requests.get(f"{API_BASE}/race-list?date={date}")
        schedules = resp.json()

        day_totals = {
            "races": 0, "tansho": 0, "fukusho": 0, "umaren": 0,
            "wide": 0, "sanrenpuku": 0, "sanrentan": 0,
            "winner_in_top3": 0, "winner_in_top5": 0, "winner_in_top6": 0,
            "top3_coverage": 0,
        }

        for schedule in schedules:
            course_name = schedule["name"]
            print(f"\n--- {course_name} ---")

            course_totals = {
                "races": 0, "tansho": 0, "fukusho": 0, "umaren": 0,
                "wide": 0, "sanrenpuku": 0, "sanrentan": 0,
                "winner_in_top3": 0, "winner_in_top5": 0, "winner_in_top6": 0,
                "top3_coverage": 0,
            }

            for race in schedule.get("races", []):
                race_id = race.get("race_id", race.get("raceId", ""))
                race_num = race.get("race_number", race.get("raceNumber", 0))
                race_name = race.get("race_name", race.get("raceName", ""))

                print(f"\n  {race_num}R {race_name} (ID: {race_id})")

                # Get AI predictions
                try:
                    card_resp = requests.get(f"{API_BASE}/racecard/{race_id}", timeout=30)
                    card_data = card_resp.json()
                    predictions = card_data.get("predictions", [])
                except Exception as e:
                    print(f"    [SKIP] AI prediction failed: {e}")
                    continue

                time.sleep(1)

                # Get actual results
                actual = fetch_actual_results(race_id)
                if not actual:
                    print(f"    [SKIP] No results available")
                    time.sleep(1)
                    continue

                time.sleep(1)

                # Check
                check = check_predictions(predictions, actual)
                if not check:
                    print(f"    [SKIP] Could not check predictions")
                    continue

                # Display
                mark_str = " ".join([f"{hn}{check['marks'].get(hn,'')}" for hn in check['ai_top6']])
                actual_str = " ".join([f"{h}({p}着)" for h, p in sorted(actual.items(), key=lambda x: x[1])[:3]])

                print(f"    AI予想: {mark_str}")
                print(f"    実結果: {actual_str}")
                print(f"    単勝{'◎' if check['tansho_hit'] else '×'} "
                      f"複勝{'◎' if check['fukusho_hit'] else '×'} "
                      f"馬連{'◎' if check['umaren_hit'] else '×'} "
                      f"ワイド{'◎' if check['wide_hit'] else '×'} "
                      f"3連複{'◎' if check['sanrenpuku_hit'] else '×'} "
                      f"3連単{'◎' if check['sanrentan_hit'] else '×'}")
                print(f"    勝馬 in Top3: {'○' if check['winner_in_top3'] else '×'} "
                      f"Top5: {'○' if check['winner_in_top5'] else '×'} "
                      f"Top6: {'○' if check['winner_in_top6'] else '×'} "
                      f"| 上位3頭カバー: {check['top3_in_marked']}/3")

                # Tally
                course_totals["races"] += 1
                for key in ["tansho", "fukusho", "umaren", "wide", "sanrenpuku", "sanrentan"]:
                    if check[f"{key}_hit"]:
                        course_totals[key] += 1
                if check["winner_in_top3"]: course_totals["winner_in_top3"] += 1
                if check["winner_in_top5"]: course_totals["winner_in_top5"] += 1
                if check["winner_in_top6"]: course_totals["winner_in_top6"] += 1
                course_totals["top3_coverage"] += check["top3_in_marked"]

            # Course summary
            n = course_totals["races"]
            if n > 0:
                print(f"\n  [{course_name} 集計] {n}レース")
                print(f"    単勝的中: {course_totals['tansho']}/{n} ({course_totals['tansho']/n*100:.0f}%)")
                print(f"    複勝的中: {course_totals['fukusho']}/{n} ({course_totals['fukusho']/n*100:.0f}%)")
                print(f"    馬連的中: {course_totals['umaren']}/{n} ({course_totals['umaren']/n*100:.0f}%)")
                print(f"    ワイド的中: {course_totals['wide']}/{n} ({course_totals['wide']/n*100:.0f}%)")
                print(f"    3連複的中: {course_totals['sanrenpuku']}/{n} ({course_totals['sanrenpuku']/n*100:.0f}%)")
                print(f"    3連単的中: {course_totals['sanrentan']}/{n} ({course_totals['sanrentan']/n*100:.0f}%)")
                print(f"    勝馬Top3率: {course_totals['winner_in_top3']}/{n} ({course_totals['winner_in_top3']/n*100:.0f}%)")
                print(f"    勝馬Top6率: {course_totals['winner_in_top6']}/{n} ({course_totals['winner_in_top6']/n*100:.0f}%)")
                print(f"    上位3頭カバー率: {course_totals['top3_coverage']}/{n*3} ({course_totals['top3_coverage']/(n*3)*100:.0f}%)")

            # Add to day totals
            for key in day_totals:
                day_totals[key] += course_totals[key]

        # Day summary
        n = day_totals["races"]
        if n > 0:
            print(f"\n{'='*70}")
            print(f"  {DATE_LABELS[date]} 総合結果 ({n}レース)")
            print(f"{'='*70}")
            print(f"    単勝的中率: {day_totals['tansho']}/{n} ({day_totals['tansho']/n*100:.1f}%)")
            print(f"    複勝的中率: {day_totals['fukusho']}/{n} ({day_totals['fukusho']/n*100:.1f}%)")
            print(f"    馬連的中率: {day_totals['umaren']}/{n} ({day_totals['umaren']/n*100:.1f}%)")
            print(f"    ワイド的中率: {day_totals['wide']}/{n} ({day_totals['wide']/n*100:.1f}%)")
            print(f"    3連複的中率: {day_totals['sanrenpuku']}/{n} ({day_totals['sanrenpuku']/n*100:.1f}%)")
            print(f"    3連単的中率: {day_totals['sanrentan']}/{n} ({day_totals['sanrentan']/n*100:.1f}%)")
            print(f"    勝馬Top3率: {day_totals['winner_in_top3']}/{n} ({day_totals['winner_in_top3']/n*100:.1f}%)")
            print(f"    勝馬Top6率: {day_totals['winner_in_top6']}/{n} ({day_totals['winner_in_top6']/n*100:.1f}%)")
            print(f"    上位3頭カバー率: {day_totals['top3_coverage']}/{n*3} ({day_totals['top3_coverage']/(n*3)*100:.1f}%)")

        # Add to grand totals
        for key in grand_totals:
            grand_totals[key] += day_totals[key]

    # Grand summary
    n = grand_totals["races"]
    if n > 0:
        print(f"\n{'='*70}")
        print(f"  全体総合結果 ({n}レース)")
        print(f"{'='*70}")
        print(f"    単勝的中率: {grand_totals['tansho']}/{n} ({grand_totals['tansho']/n*100:.1f}%)")
        print(f"    複勝的中率: {grand_totals['fukusho']}/{n} ({grand_totals['fukusho']/n*100:.1f}%)")
        print(f"    馬連的中率: {grand_totals['umaren']}/{n} ({grand_totals['umaren']/n*100:.1f}%)")
        print(f"    ワイド的中率: {grand_totals['wide']}/{n} ({grand_totals['wide']/n*100:.1f}%)")
        print(f"    3連複的中率: {grand_totals['sanrenpuku']}/{n} ({grand_totals['sanrenpuku']/n*100:.1f}%)")
        print(f"    3連単的中率: {grand_totals['sanrentan']}/{n} ({grand_totals['sanrentan']/n*100:.1f}%)")
        print(f"    勝馬Top3率: {grand_totals['winner_in_top3']}/{n} ({grand_totals['winner_in_top3']/n*100:.1f}%)")
        print(f"    勝馬Top6率: {grand_totals['winner_in_top6']}/{n} ({grand_totals['winner_in_top6']/n*100:.1f}%)")
        print(f"    上位3頭カバー率: {grand_totals['top3_coverage']}/{n*3} ({grand_totals['top3_coverage']/(n*3)*100:.1f}%)")

    print("\n完了!")


if __name__ == "__main__":
    main()
