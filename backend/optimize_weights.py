"""Optimize analytical factor weights to maximize prediction accuracy.

Tests many weight combinations against actual 3/28 + 3/29 race results.
Goal: Find the weight distribution that maximizes win rate WITHOUT relying on market odds.

Strategy:
- Market weight should be LOW (AI should predict independently)
- Test many analytical weight distributions
- Evaluate on: 単勝, 複勝, ワイド, 馬連, 3連複 hit rates
- Combined metric: weighted sum of hit rates
"""
import requests
import time
import itertools
import random
import json
from bs4 import BeautifulSoup

API_BASE = "http://localhost:8000/api"
NETKEIBA_BASE = "https://race.netkeiba.com/race"

# Import factor calculators directly
import sys
sys.path.insert(0, "/Users/atsushi.furutani/Claude Code/jra-prediction-app")
from backend.predictor.factors import (
    calc_market_score, calc_course_affinity, calc_distance_aptitude,
    calc_age_and_sex, calc_weight_carried, calc_jockey_ability,
    calc_trainer_ability, calc_horse_weight_change, calc_past_performance,
    calc_track_condition_affinity, calc_track_direction, calc_track_specific,
)


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
            pos_text = tds[0].get_text(strip=True)
            try:
                finish_pos = int(pos_text)
            except:
                continue
            try:
                horse_num = int(tds[2].get_text(strip=True))
            except:
                continue
            results[horse_num] = finish_pos
        return results
    except:
        return {}


def compute_scores(race_data, entries, analytical_weights, market_weight, analytical_weight_total):
    """Compute prediction scores with given weights."""
    surface = race_data.get("surface", "芝")
    distance = race_data.get("distance", 2000)
    head_count = len([e for e in entries if not e.get("isScratched")])
    all_weights_list = [e.get("weightCarried", 0) for e in entries if not e.get("isScratched")]
    track_condition = race_data.get("trackCondition", "")
    course_detail = race_data.get("courseDetail", "")
    racecourse_code = race_data.get("racecourseCode", "")

    scored = []
    for entry in entries:
        if entry.get("isScratched"):
            continue

        sire = entry.get("sireName", "")
        jockey = entry.get("jockeyName", "")
        trainer = entry.get("trainerName", "")
        age_str = entry.get("age", "")
        weight = entry.get("weightCarried", 0)
        odds = entry.get("odds")
        popularity = entry.get("popularity")
        horse_weight = entry.get("horseWeight", "")
        past_races = entry.get("pastRaces", [])

        factors = {
            "marketScore": calc_market_score(odds, popularity, head_count),
            "pastPerformance": calc_past_performance(past_races),
            "jockeyAbility": calc_jockey_ability(jockey),
            "courseAffinity": calc_course_affinity(sire, surface),
            "distanceAptitude": calc_distance_aptitude(sire, distance),
            "trainerAbility": calc_trainer_ability(trainer),
            "trackCondition": calc_track_condition_affinity(sire, track_condition),
            "trackDirection": calc_track_direction(past_races, course_detail),
            "trackSpecific": calc_track_specific(past_races, racecourse_code),
            "ageAndSex": calc_age_and_sex(age_str),
            "weightCarried": calc_weight_carried(weight, all_weights_list),
            "horseWeightChange": calc_horse_weight_change(horse_weight),
        }

        # Analytical score
        analytical = sum(
            factors[k] * analytical_weights.get(k, 0)
            for k in analytical_weights
        )

        market = factors["marketScore"]

        # Final score: blend
        final_score = market * market_weight + analytical * analytical_weight_total

        scored.append({
            "horseNumber": entry["horseNumber"],
            "score": final_score,
        })

    scored.sort(key=lambda x: -x["score"])
    return scored


def evaluate(scored, actual_results):
    """Evaluate prediction against actual results."""
    if not actual_results or len(scored) < 3:
        return None

    sorted_actual = sorted(actual_results.items(), key=lambda x: x[1])
    winner = sorted_actual[0][0]
    top2 = set(h for h, _ in sorted_actual[:2])
    top3 = set(h for h, _ in sorted_actual[:3])

    ai_top1 = scored[0]["horseNumber"]
    ai_top2 = set(s["horseNumber"] for s in scored[:2])
    ai_top3 = set(s["horseNumber"] for s in scored[:3])
    ai_top5 = set(s["horseNumber"] for s in scored[:5])
    ai_top6 = set(s["horseNumber"] for s in scored[:6])

    return {
        "tansho": ai_top1 == winner,
        "fukusho": len(ai_top3 & top3) >= 1,
        "umaren": ai_top2 == top2,
        "wide": len(ai_top3 & top3) >= 2,
        "sanrenpuku": ai_top3 == top3,
        "winner_in_top3": winner in ai_top3,
        "winner_in_top6": winner in ai_top6,
        "top3_coverage": len(ai_top6 & top3),
    }


def collect_race_data():
    """Collect all race data and actual results for optimization."""
    print("Collecting race data for 3/28 and 3/29...")
    all_races = []

    for date in ["20260328", "20260329"]:
        resp = requests.get(f"{API_BASE}/race-list?date={date}")
        schedules = resp.json()

        for schedule in schedules:
            for race in schedule.get("races", []):
                race_id = race.get("race_id", race.get("raceId", ""))
                race_num = race.get("race_number", race.get("raceNumber", 0))

                # Get race card data
                try:
                    card_resp = requests.get(f"{API_BASE}/racecard/{race_id}", timeout=30)
                    card_data = card_resp.json()
                except:
                    continue
                time.sleep(0.5)

                # Get actual results
                actual = fetch_actual_results(race_id)
                if not actual:
                    time.sleep(0.5)
                    continue
                time.sleep(0.5)

                all_races.append({
                    "race_id": race_id,
                    "race_num": race_num,
                    "date": date,
                    "course": schedule["name"],
                    "race_info": card_data.get("raceInfo", {}),
                    "entries": card_data.get("entries", []),
                    "actual": actual,
                })
                print(f"  Collected: {schedule['name']} {race_num}R ({date})")

    print(f"\nTotal races collected: {len(all_races)}")
    return all_races


def test_weights(all_races, analytical_weights, market_weight, analytical_weight_total):
    """Test a specific weight configuration across all races."""
    totals = {
        "races": 0, "tansho": 0, "fukusho": 0, "umaren": 0,
        "wide": 0, "sanrenpuku": 0, "winner_in_top3": 0,
        "winner_in_top6": 0, "top3_coverage": 0,
    }

    for race in all_races:
        scored = compute_scores(
            race["race_info"], race["entries"],
            analytical_weights, market_weight, analytical_weight_total
        )
        result = evaluate(scored, race["actual"])
        if result is None:
            continue

        totals["races"] += 1
        for key in ["tansho", "fukusho", "umaren", "wide", "sanrenpuku", "winner_in_top3", "winner_in_top6"]:
            if result[key]:
                totals[key] += 1
        totals["top3_coverage"] += result["top3_coverage"]

    return totals


def composite_score(totals):
    """Compute a composite score that values practical betting outcomes.

    Weights: 単勝(25%) + ワイド(25%) + 馬連(20%) + 3連複(15%) + 複勝(10%) + Top3率(5%)
    """
    n = max(totals["races"], 1)
    return (
        (totals["tansho"] / n) * 0.25 +
        (totals["wide"] / n) * 0.25 +
        (totals["umaren"] / n) * 0.20 +
        (totals["sanrenpuku"] / n) * 0.15 +
        (totals["fukusho"] / n) * 0.10 +
        (totals["winner_in_top3"] / n) * 0.05
    )


def main():
    # Collect data
    all_races = collect_race_data()
    if len(all_races) < 10:
        print("Not enough races for optimization")
        return

    # Define factor keys for analytical weights
    factor_keys = [
        "pastPerformance", "jockeyAbility", "courseAffinity", "distanceAptitude",
        "trainerAbility", "trackCondition", "trackDirection", "trackSpecific",
        "ageAndSex", "weightCarried", "horseWeightChange",
    ]

    # =========================================================================
    # Phase 1: Test market weight levels (0%, 10%, 15%, 20%, 25%, 30%)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 1: Finding optimal market vs analytical balance")
    print("=" * 70)

    # Current analytical weights (normalized to sum to 1.0)
    base_analytical = {
        "pastPerformance": 0.22,
        "jockeyAbility": 0.18,
        "courseAffinity": 0.12,
        "distanceAptitude": 0.10,
        "trainerAbility": 0.10,
        "trackCondition": 0.08,
        "trackDirection": 0.06,
        "trackSpecific": 0.06,
        "ageAndSex": 0.04,
        "weightCarried": 0.02,
        "horseWeightChange": 0.02,
    }

    best_market_weight = 0
    best_score = -1
    market_results = {}

    for mw_pct in [0, 5, 10, 15, 20, 25, 30, 35, 40]:
        mw = mw_pct / 100.0
        aw = 1.0 - mw
        totals = test_weights(all_races, base_analytical, mw, aw)
        score = composite_score(totals)
        n = totals["races"]

        market_results[mw_pct] = totals

        print(f"\n  Market={mw_pct}% Analytical={100-mw_pct}%:")
        print(f"    単勝: {totals['tansho']}/{n} ({totals['tansho']/n*100:.1f}%) "
              f"複勝: {totals['fukusho']}/{n} ({totals['fukusho']/n*100:.1f}%) "
              f"馬連: {totals['umaren']}/{n} ({totals['umaren']/n*100:.1f}%)")
        print(f"    ワイド: {totals['wide']}/{n} ({totals['wide']/n*100:.1f}%) "
              f"3連複: {totals['sanrenpuku']}/{n} ({totals['sanrenpuku']/n*100:.1f}%) "
              f"Top3: {totals['winner_in_top3']}/{n} ({totals['winner_in_top3']/n*100:.1f}%)")
        print(f"    Composite: {score:.4f}")

        if score > best_score:
            best_score = score
            best_market_weight = mw_pct

    print(f"\n  >>> Best market weight: {best_market_weight}%")

    # =========================================================================
    # Phase 2: Optimize individual analytical factor weights
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 2: Optimizing individual factor weights")
    print("=" * 70)

    mw = best_market_weight / 100.0
    aw = 1.0 - mw

    # Test many random weight distributions
    best_weights = dict(base_analytical)
    best_composite = composite_score(test_weights(all_races, best_weights, mw, aw))
    print(f"\n  Baseline composite: {best_composite:.4f}")

    # Random search with constraints
    random.seed(42)
    num_trials = 500

    for trial in range(num_trials):
        # Generate random weights
        raw = {}
        for k in factor_keys:
            # Use base weight as center, with some random variation
            base = base_analytical[k]
            raw[k] = max(0.01, base + random.gauss(0, base * 0.5))

        # Normalize to sum to 1.0
        total = sum(raw.values())
        weights = {k: v / total for k, v in raw.items()}

        totals = test_weights(all_races, weights, mw, aw)
        score = composite_score(totals)

        if score > best_composite:
            best_composite = score
            best_weights = dict(weights)
            n = totals["races"]
            print(f"\n  Trial {trial}: NEW BEST composite={score:.4f}")
            print(f"    単勝: {totals['tansho']}/{n} ({totals['tansho']/n*100:.1f}%) "
                  f"馬連: {totals['umaren']}/{n} ({totals['umaren']/n*100:.1f}%) "
                  f"ワイド: {totals['wide']}/{n} ({totals['wide']/n*100:.1f}%) "
                  f"3連複: {totals['sanrenpuku']}/{n} ({totals['sanrenpuku']/n*100:.1f}%)")

    # =========================================================================
    # Phase 3: Fine-tune around the best found weights
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 3: Fine-tuning best weights")
    print("=" * 70)

    for trial in range(300):
        raw = {}
        for k in factor_keys:
            base = best_weights[k]
            raw[k] = max(0.01, base + random.gauss(0, base * 0.2))

        total = sum(raw.values())
        weights = {k: v / total for k, v in raw.items()}

        totals = test_weights(all_races, weights, mw, aw)
        score = composite_score(totals)

        if score > best_composite:
            best_composite = score
            best_weights = dict(weights)
            n = totals["races"]
            print(f"\n  Fine-tune {trial}: NEW BEST composite={score:.4f}")
            print(f"    単勝: {totals['tansho']}/{n} ({totals['tansho']/n*100:.1f}%) "
                  f"馬連: {totals['umaren']}/{n} ({totals['umaren']/n*100:.1f}%) "
                  f"ワイド: {totals['wide']}/{n} ({totals['wide']/n*100:.1f}%) "
                  f"3連複: {totals['sanrenpuku']}/{n} ({totals['sanrenpuku']/n*100:.1f}%)")

    # =========================================================================
    # Final Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL OPTIMAL WEIGHTS")
    print("=" * 70)

    print(f"\nMarket Weight: {best_market_weight}%")
    print(f"Analytical Weight: {100 - best_market_weight}%")
    print(f"\nAnalytical Factor Breakdown (sum = 1.0):")

    # Sort by weight descending
    sorted_weights = sorted(best_weights.items(), key=lambda x: -x[1])
    for k, v in sorted_weights:
        print(f"  {k:25s}: {v:.4f} ({v*100:.1f}%)")

    # Final test
    totals = test_weights(all_races, best_weights, mw, aw)
    n = totals["races"]
    print(f"\nFinal Performance ({n} races):")
    print(f"  単勝的中率: {totals['tansho']}/{n} ({totals['tansho']/n*100:.1f}%)")
    print(f"  複勝的中率: {totals['fukusho']}/{n} ({totals['fukusho']/n*100:.1f}%)")
    print(f"  馬連的中率: {totals['umaren']}/{n} ({totals['umaren']/n*100:.1f}%)")
    print(f"  ワイド的中率: {totals['wide']}/{n} ({totals['wide']/n*100:.1f}%)")
    print(f"  3連複的中率: {totals['sanrenpuku']}/{n} ({totals['sanrenpuku']/n*100:.1f}%)")
    print(f"  勝馬Top3率: {totals['winner_in_top3']}/{n} ({totals['winner_in_top3']/n*100:.1f}%)")
    print(f"  勝馬Top6率: {totals['winner_in_top6']}/{n} ({totals['winner_in_top6']/n*100:.1f}%)")
    print(f"  上位3頭カバー率: {totals['top3_coverage']}/{n*3} ({totals['top3_coverage']/(n*3)*100:.1f}%)")

    # Output as Python dict for easy copy
    print("\n\n# Python code to paste into scoring.py:")
    print(f"MARKET_WEIGHT = {mw}")
    print(f"ANALYTICAL_WEIGHT = {aw}")
    print("ANALYTICAL_WEIGHTS = {")
    for k, v in sorted_weights:
        print(f'    "{k}": {v:.4f},')
    print("}")


if __name__ == "__main__":
    main()
