"""Optimize prediction weights using collected historical data.

Reads historical_races.json and runs weight optimization.
Much faster than live scraping since all data is pre-collected.
"""
import json
import os
import sys
import random
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.predictor.factors import (
    calc_market_score, calc_course_affinity, calc_distance_aptitude,
    calc_age_and_sex, calc_weight_carried, calc_jockey_ability,
    calc_trainer_ability, calc_horse_weight_change, calc_past_performance,
    calc_track_condition_affinity, calc_track_direction, calc_track_specific,
)

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "historical_races.json")

FACTOR_KEYS = [
    "pastPerformance", "jockeyAbility", "courseAffinity", "distanceAptitude",
    "trainerAbility", "trackCondition", "trackDirection", "trackSpecific",
    "ageAndSex", "weightCarried", "horseWeightChange",
]


def compute_factors(race_info, entries):
    """Compute all factors for a race's entries."""
    surface = race_info.get("surface", "芝")
    distance = race_info.get("distance", 2000)
    head_count = len([e for e in entries if not e.get("isScratched")])
    all_weights = [e.get("weightCarried", 0) for e in entries if not e.get("isScratched")]
    track_condition = race_info.get("trackCondition", "")
    course_detail = race_info.get("courseDetail", "")
    racecourse_code = race_info.get("racecourseCode", "")

    results = []
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
            "weightCarried": calc_weight_carried(weight, all_weights),
            "horseWeightChange": calc_horse_weight_change(horse_weight),
        }
        results.append({
            "horseNumber": entry["horseNumber"],
            "factors": factors,
        })
    return results


def score_horses(factor_data, analytical_weights, market_weight, analytical_weight_total):
    """Score horses with given weights."""
    scored = []
    for d in factor_data:
        analytical = sum(
            d["factors"][k] * analytical_weights.get(k, 0)
            for k in FACTOR_KEYS
        )
        market = d["factors"]["marketScore"]
        score = market * market_weight + analytical * analytical_weight_total
        scored.append({"horseNumber": d["horseNumber"], "score": score})
    scored.sort(key=lambda x: -x["score"])
    return scored


def evaluate(scored, actual_results):
    """Evaluate predictions against actual results."""
    if not actual_results or len(scored) < 3:
        return None

    sorted_actual = sorted(actual_results.items(), key=lambda x: x[1])
    winner = sorted_actual[0][0]
    top2 = set(h for h, _ in sorted_actual[:2])
    top3 = set(h for h, _ in sorted_actual[:3])

    ai_top1 = scored[0]["horseNumber"]
    ai_top2 = set(s["horseNumber"] for s in scored[:2])
    ai_top3 = set(s["horseNumber"] for s in scored[:3])
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


def composite_score(totals):
    """Composite metric: 単勝(25%) + ワイド(25%) + 馬連(20%) + 3連複(15%) + 複勝(10%) + Top3(5%)"""
    n = max(totals["races"], 1)
    return (
        (totals["tansho"] / n) * 0.25 +
        (totals["wide"] / n) * 0.25 +
        (totals["umaren"] / n) * 0.20 +
        (totals["sanrenpuku"] / n) * 0.15 +
        (totals["fukusho"] / n) * 0.10 +
        (totals["winner_in_top3"] / n) * 0.05
    )


def test_weights(precomputed, analytical_weights, market_weight, analytical_weight_total):
    """Test weights across all precomputed race data."""
    totals = {
        "races": 0, "tansho": 0, "fukusho": 0, "umaren": 0,
        "wide": 0, "sanrenpuku": 0, "winner_in_top3": 0,
        "winner_in_top6": 0, "top3_coverage": 0,
    }

    for factor_data, actual in precomputed:
        scored = score_horses(factor_data, analytical_weights, market_weight, analytical_weight_total)
        result = evaluate(scored, actual)
        if result is None:
            continue
        totals["races"] += 1
        for key in ["tansho", "fukusho", "umaren", "wide", "sanrenpuku", "winner_in_top3", "winner_in_top6"]:
            if result[key]:
                totals[key] += 1
        totals["top3_coverage"] += result["top3_coverage"]

    return totals


def main():
    print("Loading historical data...")
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: {DATA_FILE} not found. Run collect_fast.py first.")
        return

    with open(DATA_FILE, "r") as f:
        all_races = json.load(f)

    print(f"Total races: {len(all_races)}")

    # Precompute factors for all races (this is the slow part, do it once)
    print("Precomputing factors...")
    precomputed = []
    skipped = 0
    for race in all_races:
        actual = race.get("results", {})
        if not actual:
            skipped += 1
            continue
        # Convert string keys to int
        actual_int = {int(k): v for k, v in actual.items()}

        factor_data = compute_factors(race["race_info"], race["entries"])
        if len(factor_data) < 3:
            skipped += 1
            continue

        precomputed.append((factor_data, actual_int))

    print(f"Valid races for optimization: {len(precomputed)} (skipped {skipped})")

    if len(precomputed) < 20:
        print("Not enough data for optimization")
        return

    # Current weights (baseline)
    base_analytical = {
        "courseAffinity": 0.2228,
        "pastPerformance": 0.2048,
        "trackCondition": 0.1413,
        "trackDirection": 0.1398,
        "distanceAptitude": 0.0996,
        "jockeyAbility": 0.0602,
        "ageAndSex": 0.0529,
        "trainerAbility": 0.0336,
        "horseWeightChange": 0.0212,
        "weightCarried": 0.0146,
        "trackSpecific": 0.0093,
    }

    # =========================================================================
    # Phase 1: Market weight sweep
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"Phase 1: Market weight optimization ({len(precomputed)} races)")
    print(f"{'='*70}")

    best_mw = 0
    best_score = -1

    for mw_pct in range(0, 55, 5):
        mw = mw_pct / 100.0
        aw = 1.0 - mw
        totals = test_weights(precomputed, base_analytical, mw, aw)
        score = composite_score(totals)
        n = totals["races"]

        print(f"  Market={mw_pct:2d}%: "
              f"単勝{totals['tansho']:4d}/{n}({totals['tansho']/n*100:5.1f}%) "
              f"馬連{totals['umaren']:4d}/{n}({totals['umaren']/n*100:5.1f}%) "
              f"ワイド{totals['wide']:4d}/{n}({totals['wide']/n*100:5.1f}%) "
              f"3連複{totals['sanrenpuku']:4d}/{n}({totals['sanrenpuku']/n*100:5.1f}%) "
              f"Comp={score:.4f}")

        if score > best_score:
            best_score = score
            best_mw = mw_pct

    print(f"\n  >>> Best market weight: {best_mw}%")

    # =========================================================================
    # Phase 2: Random search for analytical weights
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"Phase 2: Analytical weight optimization (1000 trials)")
    print(f"{'='*70}")

    mw = best_mw / 100.0
    aw = 1.0 - mw

    best_weights = dict(base_analytical)
    best_composite = composite_score(test_weights(precomputed, best_weights, mw, aw))
    print(f"  Baseline composite: {best_composite:.4f}")

    random.seed(42)
    for trial in range(1000):
        raw = {}
        for k in FACTOR_KEYS:
            base = base_analytical[k]
            raw[k] = max(0.005, base + random.gauss(0, base * 0.6))
        total = sum(raw.values())
        weights = {k: v / total for k, v in raw.items()}

        totals = test_weights(precomputed, weights, mw, aw)
        score = composite_score(totals)

        if score > best_composite:
            best_composite = score
            best_weights = dict(weights)
            n = totals["races"]
            print(f"  Trial {trial:4d}: composite={score:.4f} "
                  f"単勝{totals['tansho']/n*100:.1f}% "
                  f"馬連{totals['umaren']/n*100:.1f}% "
                  f"ワイド{totals['wide']/n*100:.1f}% "
                  f"3連複{totals['sanrenpuku']/n*100:.1f}%")

    # =========================================================================
    # Phase 3: Fine-tune
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"Phase 3: Fine-tuning (500 trials)")
    print(f"{'='*70}")

    for trial in range(500):
        raw = {}
        for k in FACTOR_KEYS:
            base = best_weights[k]
            raw[k] = max(0.005, base + random.gauss(0, base * 0.15))
        total = sum(raw.values())
        weights = {k: v / total for k, v in raw.items()}

        totals = test_weights(precomputed, weights, mw, aw)
        score = composite_score(totals)

        if score > best_composite:
            best_composite = score
            best_weights = dict(weights)
            n = totals["races"]
            print(f"  Fine {trial:4d}: composite={score:.4f} "
                  f"単勝{totals['tansho']/n*100:.1f}% "
                  f"馬連{totals['umaren']/n*100:.1f}% "
                  f"ワイド{totals['wide']/n*100:.1f}%")

    # =========================================================================
    # Results
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"FINAL OPTIMAL WEIGHTS ({len(precomputed)} races)")
    print(f"{'='*70}")

    print(f"\nMARKET_WEIGHT = {mw}")
    print(f"ANALYTICAL_WEIGHT = {aw}")
    print(f"\nANALYTICAL_WEIGHTS = {{")
    for k, v in sorted(best_weights.items(), key=lambda x: -x[1]):
        print(f'    "{k}": {v:.4f},  # {v*100:.1f}%')
    print("}")

    # Final performance
    totals = test_weights(precomputed, best_weights, mw, aw)
    n = totals["races"]
    print(f"\nFinal Performance:")
    print(f"  単勝: {totals['tansho']}/{n} ({totals['tansho']/n*100:.1f}%)")
    print(f"  複勝: {totals['fukusho']}/{n} ({totals['fukusho']/n*100:.1f}%)")
    print(f"  馬連: {totals['umaren']}/{n} ({totals['umaren']/n*100:.1f}%)")
    print(f"  ワイド: {totals['wide']}/{n} ({totals['wide']/n*100:.1f}%)")
    print(f"  3連複: {totals['sanrenpuku']}/{n} ({totals['sanrenpuku']/n*100:.1f}%)")
    print(f"  勝馬Top3: {totals['winner_in_top3']}/{n} ({totals['winner_in_top3']/n*100:.1f}%)")
    print(f"  勝馬Top6: {totals['winner_in_top6']}/{n} ({totals['winner_in_top6']/n*100:.1f}%)")


if __name__ == "__main__":
    main()
