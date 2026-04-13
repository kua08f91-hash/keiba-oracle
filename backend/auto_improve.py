"""Automatic model improvement pipeline.

Collects recent race results, evaluates current model performance,
retrains if accuracy drops, and logs all metrics for tracking.

Usage:
    /usr/bin/python3 -m backend.auto_improve

Recommended: Run via cron every Monday at 6:00
    0 6 * * 1 cd "/Users/atsushi.furutani/Claude Code/jra-prediction-app" && /usr/bin/python3 -m backend.auto_improve >> /tmp/keiba-improve.log 2>&1
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import re
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from bs4 import BeautifulSoup

from backend.database.db import init_db
from backend.scraper.netkeiba import fetch_race_list, fetch_race_card
from backend.scraper.odds import estimate_from_entries, fetch_combination_odds
from backend.predictor.ml_scoring import MLScoringModel
from backend.predictor.bet_optimizer import optimize_bets, scores_to_probabilities, detect_race_pattern

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST_FILE = os.path.join(BASE_DIR, "data", "historical_races.json")
PERF_LOG = os.path.join(BASE_DIR, "data", "performance_log.json")
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
COURSE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}


def fetch_result_positions(race_id: str) -> dict:
    """Fetch finish positions from db.netkeiba.com result page."""
    url = f"https://db.netkeiba.com/race/{race_id}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = resp.apparent_encoding or "UTF-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        results = {}
        rows = soup.select("tr.HorseList")
        for row in rows:
            tds = row.select("td")
            if len(tds) < 3:
                continue
            # Position is usually first td
            pos_text = tds[0].get_text(strip=True)
            pos_match = re.match(r"(\d+)", pos_text)
            if not pos_match:
                continue
            pos = int(pos_match.group(1))
            # Horse number
            for td in tds:
                cls = td.get("class", [])
                if "Num" in cls and "Txt_C" in cls:
                    try:
                        hn = int(td.get_text(strip=True))
                        results[str(hn)] = pos
                    except ValueError:
                        pass
                    break
        return results
    except Exception as e:
        logging.warning("fetch_result_positions failed for %s: %s", race_id, e)
        return {}


def collect_recent_results(weeks_back: int = 2) -> list:
    """Collect race results from recent weekends and add to historical data."""
    today = datetime.now()
    new_races = []

    for w in range(weeks_back):
        # Find Saturday and Sunday
        days_since_sat = (today.weekday() - 5) % 7
        sat = today - timedelta(days=days_since_sat + 7 * w)
        sun = sat + timedelta(days=1)

        for dt in [sat, sun]:
            date_str = dt.strftime("%Y%m%d")
            print(f"  Checking {date_str}...")
            time.sleep(2)

            schedules = fetch_race_list(date_str)
            if not schedules:
                continue

            for schedule in schedules:
                for race in schedule.get("races", []):
                    race_id = race.get("race_id", "")
                    if not race_id:
                        continue

                    # Fetch race card
                    data = fetch_race_card(race_id)
                    if not data:
                        continue

                    # Fetch results
                    time.sleep(1)
                    results = fetch_result_positions(race_id)
                    if not results:
                        continue

                    new_races.append({
                        "race_id": race_id,
                        "date": date_str,
                        "race_info": data["race_info"],
                        "entries": data["entries"],
                        "results": results,
                    })
                    print(f"    + {COURSE_MAP.get(race_id[4:6], '??')} {race.get('race_number', '?')}R: {len(results)} results")

    return new_races


def update_historical_data(new_races: list) -> int:
    """Add new races to historical_races.json, avoiding duplicates."""
    if os.path.exists(HIST_FILE):
        with open(HIST_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []

    existing_ids = {r.get("race_id") for r in existing}
    added = 0

    for race in new_races:
        if race["race_id"] not in existing_ids:
            # Clean entries for storage (remove large fields)
            clean_entries = []
            for e in race["entries"]:
                clean_entries.append({
                    "horseNumber": e.get("horseNumber"),
                    "horseName": e.get("horseName", ""),
                    "age": e.get("age", ""),
                    "weightCarried": e.get("weightCarried", 0),
                    "jockeyName": e.get("jockeyName", ""),
                    "trainerName": e.get("trainerName", ""),
                    "odds": e.get("odds"),
                    "popularity": e.get("popularity"),
                    "horseWeight": e.get("horseWeight", ""),
                    "sireName": e.get("sireName", ""),
                    "damName": e.get("damName", ""),
                    "pastRaces": e.get("pastRaces", []),
                    "isScratched": e.get("isScratched", False),
                })

            existing.append({
                "race_id": race["race_id"],
                "date": race["date"],
                "race_info": race["race_info"],
                "entries": clean_entries,
                "results": race["results"],
            })
            added += 1

    if added > 0:
        with open(HIST_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False)

    return added


def evaluate_current_model(new_races: list) -> dict:
    """Evaluate current model on recent races."""
    predictor = MLScoringModel()
    total = 0
    tansho = 0
    umaren = 0
    wide = 0
    sanrenpuku = 0
    w_top3 = 0

    for race in new_races:
        entries = race["entries"]
        race_info = race["race_info"]
        results = race["results"]

        if len(entries) < 3 or not results:
            continue

        predictions = predictor.predict(race_info, entries)
        if len(predictions) < 3:
            continue

        ranked = sorted(
            [p for p in predictions if p["score"] > 0],
            key=lambda p: -p["score"]
        )
        if len(ranked) < 3:
            continue

        total += 1
        ai_1 = ranked[0]["horseNumber"]
        ai_2 = {ranked[0]["horseNumber"], ranked[1]["horseNumber"]}
        ai_3 = {h["horseNumber"] for h in ranked[:3]}

        actual_1st = [int(hn) for hn, pos in results.items() if pos == 1]
        actual_top2 = {int(hn) for hn, pos in results.items() if pos <= 2}
        actual_top3 = {int(hn) for hn, pos in results.items() if pos <= 3}

        if not actual_1st:
            total -= 1
            continue

        winner = actual_1st[0]
        if ai_1 == winner:
            tansho += 1
        if ai_2 == actual_top2:
            umaren += 1
        if len(ai_3 & actual_top3) >= 2:
            wide += 1
        if ai_3 == actual_top3:
            sanrenpuku += 1
        if winner in ai_3:
            w_top3 += 1

    if total == 0:
        return {}

    metrics = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "races": total,
        "tansho": round(tansho / total * 100, 1),
        "umaren": round(umaren / total * 100, 1),
        "wide": round(wide / total * 100, 1),
        "sanrenpuku": round(sanrenpuku / total * 100, 1),
        "winner_in_top3": round(w_top3 / total * 100, 1),
    }
    return metrics


def log_performance(metrics: dict):
    """Append metrics to performance log."""
    if os.path.exists(PERF_LOG):
        with open(PERF_LOG, "r", encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = []

    log.append(metrics)

    with open(PERF_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def should_retrain(metrics: dict) -> bool:
    """Determine if model should be retrained based on performance drop."""
    if not metrics:
        return False

    # Retrain if win accuracy drops below 25% or wide below 40%
    if metrics.get("tansho", 0) < 25:
        print(f"  !! 単勝精度 {metrics['tansho']}% < 25% threshold")
        return True
    if metrics.get("wide", 0) < 40:
        print(f"  !! ワイド精度 {metrics['wide']}% < 40% threshold")
        return True

    # Check trend: if last 3 evaluations all declining
    if os.path.exists(PERF_LOG):
        with open(PERF_LOG, "r", encoding="utf-8") as f:
            log = json.load(f)
        if len(log) >= 3:
            recent = log[-3:]
            tansho_trend = [r.get("tansho", 0) for r in recent]
            if tansho_trend == sorted(tansho_trend, reverse=True):
                print(f"  !! 3回連続精度低下: {tansho_trend}")
                return True

    return False


def retrain_model():
    """Retrain the ML model with updated historical data."""
    print("\n  Retraining model...")
    from backend.train_model import main as train_main
    train_main()
    print("  Model retrained successfully")


def main():
    init_db()

    print("=" * 60)
    print(f"  KEIBA ORACLE - Auto Improvement Pipeline")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Step 1: Collect recent race results
    print("\n[1/4] Collecting recent race results...")
    new_races = collect_recent_results(weeks_back=2)
    print(f"  Collected {len(new_races)} races with results")

    if not new_races:
        print("  No new races to process. Done.")
        return

    # Step 2: Add to historical data
    print("\n[2/4] Updating historical data...")
    added = update_historical_data(new_races)
    total = 0
    if os.path.exists(HIST_FILE):
        with open(HIST_FILE, "r", encoding="utf-8") as f:
            total = len(json.load(f))
    print(f"  Added {added} new races (total: {total})")

    # Step 3: Evaluate current model
    print("\n[3/4] Evaluating current model on recent races...")
    metrics = evaluate_current_model(new_races)
    if metrics:
        print(f"  Races evaluated: {metrics['races']}")
        print(f"  単勝: {metrics['tansho']}%")
        print(f"  馬連: {metrics['umaren']}%")
        print(f"  ワイド: {metrics['wide']}%")
        print(f"  3連複: {metrics['sanrenpuku']}%")
        print(f"  Winner in top-3: {metrics['winner_in_top3']}%")
        log_performance(metrics)
    else:
        print("  Could not evaluate (insufficient data)")

    # Step 4: Retrain if needed
    print("\n[4/4] Checking if retraining is needed...")
    if should_retrain(metrics):
        print("  → Retraining triggered!")
        retrain_model()
    else:
        print("  → Model performance OK, no retraining needed")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Pipeline complete")
    if metrics:
        print(f"  Current accuracy: 単勝{metrics.get('tansho', '?')}% / ワイド{metrics.get('wide', '?')}%")
    print(f"  Historical races: {total}")
    print(f"  Performance log: {PERF_LOG}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
