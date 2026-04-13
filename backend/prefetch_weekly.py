"""Pre-fetch and cache race data for upcoming weekends.

Runs automatically to ensure race cards, predictions, and odds
are cached before race day for fast page loads.

Usage:
    /usr/bin/python3 -m backend.prefetch_weekly

Recommended: Run via cron every Thursday at 21:00
    0 21 * * 4 cd "/Users/atsushi.furutani/Claude Code/jra-prediction-app" && /usr/bin/python3 -m backend.prefetch_weekly >> /tmp/keiba-prefetch.log 2>&1
"""
from __future__ import annotations

import sys
import os
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database.db import init_db
from backend.scraper.netkeiba import fetch_race_list, fetch_race_card
from backend.scraper.odds import estimate_from_entries, fetch_combination_odds
from backend.predictor.ml_scoring import MLScoringModel
from backend.predictor.bet_optimizer import optimize_bets

COURSE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}


def get_target_dates():
    """Get race dates for last week, this week, and next week (Sat/Sun)."""
    today = datetime.now()
    dates = []
    # Scan from last Monday to next next Sunday
    start = today - timedelta(days=today.weekday() + 7)  # Last week Monday
    end = today + timedelta(days=(6 - today.weekday()) + 7)  # Next week Sunday

    current = start
    while current <= end:
        if current.weekday() in (5, 6):  # Sat, Sun
            dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return dates


def main():
    init_db()
    predictor = MLScoringModel()

    print("=" * 60)
    print(f"  KEIBA ORACLE - Weekly Pre-fetch")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    target_dates = get_target_dates()
    print(f"\nTarget dates: {', '.join(target_dates)}")

    total_races = 0
    total_cached = 0

    for date_str in target_dates:
        print(f"\n--- {date_str} ---")
        time.sleep(2)

        schedules = fetch_race_list(date_str)
        if not schedules:
            print("  No races")
            continue

        race_ids = []
        for s in schedules:
            for r in s.get("races", []):
                rid = r.get("race_id", "")
                if rid:
                    race_ids.append((rid, COURSE_MAP.get(rid[4:6], "??"), r.get("race_number", 0)))

        print(f"  {len(race_ids)} races at {', '.join(s['name'] for s in schedules)}")

        for race_id, course, rnum in race_ids:
            total_races += 1
            try:
                # Fetch race card (triggers scraping + caching)
                data = fetch_race_card(race_id)
                if not data:
                    print(f"  x {course}{rnum:2d}R: failed")
                    continue

                entries = data.get("entries", [])
                race_info = data.get("race_info", {})

                # Generate predictions (triggers ML model)
                predictions = predictor.predict(race_info, entries)

                # Pre-fetch odds
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

                # Generate optimized bets
                optimize_bets(predictions, odds_data, race_info)

                total_cached += 1
                print(f"  + {course}{rnum:2d}R: {len(entries)} entries, {len(predictions)} predictions")

            except Exception as e:
                print(f"  x {course}{rnum:2d}R: {e}")

    print(f"\n{'='*60}")
    print(f"  Complete: {total_cached}/{total_races} races cached")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
