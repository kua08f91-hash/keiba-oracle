"""Race-day live refresh: update predictions 30 min before each race.

On race days (Sat/Sun), runs continuously from 9:30 to 16:30.
For each race starting within the next 30-35 minutes:
  1. Clear cached data for that race
  2. Re-scrape latest odds, horse weight, track condition
  3. Re-generate AI predictions with fresh data
  4. Re-calculate optimized bets

Usage:
    /usr/bin/python3 -m backend.refresh_raceday

Recommended cron (Sat/Sun 9:30-16:30, every 5 minutes):
    */5 9-16 * * 6,0 cd "/Users/atsushi.furutani/Claude Code/jra-prediction-app" && /usr/bin/python3 -m backend.refresh_raceday >> /tmp/keiba-raceday.log 2>&1
"""
from __future__ import annotations

import logging
import sys
import os
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database.db import init_db, get_session
from backend.database.models import Race, HorseEntry
from backend.scraper.netkeiba import fetch_race_list, fetch_race_card
from backend.scraper.odds import estimate_from_entries, fetch_combination_odds
from backend.predictor.ml_scoring import MLScoringModel
from backend.predictor.bet_optimizer import optimize_bets

COURSE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}


def clear_race_cache(race_id: str):
    """Remove cached data for a specific race to force fresh scrape."""
    db = get_session()
    try:
        db.query(HorseEntry).filter(HorseEntry.race_id == race_id).delete()
        db.query(Race).filter(Race.race_id == race_id).delete()
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def _fetch_live_odds(race_id: str) -> dict:
    """Fetch live win odds from netkeiba API."""
    import requests as _req
    import json as _json
    try:
        url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1&action=init"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"https://race.netkeiba.com/odds/index.html?race_id={race_id}",
        }
        r = _req.get(url, headers=headers, timeout=10)
        d = _json.loads(r.text)
        odds_data = d.get("data", {})
        if not isinstance(odds_data, dict):
            return {}
        tansho = odds_data.get("odds", {}).get("1", {})
        result = {}
        for hn_str, vals in tansho.items():
            if isinstance(vals, list) and len(vals) >= 3:
                try:
                    result[int(hn_str)] = {"odds": float(vals[0]), "popularity": int(vals[2])}
                except (ValueError, IndexError):
                    pass
        return result
    except Exception:
        return {}


def parse_time(time_str: str):
    """Parse 'HH:MM' to today's datetime. Returns None on failure."""
    try:
        h, m = time_str.split(":")
        h_int, m_int = int(h), int(m)
        if not (0 <= h_int <= 23 and 0 <= m_int <= 59):
            return None
        now = datetime.now()
        return now.replace(hour=h_int, minute=m_int, second=0, microsecond=0)
    except Exception as e:
        logging.debug("parse_time failed for '%s': %s", time_str, e)
        return None


def main():
    init_db()
    predictor = MLScoringModel()
    now = datetime.now()
    today_str = now.strftime("%Y%m%d")

    print(f"[{now.strftime('%H:%M:%S')}] Race-day refresh for {today_str}")

    # Check if today is a race day
    if now.weekday() not in (5, 6):  # Sat=5, Sun=6
        print("  Not a race day (Sat/Sun). Skipping.")
        return

    # Get today's schedule
    schedules = fetch_race_list(today_str)
    if not schedules:
        print("  No races today.")
        return

    # Two windows:
    #   Wide window (20-10 min before post): detected for 1-min rapid refresh
    #   Predictions are FROZEN at 10 min before post — no updates after that
    wide_start = now + timedelta(minutes=10)
    wide_end = now + timedelta(minutes=20)
    refreshed = 0

    # Collect races in the rapid-refresh window (20-10 min before post)
    rapid_races = []
    for schedule in schedules:
        for race in schedule.get("races", []):
            start_dt = parse_time(race.get("start_time", ""))
            if start_dt and wide_start <= start_dt <= wide_end:
                rapid_races.append((schedule.get("name", ""), race))

    # If races found in rapid window, run 1-min interval updates for 10 minutes
    if rapid_races:
        import time as _time
        print(f"  Rapid refresh mode: {len(rapid_races)} race(s) in 20-10min window")
        for tick in range(10):  # 10 ticks x 60s = 10 minutes
            tick_now = datetime.now()
            for course_name, race in rapid_races:
                race_id = race.get("race_id", "")
                rnum = race.get("race_number", 0)
                start_dt = parse_time(race.get("start_time", ""))
                if not start_dt or not race_id:
                    continue
                mins_left = int((start_dt - tick_now).total_seconds() / 60)
                if mins_left < 10:
                    print(f"    FROZEN (< 10 min to post) - skipping")
                    continue

                print(f"\n  [{tick_now.strftime('%H:%M:%S')}] {course_name} {rnum}R - {mins_left}min to post")

                # Clear cache + fresh scrape
                clear_race_cache(race_id)
                data = fetch_race_card(race_id)
                if not data:
                    continue
                entries = data.get("entries", [])
                race_info = data.get("race_info", {})

                # Live odds
                live_odds = _fetch_live_odds(race_id)
                if live_odds:
                    for entry in entries:
                        hn = entry["horseNumber"]
                        if hn in live_odds:
                            entry["odds"] = live_odds[hn]["odds"]
                            entry["popularity"] = live_odds[hn]["popularity"]
                    db = get_session()
                    try:
                        from backend.database.models import HorseEntry as HE
                        for he in db.query(HE).filter(HE.race_id == race_id).all():
                            if he.horse_number in live_odds:
                                he.odds = live_odds[he.horse_number]["odds"]
                                he.popularity = live_odds[he.horse_number]["popularity"]
                        db.commit()
                    except Exception:
                        db.rollback()
                    finally:
                        db.close()

                predictions = predictor.predict(race_info, entries)
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
                bets = optimize_bets(predictions, odds_data, race_info)
                print(f"    Updated: {len(live_odds)} odds, {len(bets)} bets, track: {race_info.get('trackCondition','?')}")
                refreshed += 1

            if tick < 9:
                _time.sleep(60)  # Wait 1 minute before next tick

        print(f"\n  Rapid refresh complete: {refreshed} updates")
        return

    # Standard single-pass for races outside rapid window
    for schedule in schedules:
        course = schedule.get("name", "??")
        for race in schedule.get("races", []):
            race_id = race.get("race_id", "")
            start_time_str = race.get("start_time", "")
            rnum = race.get("race_number", 0)

            if not race_id or not start_time_str:
                continue

            start_dt = parse_time(start_time_str)
            if not start_dt:
                continue

            # Check if race starts within refresh window
            if not (window_start <= start_dt <= window_end):
                continue

            mins_to_start = int((start_dt - now).total_seconds() / 60)
            print(f"\n  {course} {rnum}R ({start_time_str}) - {mins_to_start}min to post")

            # Step 1: Clear cache
            print(f"    Clearing cache...")
            clear_race_cache(race_id)

            # Step 2: Fresh scrape
            print(f"    Scraping latest data...")
            data = fetch_race_card(race_id)
            if not data:
                print(f"    !! Failed to fetch race card")
                continue

            entries = data.get("entries", [])
            race_info = data.get("race_info", {})
            condition = race_info.get("trackCondition", "")
            print(f"    Entries: {len(entries)}, Track: {condition or 'N/A'}")

            # Step 2b: Inject live odds from netkeiba API
            print(f"    Fetching live odds...")
            live_odds = _fetch_live_odds(race_id)
            if live_odds:
                for entry in entries:
                    hn = entry["horseNumber"]
                    if hn in live_odds:
                        entry["odds"] = live_odds[hn]["odds"]
                        entry["popularity"] = live_odds[hn]["popularity"]
                # Update DB cache
                db = get_session()
                try:
                    from backend.database.models import HorseEntry as HE
                    for he in db.query(HE).filter(HE.race_id == race_id).all():
                        if he.horse_number in live_odds:
                            he.odds = live_odds[he.horse_number]["odds"]
                            he.popularity = live_odds[he.horse_number]["popularity"]
                    db.commit()
                except Exception:
                    db.rollback()
                finally:
                    db.close()
                print(f"    Live odds: {len(live_odds)} horses updated")

            # Step 3: Fresh predictions
            print(f"    Generating predictions...")
            predictions = predictor.predict(race_info, entries)

            # Step 4: Fresh odds + optimized bets
            print(f"    Fetching combination odds...")
            odds_data = estimate_from_entries(entries) or {}
            try:
                real = fetch_combination_odds(race_id)
                if real:
                    for k, el in real.items():
                        if k in odds_data:
                            rhs = [frozenset(e["horses"]) for e in el]
                            odds_data[k] = el + [e for e in odds_data[k]
                                                 if frozenset(e["horses"]) not in rhs]
                        else:
                            odds_data[k] = el
            except Exception:
                pass

            bets = optimize_bets(predictions, odds_data, race_info)

            # Summary
            top = sorted([p for p in predictions if p["score"] > 0],
                         key=lambda p: -p["score"])
            if top:
                marks = " ".join(f"{p['mark']}{p['horseNumber']}" for p in top[:6] if p["mark"])
                print(f"    Predictions: {marks}")
            if bets:
                bet_str = ", ".join(f"{b['typeLabel']}" for b in bets[:3])
                print(f"    Top bets: {bet_str}")

            refreshed += 1
            time.sleep(1)

    if refreshed == 0:
        print("  No races in refresh window (25-35 min before post)")
    else:
        print(f"\n  Refreshed {refreshed} race(s)")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Done")


if __name__ == "__main__":
    main()
