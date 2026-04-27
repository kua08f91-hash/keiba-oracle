"""Export predictions as static JSON for GitHub Pages.

Generates predictions for upcoming races and saves to docs/data/
so the preview site works without a running backend.

Usage:
    /usr/bin/python3 -m backend.export_predictions

Recommended: Run via cron Saturday 08:00
"""
from __future__ import annotations

import json
import os
import sys
import time
import random
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database.db import init_db
from backend.scraper.netkeiba import fetch_race_list, fetch_race_card
from backend.scraper.odds import estimate_from_entries, fetch_combination_odds, fetch_live_combination_odds
from backend.predictor.ml_scoring import MLScoringModel
from backend.predictor.bet_optimizer import (
    optimize_bets, detect_race_pattern, scores_to_probabilities,
    generate_candidates, monte_carlo_finish, estimate_hit_probabilities,
    find_odds_for_bet, implied_fair_odds, pick_longshot,
)

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "data")
API_H = {"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"}


def fetch_live_odds(rid):
    try:
        url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={rid}&type=1&action=init"
        r = requests.get(url, headers={**API_H, "Referer": f"https://race.netkeiba.com/odds/index.html?race_id={rid}"}, timeout=10)
        d = json.loads(r.text)
        t = d.get("data", {}).get("odds", {}).get("1", {}) if isinstance(d.get("data"), dict) else {}
        res = {}
        for h, v in t.items():
            if isinstance(v, list) and len(v) >= 3:
                try:
                    res[int(h)] = {"odds": float(v[0]), "popularity": int(v[2])}
                except:
                    pass
        return res
    except:
        return {}


def fetch_combination_odds_live(rid):
    """Fetch real-time combination odds — delegates to shared utility."""
    return fetch_live_combination_odds(rid)


def main():
    init_db()
    predictor = MLScoringModel()
    os.makedirs(DOCS_DIR, exist_ok=True)

    from datetime import datetime, timedelta
    today = datetime.now()
    # Export for this weekend (Sat + Sun)
    dates = []
    for delta in range(0, 8):
        d = today + timedelta(days=delta)
        if d.weekday() in (5, 6):  # Sat, Sun
            dates.append(d.strftime("%Y%m%d"))
    if not dates:
        # If today is Sat/Sun, include today
        if today.weekday() in (5, 6):
            dates.append(today.strftime("%Y%m%d"))

    all_dates_data = []

    for ds in dates:
        dl = f"{int(ds[4:6])}/{int(ds[6:8])}"
        print(f"\n{dl}:")
        schedules = fetch_race_list(ds)
        if not schedules:
            continue

        day_data = {"date": ds, "display": dl, "courses": []}

        for s in schedules:
            course_data = {"name": s["name"], "code": s["code"], "races": []}

            for race in sorted(s.get("races", []), key=lambda r: r.get("race_number", 0)):
                rid = race["race_id"]
                rnum = race["race_number"]
                rname = race.get("race_name", "")

                data = fetch_race_card(rid)
                if not data:
                    continue

                entries = data["entries"]
                info = data["race_info"]

                # Validate frame numbers — re-fetch if missing
                non_scratched = [e for e in entries if not e.get("isScratched")]
                zero_frames = sum(1 for e in non_scratched if e.get("frameNumber", 0) == 0)
                if non_scratched and zero_frames > len(non_scratched) * 0.5:
                    print(f"  ⚠ {s['name']}{rnum:2d}R: {zero_frames}/{len(non_scratched)} entries have frame=0, re-fetching...")
                    data2 = fetch_race_card(rid, force_refresh=True)
                    if data2 and data2["entries"]:
                        entries = data2["entries"]
                        info = data2["race_info"]
                        data = data2

                # Live odds
                time.sleep(0.5)
                live = fetch_live_odds(rid)
                for e in entries:
                    if e["horseNumber"] in live:
                        e["odds"] = live[e["horseNumber"]]["odds"]
                        e["popularity"] = live[e["horseNumber"]]["popularity"]

                # Predictions
                preds = predictor.predict(info, entries)
                ranked = sorted([p for p in preds if p["score"] > 0], key=lambda p: -p["score"])

                # Fetch ALL real odds from netkeiba API (tansho/fukusho/umaren/wide/sanrenpuku/sanrentan)
                time.sleep(0.5)
                od = estimate_from_entries(entries) or {}
                live_od = fetch_live_combination_odds(rid, include_win_place=True)
                if live_od:
                    od.update(live_od)  # Real odds override estimates

                bets = optimize_bets(preds, od, info, entries=entries)

                # Pattern
                head_count = info.get("headCount", 16)
                probs = scores_to_probabilities(preds, head_count)
                pattern = detect_race_pattern(probs) if len(probs) >= 3 else ""

                # Longshot
                longshot = None
                if len(probs) >= 3:
                    rng = random.Random(42)
                    cands = generate_candidates(probs, top_n=min(5, len(probs)))
                    fin = monte_carlo_finish(probs, 5000, rng=rng)
                    cands = estimate_hit_probabilities(fin, cands)
                    for c in cands:
                        oi = find_odds_for_bet(c, od)
                        if oi:
                            c["odds"] = oi["odds"]
                            c["ev"] = c["hitProb"] * oi["odds"] - 1
                        else:
                            est = implied_fair_odds(c["hitProb"])
                            c["odds"] = round(est, 1)
                            c["ev"] = c["hitProb"] * est - 1
                    longshot = pick_longshot(cands, bets, probs)

                # Build race data
                race_data = {
                    "raceId": rid,
                    "raceNumber": rnum,
                    "raceName": rname,
                    "raceInfo": info,
                    "entries": [{
                        "horseNumber": e["horseNumber"],
                        "frameNumber": e.get("frameNumber", 0),
                        "horseName": e.get("horseName", ""),
                        "jockeyName": e.get("jockeyName", ""),
                        "trainerName": e.get("trainerName", ""),
                        "age": e.get("age", ""),
                        "weightCarried": e.get("weightCarried", 0),
                        "horseWeight": e.get("horseWeight", ""),
                        "odds": e.get("odds"),
                        "popularity": e.get("popularity"),
                        "isScratched": e.get("isScratched", False),
                        "sireName": e.get("sireName", ""),
                        "damName": e.get("damName", ""),
                    } for e in entries],
                    "predictions": [{
                        "horseNumber": p["horseNumber"],
                        "score": p["score"],
                        "mark": p["mark"],
                        "factors": p.get("factors", {}),
                    } for p in preds],
                    "bets": bets,
                    "longshot": longshot,
                    "pattern": pattern,
                }
                course_data["races"].append(race_data)
                print(f"  {s['name']}{rnum:2d}R: {len(entries)}頭 {len(bets)}買目")

            if course_data["races"]:
                day_data["courses"].append(course_data)

        if day_data["courses"]:
            all_dates_data.append(day_data)

    # Collect last week's performance stats
    perf = _load_performance_stats()

    # Save with export timestamp for stale detection
    from datetime import datetime
    perf["exportedAt"] = datetime.now().isoformat()

    output = {
        "predictions": all_dates_data,
        "performance": perf,
    }
    output_file = os.path.join(DOCS_DIR, "predictions.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    print(f"\nExported to {output_file} ({os.path.getsize(output_file) // 1024} KB)")


def _load_performance_stats() -> dict:
    """Load recent performance stats from performance_log.json."""
    perf_file = os.path.join(os.path.dirname(DOCS_DIR), "data", "performance_log.json")
    try:
        if os.path.exists(perf_file):
            with open(perf_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
            if logs and isinstance(logs, list):
                latest = logs[-1]
                return {
                    "date": latest.get("date", ""),
                    "tansho": latest.get("tansho", 0),
                    "wide": latest.get("wide", 0),
                    "winnerInTop3": latest.get("winner_in_top3", 0),
                    "races": latest.get("races", 0),
                }
    except Exception:
        pass
    return {}


if __name__ == "__main__":
    main()
