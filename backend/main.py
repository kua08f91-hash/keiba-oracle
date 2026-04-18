"""FastAPI backend for JRA prediction app.

Serves predictions from DB (populated by realtime_worker) when available,
falls back to live computation otherwise.
"""
from __future__ import annotations

import json
import logging
import sys
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Ensure backend package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.database.db import init_db, get_session
from backend.database.models import (
    Race, HorseEntry, OddsSnapshot, CombinationOdds,
    PredictionsCache, RaceStatus,
)
from backend.scraper.netkeiba import fetch_race_list, fetch_race_card, fetch_pedigree_batch
from backend.scraper.odds import fetch_combination_odds, fetch_live_combination_odds
from backend.predictor.ml_scoring import MLScoringModel
from backend.predictor.bet_optimizer import (
    optimize_bets, detect_race_pattern, scores_to_probabilities,
    generate_candidates, monte_carlo_finish, estimate_hit_probabilities,
    find_odds_for_bet, implied_fair_odds, pick_longshot,
)

app = FastAPI(title="JRA Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://localhost:8080", "null",
        "https://kua08f91-hash.github.io",  # GitHub Pages
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = MLScoringModel()

COURSE_MAP = {
    "01":"札幌","02":"函館","03":"福島","04":"新潟","05":"東京",
    "06":"中山","07":"中京","08":"京都","09":"阪神","10":"小倉",
}


def _race_list_from_db(date_str: str) -> list:
    """Build race list from DB cache (fallback when netkeiba is rate-limited)."""
    from collections import defaultdict
    db = get_session()
    try:
        races = db.query(Race).filter(Race.date == date_str).order_by(Race.race_id).all()
        if not races:
            return []
        by_course = defaultdict(list)
        for r in races:
            code = r.racecourse_code or r.race_id[4:6]
            by_course[code].append({
                "race_id": r.race_id, "raceId": r.race_id,
                "race_number": r.race_number, "raceNumber": r.race_number,
                "race_name": r.race_name, "raceName": r.race_name,
                "start_time": r.start_time or "", "grade": r.grade,
            })
        # Sort courses: main venues first (東京/中山, 京都/阪神), then locals
        COURSE_ORDER = {"05":0,"06":1,"08":2,"09":3,"01":4,"02":5,"03":6,"04":7,"07":8,"10":9}
        return [{"code": code, "name": COURSE_MAP.get(code, code),
                 "races": sorted(rs, key=lambda x: x["race_number"])}
                for code, rs in sorted(by_course.items(), key=lambda x: COURSE_ORDER.get(x[0], 99))]
    finally:
        db.close()


def _get_cached_predictions(race_id: str):
    """Return cached predictions from DB if available."""
    db = get_session()
    try:
        cache = db.query(PredictionsCache).filter(
            PredictionsCache.race_id == race_id
        ).first()
        if not cache or not cache.predictions_json:
            return None
        return {
            "predictions": json.loads(cache.predictions_json),
            "bets": json.loads(cache.bets_json) if cache.bets_json else [],
            "longshot": json.loads(cache.longshot_json) if cache.longshot_json else None,
            "pattern": cache.pattern or "",
            "frozen": cache.frozen,
            "updated_at": cache.updated_at.isoformat() if cache.updated_at else None,
        }
    except Exception:
        return None
    finally:
        db.close()


def _fetch_live_combination_odds(race_id: str, fallback_odds: dict) -> dict:
    """Fetch real-time odds for all types including tansho/fukusho range."""
    return fetch_live_combination_odds(race_id, fallback_odds, include_win_place=True)


@app.on_event("startup")
def startup():
    init_db()


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/race-list")
def get_race_list(date: str):
    """Get available races for a given date (YYYYMMDD)."""
    if not date or len(date) != 8 or not date.isdigit():
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYYMMDD.")

    schedules = fetch_race_list(date)
    if not schedules:
        schedules = _race_list_from_db(date)
    return schedules


@app.get("/api/racecard/{race_id}")
def get_race_card(race_id: str):
    """Get race card with predictions for a given race ID.

    Priority: DB cache (from realtime_worker) → live computation fallback.
    """
    if not race_id or len(race_id) < 10:
        raise HTTPException(status_code=400, detail="Invalid race ID.")

    data = fetch_race_card(race_id)
    if not data:
        raise HTTPException(status_code=404, detail="Race not found or scraping failed.")

    entries = data["entries"]

    # Try DB cache first (populated by realtime_worker)
    cached = _get_cached_predictions(race_id)
    if cached and cached["predictions"]:
        # Inject latest DB odds into entries
        db = get_session()
        try:
            for he in db.query(HorseEntry).filter(HorseEntry.race_id == race_id).all():
                for e in entries:
                    if e["horseNumber"] == he.horse_number and he.odds:
                        e["odds"] = he.odds
                        e["popularity"] = he.popularity
        finally:
            db.close()

        return {
            "raceInfo": data["race_info"],
            "entries": entries,
            "predictions": cached["predictions"],
            "frozen": cached["frozen"],
            "updatedAt": cached["updated_at"],
        }

    # Fallback: live computation (worker not running or no cache yet)
    # Fetch live odds
    try:
        import requests as _req
        url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1&action=init"
        r = _req.get(url, headers={
            "User-Agent": "Mozilla/5.0",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"https://race.netkeiba.com/odds/index.html?race_id={race_id}",
        }, timeout=5)
        d = json.loads(r.text)
        tansho = d.get("data", {}).get("odds", {}).get("1", {}) if isinstance(d.get("data"), dict) else {}
        if tansho:
            for e in entries:
                hn_str = str(e["horseNumber"]).zfill(2)
                if hn_str in tansho:
                    vals = tansho[hn_str]
                    if isinstance(vals, list) and len(vals) >= 3:
                        try:
                            e["odds"] = float(vals[0])
                            e["popularity"] = int(vals[2])
                        except (ValueError, IndexError):
                            pass
    except Exception:
        pass

    predictions = predictor.predict(data["race_info"], entries)

    return {
        "raceInfo": data["race_info"],
        "entries": entries,
        "predictions": predictions,
        "frozen": False,
        "updatedAt": None,
    }


@app.get("/api/odds/{race_id}")
def get_odds(race_id: str):
    """Get combination odds (馬連, ワイド, 3連複, 3連単) for a race.

    Returns payout data from db.netkeiba.com for completed races,
    or estimated odds based on individual horse odds for future races.
    """
    if not race_id or len(race_id) < 10:
        raise HTTPException(status_code=400, detail="Invalid race ID.")

    # First try db.netkeiba for actual payouts
    odds_data = fetch_combination_odds(race_id)
    if odds_data:
        return odds_data

    # Fallback: estimate from individual odds in the race card
    data = fetch_race_card(race_id)
    if data and data.get("entries"):
        from backend.scraper.odds import estimate_from_entries
        return estimate_from_entries(data["entries"])

    return {}


@app.get("/api/optimized-bets/{race_id}")
def get_optimized_bets(race_id: str):
    """Get optimized betting suggestions for a race.

    Priority: DB cache (from realtime_worker) → live computation fallback.
    """
    if not race_id or len(race_id) < 10:
        raise HTTPException(status_code=400, detail="Invalid race ID.")

    # Try DB cache first
    cached = _get_cached_predictions(race_id)
    if cached and cached["bets"]:
        return {
            "bets": cached["bets"],
            "longshot": cached["longshot"],
            "pattern": cached["pattern"],
            "raceId": race_id,
            "frozen": cached["frozen"],
            "updatedAt": cached["updated_at"],
        }

    # Fallback: live computation
    data = fetch_race_card(race_id)
    if not data:
        raise HTTPException(status_code=404, detail="Race not found.")

    try:
        predictions = predictor.predict(data["race_info"], data["entries"])

        from backend.scraper.odds import estimate_from_entries
        odds_data = estimate_from_entries(data["entries"]) or {}
        try:
            odds_data = _fetch_live_combination_odds(race_id, odds_data)
        except Exception:
            pass

        bets = optimize_bets(predictions, odds_data, data["race_info"])

        head_count = data["race_info"].get("headCount", 16)
        probs = scores_to_probabilities(predictions, head_count)
        pattern = detect_race_pattern(probs)

        import random
        rng = random.Random(42)
        all_candidates = generate_candidates(probs, top_n=min(5, len(probs)))
        finishes = monte_carlo_finish(probs, 5000, rng=rng)
        all_candidates = estimate_hit_probabilities(finishes, all_candidates)
        for c in all_candidates:
            oi = find_odds_for_bet(c, odds_data)
            if oi:
                c["odds"] = oi["odds"]
                c["payout"] = oi["payout"]
                c["ev"] = c["hitProb"] * oi["odds"] - 1.0
            else:
                est = implied_fair_odds(c["hitProb"])
                c["odds"] = round(est, 1)
                c["payout"] = int(est * 100)
                c["ev"] = c["hitProb"] * est - 1.0
        longshot = pick_longshot(all_candidates, bets, probs)

        return {
            "bets": bets,
            "longshot": longshot,
            "pattern": pattern,
            "raceId": race_id,
            "frozen": False,
            "updatedAt": None,
        }
    except Exception as e:
        logger.error("Error in optimized-bets for %s: %s", race_id, e)
        return {"bets": [], "pattern": "", "raceId": race_id}


@app.get("/api/race-dates")
def get_race_dates(weeks: int = 3):
    """Find JRA race dates for the next N weeks by scanning netkeiba.

    Returns list of {date, dayOfWeek, label} for dates that have races.
    """
    today = datetime.now()
    start_date = today - timedelta(days=7)  # Include last week
    end_date = today + timedelta(weeks=weeks)
    race_dates = []

    # JRA races are primarily on Sat/Sun, occasionally weekdays for special events
    scan_dates = set()
    current = start_date
    while current <= end_date:
        # Always check Sat(5) and Sun(6)
        if current.weekday() in (5, 6):
            scan_dates.add(current)
        # Also check today and tomorrow
        if (current - today).days <= 1:
            scan_dates.add(current)
        # Check Fridays and Mondays (holiday racing)
        if current.weekday() in (0, 4):
            scan_dates.add(current)
        current += timedelta(days=1)

    dow_names = ["月", "火", "水", "木", "金", "土", "日"]
    for dt in sorted(scan_dates):
        date_str = dt.strftime("%Y%m%d")
        schedules = fetch_race_list(date_str)
        # Fallback: check DB cache if netkeiba is rate-limited
        if not schedules:
            schedules = _race_list_from_db(date_str)
        if schedules and any(s.get("races") for s in schedules):
            dow = dow_names[dt.weekday()]
            today_monday = today - timedelta(days=today.weekday())
            dt_monday = dt - timedelta(days=dt.weekday())
            week_diff = (dt_monday - today_monday).days // 7
            if week_diff < 0:
                week_label = "先週"
            elif week_diff == 0:
                week_label = "今週"
            elif week_diff == 1:
                week_label = "来週"
            else:
                week_label = "再来週"

            course_names = [s["name"] for s in schedules if s.get("races")]
            race_dates.append({
                "date": date_str,
                "display": f"{dt.month}/{dt.day}({dow})",
                "weekLabel": week_label,
                "courses": course_names,
            })

    return race_dates


@app.get("/api/pedigree/{race_id}")
def get_pedigree(race_id: str):
    """Fetch pedigree data for horses in a race (async enrichment)."""
    data = fetch_race_card(race_id)
    if not data:
        raise HTTPException(status_code=404, detail="Race not found.")

    entries = data["entries"]
    horse_ids = [
        {"horseNumber": e["horseNumber"], "horseId": e["horseId"]}
        for e in entries
        if e.get("horseId") and not e.get("isScratched") and not e.get("sireName")
    ]

    pedigrees = fetch_pedigree_batch(horse_ids)

    for entry in entries:
        hn = entry["horseNumber"]
        if hn in pedigrees:
            entry["sireName"] = pedigrees[hn].get("sire", "")
            entry["damName"] = pedigrees[hn].get("dam", "")

    predictions = predictor.predict(data["race_info"], entries)

    return {
        "raceInfo": data["race_info"],
        "entries": entries,
        "predictions": predictions,
    }


@app.get("/api/odds-history/{race_id}")
def get_odds_history(race_id: str):
    """Get odds time-series from DB (populated by realtime_worker)."""
    if not race_id or len(race_id) < 10:
        raise HTTPException(status_code=400, detail="Invalid race ID.")

    db = get_session()
    try:
        snapshots = (
            db.query(OddsSnapshot)
            .filter(OddsSnapshot.race_id == race_id)
            .order_by(OddsSnapshot.captured_at)
            .all()
        )
        if not snapshots:
            return {"raceId": race_id, "history": []}

        history = {}
        for s in snapshots:
            hn = s.horse_number
            if hn not in history:
                history[hn] = []
            history[hn].append({
                "odds": s.odds,
                "popularity": s.popularity,
                "capturedAt": s.captured_at.isoformat() if s.captured_at else None,
            })

        return {"raceId": race_id, "history": history}
    finally:
        db.close()


@app.get("/api/race-status")
def get_race_status(date: str = ""):
    """Get race statuses for a date (default: today)."""
    if not date:
        date = datetime.now().strftime("%Y%m%d")

    db = get_session()
    try:
        statuses = db.query(RaceStatus).filter(
            RaceStatus.race_id.like(f"{date}%")
        ).all()

        result = []
        for s in statuses:
            cache = db.query(PredictionsCache).filter(
                PredictionsCache.race_id == s.race_id
            ).first()
            result.append({
                "raceId": s.race_id,
                "status": s.status,
                "startTime": s.start_time,
                "lastOddsUpdate": s.last_odds_update.isoformat() if s.last_odds_update else None,
                "lastPredictionUpdate": s.last_prediction_update.isoformat() if s.last_prediction_update else None,
                "frozen": cache.frozen if cache else False,
                "hasPredictions": bool(cache and cache.predictions_json),
            })

        return {"date": date, "races": result}
    finally:
        db.close()
