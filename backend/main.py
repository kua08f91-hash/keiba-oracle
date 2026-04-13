"""FastAPI backend for JRA prediction app."""
import sys
import os
from datetime import datetime, timedelta

# Ensure backend package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.database.db import init_db
from backend.scraper.netkeiba import fetch_race_list, fetch_race_card, fetch_pedigree_batch
from backend.scraper.odds import fetch_combination_odds
from backend.predictor.ml_scoring import MLScoringModel
from backend.predictor.bet_optimizer import optimize_bets, detect_race_pattern, scores_to_probabilities, generate_candidates, monte_carlo_finish, estimate_hit_probabilities, find_odds_for_bet, implied_fair_odds, pick_longshot

app = FastAPI(title="JRA Prediction API")


def _fetch_live_combination_odds(race_id: str, fallback_odds: dict) -> dict:
    """Fetch real-time odds for all bet types from netkeiba API."""
    import requests as _req
    import json as _json
    import re as _re

    API_H = {
        "User-Agent": "Mozilla/5.0",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"https://race.netkeiba.com/odds/index.html?race_id={race_id}",
    }
    # Correct type mapping (verified against JRA official):
    # 1=単勝, 2=複勝, 3=枠連, 4=馬連, 5=ワイド, 6=馬単, 7=3連複, 8=3連単
    TYPE_MAP = {
        4: "umaren",
        5: "wide",
        7: "sanrenpuku",
        8: "sanrentan",
    }

    def _parse_horse_nums(key_str):
        """Parse '0508' -> [5,8] or '050812' -> [5,8,12]"""
        nums = []
        i = 0
        while i < len(key_str):
            if i + 1 < len(key_str):
                nums.append(int(key_str[i:i+2]))
                i += 2
            else:
                i += 1
        return nums

    result = dict(fallback_odds)

    for api_type, bet_type in TYPE_MAP.items():
        try:
            url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type={api_type}&action=init"
            r = _req.get(url, headers=API_H, timeout=5)
            d = _json.loads(r.text)
            data = d.get("data", {})
            if not isinstance(data, dict):
                continue
            odds_dict = data.get("odds", {}).get(str(api_type), {})
            if not odds_dict:
                continue

            entries = []
            for combo_key, vals in odds_dict.items():
                if not isinstance(vals, list) or len(vals) < 1:
                    continue
                odds_str = vals[0]
                try:
                    odds_val = float(odds_str.replace(",", ""))
                except (ValueError, TypeError):
                    continue
                if odds_val <= 0:
                    continue
                horses = _parse_horse_nums(combo_key)
                if len(horses) >= 2:
                    entries.append({
                        "horses": horses,
                        "odds": odds_val,
                        "payout": int(odds_val * 100),
                    })

            if entries:
                result[bet_type] = entries
        except Exception:
            pass

    return result

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "null"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = MLScoringModel()


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
    return schedules


@app.get("/api/racecard/{race_id}")
def get_race_card(race_id: str):
    """Get race card with predictions for a given race ID."""
    if not race_id or len(race_id) < 10:
        raise HTTPException(status_code=400, detail="Invalid race ID.")

    data = fetch_race_card(race_id)
    if not data:
        raise HTTPException(status_code=404, detail="Race not found or scraping failed.")

    # Inject live odds only if entries completely lack odds (first load)
    # Predictions FROZEN 10 min before post, but odds always updated
    entries = data["entries"]
    race_frozen = False
    try:
        start_time_str = data["race_info"].get("startTime", "")
        if start_time_str:
            h, m = start_time_str.split(":")
            from datetime import datetime as _dt
            race_start = _dt.now().replace(hour=int(h), minute=int(m), second=0)
            mins_to_post = (race_start - _dt.now()).total_seconds() / 60
            if mins_to_post < 10:
                race_frozen = True
    except Exception:
        pass

    # Odds handling:
    # - Not frozen: fetch live odds, update entries + DB
    # - Frozen: use DB cached odds only (no external request = fast + stable)
    has_odds = any(e.get("odds") for e in entries if not e.get("isScratched"))
    if not race_frozen:
        try:
            import requests as _req
            import json as _json
            url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1&action=init"
            r = _req.get(url, headers={
                "User-Agent": "Mozilla/5.0",
                "X-Requested-With": "XMLHttpRequest",
                "Referer": f"https://race.netkeiba.com/odds/index.html?race_id={race_id}",
            }, timeout=5)
            d = _json.loads(r.text)
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
                # Persist to DB so subsequent calls use cached odds
                try:
                    from backend.database.db import get_session
                    from backend.database.models import HorseEntry as HE
                    db = get_session()
                    try:
                        for he in db.query(HE).filter(HE.race_id == race_id).all():
                            hn_str = str(he.horse_number).zfill(2)
                            if hn_str in tansho:
                                vals = tansho[hn_str]
                                if isinstance(vals, list) and len(vals) >= 3:
                                    try:
                                        he.odds = float(vals[0])
                                        he.popularity = int(vals[2])
                                    except (ValueError, IndexError):
                                        pass
                        db.commit()
                    finally:
                        db.close()
                except Exception:
                    pass
        except Exception:
            pass

    # Generate predictions
    if race_frozen:
        # FROZEN: use DB cached odds for both display and prediction (no external calls)
        try:
            from backend.database.db import get_session
            from backend.database.models import HorseEntry as HE
            db = get_session()
            try:
                db_map = {he.horse_number: (he.odds, he.popularity)
                          for he in db.query(HE).filter(HE.race_id == race_id).all()}
            finally:
                db.close()
            for e in entries:
                if e["horseNumber"] in db_map:
                    e["odds"], e["popularity"] = db_map[e["horseNumber"]]
        except Exception:
            pass
        predictions = predictor.predict(data["race_info"], entries)
    else:
        predictions = predictor.predict(data["race_info"], entries)

    return {
        "raceInfo": data["race_info"],
        "entries": entries,
        "predictions": predictions,
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
    """Get dynamically optimized betting suggestions for a race.

    Uses Monte Carlo simulation + EV optimization to select the best
    5 bets per race, adapting to each race's score distribution and odds.
    """
    if not race_id or len(race_id) < 10:
        raise HTTPException(status_code=400, detail="Invalid race ID.")

    # Get race card and predictions
    data = fetch_race_card(race_id)
    if not data:
        raise HTTPException(status_code=404, detail="Race not found.")

    try:
        predictions = predictor.predict(data["race_info"], data["entries"])

        # Get real-time odds from netkeiba API for all bet types
        from backend.scraper.odds import estimate_from_entries
        odds_data = estimate_from_entries(data["entries"]) or {}

        try:
            odds_data = _fetch_live_combination_odds(race_id, odds_data)
        except Exception as e:
            print(f"Warning: live odds fetch failed for {race_id}: {e}")

        # Run optimizer
        bets = optimize_bets(predictions, odds_data, data["race_info"])

        # Detect race pattern for UI
        head_count = data["race_info"].get("headCount", 16)
        probs = scores_to_probabilities(predictions, head_count)
        pattern = detect_race_pattern(probs)

        # Pick longshot (穴場券)
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
        }
    except Exception as e:
        print(f"Error in optimized-bets for {race_id}: {e}")
        # Return empty bets rather than 500
        return {
            "bets": [],
            "pattern": "",
            "raceId": race_id,
        }


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

    for dt in sorted(scan_dates):
        date_str = dt.strftime("%Y%m%d")
        schedules = fetch_race_list(date_str)
        if schedules and any(s.get("races") for s in schedules):
            dow_names = ["月", "火", "水", "木", "金", "土", "日"]
            dow = dow_names[dt.weekday()]

            # Determine week label
            # "今週" = current Mon-Sun block
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

    # Re-generate predictions with pedigree data
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
