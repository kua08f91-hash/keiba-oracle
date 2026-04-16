"""Real-time data pipeline worker for race day.

Continuously fetches odds, updates predictions, and manages race state.
Runs as a background process on race days.

Usage:
    /usr/bin/python3 -m backend.realtime_worker

Flow:
    1. Import: Fetch all odds (type 1,4,5,7,8) from netkeiba API → save to DB
    2. Analyze: Generate predictions from latest data → save to DB
    3. Repeat: 1-min intervals for 20-10min before post, 5-min otherwise
    4. Freeze: Lock predictions at 10min before post
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from backend.database.db import init_db, get_session
from backend.database.models import (
    Race, HorseEntry, OddsSnapshot, CombinationOdds,
    PredictionsCache, RaceStatus,
)
from backend.scraper.netkeiba import fetch_race_list, fetch_race_card
from backend.scraper.odds import estimate_from_entries
from backend.predictor.ml_scoring import MLScoringModel
from backend.predictor.bet_optimizer import (
    optimize_bets, detect_race_pattern, scores_to_probabilities,
    generate_candidates, monte_carlo_finish, estimate_hit_probabilities,
    find_odds_for_bet, implied_fair_odds, pick_longshot,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

API_H = {"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"}
COURSE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}


class RealtimeWorker:
    def __init__(self):
        init_db()
        self.predictor = MLScoringModel()
        self.today = datetime.now().strftime("%Y%m%d")

    # ─── Step 1: Data Import ───

    def fetch_win_odds(self, race_id: str) -> dict:
        """Fetch type 1 (単勝) odds from netkeiba API."""
        try:
            url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1&action=init"
            r = requests.get(url, headers={**API_H, "Referer": f"https://race.netkeiba.com/odds/index.html?race_id={race_id}"}, timeout=10)
            d = json.loads(r.text)
            tansho = d.get("data", {}).get("odds", {}).get("1", {}) if isinstance(d.get("data"), dict) else {}
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

    def fetch_combination_odds(self, race_id: str) -> dict:
        """Fetch types 4,5,7,8 (馬連/ワイド/3連複/3連単) from netkeiba API."""
        TYPE_MAP = {4: "umaren", 5: "wide", 7: "sanrenpuku", 8: "sanrentan"}
        result = {}
        for api_type, bet_type in TYPE_MAP.items():
            try:
                url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type={api_type}&action=init"
                r = requests.get(url, headers={**API_H, "Referer": f"https://race.netkeiba.com/odds/index.html?race_id={race_id}"}, timeout=5)
                d = json.loads(r.text)
                odds_dict = d.get("data", {}).get("odds", {}).get(str(api_type), {}) if isinstance(d.get("data"), dict) else {}
                entries = []
                for combo_key, vals in odds_dict.items():
                    if not isinstance(vals, list) or len(vals) < 1:
                        continue
                    try:
                        odds_val = float(vals[0].replace(",", ""))
                    except (ValueError, TypeError):
                        continue
                    if odds_val <= 0:
                        continue
                    nums = []
                    i = 0
                    while i < len(combo_key):
                        if i + 1 < len(combo_key):
                            nums.append(int(combo_key[i:i + 2]))
                            i += 2
                        else:
                            i += 1
                    if len(nums) >= 2:
                        entries.append({"horses": nums, "odds": odds_val, "payout": int(odds_val * 100)})
                if entries:
                    result[bet_type] = entries
            except Exception:
                pass
        return result

    def save_odds_to_db(self, race_id: str, win_odds: dict, combo_odds: dict):
        """Save odds snapshot to DB."""
        now = datetime.utcnow()
        db = get_session()
        try:
            # Win odds snapshots
            for hn, data in win_odds.items():
                db.add(OddsSnapshot(
                    race_id=race_id, horse_number=hn, odds_type="tansho",
                    odds=data["odds"], popularity=data["popularity"], captured_at=now,
                ))

            # Update HorseEntry with latest odds
            for he in db.query(HorseEntry).filter(HorseEntry.race_id == race_id).all():
                if he.horse_number in win_odds:
                    he.odds = win_odds[he.horse_number]["odds"]
                    he.popularity = win_odds[he.horse_number]["popularity"]

            # Combination odds (replace old, keep latest)
            db.query(CombinationOdds).filter(CombinationOdds.race_id == race_id).delete()
            for bet_type, entries in combo_odds.items():
                for e in entries:
                    key = "-".join(f"{h:02d}" for h in e["horses"])
                    db.add(CombinationOdds(
                        race_id=race_id, bet_type=bet_type, horses_key=key,
                        odds=e["odds"], captured_at=now,
                    ))

            # Update race status
            status = db.query(RaceStatus).filter(RaceStatus.race_id == race_id).first()
            if status:
                status.last_odds_update = now

            db.commit()
        except Exception as e:
            db.rollback()
            logger.error("save_odds_to_db failed: %s", e)
        finally:
            db.close()

    # ─── Step 2: Prediction ───

    def generate_and_save_predictions(self, race_id: str):
        """Generate predictions from latest DB data and save."""
        data = fetch_race_card(race_id)
        if not data:
            return

        entries = data["entries"]
        info = data["race_info"]

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

        # Predictions
        preds = self.predictor.predict(info, entries)

        # Build odds data from DB
        od = estimate_from_entries(entries) or {}
        db = get_session()
        try:
            for co in db.query(CombinationOdds).filter(CombinationOdds.race_id == race_id).all():
                horses = [int(h) for h in co.horses_key.split("-")]
                if co.bet_type not in od:
                    od[co.bet_type] = []
                od[co.bet_type].append({"horses": horses, "odds": co.odds, "payout": int(co.odds * 100)})
        finally:
            db.close()

        # Optimize bets
        bets = optimize_bets(preds, od, info)

        # Pattern + Longshot
        head_count = info.get("headCount", 16)
        probs = scores_to_probabilities(preds, head_count)
        pattern = detect_race_pattern(probs) if len(probs) >= 3 else ""
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

        # Save to DB
        now = datetime.utcnow()
        db = get_session()
        try:
            cache = db.query(PredictionsCache).filter(PredictionsCache.race_id == race_id).first()
            if not cache:
                cache = PredictionsCache(race_id=race_id)
                db.add(cache)
            cache.predictions_json = json.dumps(preds, ensure_ascii=False, default=str)
            cache.bets_json = json.dumps(bets, ensure_ascii=False, default=str)
            cache.longshot_json = json.dumps(longshot, ensure_ascii=False, default=str) if longshot else None
            cache.pattern = pattern
            cache.updated_at = now

            status = db.query(RaceStatus).filter(RaceStatus.race_id == race_id).first()
            if status:
                status.last_prediction_update = now

            db.commit()
        except Exception as e:
            db.rollback()
            logger.error("save_predictions failed: %s", e)
        finally:
            db.close()

    # ─── Step 3 & 4: Race Management ───

    def freeze_race(self, race_id: str):
        """Lock predictions — no more updates."""
        db = get_session()
        try:
            cache = db.query(PredictionsCache).filter(PredictionsCache.race_id == race_id).first()
            if cache and not cache.frozen:
                cache.frozen = True
                logger.info("  FROZEN: %s", race_id)
            status = db.query(RaceStatus).filter(RaceStatus.race_id == race_id).first()
            if status:
                status.status = "frozen"
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()

    def init_race_statuses(self):
        """Initialize race statuses for today."""
        schedules = fetch_race_list(self.today)
        if not schedules:
            logger.info("No races today")
            return []

        race_ids = []
        db = get_session()
        try:
            for s in schedules:
                course = s["name"]
                for race in s.get("races", []):
                    rid = race["race_id"]
                    rnum = race["race_number"]
                    stime = race.get("start_time", "")
                    race_ids.append(rid)

                    # Ensure race card is cached
                    fetch_race_card(rid)

                    # Upsert race status
                    status = db.query(RaceStatus).filter(RaceStatus.race_id == rid).first()
                    if not status:
                        status = RaceStatus(race_id=rid, status="upcoming", start_time=stime)
                        db.add(status)
                    else:
                        status.start_time = stime
                    logger.info("  %s%2dR %s (%s)", course, rnum, race.get("race_name", ""), stime)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error("init_race_statuses failed: %s", e)
        finally:
            db.close()

        return race_ids

    def get_minutes_to_post(self, race_id: str) -> float:
        """Get minutes until race start."""
        db = get_session()
        try:
            status = db.query(RaceStatus).filter(RaceStatus.race_id == race_id).first()
            if not status or not status.start_time:
                return 999
            try:
                h, m = status.start_time.split(":")
                now = datetime.now()
                start = now.replace(hour=int(h), minute=int(m), second=0, microsecond=0)
                return (start - now).total_seconds() / 60
            except Exception:
                return 999
        finally:
            db.close()

    def is_frozen(self, race_id: str) -> bool:
        db = get_session()
        try:
            cache = db.query(PredictionsCache).filter(PredictionsCache.race_id == race_id).first()
            return cache.frozen if cache else False
        finally:
            db.close()

    # ─── Main Loop ───

    def run(self):
        logger.info("=" * 50)
        logger.info("KEIBA ORACLE Realtime Worker")
        logger.info("Date: %s", self.today)
        logger.info("=" * 50)

        # Check if race day
        now = datetime.now()
        if now.weekday() not in (5, 6):  # Sat=5, Sun=6
            logger.info("Not a race day. Exiting.")
            return

        # Initialize
        logger.info("Initializing race statuses...")
        race_ids = self.init_race_statuses()
        if not race_ids:
            return

        logger.info("Tracking %d races", len(race_ids))
        logger.info("Starting main loop...\n")

        while True:
            any_active = False

            for rid in race_ids:
                mins = self.get_minutes_to_post(rid)
                course = COURSE_MAP.get(rid[4:6], "??")
                rnum = int(rid[10:12])

                if mins > 30:
                    continue  # Too early
                if mins < -10:
                    continue  # Long past

                any_active = True

                if mins <= 10 and not self.is_frozen(rid):
                    # Freeze point — final odds update then lock
                    logger.info("%s%2dR: Final update + FREEZE (%.0fmin)", course, rnum, mins)
                    win_odds = self.fetch_win_odds(rid)
                    combo_odds = self.fetch_combination_odds(rid)
                    if win_odds:
                        self.save_odds_to_db(rid, win_odds, combo_odds)
                    self.generate_and_save_predictions(rid)
                    self.freeze_race(rid)
                    continue

                if self.is_frozen(rid):
                    continue  # Already frozen, skip

                # Active: fetch odds + update predictions
                logger.info("%s%2dR: Update (%.0fmin to post)", course, rnum, mins)
                win_odds = self.fetch_win_odds(rid)
                combo_odds = self.fetch_combination_odds(rid)
                if win_odds:
                    self.save_odds_to_db(rid, win_odds, combo_odds)
                    self.generate_and_save_predictions(rid)
                    time.sleep(0.5)

            if not any_active:
                # Check if all races are done
                all_done = all(self.get_minutes_to_post(r) < -10 for r in race_ids)
                if all_done:
                    logger.info("All races finished. Exiting.")
                    break

            # Sleep interval: 60s if any race within 20min, else 300s
            min_mins = min((self.get_minutes_to_post(r) for r in race_ids if self.get_minutes_to_post(r) > -10), default=999)
            interval = 60 if min_mins <= 20 else 300
            logger.info("  Next check in %ds (nearest race: %.0fmin)\n", interval, min_mins)
            time.sleep(interval)


def main():
    worker = RealtimeWorker()
    worker.run()


if __name__ == "__main__":
    main()
