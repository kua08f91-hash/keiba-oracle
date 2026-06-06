"""Phase 1: Cache all race data + results + payouts for 3-4月 to local JSON.

This avoids repeated netkeiba API calls during parameter tuning.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from bs4 import BeautifulSoup
from backend.scraper.netkeiba import fetch_race_list, fetch_race_card
from backend.scraper.odds import estimate_from_entries, fetch_combination_odds
from backend.predictor.ml_scoring import MLScoringModel
from backend.database.db import init_db

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
COURSE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}

_session = requests.Session()
_session.headers.update(HEADERS)
predictor = MLScoringModel()

CACHE_FILE = os.path.join(os.path.dirname(__file__), "cached_races_mar_apr.json")


def fetch_payouts(race_id):
    url = f"https://db.netkeiba.com/race/{race_id}/"
    try:
        resp = _session.get(url, timeout=20)
        resp.encoding = resp.apparent_encoding or "UTF-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        payouts = {}
        for table in soup.select("table.pay_table_01"):
            for row in table.select("tr"):
                th = row.select_one("th")
                tds = row.select("td")
                if not th or len(tds) < 2:
                    continue
                label = th.get_text(strip=True)
                combos = tds[0].get_text("|", strip=True).split("|")
                amounts = tds[1].get_text("|", strip=True).split("|")
                entries = []
                for combo, amount in zip(combos, amounts):
                    try:
                        amt = int(amount.replace(",", ""))
                    except ValueError:
                        continue
                    nums = [int(n) for n in re.findall(r"\d+", combo)]
                    if nums:
                        entries.append({"nums": nums, "amount": amt})
                if entries:
                    payouts[label] = entries
        return payouts
    except Exception:
        return {}


def main():
    init_db()

    # Check if partial cache exists
    cached = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cached = json.load(f)
        print(f"Loaded {len(cached)} cached races")

    # Collect race dates
    print("Collecting race dates for 3-4月...")
    all_dates = {}
    d = date(2026, 3, 1)
    end = date(2026, 4, 30)
    while d <= end:
        if d.weekday() in (5, 6):
            ds = d.strftime("%Y%m%d")
            time.sleep(2)
            schedules = fetch_race_list(ds)
            if schedules:
                ids = []
                for s in schedules:
                    for r in s.get("races", []):
                        rid = r.get("race_id", "")
                        if rid:
                            ids.append(rid)
                if ids:
                    all_dates[ds] = ids
                    print(f"  {ds}: {len(ids)} races")
        d += timedelta(days=1)

    total = sum(len(v) for v in all_dates.values())
    print(f"\nTotal: {total} races, {len(cached)} already cached\n")

    processed = 0
    for ds in sorted(all_dates.keys()):
        for race_id in all_dates[ds]:
            if race_id in cached:
                processed += 1
                continue

            course = COURSE_MAP.get(race_id[4:6], "??")
            rnum = int(race_id[10:12])
            processed += 1

            # Fetch race card
            try:
                data = fetch_race_card(race_id)
            except Exception:
                data = None
            if not data or len(data.get("entries", [])) < 3:
                print(f"  [{processed}/{total}] {course}{rnum:2d}R: skip (no data)")
                continue

            entries = data["entries"]
            info = data["race_info"]

            # Predictions
            try:
                preds = predictor.predict(info, entries)
            except Exception:
                continue
            if len(preds) < 3:
                continue

            # Odds
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

            # Payouts
            time.sleep(2)
            payouts = None
            for attempt in range(3):
                payouts = fetch_payouts(race_id)
                if payouts:
                    break
                time.sleep(5 * (attempt + 1))

            if not payouts:
                print(f"  [{processed}/{total}] {course}{rnum:2d}R: skip (no payouts)")
                continue

            # Cache this race
            cached[race_id] = {
                "race_id": race_id,
                "date": ds,
                "course": course,
                "rnum": rnum,
                "info": info,
                "entries": [{
                    "horseNumber": e["horseNumber"],
                    "frameNumber": e.get("frameNumber", 0),
                    "horseName": e.get("horseName", ""),
                    "odds": e.get("odds"),
                    "popularity": e.get("popularity"),
                    "isScratched": e.get("isScratched", False),
                } for e in entries],
                "predictions": [{
                    "horseNumber": p["horseNumber"],
                    "score": p["score"],
                    "mark": p.get("mark", ""),
                } for p in preds],
                "odds_data": odds_data,
                "payouts": payouts,
            }

            print(f"  [{processed}/{total}] {course}{rnum:2d}R: cached ({len(payouts)} payout types)")

            # Save periodically
            if processed % 20 == 0:
                with open(CACHE_FILE, "w") as f:
                    json.dump(cached, f, ensure_ascii=False)

            time.sleep(3)

    # Final save
    with open(CACHE_FILE, "w") as f:
        json.dump(cached, f, ensure_ascii=False)

    print(f"\nCached {len(cached)} races to {CACHE_FILE}")
    print(f"File size: {os.path.getsize(CACHE_FILE) // 1024} KB")


if __name__ == "__main__":
    main()
