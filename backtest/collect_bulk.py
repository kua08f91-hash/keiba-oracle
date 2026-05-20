"""Bulk historical data collector — faster by reducing API calls.

Strategy: For each race date, fetch race list once, then for each race:
  1. fetch_race_card (uses DB cache with 30-day TTL)
  2. Positions from result page
  3. Payouts from db.netkeiba
  4. Odds estimated from entries

Optimizations vs collect_fast.py:
  - 0.5s sleep between API calls (down from 1-2s)
  - Skip already-collected race IDs (resume)
  - Save every 10 races
  - Parallel date fetching where possible

Usage:
    /usr/bin/python3 -m backtest.collect_bulk --start 20240101 --end 20241231
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
import time
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from bs4 import BeautifulSoup
from backend.database.db import init_db
from backend.scraper.netkeiba import fetch_race_list, fetch_race_card
from backend.scraper.odds import estimate_from_entries

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

_session = requests.Session()
_session.headers.update(HEADERS)


def fetch_positions(race_id):
    r = _session.get(f"https://race.netkeiba.com/race/result.html?race_id={race_id}", timeout=15)
    r.encoding = r.apparent_encoding or "UTF-8"
    soup = BeautifulSoup(r.text, "lxml")
    pos = {}
    for row in soup.select(".HorseList"):
        tds = row.select("td")
        if len(tds) < 3:
            continue
        try:
            pos[int(tds[2].get_text(strip=True))] = int(tds[0].get_text(strip=True))
        except (ValueError, IndexError):
            pass
    return pos


def fetch_payouts(race_id):
    r = _session.get(f"https://db.netkeiba.com/race/{race_id}/", timeout=15)
    r.encoding = r.apparent_encoding or "UTF-8"
    soup = BeautifulSoup(r.text, "html.parser")
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
            for combo, amount in zip(combos, amounts):
                try:
                    amt = int(amount.replace(",", ""))
                except ValueError:
                    continue
                nums = [int(n) for n in re.findall(r"\d+", combo)]
                if nums:
                    payouts.setdefault(label, []).append({"nums": nums, "amount": amt})
    return payouts


def collect_range(start: date, end: date, cache_name: str):
    cache_file = os.path.join(CACHE_DIR, f"{cache_name}.pkl")

    races = []
    collected_rids = set()
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            races = pickle.load(f)
        collected_rids = {r["rid"] for r in races}
        print(f"Resuming: {len(races)} already collected")

    init_db()
    last_save = len(races)

    d = start
    while d <= end:
        if d.weekday() not in (5, 6):
            d += timedelta(days=1)
            continue

        ds = d.strftime("%Y%m%d")
        time.sleep(0.5)
        try:
            schedules = fetch_race_list(ds)
        except Exception as e:
            print(f"  {ds}: list error {e}", flush=True)
            time.sleep(3)
            d += timedelta(days=1)
            continue

        if not schedules:
            d += timedelta(days=1)
            continue

        day_count = 0
        for s in schedules:
            for r in s.get("races", []):
                rid = r.get("race_id", "")
                if not rid or rid in collected_rids:
                    continue

                try:
                    # 1. Race card
                    data = fetch_race_card(rid)
                    if not data or len(data.get("entries", [])) < 3:
                        continue

                    # 2. Positions
                    time.sleep(0.5)
                    positions = fetch_positions(rid)
                    if not positions:
                        continue

                    # 3. Payouts
                    time.sleep(0.5)
                    payouts = fetch_payouts(rid)

                    # 4. Odds
                    od = estimate_from_entries(data["entries"]) or {}

                    races.append({
                        "rid": rid, "date": ds,
                        "name": f"{s['name']}{r['race_number']:2d}R",
                        "entries": data["entries"],
                        "info": data["race_info"],
                        "positions": positions,
                        "payouts": payouts,
                        "odds_data": od,
                    })
                    collected_rids.add(rid)
                    day_count += 1

                except Exception as e:
                    print(f"  {rid}: {e}", flush=True)
                    time.sleep(3)
                    continue

                # Save every 10 races
                if len(races) - last_save >= 10:
                    with open(cache_file, "wb") as f:
                        pickle.dump(races, f)
                    last_save = len(races)

        if day_count > 0:
            print(f"  {ds}: +{day_count}R (total {len(races)}R)", flush=True)

        d += timedelta(days=1)

    # Final save
    with open(cache_file, "wb") as f:
        pickle.dump(races, f)
    print(f"\nDone: {len(races)}R saved to {cache_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYYMMDD")
    parser.add_argument("--end", required=True, help="YYYYMMDD")
    parser.add_argument("--name", default="", help="Cache file name (default: hist_YYYY)")
    args = parser.parse_args()

    s = date(int(args.start[:4]), int(args.start[4:6]), int(args.start[6:8]))
    e = date(int(args.end[:4]), int(args.end[4:6]), int(args.end[6:8]))
    name = args.name or f"hist_{s.year}"

    print(f"Collecting {s} → {e} (cache: {name})")
    collect_range(s, e, name)


if __name__ == "__main__":
    main()
