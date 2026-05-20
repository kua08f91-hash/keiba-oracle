"""Fast historical data collector with resume and incremental save.

Collects race cards + results for backtesting.
Uses db.netkeiba.com for payouts (more reliable than race.netkeiba result.html).
Saves every 20 races for crash recovery.

Usage:
    /usr/bin/python3 -m backtest.collect_fast --year 2024
    /usr/bin/python3 -m backtest.collect_fast --year 2025
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


def fetch_db_payouts(race_id: str) -> dict:
    """Fetch payouts from db.netkeiba.com (reliable for past races)."""
    url = f"https://db.netkeiba.com/race/{race_id}/"
    try:
        r = _session.get(url, timeout=15)
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
    except Exception:
        return {}


def fetch_positions(race_id: str) -> dict:
    """Fetch finish positions from race.netkeiba.com result page."""
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    try:
        r = _session.get(url, timeout=15)
        r.encoding = r.apparent_encoding or "UTF-8"
        soup = BeautifulSoup(r.text, "lxml")
        positions = {}
        for row in soup.select(".HorseList"):
            tds = row.select("td")
            if len(tds) < 3:
                continue
            try:
                positions[int(tds[2].get_text(strip=True))] = int(tds[0].get_text(strip=True))
            except (ValueError, IndexError):
                pass
        return positions
    except Exception:
        return {}


def collect_year(year: int):
    """Collect all races for a year with resume support."""
    cache_file = os.path.join(CACHE_DIR, f"hist_{year}.pkl")

    # Resume from existing
    races = []
    collected_rids = set()
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            races = pickle.load(f)
        collected_rids = {r["rid"] for r in races}
        print(f"Resuming {year}: {len(races)} already collected")

    init_db()

    d = date(year, 1, 1)
    end_d = date(year, 12, 31)
    last_save = len(races)

    while d <= end_d:
        if d.weekday() not in (5, 6):
            d += timedelta(days=1)
            continue

        ds = d.strftime("%Y%m%d")
        time.sleep(1)

        try:
            schedules = fetch_race_list(ds)
        except Exception as e:
            print(f"  {ds}: list error {e}")
            time.sleep(5)
            d += timedelta(days=1)
            continue

        if not schedules:
            d += timedelta(days=1)
            continue

        for s in schedules:
            for r in s.get("races", []):
                rid = r.get("race_id", "")
                if not rid or rid in collected_rids:
                    continue

                try:
                    data = fetch_race_card(rid)
                    if not data or len(data.get("entries", [])) < 3:
                        continue

                    time.sleep(1)
                    positions = fetch_positions(rid)
                    if not positions:
                        continue

                    time.sleep(1)
                    payouts = fetch_db_payouts(rid)

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

                    if len(races) % 10 == 0:
                        print(f"  {ds} {s['name']}{r['race_number']:2d}R [{len(races)}R]", flush=True)

                    # Save every 20 races
                    if len(races) - last_save >= 20:
                        with open(cache_file, "wb") as f:
                            pickle.dump(races, f)
                        last_save = len(races)

                except Exception as e:
                    print(f"  {rid}: error {e}")
                    time.sleep(5)

        d += timedelta(days=1)

    # Final save
    with open(cache_file, "wb") as f:
        pickle.dump(races, f)
    print(f"\n{year}: {len(races)}R saved to {cache_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()
    collect_year(args.year)


if __name__ == "__main__":
    main()
