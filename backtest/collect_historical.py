"""Collect historical race data for backtest (Phase 0-F).

Fetches race cards + results + payouts from netkeiba for multiple years.
Saves to backtest/cache/ as pickle files per year-month.
Designed for long-running background execution with resume support.

Usage:
    /usr/bin/python3 -m backtest.collect_historical --year 2024
    /usr/bin/python3 -m backtest.collect_historical --year 2025
    /usr/bin/python3 -m backtest.collect_historical --year 2024 --month 6
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


def fetch_result_payouts(race_id: str) -> tuple:
    """Fetch finish positions and payouts."""
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    r = _session.get(url, timeout=20)
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

    payouts = {}
    pay_back = soup.select_one(".Result_Pay_Back")
    if pay_back:
        for tr in pay_back.select("tr"):
            th = tr.select_one("th")
            if not th:
                continue
            label = th.get_text(strip=True)
            result_td = tr.select_one("td.Result")
            payout_td = tr.select_one("td.Payout")
            if not result_td or not payout_td:
                continue
            uls = result_td.select("ul")
            cbs = []
            if uls:
                for ul in uls:
                    nums = [int(s.get_text(strip=True)) for s in ul.select("span") if s.get_text(strip=True).isdigit()]
                    if nums:
                        cbs.append(nums)
            else:
                nums = [int(s.get_text(strip=True)) for s in result_td.select("span") if s.get_text(strip=True).isdigit()]
                if nums:
                    cbs = [nums]
            if label == "複勝" and len(cbs) == 1 and len(cbs[0]) > 1:
                cbs = [[n] for n in cbs[0]]
            pt = re.sub(r"<br\s*/?>", "|", payout_td.decode_contents())
            pt = re.sub(r"<[^>]+>", "", pt)
            amounts = []
            for part in pt.split("|"):
                m = re.search(r"([\d,]+)円", part.strip())
                if m:
                    amounts.append(int(m.group(1).replace(",", "")))
            if cbs and amounts:
                for i, combo in enumerate(cbs):
                    amt = amounts[i] if i < len(amounts) else amounts[-1]
                    payouts.setdefault(label, []).append({"nums": combo, "amount": amt})

    return positions, payouts


def collect_month(year: int, month: int) -> list:
    """Collect all races for a given month. Returns list of race dicts."""
    cache_file = os.path.join(CACHE_DIR, f"hist_{year}{month:02d}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            races = pickle.load(f)
        print(f"  {year}/{month:02d}: {len(races)}R (cached)")
        return races

    init_db()
    races = []
    d = date(year, month, 1)
    end_month = month + 1 if month < 12 else 1
    end_year = year if month < 12 else year + 1
    end_d = date(end_year, end_month, 1)

    while d < end_d:
        if d.weekday() in (5, 6):  # JRA races on Sat/Sun
            ds = d.strftime("%Y%m%d")
            time.sleep(2)
            try:
                schedules = fetch_race_list(ds)
            except Exception as e:
                print(f"    {ds}: race list error {e}")
                d += timedelta(days=1)
                continue

            if not schedules:
                d += timedelta(days=1)
                continue

            for s in schedules:
                for r in s.get("races", []):
                    rid = r.get("race_id", "")
                    if not rid:
                        continue
                    try:
                        data = fetch_race_card(rid)
                        if not data or len(data.get("entries", [])) < 3:
                            continue

                        time.sleep(2)
                        positions, payouts = fetch_result_payouts(rid)
                        if not positions:
                            continue

                        time.sleep(1)
                        od = estimate_from_entries(data["entries"]) or {}

                        races.append({
                            "rid": rid,
                            "date": ds,
                            "name": f"{s['name']}{r['race_number']:2d}R",
                            "entries": data["entries"],
                            "info": data["race_info"],
                            "positions": positions,
                            "payouts": payouts,
                            "odds_data": od,
                        })

                        if len(races) % 10 == 0:
                            print(f"    {ds} {s['name']}{r['race_number']:2d}R [{len(races)}R total]")

                    except Exception as e:
                        print(f"    {rid}: error {e}")
                        time.sleep(10)
                        continue

        d += timedelta(days=1)

    # Save
    with open(cache_file, "wb") as f:
        pickle.dump(races, f)
    print(f"  {year}/{month:02d}: {len(races)}R collected")
    return races


def main():
    parser = argparse.ArgumentParser(description="Collect historical race data")
    parser.add_argument("--year", type=int, required=True, help="Year (e.g. 2024)")
    parser.add_argument("--month", type=int, default=0, help="Month (0=all)")
    args = parser.parse_args()

    print(f"Collecting {args.year}" + (f"/{args.month:02d}" if args.month else " full year"))

    total = 0
    months = [args.month] if args.month else list(range(1, 13))

    for m in months:
        races = collect_month(args.year, m)
        total += len(races)

    print(f"\nTotal: {total}R for {args.year}")
    print(f"Saved to backtest/cache/hist_{args.year}*.pkl")


if __name__ == "__main__":
    main()
