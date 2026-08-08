"""Fetch July-August race results from netkeiba and validate D5 parameters.

Usage: python3 scripts/fetch_july_results.py
Run when netkeiba is accessible (daytime hours).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
ARCHIVE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "data", "archive")
CACHE_FILE = "/tmp/keiba_results_cache.json"
OUTPUT_FILE = "/tmp/all_results_jul_aug.json"

BET_UNIT = 500


def fetch_result(race_id: str, cache: dict) -> dict:
    if race_id in cache:
        return cache[race_id]
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "UTF-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        positions = {}
        for row in soup.select("tr.HorseList"):
            tds = row.select("td")
            if len(tds) < 3:
                continue
            m = re.match(r"(\d+)", tds[0].get_text(strip=True))
            if not m:
                continue
            try:
                positions[int(tds[2].get_text(strip=True))] = int(m.group(1))
            except ValueError:
                pass
        payouts = {}
        for table in soup.select(".FullWrap table"):
            for row in table.select("tr"):
                th = row.select_one("th")
                tds_p = row.select("td")
                if th and len(tds_p) >= 2:
                    payouts[th.get_text(strip=True)] = {"payout": tds_p[1].get_text(strip=True)}
        cache[race_id] = {"positions": {str(k): v for k, v in positions.items()}, "payouts": payouts}
        return cache[race_id]
    except Exception as e:
        print(f"  Error {race_id}: {e}")
        return {"positions": {}, "payouts": {}}


def collect_results():
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cache = json.load(f)

    dates = sorted([
        fn.replace("predictions_", "").replace(".json", "")
        for fn in os.listdir(ARCHIVE)
        if fn.startswith("predictions_2026") and fn.endswith(".json")
    ])

    all_results = []
    for date in dates:
        fpath = os.path.join(ARCHIVE, f"predictions_{date}.json")
        with open(fpath) as f:
            day = json.load(f)
        courses = day.get("courses", [])
        n = 0
        for course in courses:
            for r in course.get("races", []):
                rid = r.get("raceId", "")
                if not rid:
                    continue
                cached = rid in cache
                result = fetch_result(rid, cache)
                if not cached:
                    time.sleep(0.3)
                pos = {int(k): v for k, v in result["positions"].items()} if result["positions"] else {}
                if not pos:
                    continue
                payouts = result["payouts"]
                preds = sorted(r.get("predictions", []), key=lambda x: x.get("score", 0), reverse=True)
                if not preds or preds[0].get("score", 0) <= 0:
                    continue

                up = int(payouts.get("馬単", {}).get("payout", "0").replace(",", "").replace("円", "") or 0)
                mp = int(payouts.get("馬連", {}).get("payout", "0").replace(",", "").replace("円", "") or 0)
                gap = preds[0]["score"] - preds[1]["score"] if len(preds) > 1 else 0

                all_results.append({
                    "date": date,
                    "course": course.get("name", ""),
                    "rnum": r.get("raceNumber", ""),
                    "honmei_hn": preds[0]["horseNumber"],
                    "honmei_score": preds[0]["score"],
                    "honmei_pos": pos.get(preds[0]["horseNumber"], 99),
                    "preds": [[p["horseNumber"], p.get("score", 0)] for p in preds],
                    "positions": {str(k): v for k, v in pos.items()},
                    "umatan_p": up,
                    "umaren_p": mp,
                    "wide_p": int(mp * 0.5),
                    "gap": round(gap, 2),
                })
                n += 1
        print(f"{date}: {n}R")

    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, ensure_ascii=False)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, ensure_ascii=False)
    print(f"\nTotal: {len(all_results)}R saved to {OUTPUT_FILE}")
    return all_results


def simulate(data, score_min, ut, ur, wd):
    inv = 0; ret = 0; hits = 0; nr = 0
    for r in data:
        if r["honmei_score"] < score_min:
            continue
        nr += 1
        hn = r["honmei_hn"]
        pos = {int(k): v for k, v in r["positions"].items()}
        preds = r["preds"]  # [[hn, score], ...]
        for i in range(1, min(ut + 1, len(preds))):
            inv += BET_UNIT
            if pos.get(hn) == 1 and pos.get(preds[i][0]) == 2:
                ret += r["umatan_p"] * BET_UNIT / 100; hits += 1
        for i in range(1, min(ur + 1, len(preds))):
            inv += BET_UNIT
            t2 = {h for h, p in pos.items() if p <= 2}
            if hn in t2 and preds[i][0] in t2:
                ret += r["umaren_p"] * BET_UNIT / 100; hits += 1
        for i in range(1, min(wd + 1, len(preds))):
            inv += BET_UNIT
            t3 = {h for h, p in pos.items() if p <= 3}
            if hn in t3 and preds[i][0] in t3:
                ret += r["wide_p"] * BET_UNIT / 100; hits += 1
    roi = ret / inv * 100 if inv > 0 else 0
    return nr, inv, ret, roi, hits


def run_simulation(data):
    print("\n" + "=" * 80)
    print(f"  D5パラメータ検証 ({len(data)}R)")
    print("=" * 80)
    print(f"  {'◎min':>5} {'馬単':>4} {'馬連':>4} {'ﾜｲﾄﾞ':>4} {'点/R':>4} {'R数':>4} {'投資':>10} {'回収':>10} {'ROI':>7} {'的中':>4}")

    results = []
    for sm in [75, 76, 77, 78, 79, 80]:
        for ut in [1, 2, 3]:
            for ur in [1, 2, 3]:
                for wd in [2, 3, 4]:
                    ppr = ut + ur + wd
                    if ppr < 4 or ppr > 10:
                        continue
                    nr, inv, ret, roi, hits = simulate(data, sm, ut, ur, wd)
                    if nr < 3 or inv == 0:
                        continue
                    results.append((sm, ut, ur, wd, ppr, nr, inv, ret, roi, hits))

    results.sort(key=lambda x: -x[8])
    for r in results[:15]:
        sm, ut, ur, wd, ppr, nr, inv, ret, roi, hits = r
        profit = ret - inv
        print(f"  >={sm:>3} {ut:>4} {ur:>4} {wd:>4} {ppr:>4} {nr:>4} "
              f"¥{inv:>8,.0f} ¥{ret:>8,.0f} {roi:>6.0f}% {hits:>4}  ({'+' if profit >= 0 else ''}¥{profit:,.0f})")


if __name__ == "__main__":
    data = collect_results()
    if data:
        run_simulation(data)
    else:
        print("No data collected. Is netkeiba accessible?")
