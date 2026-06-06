"""Cache April 2026 race data + results + payouts for fast simulation."""
from __future__ import annotations
import json, os, pickle, re, sys, time
from datetime import date, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from bs4 import BeautifulSoup
from backend.database.db import init_db
from backend.scraper.netkeiba import fetch_race_list, fetch_race_card
from backend.scraper.odds import estimate_from_entries, fetch_live_combination_odds

API_H = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
_session = requests.Session()
_session.headers.update(API_H)
CACHE_FILE = os.path.join(os.path.dirname(__file__), "cached_april_races.pkl")

def fetch_result_payouts(race_id):
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    r = _session.get(url, timeout=15)
    r.encoding = r.apparent_encoding or "UTF-8"
    soup = BeautifulSoup(r.text, "lxml")
    positions = {}
    for row in soup.select(".HorseList"):
        tds = row.select("td")
        if len(tds) < 3: continue
        try: positions[int(tds[2].get_text(strip=True))] = int(tds[0].get_text(strip=True))
        except: pass
    payouts = {}
    pay_back = soup.select_one(".Result_Pay_Back")
    if pay_back:
        for tr in pay_back.select("tr"):
            th = tr.select_one("th")
            if not th: continue
            label = th.get_text(strip=True)
            result_td = tr.select_one("td.Result")
            payout_td = tr.select_one("td.Payout")
            if not result_td or not payout_td: continue
            uls = result_td.select("ul")
            cbs = []
            if uls:
                for ul in uls:
                    nums = [int(s.get_text(strip=True)) for s in ul.select("span") if s.get_text(strip=True).isdigit()]
                    if nums: cbs.append(nums)
            else:
                nums = [int(s.get_text(strip=True)) for s in result_td.select("span") if s.get_text(strip=True).isdigit()]
                if nums: cbs = [nums]
            if label == "複勝" and len(cbs)==1 and len(cbs[0])>1:
                cbs = [[n] for n in cbs[0]]
            pt = re.sub(r"<br\s*/?>", "|", payout_td.decode_contents())
            pt = re.sub(r"<[^>]+>", "", pt)
            amounts = []
            for part in pt.split("|"):
                m = re.search(r"([\d,]+)円", part.strip())
                if m: amounts.append(int(m.group(1).replace(",", "")))
            if cbs and amounts:
                for i, combo in enumerate(cbs):
                    amt = amounts[i] if i < len(amounts) else amounts[-1]
                    payouts.setdefault(label, []).append({"nums": combo, "amount": amt})
    return positions, payouts

def main():
    init_db()
    races = []
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            races = pickle.load(f)
        print(f"Loaded {len(races)} cached races")
    cached_rids = {r["rid"] for r in races}

    d = date(2026, 4, 1)
    end = date(2026, 4, 30)
    all_ids = []
    while d <= end:
        if d.weekday() in (5, 6):
            ds = d.strftime("%Y%m%d")
            time.sleep(2)
            schedules = fetch_race_list(ds)
            if schedules:
                for s in schedules:
                    for r in s.get("races", []):
                        rid = r.get("race_id", "")
                        if rid:
                            all_ids.append((rid, s["name"], r["race_number"], ds))
        d += timedelta(days=1)

    print(f"Total: {len(all_ids)} races, {len(cached_rids)} cached")

    for i, (rid, course, rnum, ds) in enumerate(all_ids):
        if rid in cached_rids:
            continue
        try:
            data = fetch_race_card(rid)
            if not data or len(data.get("entries", [])) < 3:
                print(f"  [{i+1}/{len(all_ids)}] {course}{rnum:2d}R: skip")
                continue
            time.sleep(2)
            positions, payouts = fetch_result_payouts(rid)
            if not positions:
                print(f"  [{i+1}/{len(all_ids)}] {course}{rnum:2d}R: no results")
                continue
            time.sleep(1)
            od = estimate_from_entries(data["entries"]) or {}
            try:
                live_od = fetch_live_combination_odds(rid, include_win_place=True)
                if live_od: od.update(live_od)
            except: pass
        except Exception as e:
            print(f"  [{i+1}/{len(all_ids)}] {course}{rnum:2d}R: error {e}")
            time.sleep(10)
            continue

        races.append({
            "rid": rid, "date": ds, "name": f"{course}{rnum:2d}R",
            "entries": data["entries"], "info": data["race_info"],
            "positions": positions, "payouts": payouts, "odds_data": od,
        })
        print(f"  [{i+1}/{len(all_ids)}] {course}{rnum:2d}R: cached")

        if len(races) % 20 == 0:
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(races, f)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(races, f)
    print(f"\nCached {len(races)} races to {CACHE_FILE}")

if __name__ == "__main__":
    main()
