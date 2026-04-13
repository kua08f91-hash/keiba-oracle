"""Fast historical data collection - results + pedigree + past races.

Collects from netkeiba result pages and shutuba_past pages.
Saves to JSON for offline weight optimization.

More efficient approach:
- result.html: finish positions, odds, popularity, horse weight, jockey, trainer, age
- shutuba_past.html: sire name, dam name, past 5 race results
- Both pages per race = complete factor data for optimization
"""
import requests
import time
import json
import os
import sys
import re
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

NETKEIBA_BASE = "https://race.netkeiba.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "historical_races.json")


def get_jra_race_dates(months_back=36):
    """Generate weekend dates for the past N months."""
    dates = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back * 30)
    current = start_date
    while current <= end_date:
        if current.weekday() in (5, 6):
            dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return dates


def fetch_race_ids(date_str):
    """Fetch JRA race IDs for a given date."""
    url = f"{NETKEIBA_BASE}/top/race_list_sub.html?kaisai_date={date_str}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "UTF-8"
        all_ids = re.findall(r"race_id=(\d+)", resp.text)
        race_ids = []
        for rid in all_ids:
            if len(rid) >= 12 and rid[4:6] in ["01","02","03","04","05","06","07","08","09","10"]:
                if rid not in race_ids:
                    race_ids.append(rid)
        return race_ids
    except:
        return []


def fetch_race_result(race_id):
    """Fetch complete race data from result page."""
    url = f"{NETKEIBA_BASE}/race/result.html?race_id={race_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")

        # Race info
        race_info = {"raceId": race_id, "racecourseCode": race_id[4:6]}

        race_data01 = soup.select_one(".RaceData01")
        if race_data01:
            text = race_data01.get_text()
            m = re.search(r"(芝|ダ|障)(\d+)m", text)
            if m:
                race_info["surface"] = {"芝": "芝", "ダ": "ダート", "障": "障害"}.get(m.group(1), "芝")
                race_info["distance"] = int(m.group(2))
            if "右" in text:
                race_info["courseDetail"] = "右"
            elif "左" in text:
                race_info["courseDetail"] = "左"
            else:
                race_info["courseDetail"] = ""
            tc = re.search(r"馬場:(良|稍重|重|不良|稍|不)", text)
            if tc:
                tc_map = {"良": "良", "稍": "稍重", "稍重": "稍重", "重": "重", "不": "不良", "不良": "不良"}
                race_info["trackCondition"] = tc_map.get(tc.group(1), "")
            else:
                race_info["trackCondition"] = ""

        # Results table
        table = soup.select_one("table.RaceTable01")
        if not table:
            return None

        results = {}
        entries = []
        for row in table.select("tr.HorseList"):
            tds = row.select("td")
            if len(tds) < 11:
                continue
            try:
                finish_pos = int(tds[0].get_text(strip=True))
                horse_num = int(tds[2].get_text(strip=True))
            except:
                continue

            results[horse_num] = finish_pos

            entry = {
                "horseNumber": horse_num,
                "horseName": tds[3].get_text(strip=True),
                "age": tds[4].get_text(strip=True),
                "weightCarried": 0,
                "jockeyName": tds[6].get_text(strip=True),
                "trainerName": tds[13].get_text(strip=True) if len(tds) > 13 else "",
                "odds": None,
                "popularity": None,
                "horseWeight": tds[14].get_text(strip=True) if len(tds) > 14 else "",
                "sireName": "",
                "damName": "",
                "pastRaces": [],
                "isScratched": False,
            }
            try:
                entry["weightCarried"] = float(tds[5].get_text(strip=True))
            except:
                pass
            try:
                entry["odds"] = float(tds[10].get_text(strip=True))
            except:
                pass
            try:
                entry["popularity"] = int(tds[9].get_text(strip=True))
            except:
                pass
            entries.append(entry)

        if not results:
            return None
        return race_info, entries, results
    except:
        return None


def fetch_pedigree_past(race_id):
    """Fetch pedigree + past race data from shutuba_past page.

    Page structure:
    - Each horse has multiple tr.HorseList rows
    - Horse info row has .Horse_Info with .Horse02(name), .Horse03(dam), .Horse04(sire)
    - Past race rows have td.Past with .Data01(date/track/rank) and .Data02(detail)
    """
    url = f"{NETKEIBA_BASE}/race/shutuba_past.html?race_id={race_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")

        data = {}
        current_horse = None

        for row in soup.select("tr.HorseList"):
            # Check if this row has horse info (name, pedigree)
            horse_info = row.select_one(".Horse_Info")
            if horse_info:
                # Get horse name
                name_el = horse_info.select_one(".Horse02 a")
                if not name_el:
                    name_el = horse_info.select_one("a[href*='horse']")
                if name_el:
                    current_horse = name_el.get_text(strip=True)
                else:
                    current_horse = None
                    continue

                # Get sire from Horse04 (in parentheses)
                sire_el = horse_info.select_one(".Horse04")
                sire = sire_el.get_text(strip=True).strip("()（）") if sire_el else ""

                # Get dam from Horse03
                dam_el = horse_info.select_one(".Horse03")
                dam = dam_el.get_text(strip=True) if dam_el else ""

                if current_horse not in data:
                    data[current_horse] = {"sire": sire, "dam": dam, "pastRaces": []}

            # Check for past race data in this row
            past_tds = row.select("td.Past")
            if past_tds and current_horse and current_horse in data:
                for td in past_tds:
                    pr = _parse_past_td_v2(td)
                    if pr:
                        # Avoid duplicates
                        if pr not in data[current_horse]["pastRaces"]:
                            data[current_horse]["pastRaces"].append(pr)

        # Limit to 5 past races per horse
        for horse in data:
            data[horse]["pastRaces"] = data[horse]["pastRaces"][:5]

        return data
    except:
        return {}


def _parse_past_td_v2(td):
    """Parse a past race td element (v2 structure)."""
    data01 = td.select_one(".Data01")
    data02 = td.select_one(".Data02")

    if not data01:
        return None

    pr = {}

    # Finish position from .Num span in Data01
    num_el = data01.select_one(".Num")
    if num_el:
        try:
            pr["pos"] = int(num_el.get_text(strip=True))
        except:
            return None
    else:
        return None

    # Track name from Data01 text
    spans = data01.select("span")
    if spans:
        date_track = spans[0].get_text(strip=True)
        for tn in ["札幌","函館","福島","新潟","東京","中山","中京","京都","阪神","小倉"]:
            if tn in date_track:
                pr["track"] = tn
                break

    # Detail from Data02
    if data02:
        detail = data02.get_text(strip=True)
        m = re.search(r"(芝|ダ)(\d+)", detail)
        if m:
            pr["surface"] = "芝" if m.group(1) == "芝" else "ダート"
            pr["distance"] = int(m.group(2))
        if "右" in detail:
            pr["direction"] = "右"
        elif "左" in detail:
            pr["direction"] = "左"

    return pr if pr.get("pos", 0) > 0 else None


def main():
    months = 36
    if "--months" in sys.argv:
        idx = sys.argv.index("--months")
        months = int(sys.argv[idx + 1])

    print(f"Collecting JRA race data for the past {months} months...")
    dates = get_jra_race_dates(months)
    print(f"Weekend dates to check: {len(dates)}")

    # Load existing
    all_races = []
    existing_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            all_races = json.load(f)
            existing_ids = {r["race_id"] for r in all_races}
            print(f"Loaded {len(all_races)} existing races")

    total_new = 0
    for i, date_str in enumerate(dates):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"\n[{i+1}/{len(dates)}] {date_str}...", end="", flush=True)

        race_ids = fetch_race_ids(date_str)
        time.sleep(0.8)

        if not race_ids:
            continue

        new_count = 0
        for race_id in race_ids:
            if race_id in existing_ids:
                continue

            # Fetch result
            result_data = fetch_race_result(race_id)
            if result_data is None:
                time.sleep(0.5)
                continue
            time.sleep(0.8)

            race_info, entries, results = result_data

            # Fetch pedigree + past races
            pedigree = fetch_pedigree_past(race_id)
            time.sleep(0.8)

            if pedigree:
                for entry in entries:
                    name = entry["horseName"]
                    if name in pedigree:
                        p = pedigree[name]
                        entry["sireName"] = p.get("sire", "")
                        entry["damName"] = p.get("dam", "")
                        entry["pastRaces"] = p.get("pastRaces", [])

            all_races.append({
                "race_id": race_id,
                "date": date_str,
                "race_info": race_info,
                "entries": entries,
                "results": {str(k): v for k, v in results.items()},
            })
            existing_ids.add(race_id)
            new_count += 1
            total_new += 1

        if new_count > 0:
            print(f" +{new_count}", end="", flush=True)

        # Save after each day that had races
        if new_count > 0 and total_new >= 36:
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            with open(OUTPUT_FILE, "w") as f:
                json.dump(all_races, f, ensure_ascii=False)
            print(f"\n  [SAVED] {len(all_races)} total races ({total_new} new)")

    # Final save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_races, f, ensure_ascii=False)

    print(f"\n\n{'='*60}")
    print(f"Collection complete!")
    print(f"Total races: {len(all_races)}")
    print(f"New races added: {total_new}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
