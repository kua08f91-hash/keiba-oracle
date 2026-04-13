"""Collect historical JRA race data for weight optimization.

Scrapes race results from netkeiba for the past 3 years.
Saves results to a JSON file for offline optimization.

Usage:
  python3 backend/collect_historical.py [--months 36]
"""
import requests
import time
import json
import os
import sys
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_BASE = "http://localhost:8000/api"
NETKEIBA_BASE = "https://race.netkeiba.com/race"
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "historical_races.json")

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def get_jra_race_dates(months_back=36):
    """Generate JRA race dates (weekends + holidays) for the past N months."""
    dates = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back * 30)

    current = start_date
    while current <= end_date:
        # JRA races are typically Sat/Sun
        if current.weekday() in (5, 6):  # Sat=5, Sun=6
            dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)

    return dates


def fetch_race_list_from_netkeiba(date_str):
    """Fetch race list from netkeiba for a given date.

    Uses race_list_sub.html which is the AJAX-loaded content
    containing actual race_id links.
    """
    import re
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "UTF-8"

        # Extract all race_id values from the HTML
        all_ids = re.findall(r"race_id=(\d+)", resp.text)

        race_ids = []
        for rid in all_ids:
            # Only JRA races (course code 01-10)
            if len(rid) >= 12 and rid[4:6] in ["01","02","03","04","05","06","07","08","09","10"]:
                if rid not in race_ids:
                    race_ids.append(rid)

        return race_ids
    except Exception as e:
        print(f"  [WARN] Failed to fetch race list for {date_str}: {e}")
        return []


def fetch_result_and_entries(race_id):
    """Fetch race result and basic entry data from netkeiba result page.

    Returns (race_info_dict, entries_list, results_dict) or None on failure.
    """
    url = f"{NETKEIBA_BASE}/result.html?race_id={race_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")

        # Get race info
        race_info = {"raceId": race_id}

        # Race name
        race_name_el = soup.select_one(".RaceName")
        if race_name_el:
            race_info["raceName"] = race_name_el.get_text(strip=True)

        # Race data (distance, surface, etc.)
        race_data = soup.select_one(".RaceData01")
        if race_data:
            text = race_data.get_text()
            import re
            # Surface + distance
            m = re.search(r"(芝|ダ|障)(\d+)m", text)
            if m:
                surface_map = {"芝": "芝", "ダ": "ダート", "障": "障害"}
                race_info["surface"] = surface_map.get(m.group(1), "芝")
                race_info["distance"] = int(m.group(2))

            # Direction
            if "右" in text:
                race_info["courseDetail"] = "右"
            elif "左" in text:
                race_info["courseDetail"] = "左"
            else:
                race_info["courseDetail"] = ""

            # Track condition
            tc_match = re.search(r"馬場:(良|稍重|重|不良|稍|不)", text)
            if tc_match:
                tc = tc_match.group(1)
                tc_map = {"良": "良", "稍": "稍重", "稍重": "稍重", "重": "重", "不": "不良", "不良": "不良"}
                race_info["trackCondition"] = tc_map.get(tc, "")
            else:
                race_info["trackCondition"] = ""

        # Racecourse code from race_id
        race_info["racecourseCode"] = race_id[4:6]

        # Parse result table
        table = soup.select_one("table.RaceTable01")
        if not table:
            return None

        results = {}
        entries = []
        rows = table.select("tr.HorseList")

        for row in rows:
            tds = row.select("td")
            if len(tds) < 14:
                continue

            # Finish position
            pos_text = tds[0].get_text(strip=True)
            try:
                finish_pos = int(pos_text)
            except:
                continue

            # Horse number
            try:
                horse_num = int(tds[2].get_text(strip=True))
            except:
                continue

            # Horse name
            horse_name = tds[3].get_text(strip=True)

            # Age/sex
            age_str = tds[4].get_text(strip=True)

            # Weight carried
            try:
                weight_carried = float(tds[5].get_text(strip=True))
            except:
                weight_carried = 0

            # Jockey
            jockey_name = tds[6].get_text(strip=True)

            # Trainer
            trainer_name = ""
            if len(tds) > 13:
                trainer_name = tds[13].get_text(strip=True)

            # Odds
            odds = None
            if len(tds) > 10:
                try:
                    odds = float(tds[10].get_text(strip=True))
                except:
                    pass

            # Popularity
            popularity = None
            if len(tds) > 9:
                try:
                    popularity = int(tds[9].get_text(strip=True))
                except:
                    pass

            # Horse weight
            horse_weight = ""
            if len(tds) > 14:
                horse_weight = tds[14].get_text(strip=True)

            results[horse_num] = finish_pos

            # Get sire name from horse detail (link)
            horse_link = tds[3].select_one("a")
            horse_id = ""
            if horse_link:
                href = horse_link.get("href", "")
                import re
                m = re.search(r"/horse/(\w+)", href)
                if m:
                    horse_id = m.group(1)

            entries.append({
                "horseNumber": horse_num,
                "horseName": horse_name,
                "horseId": horse_id,
                "age": age_str,
                "weightCarried": weight_carried,
                "jockeyName": jockey_name,
                "trainerName": trainer_name,
                "odds": odds,
                "popularity": popularity,
                "horseWeight": horse_weight,
                "sireName": "",  # Will need separate fetch for pedigree
                "damName": "",
                "isScratched": False,
                "pastRaces": [],  # Will need separate fetch
            })

        if not results or not entries:
            return None

        return race_info, entries, results

    except Exception as e:
        return None


def fetch_pedigree_and_past(race_id):
    """Fetch pedigree + past race data from shutuba_past.html."""
    url = f"{NETKEIBA_BASE}/shutuba_past.html?race_id={race_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")

        data = {}  # horse_name -> {sire, dam, pastRaces}

        rows = soup.select("tr.HorseList")
        for row in rows:
            # Horse name
            name_el = row.select_one(".HorseName a")
            if not name_el:
                continue
            horse_name = name_el.get_text(strip=True)

            # Pedigree
            breed_els = row.select(".Pedigree a")
            sire = breed_els[0].get_text(strip=True) if len(breed_els) > 0 else ""
            dam = breed_els[1].get_text(strip=True) if len(breed_els) > 1 else ""

            # Past races
            past_races = []
            past_tds = row.select("td.Past")
            for td in past_tds[:5]:
                race_data = _parse_past_td(td)
                if race_data:
                    past_races.append(race_data)

            data[horse_name] = {
                "sire": sire,
                "dam": dam,
                "pastRaces": past_races,
            }

        return data
    except:
        return {}


def _parse_past_td(td):
    """Parse a past race td element."""
    import re
    text = td.get_text("\n", strip=True)
    if not text or text == "-":
        return None

    result = {}

    # Finish position (着順)
    rank_el = td.select_one(".Rank")
    if rank_el:
        try:
            result["pos"] = int(rank_el.get_text(strip=True))
        except:
            result["pos"] = 0
    else:
        # Try first line
        lines = text.split("\n")
        try:
            result["pos"] = int(lines[0])
        except:
            result["pos"] = 0

    # Track name
    track_el = td.select_one(".Data02")
    if track_el:
        track_text = track_el.get_text(strip=True)
        # Extract track name (e.g., "中山", "阪神")
        for tn in ["札幌","函館","福島","新潟","東京","中山","中京","京都","阪神","小倉"]:
            if tn in track_text:
                result["track"] = tn
                break

        # Surface + distance
        m = re.search(r"(芝|ダ)(\d+)", track_text)
        if m:
            result["surface"] = "芝" if m.group(1) == "芝" else "ダート"
            result["distance"] = int(m.group(2))

        # Direction
        if "右" in track_text:
            result["direction"] = "右"
        elif "左" in track_text:
            result["direction"] = "左"

    return result if result.get("pos", 0) > 0 else None


def main():
    months = 36
    if len(sys.argv) > 1 and sys.argv[1] == "--months":
        months = int(sys.argv[2])

    print(f"Collecting JRA race data for the past {months} months...")

    # Generate dates
    dates = get_jra_race_dates(months)
    print(f"Total weekend dates to check: {len(dates)}")

    # Load existing data if available
    all_races = []
    existing_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            all_races = json.load(f)
            existing_ids = {r["race_id"] for r in all_races}
            print(f"Loaded {len(all_races)} existing races")

    # Process each date
    total_new = 0
    for i, date_str in enumerate(dates):
        print(f"\n[{i+1}/{len(dates)}] {date_str}...", end="", flush=True)

        race_ids = fetch_race_list_from_netkeiba(date_str)
        time.sleep(1)

        if not race_ids:
            print(f" no races")
            continue

        new_count = 0
        for race_id in race_ids:
            if race_id in existing_ids:
                continue

            # Fetch result + entries
            result_data = fetch_result_and_entries(race_id)
            if result_data is None:
                time.sleep(0.5)
                continue
            time.sleep(1)

            race_info, entries, results = result_data

            # Fetch pedigree + past races
            pedigree_data = fetch_pedigree_and_past(race_id)
            time.sleep(1)

            # Merge pedigree into entries
            if pedigree_data:
                for entry in entries:
                    name = entry["horseName"]
                    if name in pedigree_data:
                        p = pedigree_data[name]
                        entry["sireName"] = p.get("sire", "")
                        entry["damName"] = p.get("dam", "")
                        entry["pastRaces"] = p.get("pastRaces", [])

            race_record = {
                "race_id": race_id,
                "date": date_str,
                "race_info": race_info,
                "entries": entries,
                "results": results,
            }
            all_races.append(race_record)
            existing_ids.add(race_id)
            new_count += 1
            total_new += 1

        print(f" {len(race_ids)} races found, {new_count} new")

        # Save periodically
        if total_new > 0 and total_new % 50 == 0:
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            with open(OUTPUT_FILE, "w") as f:
                json.dump(all_races, f, ensure_ascii=False)
            print(f"  [SAVED] {len(all_races)} total races")

    # Final save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_races, f, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Collection complete!")
    print(f"Total races: {len(all_races)}")
    print(f"New races added: {total_new}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
