"""Scrape race data from netkeiba.com."""
from __future__ import annotations

import json
import logging
import re
import time
import requests
from datetime import datetime, timedelta
from typing import Optional
from bs4 import BeautifulSoup
logger = logging.getLogger(__name__)

from .parser import parse_race_list, parse_race_card
from ..database.db import get_session
from ..database.models import Race, HorseEntry

SCRAPE_DELAY = 5  # seconds between requests
CACHE_TTL = timedelta(days=30)

SESSION_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(SESSION_HEADERS)
    return s


def fetch_race_list(date_str: str) -> list:
    """Fetch available races for a given date (YYYYMMDD format)."""
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"
    session = _make_session()
    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "UTF-8"
        schedules = parse_race_list(resp.text)
        for schedule in schedules:
            schedule["races"] = [r for r in schedule["races"] if r["race_number"] > 0]
        return schedules
    except requests.RequestException as e:
        logger.error("Error fetching race list: %s", e)
        return []


def fetch_race_card(race_id: str, force_refresh: bool = False) -> Optional[dict]:
    """Fetch race card (出馬表) for a given race ID.

    Checks cache first, scrapes if needed.
    If cached data has incomplete frame numbers (majority = 0), invalidates
    cache and re-scrapes — this handles early prefetch before JRA assigns枠番.
    Returns {race_info, entries} or None on error.
    """
    # Check cache
    if not force_refresh:
        db = get_session()
        try:
            cached_race = db.query(Race).filter(Race.race_id == race_id).first()
            if cached_race and cached_race.scraped_at:
                age = datetime.utcnow() - cached_race.scraped_at
                if age < CACHE_TTL:
                    entries = db.query(HorseEntry).filter(
                        HorseEntry.race_id == race_id
                    ).all()
                    if entries:
                        # Check if frame numbers are populated
                        non_scratched = [e for e in entries if not e.is_scratched]
                        zero_frames = sum(1 for e in non_scratched if e.frame_number == 0)
                        if non_scratched and zero_frames > len(non_scratched) * 0.5:
                            # Majority of entries have frame_number=0 — stale cache
                            logger.info(
                                "Race %s: %d/%d entries have frame=0, invalidating cache",
                                race_id, zero_frames, len(non_scratched),
                            )
                        else:
                            return _format_cached(cached_race, entries)
        finally:
            db.close()

    # Scrape shutuba page
    session = _make_session()
    try:
        time.sleep(SCRAPE_DELAY)
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "UTF-8"
        data = parse_race_card(resp.text)

        if not data or not data.get("entries"):
            return None

        data["race_info"]["raceId"] = race_id
        data["race_info"]["racecourseCode"] = race_id[4:6] if len(race_id) >= 6 else ""

        # Fetch pedigree and past races from shutuba_past page (single request for all horses)
        pedigree_map = _fetch_pedigree_from_shutuba_past(session, race_id)
        if pedigree_map:
            for entry in data["entries"]:
                name = entry["horseName"]
                if name in pedigree_map:
                    p = pedigree_map[name]
                    entry["sireName"] = p.get("sire", "")
                    entry["damName"] = p.get("dam", "")
                    entry["broodmareSire"] = p.get("bms", "")
                    entry["pastRaces"] = p.get("pastRaces", [])

        # Fetch odds, popularity, and horse weight from result page
        result_data = _fetch_result_data(session, race_id)
        if result_data:
            for entry in data["entries"]:
                hn = entry["horseNumber"]
                if hn in result_data:
                    rd = result_data[hn]
                    if rd.get("odds") is not None:
                        entry["odds"] = rd["odds"]
                    if rd.get("popularity") is not None:
                        entry["popularity"] = rd["popularity"]
                    if rd.get("weight") and not entry.get("horseWeight"):
                        entry["horseWeight"] = rd["weight"]

        # Cache in DB
        _cache_race_card(race_id, data)
        return data
    except requests.RequestException as e:
        logger.error("Error fetching race card: %s", e)
        return None


def fetch_pedigree_batch(horse_ids: list) -> dict:
    """Fetch pedigree for multiple horses from shutuba_past page.
    Returns {horseNumber: {sire, dam}}.
    """
    # This is now a no-op since pedigree is fetched inline
    return {}


def _fetch_result_data(session: requests.Session, race_id: str) -> dict:
    """Fetch odds, popularity, and horse weight from result page.

    The result page has:
      td[2] .Num.Txt_C = horse number
      td[9] .Odds.Txt_C = popularity (人気)
      td[10] .Odds.Txt_R = odds
      td[14] .Weight = horse weight e.g. "484(-4)"

    Returns {horseNumber: {odds, popularity, weight}}.
    """
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    try:
        time.sleep(SCRAPE_DELAY)
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "UTF-8"
        soup = BeautifulSoup(resp.text, "lxml")

        result = {}
        rows = soup.select("tr.HorseList")
        if not rows:
            return {}

        for row in rows:
            tds = row.select("td")
            if len(tds) < 11:
                continue

            # Horse number: td with class Num + Txt_C (typically td[2])
            horse_num = None
            for td in tds:
                cls_list = td.get("class", [])
                if "Num" in cls_list and "Txt_C" in cls_list:
                    try:
                        horse_num = int(td.get_text(strip=True))
                    except ValueError:
                        pass
                    break

            if horse_num is None:
                continue

            # Odds and popularity from .Odds cells
            odds_val = None
            popularity_val = None
            odds_cells = row.select("td.Odds")
            for cell in odds_cells:
                cls_list = cell.get("class", [])
                text = cell.get_text(strip=True)
                if "Txt_C" in cls_list or "BgYellow" in cls_list:
                    # This is the popularity cell
                    try:
                        popularity_val = int(text)
                    except ValueError:
                        pass
                elif "Txt_R" in cls_list:
                    # This is the odds cell
                    odds_match = re.search(r"(\d+\.?\d*)", text)
                    if odds_match:
                        odds_val = float(odds_match.group(1))

            # Horse weight from .Weight cell
            weight_text = ""
            weight_td = row.select_one("td.Weight")
            if weight_td:
                weight_text = weight_td.get_text(strip=True).replace("\n", "")
                if weight_text == "--":
                    weight_text = ""

            result[horse_num] = {
                "odds": odds_val,
                "popularity": popularity_val,
                "weight": weight_text,
            }

        return result
    except Exception as e:
        logger.error("Error fetching result data: %s", e)
        return {}


def _fetch_pedigree_from_shutuba_past(session: requests.Session, race_id: str) -> dict:
    """Fetch pedigree and past race data from shutuba_past.html page.

    Single request returns sire/dam/broodmare_sire and past race results
    for ALL horses.
    Returns {horseName: {sire, dam, bms, pastRaces: [{pos, condition, surface, distance, track, direction}]}}.
    """
    url = f"https://race.netkeiba.com/race/shutuba_past.html?race_id={race_id}"
    try:
        time.sleep(SCRAPE_DELAY)
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "UTF-8"
        soup = BeautifulSoup(resp.text, "lxml")

        result = {}

        # Each .HorseList row corresponds to one horse
        rows = soup.select(".HorseList")
        for row in rows:
            # Extract pedigree from .Horse_Info
            hi = row.select_one(".Horse_Info")
            if not hi:
                continue

            s1 = hi.select_one(".Horse01")  # 父
            s2 = hi.select_one(".Horse02")  # 馬名
            s3 = hi.select_one(".Horse03")  # 母
            s4 = hi.select_one(".Horse04")  # (母父)

            sire = s1.get_text(strip=True) if s1 else ""
            name = s2.get_text(strip=True) if s2 else ""
            dam = s3.get_text(strip=True) if s3 else ""
            bms = s4.get_text(strip=True).strip("()") if s4 else ""

            # Clean up name (remove blinker marks etc.)
            name = re.sub(r"[BＢ]$", "", name).strip()

            if not name:
                continue

            # Extract past race data from td.Past elements
            past_races = []
            past_tds = row.select("td.Past")
            for td in past_tds:
                past_race = _parse_past_race_td(td)
                if past_race:
                    past_races.append(past_race)

            if sire:
                result[name] = {
                    "sire": sire,
                    "dam": dam,
                    "bms": bms,
                    "pastRaces": past_races,
                }

        # Fallback: if no rows found, try the old .Horse_Info approach
        if not result:
            for hi in soup.select(".Horse_Info"):
                s1 = hi.select_one(".Horse01")
                s2 = hi.select_one(".Horse02")
                s3 = hi.select_one(".Horse03")
                s4 = hi.select_one(".Horse04")

                sire = s1.get_text(strip=True) if s1 else ""
                name = s2.get_text(strip=True) if s2 else ""
                dam = s3.get_text(strip=True) if s3 else ""
                bms = s4.get_text(strip=True).strip("()") if s4 else ""

                name = re.sub(r"[BＢ]$", "", name).strip()

                if name and sire:
                    result[name] = {"sire": sire, "dam": dam, "bms": bms, "pastRaces": []}

        return result
    except Exception as e:
        logger.error("Error fetching pedigree from shutuba_past: %s", e)
        return {}


def _parse_past_race_td(td) -> Optional[dict]:
    """Parse a single td.Past element for past race data.

    Text format example:
      2026.02.28 阪神10仁川SLダ2000 2:04.8良16頭 8番 4人 藤岡佑介 58.514-14-14-
    Class 'Ranking_1' indicates 1st place, 'Ranking_2' = 2nd, etc.

    Extracts: pos, condition, surface, distance, track, direction, date,
    finish_time, field_size, post_position, popularity, weight_carried,
    corners (通過順), running_style (derived: 逃げ/先行/差し/追込).
    """
    text = td.get_text(strip=True)
    if not text:
        return None

    # Position from Ranking class (most reliable)
    cls_list = td.get("class", [])
    pos = 0
    for cls in cls_list:
        m = re.match(r"Ranking_(\d+)", cls)
        if m:
            pos = int(m.group(1))
            break

    if pos == 0:
        pos_match = re.search(r"(\d+)着", text)
        if pos_match:
            pos = int(pos_match.group(1))

    # Date: YYYY.MM.DD
    date_str = ""
    date_match = re.search(r"(\d{4}\.\d{2}\.\d{2})", text)
    if date_match:
        date_str = date_match.group(1)

    # Track condition
    condition = ""
    cond_match = re.search(r"(良|稍|重|不)", text)
    if cond_match:
        cond_char = cond_match.group(1)
        condition = {"良": "良", "稍": "稍重", "重": "重", "不": "不良"}.get(cond_char, "")

    # Surface
    surface = ""
    surf_match = re.search(r"(芝|ダ)", text)
    if surf_match:
        surface = surf_match.group(1)

    # Distance
    dist = 0
    dist_match = re.search(r"(?:芝|ダ)(\d{3,4})", text)
    if dist_match:
        dist = int(dist_match.group(1))

    # Track name
    track = ""
    track_names = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
    for tn in track_names:
        if tn in text:
            track = tn
            break

    track_direction_map = {
        "札幌": "右", "函館": "右", "福島": "右", "新潟": "左",
        "東京": "左", "中山": "右", "中京": "左", "京都": "右",
        "阪神": "右", "小倉": "右",
    }
    direction = track_direction_map.get(track, "")

    # Finish time: "M:SS.f" format (e.g., 2:04.8, 1:34.2)
    finish_time = ""
    time_match = re.search(r"(\d:\d{2}\.\d)", text)
    if time_match:
        finish_time = time_match.group(1)

    # Field size: "16頭"
    field_size = 0
    field_match = re.search(r"(\d{1,2})頭", text)
    if field_match:
        field_size = int(field_match.group(1))

    # Post position: "8番"
    post_position = 0
    post_match = re.search(r"(\d{1,2})番", text)
    if post_match:
        post_position = int(post_match.group(1))

    # Popularity at the time: "4人"
    past_popularity = 0
    pop_match = re.search(r"(\d{1,2})人", text)
    if pop_match:
        past_popularity = int(pop_match.group(1))

    # Weight carried: e.g., "58.5" before corner numbers
    weight_carried = 0.0
    # Match X.Y where X is 2 digits, typically 49.0 to 60.0
    wc_match = re.search(r"(\d{2}\.\d)(?=\d+-)", text)
    if wc_match:
        try:
            weight_carried = float(wc_match.group(1))
        except ValueError:
            pass

    # Corners: "14-14-14-" or "3-2-2-" (1-4 numbers separated by hyphens)
    # They appear at the end of the text, directly after the weight (e.g., "58.5" + "14-14-14-").
    # To avoid absorbing the last digit(s) of the weight into the first corner number,
    # we strip the weight prefix from the trailing portion before matching.
    corners = []
    corner_search_text = text
    if wc_match:
        # Remove the weight match from the string so the corner regex starts fresh
        corner_search_text = text[:wc_match.start()] + text[wc_match.end():]
    corner_match = re.search(r"(\d+(?:-\d+){1,3})-?$", corner_search_text)
    if corner_match:
        corner_str = corner_match.group(1)
        try:
            corners = [int(n) for n in corner_str.split("-") if n]
        except ValueError:
            corners = []

    # Derive running style from corners (脚質)
    # Based on average corner position relative to field size
    running_style = ""
    if corners and field_size > 0:
        avg_corner = sum(corners) / len(corners)
        ratio = avg_corner / field_size
        if ratio <= 0.25:
            running_style = "逃げ"      # Front-runner
        elif ratio <= 0.50:
            running_style = "先行"      # Stalker
        elif ratio <= 0.75:
            running_style = "差し"      # Midpack closer
        else:
            running_style = "追込"      # Deep closer

    return {
        "pos": pos,
        "condition": condition,
        "surface": surface,
        "distance": dist,
        "track": track,
        "direction": direction,
        "date": date_str,
        "finishTime": finish_time,
        "fieldSize": field_size,
        "postPosition": post_position,
        "popularity": past_popularity,
        "weightCarried": weight_carried,
        "corners": corners,
        "runningStyle": running_style,
    }


def _cache_race_card(race_id: str, data: dict):
    """Store scraped race card in SQLite."""
    db = get_session()
    try:
        db.query(HorseEntry).filter(HorseEntry.race_id == race_id).delete()
        db.query(Race).filter(Race.race_id == race_id).delete()

        info = data["race_info"]
        race = Race(
            race_id=race_id,
            race_name=info.get("raceName", ""),
            race_number=info.get("raceNumber", 0),
            grade=info.get("grade"),
            distance=info.get("distance", 0),
            surface=info.get("surface", ""),
            course_detail=info.get("courseDetail", ""),
            start_time=info.get("startTime", ""),
            racecourse_code=info.get("racecourseCode", ""),
            date=info.get("date", ""),
            head_count=len(data["entries"]),
            scraped_at=datetime.utcnow(),
        )
        db.add(race)

        for entry_data in data["entries"]:
            entry = HorseEntry(
                race_id=race_id,
                frame_number=entry_data["frameNumber"],
                horse_number=entry_data["horseNumber"],
                horse_name=entry_data["horseName"],
                horse_id=entry_data.get("horseId", ""),
                sire_name=entry_data.get("sireName", ""),
                dam_name=entry_data.get("damName", ""),
                coat_color=entry_data.get("coatColor", ""),
                weight_carried=entry_data.get("weightCarried", 0),
                age=entry_data.get("age", ""),
                jockey_name=entry_data.get("jockeyName", ""),
                jockey_id=entry_data.get("jockeyId", ""),
                trainer_name=entry_data.get("trainerName", ""),
                trainer_id=entry_data.get("trainerId", ""),
                horse_weight=entry_data.get("horseWeight", ""),
                odds=entry_data.get("odds"),
                popularity=entry_data.get("popularity"),
                is_scratched=entry_data.get("isScratched", False),
                brood_mare_sire=entry_data.get("broodmareSire", ""),
                past_races_json=json.dumps(entry_data.get("pastRaces", []), ensure_ascii=False),
            )
            db.add(entry)

        db.commit()
    except Exception as e:
        db.rollback()
        logger.error("Error caching race card: %s", e)
    finally:
        db.close()


def _format_cached(race: Race, entries: list) -> dict:
    """Convert cached DB records back to dict format."""
    return {
        "race_info": {
            "raceId": race.race_id,
            "raceName": race.race_name,
            "raceNumber": race.race_number,
            "grade": race.grade,
            "distance": race.distance,
            "surface": race.surface,
            "courseDetail": race.course_detail,
            "startTime": race.start_time,
            "racecourseCode": race.racecourse_code,
            "date": race.date,
            "headCount": race.head_count,
        },
        "entries": [
            {
                "frameNumber": e.frame_number,
                "horseNumber": e.horse_number,
                "horseName": e.horse_name,
                "horseId": e.horse_id,
                "sireName": e.sire_name,
                "damName": e.dam_name,
                "coatColor": e.coat_color,
                "weightCarried": e.weight_carried,
                "age": e.age,
                "jockeyName": e.jockey_name,
                "jockeyId": e.jockey_id,
                "trainerName": e.trainer_name,
                "trainerId": e.trainer_id,
                "horseWeight": e.horse_weight,
                "odds": e.odds,
                "popularity": e.popularity,
                "isScratched": e.is_scratched,
                "broodmareSire": getattr(e, "brood_mare_sire", "") or "",
                "pastRaces": json.loads(getattr(e, "past_races_json", "[]") or "[]"),
            }
            for e in entries
        ],
    }
