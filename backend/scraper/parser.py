"""Parse netkeiba HTML pages into structured data."""
from __future__ import annotations

import logging
import re
from bs4 import BeautifulSoup, Tag
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# Known JRA graded race names — used to cross-validate icon-based grade detection
# (netkeiba uses Icon_GradeType2/3 for Listed/Open races too, causing false positives)
_KNOWN_GRADED = {
    "GI": [
        "フェブラリーS","高松宮記念","大阪杯","桜花賞","皐月賞","天皇賞",
        "NHKマイルC","NHKマイル","ヴィクトリアマイル","オークス","日本ダービー",
        "東京優駿","安田記念","宝塚記念","スプリンターズS","秋華賞","菊花賞",
        "エリザベス女王杯","マイルCS","ジャパンC","チャンピオンズC","有馬記念",
        "ホープフルS","朝日杯FS","阪神JF",
        "中山グランドジャンプ","中山GJ","中山大障害",
    ],
    "GII": [
        "日経新春杯","アメリカJCC","東海S","京都記念","共同通信杯",
        "きさらぎ賞","京都牝馬S","中山記念","阪急杯","チューリップ賞","弥生賞",
        "金鯱賞","フィリーズR","フィリーズレビュー","スプリングS","日経賞",
        "阪神大賞典","毎日杯","マイラーズC","フローラS","青葉賞",
        "京王杯SC","目黒記念","京都新聞杯",
        "札幌記念","セントウルS","ローズS","オールカマー","神戸新聞杯",
        "セントライト記念","京都大賞典","府中牝馬S","富士S","スワンS",
        "デイリー杯","京阪杯","アルゼンチン共和国杯","ステイヤーズS",
        "阪神カップ","阪神C","阪神牝馬S",
        "東京ハイジャンプ","阪神スプリングジャンプ","東京オータムジャンプ","京都ハイジャンプ",
    ],
    "GIII": [
        "中山金杯","京都金杯","シンザン記念","フェアリーS","京成杯",
        "愛知杯","東京新聞杯","シルクロードS","根岸S","小倉大賞典",
        "ダイヤモンドS","アーリントンC","中山牝馬S","フラワーC","ファルコンS",
        "オーシャンS","ダービー卿CT","ダービーCT","マーガレットS","アンタレスS",
        "ニュージーランドT","NZT","AJC",
        "福島牝馬S","新潟大賞典","葵S","鳴尾記念","エプソムC",
        "函館スプリントS","マーメイドS","ユニコーンS","CBC賞","ラジオNIKKEI賞",
        "プロキオンS","七夕賞","函館記念","中京記念","小倉記念","関屋記念",
        "エルムS","北九州記念","札幌2歳S","キーンランドC","新潟記念",
        "紫苑S","レパードS","シリウスS","みやこS","武蔵野S",
        "ファンタジーS","東京スポーツ杯","京都2歳S","サウジアラビアRC",
        "ターコイズS","カペラS","中日新聞杯","チャーチルC","マーチS",
        "京成杯AH","京成杯オータムH","チャレンジC",
        "京都ジャンプS","阪神ジャンプS","新潟ジャンプS",
    ],
}


def _is_known_graded(race_name: str, grade: str) -> bool:
    """Check if race name matches a known graded race."""
    for key in _KNOWN_GRADED.get(grade, []):
        if race_name == key or race_name.startswith(key):
            return True
    return False


def parse_race_list(html: str) -> list:
    """Parse race list page and return racecourse schedules."""
    soup = BeautifulSoup(html, "lxml")
    schedules = []

    # Each racecourse is a DL block
    race_lists = soup.select("dl.RaceList_DataList")

    for dl in race_lists:
        races = []
        race_items = dl.select("dd li")

        for li in race_items:
            link = li.select_one("a")
            if not link:
                continue

            href = link.get("href", "")
            race_id_match = re.search(r"race_id=(\d+)", href)
            if not race_id_match:
                continue

            race_id = race_id_match.group(1)

            # Race number from text like "1R"
            race_num_el = link.select_one(".Race_Num")
            race_number = 0
            if race_num_el:
                num_match = re.search(r"(\d+)", race_num_el.get_text(strip=True))
                if num_match:
                    race_number = int(num_match.group(1))

            # Race name
            race_name = ""
            name_el = link.select_one(".ItemTitle")
            if name_el:
                race_name = name_el.get_text(strip=True)
            else:
                # Fallback: get text after the race number
                all_text = link.get_text(strip=True)
                name_match = re.sub(r"^\d+R\s*", "", all_text)
                race_name = name_match[:20]

            # Start time
            start_time = ""
            time_el = link.select_one(".RaceList_Itemtime")
            if time_el:
                start_time = time_el.get_text(strip=True)
            else:
                time_match = re.search(r"(\d{1,2}:\d{2})", link.get_text())
                if time_match:
                    start_time = time_match.group(1)

            if race_number > 0:
                races.append({
                    "race_id": race_id,
                    "race_number": race_number,
                    "race_name": race_name,
                    "start_time": start_time,
                })

        if races:
            code = races[0]["race_id"][4:6] if len(races[0]["race_id"]) >= 6 else "00"
            name = _code_to_name(code)
            schedules.append({"code": code, "name": name, "races": races})

    return schedules


def parse_race_card(html: str) -> dict:
    """Parse shutuba (出馬表) page."""
    soup = BeautifulSoup(html, "lxml")
    race_info = _parse_race_info(soup)
    entries = _parse_entries(soup)
    return {"race_info": race_info, "entries": entries}


def _parse_race_info(soup: BeautifulSoup) -> dict:
    """Extract race metadata from the page header."""
    info = {
        "raceId": "",
        "raceName": "",
        "raceNumber": 0,
        "grade": None,
        "distance": 0,
        "surface": "芝",
        "courseDetail": "",
        "startTime": "",
        "racecourseCode": "",
        "date": "",
        "headCount": 0,
        "trackCondition": "",
    }

    # Race name
    race_name_el = soup.select_one(".RaceName")
    if race_name_el:
        info["raceName"] = race_name_el.get_text(strip=True)
        # Grade detection: Icon_GradeType1 is reliable for GI only.
        # GII/GIII icons (Type2/3) are also used for Listed/Open races on netkeiba,
        # so we cross-validate with known graded race names to avoid false positives.
        name = info["raceName"]
        if race_name_el.select_one(".Icon_GradeType1, .Icon_GradeType15, .Icon_GradeType12"):
            info["grade"] = "GI"
        elif race_name_el.select_one(".Icon_GradeType2, .Icon_GradeType16, .Icon_GradeType13"):
            # Cross-validate: only set GII if name matches known GII races
            if _is_known_graded(name, "GII"):
                info["grade"] = "GII"
        elif race_name_el.select_one(".Icon_GradeType3, .Icon_GradeType17, .Icon_GradeType14"):
            # Cross-validate: only set GIII if name matches known GIII races
            if _is_known_graded(name, "GIII"):
                info["grade"] = "GIII"
        # Name-based fallback for known races without icons
        if info["grade"] is None:
            if _is_known_graded(name, "GI"):
                info["grade"] = "GI"
            elif _is_known_graded(name, "GII"):
                info["grade"] = "GII"
            elif _is_known_graded(name, "GIII"):
                info["grade"] = "GIII"

    # Race number
    race_num_el = soup.select_one(".RaceNum")
    if race_num_el:
        num_match = re.search(r"(\d+)", race_num_el.get_text())
        if num_match:
            info["raceNumber"] = int(num_match.group(1))

    # Race data (distance, surface, etc.)
    race_data_el = soup.select_one(".RaceData01")
    if race_data_el:
        text = race_data_el.get_text(strip=True)

        # Start time
        time_match = re.search(r"(\d{1,2}:\d{2})", text)
        if time_match:
            info["startTime"] = time_match.group(1)

        # Surface and distance
        dist_match = re.search(r"(芝|ダ|障)(\d+)m", text)
        if dist_match:
            surface_char = dist_match.group(1)
            info["distance"] = int(dist_match.group(2))
            info["surface"] = {"芝": "芝", "ダ": "ダート", "障": "障害"}.get(surface_char, "芝")

        # Course direction
        dir_match = re.search(r"\((.+?)\)", text)
        if dir_match:
            info["courseDetail"] = dir_match.group(1)

        # Track condition: parse "馬場:良" or "馬場:稍" etc.
        cond_match = re.search(r"馬場\s*[:：]\s*(良|稍重|稍|重|不良|不)", text)
        if cond_match:
            cond_char = cond_match.group(1)
            info["trackCondition"] = {
                "良": "良",
                "稍": "稍重",
                "稍重": "稍重",
                "重": "重",
                "不": "不良",
                "不良": "不良",
            }.get(cond_char, "")

    return info


def _parse_entries(soup: BeautifulSoup) -> list:
    """Extract horse entries from the shutuba table."""
    entries = []
    rows = soup.select(".HorseList")
    auto_number = 1  # For future races without assigned horse numbers

    for row in rows:
        entry = _parse_horse_row(row, auto_number)
        if entry:
            entries.append(entry)
            auto_number += 1

    return entries


def _parse_horse_row(row: Tag, auto_number: int = 0) -> Optional[dict]:
    """Parse a single horse row from the shutuba table."""
    try:
        # Check if scratched
        is_scratched = row.select_one(".Cancel_Txt") is not None

        # Frame number - class like Waku1, Waku2 etc.
        frame_number = 0
        for td in row.select("td"):
            cls_list = td.get("class", [])
            for cls in cls_list:
                m = re.match(r"Waku(\d+)", cls)
                if m:
                    frame_number = int(m.group(1))
                    break
            if frame_number:
                break

        # Horse number - class like Umaban1, Umaban2 etc.
        horse_number = 0
        for td in row.select("td"):
            cls_list = td.get("class", [])
            for cls in cls_list:
                m = re.match(r"Umaban(\d+)", cls)
                if m:
                    try:
                        horse_number = int(td.get_text(strip=True))
                    except ValueError:
                        pass
                    break
            if horse_number:
                break

        # For future races, frame/horse numbers may not be assigned yet
        # Check if this row has a horse name before skipping
        has_horse = row.select_one(".HorseInfo a") is not None
        if horse_number == 0 and not has_horse:
            return None
        if horse_number == 0:
            horse_number = auto_number  # Use sequential number

        # Horse name and ID
        horse_name = ""
        horse_id = ""
        horse_info = row.select_one(".HorseInfo a")
        if horse_info:
            horse_name = horse_info.get_text(strip=True)
            href = horse_info.get("href", "")
            id_match = re.search(r"/horse/(\w+)", href)
            if id_match:
                horse_id = id_match.group(1)

        # Age/Sex - .Barei class
        age = ""
        barei_el = row.select_one(".Barei")
        if barei_el:
            age = barei_el.get_text(strip=True)
        elif not is_scratched:
            # Find td with sex+age pattern
            for td in row.select("td"):
                text = td.get_text(strip=True)
                if re.match(r"^[牡牝セ騸]\d+$", text):
                    age = text
                    break

        # Weight carried (斤量)
        weight_carried = 0.0
        for td in row.select("td"):
            cls_list = td.get("class", [])
            if "Txt_C" in cls_list and "Barei" not in cls_list and "Popular" not in cls_list:
                text = td.get_text(strip=True)
                w_match = re.match(r"^(\d+\.?\d*)$", text)
                if w_match and 40 <= float(w_match.group(1)) <= 70:
                    weight_carried = float(w_match.group(1))
                    break

        # Jockey
        jockey_name = ""
        jockey_id = ""
        jockey_td = row.select_one("td.Jockey") or row.select_one(".Jockey")
        if jockey_td:
            jockey_a = jockey_td.select_one("a")
            if jockey_a:
                jockey_name = jockey_a.get_text(strip=True)
                href = jockey_a.get("href", "")
                id_match = re.search(r"/jockey/(\w+)", href)
                if id_match:
                    jockey_id = id_match.group(1)
            else:
                jockey_name = jockey_td.get_text(strip=True)

        # Trainer
        trainer_name = ""
        trainer_id = ""
        trainer_td = row.select_one("td.Trainer") or row.select_one(".Trainer")
        if trainer_td:
            trainer_a = trainer_td.select_one("a")
            if trainer_a:
                trainer_name = trainer_a.get_text(strip=True)
                href = trainer_a.get("href", "")
                id_match = re.search(r"/trainer/(\w+)", href)
                if id_match:
                    trainer_id = id_match.group(1)
            else:
                trainer_name = trainer_td.get_text(strip=True)
                # Remove region prefix if present
                trainer_name = re.sub(r"^(栗東|美浦)", "", trainer_name)

        # Horse weight
        horse_weight = ""
        weight_td = row.select_one("td.Weight")
        if weight_td:
            horse_weight = weight_td.get_text(strip=True).replace("\n", "")

        # Odds
        odds = None
        popularity = None
        popular_td = row.select_one(".Popular_Ninki")
        if popular_td:
            pop_match = re.search(r"(\d+)", popular_td.get_text(strip=True))
            if pop_match:
                popularity = int(pop_match.group(1))

        odds_td = row.select_one("td.Txt_R.Popular")
        if odds_td:
            odds_text = odds_td.get_text(strip=True)
            odds_match = re.search(r"(\d+\.?\d*)", odds_text)
            if odds_match:
                odds = float(odds_match.group(1))

        return {
            "frameNumber": frame_number,
            "horseNumber": horse_number,
            "horseName": horse_name,
            "horseId": horse_id,
            "sireName": "",  # Not available on shutuba page
            "damName": "",
            "coatColor": "",
            "weightCarried": weight_carried,
            "age": age,
            "jockeyName": jockey_name,
            "jockeyId": jockey_id,
            "trainerName": trainer_name,
            "trainerId": trainer_id,
            "horseWeight": horse_weight,
            "odds": odds,
            "popularity": popularity,
            "isScratched": is_scratched,
        }
    except Exception as e:
        logger.error("Error parsing horse row: %s", e)
        return None


def _code_to_name(code: str) -> str:
    mapping = {
        "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
        "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
    }
    return mapping.get(code, "不明")
