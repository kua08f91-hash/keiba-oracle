"""Backup data source: keibabook.co.jp scraper.

Used as fallback when netkeiba.com is rate-limited (400 errors).
Provides race results with finish positions and payout data.

URL pattern:
  Result: https://s.keibabook.co.jp/cyuou/seiseki/{meeting_id}
  meeting_id format: YYYY + 回次(2桁) + 場所(2桁) + 日次(2桁) + レース番号(2桁)
"""
from __future__ import annotations

import logging
import re
import time

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/125.0.0.0 Safari/537.36",
}

SCRAPE_DELAY = 2


def fetch_race_result(keibabook_id: str) -> dict:
    """Fetch race result from keibabook.

    Args:
        keibabook_id: keibabook meeting ID (e.g., '202602010711')

    Returns:
        {
            'finish': [{pos, horseNumber, horseName, jockey, time}, ...],
            'payouts': {
                '単勝': [{nums: [int], amount: int}],
                '複勝': [{nums: [int], amount: int}],
                '馬連': [{nums: [int, int], amount: int}],
                'ワイド': [{nums: [int, int], amount: int}],
                '三連複': [{nums: [int, int, int], amount: int}],
                '三連単': [{nums: [int, int, int], amount: int}],
                ...
            }
        }
        or {} on failure.
    """
    url = f"https://s.keibabook.co.jp/cyuou/seiseki/{keibabook_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200 or len(r.text) < 3000:
            return {}
        r.encoding = r.apparent_encoding or "utf-8"
        soup = BeautifulSoup(r.text, "html.parser")
        return _parse_result_page(soup)
    except Exception as e:
        logger.warning("keibabook fetch failed for %s: %s", keibabook_id, e)
        return {}


def _parse_result_page(soup: BeautifulSoup) -> dict:
    """Parse keibabook result page."""
    result = {"finish": [], "payouts": {}}

    # --- Finish order from table.seiseki ---
    for table in soup.select("table.seiseki"):
        for row in table.select("tr"):
            tds = row.select("td")
            if len(tds) < 4:
                continue
            pos_text = tds[0].get_text(strip=True)
            num_text = tds[1].get_text(strip=True)
            if not pos_text.isdigit() or not num_text.isdigit():
                continue
            # Column 3 has horse name + jockey + etc
            info_text = tds[3].get_text(strip=True) if len(tds) > 3 else ""
            # Extract horse name (first part before other info)
            horse_name = ""
            info_el = tds[3] if len(tds) > 3 else None
            if info_el:
                # Horse name is typically the first text node
                horse_name = info_text[:20].split("牡")[0].split("牝")[0].split("セ")[0].strip()

            result["finish"].append({
                "pos": int(pos_text),
                "horseNumber": int(num_text),
                "horseName": horse_name,
            })

    result["finish"].sort(key=lambda x: x["pos"])

    # --- Payouts from table.kako-haraimoshi ---
    for table in soup.select("table.kako-haraimoshi"):
        for row in table.select("tr"):
            tds = row.select("td")
            if len(tds) < 3:
                continue
            label = tds[0].get_text(strip=True)
            combo_text = tds[1].get_text(strip=True)
            amount_text = tds[2].get_text(strip=True)

            # Normalize label
            label_map = {
                "単勝": "単勝", "複勝": "複勝", "枠連": "枠連",
                "馬連": "馬連", "馬単": "馬単", "ワイド": "ワイド",
                "3連複": "三連複", "三連複": "三連複",
                "3連単": "三連単", "三連単": "三連単",
            }
            norm_label = label_map.get(label, label)
            if norm_label not in label_map.values():
                continue

            # Parse combinations and amounts
            entries = _parse_payout_row(norm_label, combo_text, amount_text)
            if entries:
                if norm_label not in result["payouts"]:
                    result["payouts"][norm_label] = []
                result["payouts"][norm_label].extend(entries)

    return result


def _parse_payout_row(label: str, combo_text: str, amount_text: str) -> list:
    """Parse a single payout row into structured entries.

    Formats from keibabook:
      単勝:   combo="4",       amount="400円"
      複勝:   combo="425",     amount="170円370円1,790円"  (3 horses packed)
      馬連:   combo="2-4",     amount="2,290円"
      ワイド:  combo="2-44-52-5", amount="960円7,300円16,820円"  (3 combos packed)
      3連複:  combo="2-4-5",   amount="88,300円"
      3連単:  combo="4-2-5",   amount="291,940円"
    """
    entries = []

    # Extract amounts
    raw_amounts = re.findall(r"[\d,]+円", amount_text)
    amounts = []
    for a in raw_amounts:
        try:
            amounts.append(int(a.replace(",", "").replace("円", "")))
        except ValueError:
            pass

    n_entries = len(amounts)
    if n_entries == 0:
        return []

    if label == "複勝":
        # "425" → [4], [2], [5] (each horse number, 1-2 digits)
        # Split based on number of amounts
        digits = combo_text.strip()
        horses = []
        i = 0
        while i < len(digits) and len(horses) < n_entries:
            if i + 1 < len(digits) and int(digits[i:i+2]) <= 18 and len(horses) < n_entries - (len(digits) - i - 2 + 1) // 1:
                # Try 2-digit if ≤ 18 and we need more entries
                if len(digits) - i - 2 >= n_entries - len(horses) - 1:
                    horses.append(int(digits[i:i+2]))
                    i += 2
                    continue
            horses.append(int(digits[i]))
            i += 1
        for j, h in enumerate(horses):
            if j < len(amounts):
                entries.append({"nums": [h], "amount": amounts[j]})

    elif label == "ワイド" and n_entries > 1:
        # "2-44-52-5" → ["2-4", "4-5", "2-5"] (multiple dash-combos packed)
        # Parse sequentially: each combo is "N-M" where N,M are 1-2 digit (1-18)
        combos = _parse_packed_combos(combo_text, n_entries, combo_size=2)
        for j, nums in enumerate(combos):
            if j < len(amounts):
                entries.append({"nums": nums, "amount": amounts[j]})

    elif "-" in combo_text:
        # Single combo with dashes: "2-4", "2-4-5", "4-2-5"
        nums = [int(n) for n in re.findall(r"\d+", combo_text)]
        if nums and amounts:
            entries.append({"nums": nums, "amount": amounts[0]})

    else:
        # Single number: "4" (単勝)
        nums = [int(n) for n in re.findall(r"\d+", combo_text)]
        if nums and amounts:
            entries.append({"nums": nums, "amount": amounts[0]})

    return entries


def _parse_packed_combos(text: str, n_combos: int, combo_size: int = 2) -> list:
    """Parse packed combo text like '2-44-52-5' into [[2,4],[4,5],[2,5]].

    Strategy: Split by '-' to get segments, extract horse numbers (1-2 digits,
    max 18) from each segment, then group flat list into pairs.
    """
    segments = text.split("-")
    # Extract individual horse numbers from all segments
    all_nums = []
    for seg in segments:
        seg = seg.strip()
        i = 0
        while i < len(seg):
            if i + 1 < len(seg) and seg[i:i+2].isdigit() and int(seg[i:i+2]) <= 18 and int(seg[i:i+2]) >= 10:
                all_nums.append(int(seg[i:i+2]))
                i += 2
            elif seg[i].isdigit():
                all_nums.append(int(seg[i]))
                i += 1
            else:
                i += 1

    # Group into combos of combo_size
    combos = []
    for i in range(0, len(all_nums) - combo_size + 1, combo_size):
        combos.append(all_nums[i:i+combo_size])
        if len(combos) >= n_combos:
            break

    return combos


def fetch_results_for_date(date_str: str, meeting_ids: dict = None) -> dict:
    """Fetch all race results for a date from keibabook.

    Args:
        date_str: YYYYMMDD
        meeting_ids: Dict of {course_name: keibabook_prefix} if known.
                     If None, attempts to discover from web search.

    Returns:
        Dict of {race_key: {top3: [int], payouts: dict}}
    """
    if not meeting_ids:
        logger.warning("keibabook meeting_ids not provided for %s", date_str)
        return {}

    all_results = {}
    for course, prefix in meeting_ids.items():
        for rnum in range(1, 13):
            kid = f"{prefix}{rnum:02d}"
            result = fetch_race_result(kid)
            if result and result.get("finish"):
                top3 = [f["horseNumber"] for f in result["finish"][:3]]
                key = f"{course}{rnum}R"
                all_results[key] = {
                    "top3": top3,
                    "finish": result["finish"],
                    "payouts": result.get("payouts", {}),
                }
            time.sleep(SCRAPE_DELAY)

    return all_results
