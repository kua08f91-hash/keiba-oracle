"""Fetch combination odds/payouts from netkeiba.

For completed races: scrapes db.netkeiba.com for actual payouts.
For future races: estimates odds from individual horse odds.
"""
from __future__ import annotations
import re
import math
from typing import Optional
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

# Shared netkeiba API type mapping (used by main.py, realtime_worker.py, etc.)
# netkeiba API type mapping: 1=単勝, 2=複勝, 3=枠連, 4=馬連, 5=ワイド, 6=馬単, 7=3連複, 8=3連単
NETKEIBA_TYPE_MAP = {4: "umaren", 5: "wide", 6: "umatan", 7: "sanrenpuku", 8: "sanrentan"}
API_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "X-Requested-With": "XMLHttpRequest",
}


def parse_combo_key(key_str):
    """Parse '0508' -> [5,8] or '050812' -> [5,8,12]. Shared utility."""
    nums = []
    i = 0
    while i < len(key_str):
        if i + 1 < len(key_str):
            nums.append(int(key_str[i:i + 2]))
            i += 2
        else:
            i += 1
    return nums


def fetch_live_combination_odds(race_id: str, fallback_odds: dict = None, include_win_place: bool = False) -> dict:
    """Fetch real-time combination odds from netkeiba API for a live race.

    Shared by main.py, realtime_worker.py, export_predictions.py.

    Args:
        include_win_place: If True, also fetch type=1 (単勝) and type=2 (複勝).
    """
    import json as _json
    type_map = dict(NETKEIBA_TYPE_MAP)
    if include_win_place:
        type_map[1] = "tansho"
        type_map[2] = "fukusho"
    result = dict(fallback_odds or {})
    for api_type, bet_type in type_map.items():
        try:
            url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type={api_type}&action=init"
            r = requests.get(url, headers={
                **API_HEADERS,
                "Referer": f"https://race.netkeiba.com/odds/index.html?race_id={race_id}",
            }, timeout=5)
            d = _json.loads(r.text)
            data = d.get("data", {})
            if not isinstance(data, dict):
                continue
            odds_dict = data.get("odds", {}).get(str(api_type), {})
            if not odds_dict:
                continue
            entries = []
            for combo_key, vals in odds_dict.items():
                if not isinstance(vals, list) or len(vals) < 1:
                    continue
                try:
                    odds_val = float(vals[0].replace(",", ""))
                except (ValueError, TypeError):
                    continue
                if odds_val <= 0:
                    continue
                horses = parse_combo_key(combo_key)
                # tansho/fukusho have 1 horse, others have 2-3
                min_horses = 1 if bet_type in ("tansho", "fukusho") else 2
                if len(horses) >= min_horses:
                    entries.append({"horses": horses, "odds": odds_val, "payout": int(odds_val * 100)})
            if entries:
                result[bet_type] = entries
        except Exception:
            pass
    return result


def fetch_combination_odds(race_id: str) -> dict:
    """Fetch actual payout data from db.netkeiba.com for completed races.

    Returns dict with keys: tansho, fukusho, umaren, wide, sanrenpuku, sanrentan
    Each value is a list of {horses: list, odds: float, payout: int, ordered: bool}
    Returns empty dict if no payout data found.
    """
    payouts = _fetch_payouts_from_db(race_id)
    return payouts or {}


def _fetch_payouts_from_db(race_id: str) -> dict | None:
    """Fetch actual payout data from db.netkeiba.com for completed races."""
    url = f"https://db.netkeiba.com/race/{race_id}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = resp.apparent_encoding or "UTF-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        pay_tables = soup.select("table.pay_table_01")
        if not pay_tables:
            return None

        result = {}
        type_map = {
            "単勝": "tansho",
            "複勝": "fukusho",
            "馬連": "umaren",
            "ワイド": "wide",
            "三連複": "sanrenpuku",
            "三連単": "sanrentan",
        }

        for table in pay_tables:
            for row in table.select("tr"):
                th = row.select_one("th")
                tds = row.select("td")
                if not th or len(tds) < 2:
                    continue

                label = th.get_text(strip=True)
                key = type_map.get(label)
                if not key:
                    continue

                combos = tds[0].get_text("|", strip=True).split("|")
                amounts = tds[1].get_text("|", strip=True).split("|")

                entries = []
                for combo, amount in zip(combos, amounts):
                    try:
                        amt = int(amount.replace(",", ""))
                        nums = [int(n) for n in re.findall(r"\d+", combo)]
                        if nums:
                            # Convert payout to odds (payout per ¥100)
                            odds = amt / 100.0
                            entries.append({
                                "horses": nums,
                                "odds": odds,
                                "payout": amt,
                                "ordered": "→" in combo,
                            })
                    except:
                        continue

                if entries:
                    result[key] = entries

        return result if result else None
    except:
        return None


def estimate_from_entries(entries: list) -> dict:
    """Estimate combination odds from individual horse odds.

    Used for future races or when payout data is unavailable.
    """
    horse_odds = {}
    for e in entries:
        if e.get("odds") and not e.get("isScratched"):
            horse_odds[e["horseNumber"]] = e["odds"]

    if len(horse_odds) < 3:
        return {}

    # Sort by odds (favorites first)
    sorted_horses = sorted(horse_odds.items(), key=lambda x: x[1])

    result = {}

    # 単勝
    result["tansho"] = [
        {"horses": [h], "odds": o, "payout": int(o * 100), "ordered": False}
        for h, o in sorted_horses[:10]
    ]

    # 複勝 (estimated as odds / 3)
    result["fukusho"] = [
        {"horses": [h], "odds": round(max(1.0, o / 3), 1), "payout": int(max(100, o / 3 * 100)), "ordered": False}
        for h, o in sorted_horses[:10]
    ]

    # 馬連 (estimated: odds1 * odds2 * 0.8 / head_count_factor)
    umaren = []
    head = len(horse_odds)
    for i, (h1, o1) in enumerate(sorted_horses[:6]):
        for j, (h2, o2) in enumerate(sorted_horses[i+1:i+6], start=i+1):
            est_odds = round(o1 * o2 * 0.8 / max(1, head / 10), 1)
            umaren.append({
                "horses": sorted([h1, h2]),
                "odds": est_odds,
                "payout": int(est_odds * 100),
                "ordered": False,
            })
    umaren.sort(key=lambda x: x["odds"])
    result["umaren"] = umaren[:15]

    # ワイド (estimated: umaren odds * 0.3)
    wide = []
    for entry in umaren:
        w_odds = round(max(1.0, entry["odds"] * 0.3), 1)
        wide.append({
            "horses": entry["horses"],
            "odds": w_odds,
            "payout": int(w_odds * 100),
            "ordered": False,
        })
    result["wide"] = wide[:15]

    # 三連複 (estimated)
    sanrenpuku = []
    for i, (h1, o1) in enumerate(sorted_horses[:5]):
        for j, (h2, o2) in enumerate(sorted_horses[i+1:i+5], start=i+1):
            for k, (h3, o3) in enumerate(sorted_horses[j+1:j+5], start=j+1):
                est_odds = round(o1 * o2 * o3 * 0.5 / max(1, (head / 8) ** 2), 1)
                sanrenpuku.append({
                    "horses": sorted([h1, h2, h3]),
                    "odds": est_odds,
                    "payout": int(est_odds * 100),
                    "ordered": False,
                })
    sanrenpuku.sort(key=lambda x: x["odds"])
    result["sanrenpuku"] = sanrenpuku[:20]

    # 三連単 (estimated: sanrenpuku * 6 for permutations, adjusted)
    sanrentan = []
    for entry in sanrenpuku[:10]:
        est_odds = round(entry["odds"] * 4.5, 1)
        sanrentan.append({
            "horses": entry["horses"],
            "odds": est_odds,
            "payout": int(est_odds * 100),
            "ordered": True,
        })
    result["sanrentan"] = sanrentan[:15]

    return result
