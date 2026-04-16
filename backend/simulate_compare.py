"""Compare scoring engines v5/v6/v7 on March 2026 with the same bet optimizer.

Usage:
    /usr/bin/python3 -m backend.simulate_compare v5
    /usr/bin/python3 -m backend.simulate_compare v6
"""
from __future__ import annotations

import re
import sys
import os
import time
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from bs4 import BeautifulSoup
from backend.scraper.netkeiba import fetch_race_list, fetch_race_card
from backend.scraper.odds import estimate_from_entries, fetch_combination_odds
from backend.predictor.bet_optimizer import optimize_bets, detect_race_pattern, scores_to_probabilities
from backend.database.db import init_db

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
}
COURSE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}

# Persistent session
_session = requests.Session()
_session.headers.update(HEADERS)


def create_predictor(version):
    """Create predictor for the specified version."""
    if version == "v5":
        from backend.predictor.scoring import WeightedScoringModel
        return WeightedScoringModel()
    elif version == "v6":
        return V6SingleMLPredictor()
    elif version == "v7":
        from backend.predictor.ml_scoring import MLScoringModel
        return MLScoringModel()
    else:
        raise ValueError(f"Unknown version: {version}")


class V6SingleMLPredictor:
    """v6: Single ML model (combined only, no analytical blend)."""

    def __init__(self):
        import joblib
        model_path = os.path.join(os.path.dirname(__file__), "predictor", "trained_model.pkl")
        bundle = joblib.load(model_path)
        self._model = bundle.get("model_combined") or bundle.get("model")
        self._columns = bundle.get("all_columns") or bundle.get("feature_columns")
        self._fallback = None

    def predict(self, race_info, entries):
        from backend.predictor.feature_engineering import (
            ALL_COLUMNS, extract_race_context, extract_horse_features, features_to_vector,
        )
        from backend.predictor.scoring import MARK_MAP, ALL_FACTOR_KEYS

        active = [e for e in entries if not e.get("isScratched")]
        if len(active) < 3:
            from backend.predictor.scoring import WeightedScoringModel
            return WeightedScoringModel().predict(race_info, entries)

        context = extract_race_context(race_info, entries)
        all_weights = [e.get("weightCarried", 0) for e in active]
        all_odds = [e.get("odds") for e in active]

        vecs = []
        factor_list = []
        horse_nums = []

        for entry in active:
            feat_dict, factors = extract_horse_features(entry, race_info, context, all_weights, all_odds)
            vecs.append(features_to_vector(feat_dict, self._columns or ALL_COLUMNS))
            factor_list.append(factors)
            horse_nums.append(entry["horseNumber"])

        X = np.array(vecs, dtype=np.float64)
        probs = self._model.predict_proba(X)[:, 1]  # Combined only, NO analytical blend

        # Same normalization as v7
        max_p, min_p = float(np.max(probs)), float(np.min(probs))
        if max_p > min_p:
            scores = [30.0 + 65.0 * (float(p) - min_p) / (max_p - min_p) for p in probs]
        else:
            scores = [60.0] * len(probs)

        predictions = []
        for entry in entries:
            if entry.get("isScratched"):
                predictions.append({"horseNumber": entry["horseNumber"], "score": 0, "mark": "", "factors": {}})

        for i, hn in enumerate(horse_nums):
            predictions.append({
                "horseNumber": hn,
                "score": round(scores[i], 2),
                "mark": "",
                "factors": {k: round(v, 1) for k, v in factor_list[i].items()},
            })

        active_preds = sorted([p for p in predictions if p["score"] > 0], key=lambda p: -p["score"])
        mark_map = {0: "◎", 1: "◯", 2: "▲", 3: "△", 4: "△"}
        for i, pred in enumerate(active_preds):
            pred["mark"] = mark_map.get(i, "")

        return predictions


def fetch_payouts(race_id):
    url = f"https://db.netkeiba.com/race/{race_id}/"
    try:
        resp = _session.get(url, timeout=20)
        resp.encoding = resp.apparent_encoding or "UTF-8"
        soup = BeautifulSoup(resp.text, "html.parser")
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
                entries_list = []
                for combo, amount in zip(combos, amounts):
                    try:
                        amt = int(amount.replace(",", ""))
                    except ValueError:
                        continue
                    nums = [int(n) for n in re.findall(r"\d+", combo)]
                    if nums:
                        entries_list.append({"nums": nums, "amount": amt})
                if entries_list:
                    payouts[label] = entries_list
        return payouts
    except Exception:
        return {}


def check_bet_hit(bet, payouts):
    type_map = {
        "tansho": "単勝", "fukusho": "複勝", "wakuren": "枠連",
        "umaren": "馬連", "umatan": "馬単", "wide": "ワイド",
        "sanrenpuku": "三連複", "sanrentan": "三連単",
    }
    label = type_map.get(bet["type"])
    if not label or label not in payouts:
        return False, 0
    horses = bet["horses"]
    for entry in payouts[label]:
        pnums = entry["nums"]
        pamt = entry["amount"]
        if bet["type"] == "tansho" and len(horses) == 1 and horses[0] in pnums:
            return True, pamt
        elif bet["type"] == "fukusho" and len(horses) == 1 and horses[0] == pnums[0]:
            return True, pamt
        elif bet["type"] in ("umaren", "wide", "wakuren", "sanrenpuku") and set(horses) == set(pnums):
            return True, pamt
        elif bet["type"] in ("umatan", "sanrentan") and horses == pnums:
            return True, pamt
    return False, 0


def get_march_race_ids():
    from datetime import date, timedelta
    results = {}
    d = date(2026, 3, 1)
    end = date(2026, 3, 31)
    while d <= end:
        if d.weekday() in (5, 6):
            ds = d.strftime("%Y%m%d")
            time.sleep(2)
            schedules = fetch_race_list(ds)
            if schedules:
                ids = []
                for s in schedules:
                    for r in s.get("races", []):
                        rid = r.get("race_id", "")
                        if rid and rid not in ids:
                            ids.append(rid)
                if ids:
                    results[ds] = ids
                    print(f"  {int(ds[4:6])}/{int(ds[6:8])}: {len(ids)} races", flush=True)
        d += timedelta(days=1)
    return results


def main():
    version = sys.argv[1] if len(sys.argv) > 1 else "v7"
    init_db()
    predictor = create_predictor(version)

    print("=" * 70, flush=True)
    print(f"  KEIBA ORACLE — {version.upper()} Engine Simulation (March 2026)", flush=True)
    print("=" * 70, flush=True)

    print("\nCollecting race dates...", flush=True)
    all_race_ids = get_march_race_ids()
    total = sum(len(v) for v in all_race_ids.values())
    print(f"\nTotal: {total} races\n", flush=True)

    all_results = []
    type_stats = {}

    for ds in sorted(all_race_ids.keys()):
        race_ids = all_race_ids[ds]
        print(f"{'─'*70}", flush=True)
        print(f"  {int(ds[4:6])}/{int(ds[6:8])} ({len(race_ids)} races)", flush=True)
        print(f"{'─'*70}", flush=True)

        for race_id in race_ids:
            course = COURSE_MAP.get(race_id[4:6], "??")
            rnum = int(race_id[10:12])

            data = fetch_race_card(race_id)
            if not data:
                continue

            entries = data.get("entries", [])
            race_info = data.get("race_info", {})
            if len(entries) < 3:
                continue

            try:
                predictions = predictor.predict(race_info, entries)
            except Exception:
                continue
            if len([p for p in predictions if p.get("score", 0) > 0]) < 3:
                continue

            odds_data = estimate_from_entries(entries) or {}
            try:
                real = fetch_combination_odds(race_id)
                if real:
                    for k, el in real.items():
                        if k in odds_data:
                            rhs = [frozenset(e["horses"]) for e in el]
                            odds_data[k] = el + [e for e in odds_data[k] if frozenset(e["horses"]) not in rhs]
                        else:
                            odds_data[k] = el
            except Exception:
                pass

            try:
                optimized = optimize_bets(predictions, odds_data, race_info, entries=entries)
            except Exception:
                continue
            if not optimized:
                continue

            payouts = None
            for attempt in range(3):
                payouts = fetch_payouts(race_id)
                if payouts:
                    break
                time.sleep(8 * (attempt + 1))
            if not payouts:
                print(f"  - {course}{rnum:2d}R: no payouts (skipped)", flush=True)
                continue

            race_bet = len(optimized) * 100
            race_payout = 0
            race_hits = []

            for bet in optimized:
                hit, amount = check_bet_hit(bet, payouts)
                bt = bet["type"]
                if bt not in type_stats:
                    type_stats[bt] = {"bets": 0, "hits": 0, "invested": 0, "returned": 0, "label": bet.get("typeLabel", bt)}
                type_stats[bt]["bets"] += 1
                type_stats[bt]["invested"] += 100
                if hit:
                    race_payout += amount
                    race_hits.append(bet.get("typeLabel", bt))
                    type_stats[bt]["hits"] += 1
                    type_stats[bt]["returned"] += amount

            profit = race_payout - race_bet
            mark = "+" if profit > 0 else (" " if profit == 0 else "-")
            hit_str = ",".join(race_hits) if race_hits else "---"
            print(f"  {mark} {course}{rnum:2d}R ¥{race_bet}→¥{race_payout:>6,} {profit:>+7,} ({hit_str})", flush=True)
            time.sleep(3)

            all_results.append({
                "date": ds, "course": course, "rnum": rnum,
                "bet": race_bet, "payout": race_payout, "profit": profit,
                "hits": race_hits,
            })

    # Summary
    n = len(all_results)
    if n == 0:
        print("\nNo results.", flush=True)
        return

    total_bet = sum(r["bet"] for r in all_results)
    total_payout = sum(r["payout"] for r in all_results)
    total_profit = total_payout - total_bet
    total_hits = sum(len(r["hits"]) for r in all_results)
    total_bets_n = sum(r["bet"] // 100 for r in all_results)
    win_races = sum(1 for r in all_results if r["profit"] > 0)
    roi = total_payout / total_bet * 100 if total_bet > 0 else 0

    print(f"\n{'='*70}", flush=True)
    print(f"  {version.upper()} Engine — March 2026 Results", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Races:     {n}", flush=True)
    print(f"  Bets:      {total_bets_n}", flush=True)
    print(f"  Invested:  ¥{total_bet:,}", flush=True)
    print(f"  Returned:  ¥{total_payout:,}", flush=True)
    print(f"  Profit:    {'+'if total_profit>=0 else ''}¥{total_profit:,}", flush=True)
    print(f"  ROI:       {roi:.1f}%", flush=True)
    print(f"  Hit Rate:  {total_hits}/{total_bets_n} ({total_hits/total_bets_n*100:.1f}%)", flush=True)
    print(f"  Win Rate:  {win_races}/{n} ({win_races/n*100:.1f}%)", flush=True)

    print(f"\n  --- Bet Type ---", flush=True)
    for bt in ["tansho", "fukusho", "wakuren", "umaren", "umatan", "wide", "sanrenpuku", "sanrentan"]:
        if bt not in type_stats:
            continue
        s = type_stats[bt]
        s_roi = s["returned"] / s["invested"] * 100 if s["invested"] > 0 else 0
        s_hit = s["hits"] / s["bets"] * 100 if s["bets"] > 0 else 0
        print(f"  {s['label']:4s}: {s['bets']:4d}点 的中{s['hits']:3d} ({s_hit:5.1f}%) ROI {s_roi:6.1f}%", flush=True)

    print(f"\n{'='*70}", flush=True)


if __name__ == "__main__":
    main()
