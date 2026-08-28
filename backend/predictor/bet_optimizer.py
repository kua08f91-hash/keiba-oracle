"""Dynamic per-race betting optimizer (D6 — Dual Layer strategy).

Two-layer EV-driven strategy:
  Layer 1 (EV Core): ◎が2-4倍帯 → 単勝/馬連/ワイド/3連複からEV上位5点
  Layer 2 (Value Shot): 全レース → 高オッズ穴狙い+longshot EV上位5点

Both layers select bets purely by EV ranking — no fixed bet types or partners.
"""
from __future__ import annotations

import math
import random
from itertools import combinations, permutations
from typing import List, Dict, Optional, Any

# Softmax temperature — calibrated so rank-1 P(win) ≈ 0.328 on historical data.
TEMPERATURE = 9.5

# Monte Carlo samples for hit probability estimation
MC_SAMPLES = 5000

# JRA takeout rate (~25%)
JRA_TAKEOUT = 0.25

# Shrinkage factor for estimated (non-real) odds — applied in EV calculation (Fix D)
ESTIMATED_ODDS_SHRINKAGE = 0.60

# Fix A: MC hitProb deflation factors by bet type.
# MC simulation overestimates combo bet probabilities vs market-implied rates.
# "mild" profile: validated on 3-4月 564R (3月102.2%, 4月145.5%, Total 122.0%)
HITPROB_DEFLATION = {
    "sanrenpuku": 0.25,
    "sanrentan": 0.30,
    "umatan": 0.35,
    "umaren": 0.55,
    "wide": 0.50,
    "tansho": 0.75,
    "fukusho": 0.80,
    "wakuren": 0.50,
}

# Minimum EV threshold — relaxed to -0.60 for more candidates (validated on 3-4月)
MIN_EV_THRESHOLD = -0.60

# Maximum bets to return per layer
MAX_BETS = 5  # D6: EV上位5点/layer

# D6 Dual Layer thresholds
SHOUBU_MIN_SCORE = 68.0       # 勝負レース判定: ◎>=68 (27ファクター対応)

# EV Core (Layer 1): ◎が2-4倍帯のレースで単勝/馬連/ワイド/3連複から選択
EV_CORE_HONMEI_ODDS_MIN = 2.0   # ◎オッズ下限
EV_CORE_HONMEI_ODDS_MAX = 4.0   # ◎オッズ上限
EV_CORE_BET_TYPES = {"tansho", "umaren", "wide", "sanrenpuku"}  # 対象券種
EV_CORE_MAX_BETS = 5             # EV上位5点

# Value Shot (Layer 2): 全レースで高オッズ穴狙い (馬単含む全券種)
VALUE_SHOT_BET_TYPES = {"tansho", "umaren", "umatan", "wide", "sanrenpuku", "sanrentan"}
VALUE_SHOT_MIN_ODDS = 20.0       # 最低オッズ (穴狙いなので20倍以上のみ)
VALUE_SHOT_MAX_BETS = 5          # EV上位5点

# オッズ上限キャップ (推定・異常値の排除)
MAX_ODDS_CAP = {
    "tansho": 200.0,
    "umaren": 500.0,
    "umatan": 1000.0,
    "wide": 200.0,
    "sanrenpuku": 1000.0,
    "sanrentan": 5000.0,
    "fukusho": 50.0,
    "wakuren": 200.0,
}

# Legacy constants for backward compat
HONMEI_WIDE_PARTNERS = 1
UMAREN_MIN_SCORE = 68.0
UMATAN_MIN_SCORE = 68.0
UMATAN_MIN_GAP = 0.0
HONMEI_UMATAN_PARTNERS = 3
HONMEI_UMAREN_PARTNERS = 2
TANPUKU_TANSHO_MIN_ODDS = 6.0
TANPUKU_FUKUSHO_MIN_ODDS = 6.0
TANPUKU_RATIO = (1, 1)

# ケリー基準 (improvement 1: 利益額6倍)
KELLY_FRACTION = 0.5  # ハーフケリー
DEFAULT_BANKROLL = 100000  # デフォルト残高
KELLY_MIN_BET = 100   # 最小賭金
KELLY_MAX_BET = 5000  # 最大賭金

# Minimum odds per type (user rule: "1倍台は提示しない")
MIN_ODDS_BY_TYPE = {
    "tansho": 2.0,
    "fukusho": 2.0,
    "wakuren": 2.0,
    "umaren": 2.0,
    "umatan": 2.0,
    "wide": 2.5,        # Low-odds ワイド drags ROI
    "sanrenpuku": 2.0,
    "sanrentan": 2.0,
}

# Bet type labels (all 8 JRA types)
BET_TYPES = {
    "tansho": "単勝",
    "fukusho": "複勝",
    "wakuren": "枠連",
    "umaren": "馬連",
    "umatan": "馬単",
    "wide": "ワイド",
    "sanrenpuku": "3連複",
    "sanrentan": "3連単",
}


def scores_to_probabilities(
    predictions: List[Dict], head_count: int, temp_adjust: float = 1.0
) -> Dict[int, float]:
    """Convert AI prediction scores to win probabilities via softmax."""
    scored = [p for p in predictions if p.get("score", 0) > 0 and not p.get("isScratched")]
    if not scored:
        return {}

    scores = {p["horseNumber"]: p["score"] for p in scored}
    t = TEMPERATURE * (max(head_count, 3) / 14) ** 0.2 * temp_adjust

    max_score = max(scores.values())
    exp_scores = {hn: math.exp((s - max_score) / t) for hn, s in scores.items()}
    total = sum(exp_scores.values())
    return {hn: v / total for hn, v in exp_scores.items()}


def monte_carlo_finish(
    probs: Dict[int, float], n_samples: int = MC_SAMPLES, rng: random.Random = None
) -> List[List[int]]:
    """Generate Monte Carlo finish order samples."""
    if rng is None:
        rng = random.Random(42)
    horses = list(probs.keys())
    weights = [probs[h] for h in horses]
    finishes = []

    for _ in range(n_samples):
        remaining = list(zip(horses, weights))
        finish = []
        for _ in range(min(len(remaining), 3)):
            total_w = sum(w for _, w in remaining)
            if total_w <= 0:
                break
            r = rng.random() * total_w
            cumulative = 0
            chosen_idx = 0
            for idx, (_, w) in enumerate(remaining):
                cumulative += w
                if cumulative >= r:
                    chosen_idx = idx
                    break
            finish.append(remaining[chosen_idx][0])
            remaining.pop(chosen_idx)
        finishes.append(finish)

    return finishes


def estimate_hit_probabilities(
    finishes: List[List[int]],
    candidates: List[Dict],
) -> List[Dict]:
    """Estimate hit probability for each candidate bet using MC finishes."""
    n = len(finishes)
    if n == 0:
        for c in candidates:
            c["hitProb"] = 0.0
        return candidates

    # Pre-build frame-based finish mapping for wakuren
    # (handled via frame_map stored in candidate)

    for candidate in candidates:
        bet_type = candidate["type"]
        horses = candidate["horses"]
        hits = 0

        if bet_type == "wakuren":
            # Frame-based: use frame_map to convert MC finishes
            frame_map = candidate.get("_frame_map", {})
            for finish in finishes:
                if len(finish) >= 2:
                    f1 = frame_map.get(finish[0], 0)
                    f2 = frame_map.get(finish[1], 0)
                    if f1 > 0 and f2 > 0 and set(horses) == set([f1, f2]):
                        hits += 1
        else:
            for finish in finishes:
                if bet_type == "tansho":
                    if len(finish) >= 1 and finish[0] == horses[0]:
                        hits += 1
                elif bet_type == "fukusho":
                    if horses[0] in finish[:3]:
                        hits += 1
                elif bet_type == "umaren":
                    if len(finish) >= 2 and set(horses) == set(finish[:2]):
                        hits += 1
                elif bet_type == "umatan":
                    if len(finish) >= 2 and finish[:2] == horses:
                        hits += 1
                elif bet_type == "wide":
                    if len(finish) >= 3 and set(horses).issubset(set(finish[:3])):
                        hits += 1
                elif bet_type == "sanrenpuku":
                    if len(finish) >= 3 and set(horses) == set(finish[:3]):
                        hits += 1
                elif bet_type == "sanrentan":
                    if len(finish) >= 3 and finish[:3] == horses:
                        hits += 1

        candidate["hitProb"] = hits / n

    return candidates


def generate_candidates(
    probs: Dict[int, float],
    top_n: int = 5,
    entries: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Generate all candidate bets from top-ranked horses across 8 JRA bet types.

    Args:
        probs: Dict mapping horseNumber -> P(win)
        top_n: Number of top horses to consider
        entries: Optional race entries (needed for 枠連 frame mapping)

    Returns:
        List of candidate bet dicts
    """
    ranked = sorted(probs.items(), key=lambda x: -x[1])
    top_horses = [h for h, _ in ranked[:top_n]]
    candidates = []

    # Build frame map from entries
    frame_map = {}
    if entries:
        for e in entries:
            if not e.get("isScratched"):
                frame_map[e.get("horseNumber", 0)] = e.get("frameNumber", 0)

    # ── 単勝: top 3 ──
    for h in top_horses[:3]:
        candidates.append({
            "type": "tansho", "typeLabel": "単勝",
            "horses": [h], "ordered": False,
        })

    # ── 複勝: top 3 ──
    for h in top_horses[:3]:
        candidates.append({
            "type": "fukusho", "typeLabel": "複勝",
            "horses": [h], "ordered": False,
        })

    # ── 枠連: top 5 horses' frame pairs ──
    if frame_map:
        seen_frame_pairs = set()
        for h1, h2 in combinations(top_horses[:5], 2):
            f1 = frame_map.get(h1, 0)
            f2 = frame_map.get(h2, 0)
            if f1 > 0 and f2 > 0:
                pair = tuple(sorted([f1, f2]))
                if pair not in seen_frame_pairs:
                    seen_frame_pairs.add(pair)
                    candidates.append({
                        "type": "wakuren", "typeLabel": "枠連",
                        "horses": list(pair), "ordered": False,
                        "_frame_map": frame_map,
                    })

    # ── 馬連: top 6 pairs + ◎×人気上位 (相手先精度向上) ──
    # Base: top 6 AI pairs
    umaren_seen = set()
    for h1, h2 in combinations(top_horses[:6], 2):
        pair = tuple(sorted([h1, h2]))
        umaren_seen.add(pair)
        candidates.append({
            "type": "umaren", "typeLabel": "馬連",
            "horses": list(pair), "ordered": False,
        })
    # ◎流し拡張: ◎(top1) × 人気1-5位馬 (AI top6外でも人気なら相手候補に)
    if entries and top_horses:
        top1 = top_horses[0]
        popular_horses = sorted(
            [e["horseNumber"] for e in entries
             if not e.get("isScratched") and e.get("popularity") and e["popularity"] <= 5],
            key=lambda hn: next((e["popularity"] for e in entries if e["horseNumber"] == hn), 99),
        )
        for ph in popular_horses:
            if ph == top1:
                continue
            pair = tuple(sorted([top1, ph]))
            if pair not in umaren_seen:
                umaren_seen.add(pair)
                candidates.append({
                    "type": "umaren", "typeLabel": "馬連",
                    "horses": list(pair), "ordered": False,
                })

    # ── 馬単: top_n ordered pairs (D2: top 7 for ◎-anchor coverage) ──
    for h1, h2 in permutations(top_horses[:top_n], 2):
        candidates.append({
            "type": "umatan", "typeLabel": "馬単",
            "horses": [h1, h2], "ordered": True,
        })

    # ── ワイド: top 6 pairs + ◎×人気上位 ──
    wide_seen = set()
    for h1, h2 in combinations(top_horses[:6], 2):
        pair = tuple(sorted([h1, h2]))
        wide_seen.add(pair)
        candidates.append({
            "type": "wide", "typeLabel": "ワイド",
            "horses": list(pair), "ordered": False,
        })
    if entries and top_horses:
        top1 = top_horses[0]
        popular_horses_w = sorted(
            [e["horseNumber"] for e in entries
             if not e.get("isScratched") and e.get("popularity") and e["popularity"] <= 5],
            key=lambda hn: next((e["popularity"] for e in entries if e["horseNumber"] == hn), 99),
        )
        for ph in popular_horses_w:
            if ph == top1:
                continue
            pair = tuple(sorted([top1, ph]))
            if pair not in wide_seen:
                wide_seen.add(pair)
                candidates.append({
                    "type": "wide", "typeLabel": "ワイド",
                    "horses": list(pair), "ordered": False,
                })

    # ── 3連複: top 6 triples (was top 4 — 的中可能性3倍) ──
    for h1, h2, h3 in combinations(top_horses[:6], 3):
        candidates.append({
            "type": "sanrenpuku", "typeLabel": "3連複",
            "horses": sorted([h1, h2, h3]), "ordered": False,
        })

    # ── 3連単: top 5 permutations (was top 4 — 的中可能性1.5倍) ──
    for perm in permutations(top_horses[:5], 3):
        candidates.append({
            "type": "sanrentan", "typeLabel": "3連単",
            "horses": list(perm), "ordered": True,
        })

    return candidates


def find_odds_for_bet(bet: Dict, odds_data: Dict) -> Optional[Dict]:
    """Look up odds for a specific bet from the odds data."""
    bet_type = bet["type"]
    horses = bet["horses"]
    ordered = bet.get("ordered", False)

    if not odds_data or bet_type not in odds_data:
        return None

    entries = odds_data[bet_type]
    for entry in entries:
        entry_horses = entry.get("horses", [])
        matched = False
        if ordered or bet_type in ("sanrentan", "umatan"):
            matched = entry_horses == horses
        else:
            matched = set(entry_horses) == set(horses)
        if matched:
            result = {"odds": entry["odds"], "payout": entry["payout"]}
            # Preserve range odds for ワイド/複勝 display
            if "oddsMin" in entry:
                result["oddsMin"] = entry["oddsMin"]
            if "oddsMax" in entry:
                result["oddsMax"] = entry["oddsMax"]
            return result

    return None


def kelly_bet_size(hit_prob: float, odds: float, bankroll: float = DEFAULT_BANKROLL) -> int:
    """Calculate bet size as unit multiplier using half-Kelly criterion.

    f = ((b+1)*p - 1) / b where b = odds - 1, p = hit probability.
    Returns unit multiplier (1-10): how many base units to bet.
    1 = standard (¥500), 2 = double (¥1,000), etc.
    Returns 1 if Kelly f <= 0 (minimum 1 unit).
    """
    if odds <= 1 or hit_prob <= 0:
        return 1
    b = odds - 1.0
    f = ((b + 1) * hit_prob - 1) / b
    f_half = f * KELLY_FRACTION
    if f_half <= 0:
        return 1
    # Scale: f_half typically 0.01-0.10 → map to 1-10 units
    # f_half * 100 gives percentage of bankroll → divide by 2 for unit count
    units = max(1, min(int(f_half * 50 + 1), 10))
    return units


def implied_fair_odds(hit_prob: float) -> float:
    """Calculate implied fair odds from hit probability, applying JRA takeout."""
    if hit_prob <= 0:
        return 1.0
    fair = 1.0 / hit_prob
    return fair * (1 - JRA_TAKEOUT)


def optimize_bets(
    predictions: List[Dict],
    odds_data: Dict,
    race_info: Dict,
    max_bets: int = MAX_BETS,
    entries: Optional[List[Dict]] = None,
    mc_samples: int = MC_SAMPLES,
) -> List[Dict]:
    """Main entry point: D5 ◎軸厚張り strategy.

    Strategy: ◎ (highest AI score horse) anchored spread across 馬単/馬連/ワイド.
    Validated on 8/8 36R backtest: ROI 110.3% (vs D4 0.0%).

    Core insight: ◎ wins 47% of races — anchor all bets on ◎ and spread
    across AI-ranked partners. No arbitrary odds range filter.

    Bet structure per race:
      - 馬単 ◎→AI 2~7位 (up to 6 points) — odds >= 5x
      - 馬連 ◎-AI 2~5位 (up to 4 points) — odds >= 3x
      - ワイド ◎-AI 2~5位 (up to 4 points) — odds >= 2.5x
    Total: up to 14 points/race.

    Args:
        predictions: List of {horseNumber, score, ...} from scoring engine
        odds_data: Dict with bet type keys
        race_info: Dict with raceId, headCount, etc.
        max_bets: Maximum number of bets to return
        entries: Optional race entries (for 枠連 frame data)
        mc_samples: MC simulation samples (used for hitProb display only).
    """
    # D5: minimum odds per bet type (ROI protection — exclude absolute favorites)
    D5_MIN_ODDS = {
        "umatan": 5.0,
        "umaren": 3.0,
        "wide": 2.5,
    }

    head_count = race_info.get("headCount", 16)
    if head_count < 3:
        return []

    probs = scores_to_probabilities(predictions, head_count)
    if len(probs) < 3:
        return []

    # AI ranking
    ai_sorted = sorted(predictions, key=lambda p: -p.get("score", 0))
    honmei = ai_sorted[0]
    honmei_hn = honmei["horseNumber"]

    # Generate all candidates for MC hitProb calculation
    candidates = generate_candidates(probs, top_n=min(7, len(probs)), entries=entries)

    # Monte Carlo for hitProb display
    rng = random.Random(42)
    finishes = monte_carlo_finish(probs, mc_samples, rng=rng)
    candidates = estimate_hit_probabilities(finishes, candidates)
    for candidate in candidates:
        candidate["hitProb"] *= HITPROB_DEFLATION.get(candidate["type"], 1.0)

    # Build candidate lookup for odds attachment
    cand_lookup = {}
    for c in candidates:
        key = (c["type"], tuple(c["horses"]))
        cand_lookup[key] = c

    selected = []
    # Tanpuku bets are collected separately so they survive the max_bets cap.
    tanpuku_bets: List[Dict] = []

    def _try_add(bet_type, horses, ordered):
        """Try to add a ◎-anchor bet if odds are available and above minimum."""
        key = (bet_type, tuple(horses))
        c = cand_lookup.get(key)
        if not c:
            # Create ad-hoc candidate if not in generated set
            c = {"type": bet_type, "typeLabel": BET_TYPES.get(bet_type, bet_type),
                 "horses": horses, "ordered": ordered, "hitProb": 0.0}
        oi = find_odds_for_bet(c, odds_data)
        if not oi:
            return
        min_odds = D5_MIN_ODDS.get(bet_type, 2.0)
        if oi["odds"] < min_odds:
            return
        c["odds"] = oi["odds"]
        c["payout"] = oi["payout"]
        c["hasRealOdds"] = True
        if "oddsMin" in oi:
            c["oddsMin"] = oi["oddsMin"]
        if "oddsMax" in oi:
            c["oddsMax"] = oi["oddsMax"]
        # EV for display
        fair_prob = 1.0 / oi["odds"]
        ai_prob = c["hitProb"]
        adj = min(0.05, max(-0.05, (ai_prob - fair_prob) * 0.3))
        c["ev"] = (fair_prob + adj) * oi["odds"] - 1.0
        selected.append(c)

    # ── Dynamic bet selection: 馬単・馬連を軸に高配当狙い ──
    honmei_score = honmei["score"]
    gap_12 = honmei_score - ai_sorted[1]["score"] if len(ai_sorted) >= 2 else 0

    # 馬単 ◎→AI 2~4位 (高配当の主力)
    if honmei_score >= UMATAN_MIN_SCORE and gap_12 >= UMATAN_MIN_GAP:
        for i in range(1, HONMEI_UMATAN_PARTNERS + 1):
            if i >= len(ai_sorted):
                break
            partner_hn = ai_sorted[i]["horseNumber"]
            _try_add("umatan", [honmei_hn, partner_hn], True)

    # 馬連 ◎-AI 2~3位 (中配当の安定枠)
    if honmei_score >= UMAREN_MIN_SCORE:
        for i in range(1, HONMEI_UMAREN_PARTNERS + 1):
            if i >= len(ai_sorted):
                break
            partner_hn = ai_sorted[i]["horseNumber"]
            pair = sorted([honmei_hn, partner_hn])
            _try_add("umaren", pair, False)

    # ワイド ◎-AI 2位のみ (保険1点)
    for i in range(1, HONMEI_WIDE_PARTNERS + 1):
        if i >= len(ai_sorted):
            break
        partner_hn = ai_sorted[i]["horseNumber"]
        pair = sorted([honmei_hn, partner_hn])
        _try_add("wide", pair, False)

    # ── 単複リスクヘッジ: ◎の単勝(6x+) + 複勝(2x+) を1:3比率 ──
    honmei_odds = 0.0
    if entries:
        for e in entries:
            if e.get("horseNumber") == honmei_hn and e.get("odds"):
                honmei_odds = e["odds"]
                break
    if not honmei_odds and odds_data and "tansho" in odds_data:
        for entry in odds_data["tansho"]:
            if entry.get("horses") == [honmei_hn]:
                honmei_odds = entry["odds"]
                break

    if honmei_odds >= TANPUKU_TANSHO_MIN_ODDS:
        # 単勝 ◎ (1単位)
        tansho_cand = cand_lookup.get(("tansho", (honmei_hn,)))
        if tansho_cand:
            oi = find_odds_for_bet(tansho_cand, odds_data)
            if oi and oi["odds"] >= TANPUKU_TANSHO_MIN_ODDS:
                c = dict(tansho_cand)
                c["odds"] = oi["odds"]
                c["payout"] = oi["payout"]
                c["hasRealOdds"] = True
                c["ev"] = c["hitProb"] * oi["odds"] - 1.0
                c["betSize"] = TANPUKU_RATIO[0]  # 1単位
                tanpuku_bets.append(c)
        # 複勝 ◎ (3単位)
        fukusho_cand = cand_lookup.get(("fukusho", (honmei_hn,)))
        if fukusho_cand:
            oi = find_odds_for_bet(fukusho_cand, odds_data)
            if oi and oi["odds"] >= TANPUKU_FUKUSHO_MIN_ODDS:
                c = dict(fukusho_cand)
                c["odds"] = oi["odds"]
                c["payout"] = oi["payout"]
                c["hasRealOdds"] = True
                if "oddsMin" in oi:
                    c["oddsMin"] = oi["oddsMin"]
                if "oddsMax" in oi:
                    c["oddsMax"] = oi["oddsMax"]
                c["ev"] = c["hitProb"] * oi["odds"] - 1.0
                c["betSize"] = TANPUKU_RATIO[1]  # 3単位
                tanpuku_bets.append(c)
    elif honmei_odds >= TANPUKU_FUKUSHO_MIN_ODDS:
        # 複勝のみ (2単位)
        fukusho_cand = cand_lookup.get(("fukusho", (honmei_hn,)))
        if fukusho_cand:
            oi = find_odds_for_bet(fukusho_cand, odds_data)
            if oi and oi["odds"] >= TANPUKU_FUKUSHO_MIN_ODDS:
                c = dict(fukusho_cand)
                c["odds"] = oi["odds"]
                c["payout"] = oi["payout"]
                c["hasRealOdds"] = True
                if "oddsMin" in oi:
                    c["oddsMin"] = oi["oddsMin"]
                if "oddsMax" in oi:
                    c["oddsMax"] = oi["oddsMax"]
                c["ev"] = c["hitProb"] * oi["odds"] - 1.0
                c["betSize"] = 2  # 2単位
                tanpuku_bets.append(c)

    # Sort combo bets by odds descending (high-value bets first for display)
    selected.sort(key=lambda x: -x.get("odds", 0))

    # Apply max_bets cap to combo bets, then append tanpuku bets unconditionally.
    # Tanpuku bets are guaranteed risk-hedge additions and must not be displaced
    # by the combo-bet cap.
    selected = selected[:max_bets]
    selected.extend(tanpuku_bets)

    # ── ケリー基準で金額傾斜 (betSize フィールド) ──
    bankroll = race_info.get("bankroll", DEFAULT_BANKROLL)
    for bet in selected:
        if "betSize" not in bet:  # 単複は既にbetSize設定済み
            k_size = kelly_bet_size(bet["hitProb"], bet.get("odds", 1), bankroll)
            if k_size > 0:
                bet["betSize"] = k_size
            else:
                bet["betSize"] = 1  # Kelly f<=0 でも最小1単位は購入

    # Clean up frame maps from all candidates
    for c in candidates:
        c.pop("_frame_map", None)

    # Assign ranks
    for i, bet in enumerate(selected):
        bet["rank"] = i + 1

    return selected


def _ev_select(candidates: List[Dict], odds_data: Dict,
               allowed_types: set, max_bets: int,
               min_odds: float = 2.0,
               require_honmei: bool = False,
               honmei_hn: int = 0) -> List[Dict]:
    """Pure EV-based selection: pick top N bets by EV from candidates.

    Args:
        candidates: Pre-computed candidates with hitProb set.
        odds_data: Real odds data dict.
        allowed_types: Set of allowed bet type strings.
        max_bets: Maximum number of bets to select.
        min_odds: Minimum odds threshold.
        require_honmei: If True, only consider bets involving honmei_hn.
        honmei_hn: Horse number of ◎ (used when require_honmei=True).

    Returns:
        List of selected bets sorted by EV descending.
    """
    scored = []
    for c in candidates:
        if c["type"] not in allowed_types:
            continue
        if require_honmei and honmei_hn not in c["horses"]:
            continue

        oi = find_odds_for_bet(c, odds_data)
        if not oi:
            continue
        if oi["odds"] < min_odds:
            continue

        raw_odds = oi["odds"]

        # オッズ上限キャップ (99999倍等の異常値を排除)
        cap = MAX_ODDS_CAP.get(c["type"], 500.0)
        if raw_odds > cap:
            continue  # キャップ超えは候補から除外

        bet = dict(c)
        bet["odds"] = raw_odds
        bet["payout"] = oi["payout"]
        bet["hasRealOdds"] = True
        if "oddsMin" in oi:
            bet["oddsMin"] = oi["oddsMin"]
        if "oddsMax" in oi:
            bet["oddsMax"] = oi["oddsMax"]

        # EV = hitProb * odds - 1
        bet["ev"] = bet["hitProb"] * raw_odds - 1.0
        scored.append(bet)

    # Sort by EV descending, pick top N
    scored.sort(key=lambda x: -x["ev"])
    selected = scored[:max_bets]

    # Kelly bet sizing
    for bet in selected:
        bet["betSize"] = kelly_bet_size(bet["hitProb"], bet.get("odds", 1))

    for i, bet in enumerate(selected):
        bet["rank"] = i + 1

    return selected


def optimize_bets_dual(
    predictions: List[Dict],
    odds_data: Dict,
    race_info: Dict,
    entries: Optional[List[Dict]] = None,
    mc_samples: int = MC_SAMPLES,
) -> Dict[str, Any]:
    """D6 Dual Layer optimizer.

    Returns dict with:
      "core_bets": Layer 1 EV Core bets (up to 5, only when ◎ is 2-4x)
      "value_bets": Layer 2 Value Shot bets (up to 5, all races)
      "longshot": Best longshot candidate
      "pattern": Race pattern label
      "layer1_active": Whether Layer 1 is active for this race
    """
    head_count = race_info.get("headCount", 16)
    if head_count < 3:
        return {"core_bets": [], "value_bets": [], "longshot": None,
                "pattern": "", "layer1_active": False}

    probs = scores_to_probabilities(predictions, head_count)
    if len(probs) < 3:
        return {"core_bets": [], "value_bets": [], "longshot": None,
                "pattern": "", "layer1_active": False}

    # AI ranking
    ai_sorted = sorted(predictions, key=lambda p: -p.get("score", 0))
    honmei_hn = ai_sorted[0]["horseNumber"]

    # Generate candidates
    candidates = generate_candidates(probs, top_n=min(7, len(probs)), entries=entries)

    # Monte Carlo
    rng = random.Random(42)
    finishes = monte_carlo_finish(probs, mc_samples, rng=rng)
    candidates = estimate_hit_probabilities(finishes, candidates)
    for c in candidates:
        c["hitProb"] *= HITPROB_DEFLATION.get(c["type"], 1.0)

    # Get ◎ odds
    honmei_odds = 0.0
    if entries:
        for e in entries:
            if e.get("horseNumber") == honmei_hn and e.get("odds"):
                honmei_odds = e["odds"]
                break
    if not honmei_odds and odds_data and "tansho" in odds_data:
        for entry in odds_data["tansho"]:
            if entry.get("horses") == [honmei_hn]:
                honmei_odds = entry["odds"]
                break

    # ── Layer 1: EV Core (◎ 2-4倍帯のみ) ──
    layer1_active = EV_CORE_HONMEI_ODDS_MIN <= honmei_odds < EV_CORE_HONMEI_ODDS_MAX
    core_bets = []
    if layer1_active:
        core_bets = _ev_select(
            candidates, odds_data,
            allowed_types=EV_CORE_BET_TYPES,
            max_bets=EV_CORE_MAX_BETS,
            min_odds=2.0,
            require_honmei=True,
            honmei_hn=honmei_hn,
        )

    # ── Layer 2: Value Shot (全レース, 高オッズ) ──
    # Exclude bets already in Layer 1
    core_keys = {(b["type"], tuple(b["horses"])) for b in core_bets}
    value_candidates = [c for c in candidates
                        if (c["type"], tuple(c["horses"])) not in core_keys]
    value_bets = _ev_select(
        value_candidates, odds_data,
        allowed_types=VALUE_SHOT_BET_TYPES,
        max_bets=VALUE_SHOT_MAX_BETS,
        min_odds=VALUE_SHOT_MIN_ODDS,
    )

    # ── Longshot (投資対象化) ──
    all_selected = core_bets + value_bets
    longshot = pick_longshot(candidates, all_selected, probs)

    # Pattern
    pattern = detect_race_pattern(probs)

    # Clean up frame maps
    for c in candidates:
        c.pop("_frame_map", None)

    return {
        "core_bets": core_bets,
        "value_bets": value_bets,
        "longshot": longshot,
        "pattern": pattern,
        "layer1_active": layer1_active,
        "honmei_odds": honmei_odds,
    }


def _diversify(
    candidates: List[Dict],
    max_bets: int,
    probs_ref: Optional[Dict[int, float]] = None,
) -> List[Dict]:
    """Balanced selection: maximize both hit rate and ROI.

    Two-phase selection:
      Phase 1 — Guaranteed anchors:
        - 1x 馬連◎流し: umaren containing top AI horse (ROI 98.3%)
        - 1x ワイド◎流し: wide containing top AI horse (ROI 101%)
      Phase 2 — Fill remaining by EV with type diversity limits.

    This ensures every race has at least one high-ROI bet and one
    high-hit-rate bet, then fills the rest optimally.
    """
    TYPE_LIMITS = {
        "wide": 2,         # Stable ROI (5/3: 101%), pair with umaren
        "umaren": 2,       # ◎流し anchor (5/3: ROI 98.3%, 25% hit rate)
        "sanrentan": 1,    # ROI engine (strict 1 slot)
        "umatan": 1,       # Exacta middle ground
        "sanrenpuku": 1,   # Trio unordered
        "tansho": 1,       # Win bet
        "fukusho": 1,      # Place bet
        "wakuren": 1,      # Frame pair
    }

    # EV sort bonuses — 馬連◎流し + ワイド重視
    TYPE_BONUS = {
        "umaren": 0.20,     # Highest: ◎流し anchor (ROI 98.3% validated)
        "wide": 0.18,       # Secondary: stable hit rate (ROI 101%)
        "sanrentan": 0.05,  # Reduced: 0/34 on 5/3
        "umatan": 0.03,     # Reduced: 0/27 on 5/3
        "tansho": 0.02,     # Win bet
        "sanrenpuku": 0.01, # Trio
        "fukusho": 0.0,     # Not prioritized
        "wakuren": 0.0,
    }

    # Filter viable candidates
    viable = []
    for c in candidates:
        min_odds = MIN_ODDS_BY_TYPE.get(c["type"], 2.0)
        if (c["ev"] > MIN_EV_THRESHOLD
                and c["hitProb"] > 0.001
                and c.get("odds", 0) >= min_odds):
            viable.append(c)

    # Apply sort bonus
    for c in viable:
        c["_sort_ev"] = c["ev"] + TYPE_BONUS.get(c["type"], 0)
    viable.sort(key=lambda x: -x["_sort_ev"])

    selected = []
    selected_ids = set()
    type_counts = {}

    def _pick(bet):
        selected.append(bet)
        selected_ids.add(id(bet))
        type_counts[bet["type"]] = type_counts.get(bet["type"], 0) + 1

    # Phase 1a: 馬連◎流し anchor — best umaren containing ◎ (top-ranked horse)
    # Fix C: Search ALL candidates (not just viable) to ensure ◎ anchor is never filtered out
    ranked_horses = sorted(probs_ref.items(), key=lambda x: -x[1]) if probs_ref else []
    top1_horse = ranked_horses[0][0] if ranked_horses else None
    if top1_horse is not None:
        honmei_umarens = [
            c for c in candidates
            if c["type"] == "umaren" and top1_horse in c["horses"]
            and c["hitProb"] > 0.001 and c.get("odds", 0) >= MIN_ODDS_BY_TYPE.get("umaren", 2.0)
        ]
        if honmei_umarens:
            best = max(honmei_umarens, key=lambda c: c.get("ev", -99))
            _pick(best)

    # Phase 1b: ワイド◎流し anchor — best wide containing ◎ (also from all candidates)
    if top1_horse is not None:
        honmei_wides = [
            c for c in candidates
            if c["type"] == "wide" and top1_horse in c["horses"]
            and c["hitProb"] > 0.001 and c.get("odds", 0) >= MIN_ODDS_BY_TYPE.get("wide", 2.5)
            and id(c) not in selected_ids
        ]
        if honmei_wides:
            best_w = max(honmei_wides, key=lambda c: c.get("ev", -99))
            _pick(best_w)
    if not any(b["type"] == "wide" for b in selected):
        for bet in viable:
            if id(bet) not in selected_ids and bet["type"] == "wide":
                _pick(bet)
                break

    # Phase 2: Fill remaining by EV with type limits
    for bet in viable:
        if len(selected) >= max_bets:
            break
        if id(bet) in selected_ids:
            continue
        t = bet["type"]
        if type_counts.get(t, 0) >= TYPE_LIMITS.get(t, 1):
            continue
        _pick(bet)

    # Fill overflow if needed — still respect TYPE_LIMITS (especially umaren=0)
    if len(selected) < max_bets:
        for bet in viable:
            if len(selected) >= max_bets:
                break
            if id(bet) in selected_ids:
                continue
            t = bet["type"]
            if type_counts.get(t, 0) >= TYPE_LIMITS.get(t, 1):
                continue
            _pick(bet)

    # Clean up
    for c in viable:
        c.pop("_sort_ev", None)
    for i, bet in enumerate(selected):
        bet["rank"] = i + 1

    return selected


def pick_longshot(
    candidates: List[Dict],
    selected_bets: List[Dict],
    probs: Dict[int, float],
) -> Optional[Dict]:
    """Pick the best high-odds longshot bet not already in the top 5."""
    MIN_LONGSHOT_ODDS = 20.0
    MAX_LONGSHOT_ODDS = 100.0
    selected_keys = set()
    for b in selected_bets:
        selected_keys.add((b["type"], tuple(b["horses"])))

    longshots = []
    for c in candidates:
        odds = c.get("odds", 0)
        if odds < MIN_LONGSHOT_ODDS or odds > MAX_LONGSHOT_ODDS:
            continue
        if c.get("hitProb", 0) < 0.005:
            continue
        if c.get("ev", -1) <= -0.3:
            continue
        key = (c["type"], tuple(c["horses"]))
        if key in selected_keys:
            continue
        longshot_score = c.get("odds", 0) * c.get("hitProb", 0)
        longshots.append((longshot_score, c))

    if not longshots:
        return None

    longshots.sort(key=lambda x: -x[0])
    best = longshots[0][1]
    best["rank"] = 0
    return best


def detect_race_pattern(probs: Dict[int, float]) -> str:
    """Detect the score distribution pattern for UI display."""
    ranked = sorted(probs.values(), reverse=True)
    if len(ranked) < 3:
        return "少頭数"

    gap_1_2 = ranked[0] - ranked[1]
    gap_2_3 = ranked[1] - ranked[2]
    spread = ranked[0] - ranked[2]

    if gap_1_2 > 0.10:
        return "本命堅軸"
    elif spread < 0.06:
        return "混戦模様"
    elif gap_2_3 > 0.08:
        return "2強対決"
    else:
        return "標準配置"


def evaluate_bet_confidence(predictions: list, race_info: dict, entries: list = None) -> str:
    """D5 dynamic: 勝負レース判定.

    Returns:
      "A" (勝負): ◎score >= SHOUBU_MIN_SCORE — buy with dynamic bet selection
      "C" (SKIP): ◎score < SHOUBU_MIN_SCORE — skip this race

    Validated on 564R (5-8月): ◎>=74 → 191R selected, ROI 224%
    """
    ai_sorted = sorted(predictions, key=lambda p: -p.get("score", 0))
    if not ai_sorted:
        return "C"

    honmei_score = ai_sorted[0].get("score", 0)

    if honmei_score >= SHOUBU_MIN_SCORE:
        return "A"

    return "C"
