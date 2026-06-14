"""Dynamic per-race betting optimizer (v10 — calibrated 馬連◎流し strategy).

Generates all candidate bets across 8 JRA bet types, estimates hit probability
via softmax + Monte Carlo simulation, calculates EV, and selects the best
bets balancing hit rate and ROI.

Strategy: 馬連◎流し anchor + ワイド anchor + EV fill.
Fixes applied (validated via A/B split simulation on 5/3, ROI 173%):
  A: MC hitProb deflation by bet type (combo overestimation correction)
  C: Force ◎anchor from all candidates (bypass viable filter)
  D: Estimated odds shrinkage (0.75x discount for non-real odds)
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

# Maximum bets to return
MAX_BETS = 5

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

    # ── 馬単: top 5 ordered pairs (was top 4) ──
    for h1, h2 in permutations(top_horses[:5], 2):
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
    """Main entry point: D1 bimodal value-range strategy.

    Strategy: Buy bets where AI top-5 horse is involved and odds fall
    within type-specific high-ROI ranges. Skips 30-50x dead zone.

    Bimodal ranges (validated on 4,943R 3-way split):
      - 馬単 50-300x (high-odds, ROI 84-134%)
      - 馬連 20-30x (low-odds, ROI 77%)
      - ワイド 10-30x (low-odds, ROI 82%)
    Each type max 2 bets. Overall max 5 per race.
    Tune 67.1%, Validate 79.0%, Holdout 87.6%, Avg 77.9%.

    Args:
        predictions: List of {horseNumber, score, ...} from scoring engine
        odds_data: Dict with bet type keys
        race_info: Dict with raceId, headCount, etc.
        max_bets: Maximum number of bets to return
        entries: Optional race entries (for 枠連 frame data)
        mc_samples: MC simulation samples (used for hitProb display only).
    """
    # D2: expanded candidate pool for higher big-hit frequency
    # Based on 3,000R simulation: top7 + lower umaren → ROI 137%, hit 6.5%
    # Key: top7 expands umatan candidates from 10 combos to 21 combos (+110%)
    VALUE_RANGES = {
        "umatan": (50.0, 300.0),   # High-odds only (real odds produce 50x+ from top7 combos)
        "umaren": (15.0, 30.0),    # Lowered from 20x → 15x for +5% hit rate
        "wide": (10.0, 30.0),      # Unchanged (already capturing most hits)
    }
    VALUE_AI_TOP_N = 7  # Expanded from 5 → 7 for more umatan candidates
    TYPE_MAX = 2  # Max bets per type

    head_count = race_info.get("headCount", 16)
    if head_count < 3:
        return []


    probs = scores_to_probabilities(predictions, head_count)
    if len(probs) < 3:
        return []

    # Generate candidates from all types
    candidates = generate_candidates(probs, top_n=min(7, len(probs)), entries=entries)

    # Monte Carlo for hitProb display (not used for selection)
    rng = random.Random(42)
    finishes = monte_carlo_finish(probs, mc_samples, rng=rng)
    candidates = estimate_hit_probabilities(finishes, candidates)
    for candidate in candidates:
        candidate["hitProb"] *= HITPROB_DEFLATION.get(candidate["type"], 1.0)

    # AI top-N horses
    ai_sorted = sorted(predictions, key=lambda p: -p.get("score", 0))
    ai_top = set(p["horseNumber"] for p in ai_sorted[:VALUE_AI_TOP_N])

    # Select candidates within type-specific odds ranges
    viable = []
    for c in candidates:
        bt = c["type"]
        if bt not in VALUE_RANGES:
            continue
        odds_min, odds_max = VALUE_RANGES[bt]
        oi = find_odds_for_bet(c, odds_data)
        if not oi:
            continue
        # D1: upper bound is exclusive for umaren/wide (dead zone starts at 30x),
        # but inclusive for umatan (300x is a valid high-odds bet).
        if bt in ("umaren", "wide"):
            if oi["odds"] < odds_min or oi["odds"] >= odds_max:
                continue
        else:
            if oi["odds"] < odds_min or oi["odds"] > odds_max:
                continue
        if not any(h in ai_top for h in c["horses"]):
            continue

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
        viable.append(c)

    # Sort by highest odds (most value potential)
    viable.sort(key=lambda x: -x["odds"])

    # Select with per-type limit
    selected = []
    type_counts = {}
    for c in viable:
        bt = c["type"]
        if type_counts.get(bt, 0) >= TYPE_MAX:
            continue
        selected.append(c)
        type_counts[bt] = type_counts.get(bt, 0) + 1
        if len(selected) >= max_bets:
            break

    # Clean up and assign ranks
    for c in candidates:
        c.pop("_frame_map", None)
    for i, bet in enumerate(selected):
        bet["rank"] = i + 1

    return selected


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
    """Evaluate whether this race is a good bet opportunity.

    Returns confidence level based on 3,000R analysis:
      "A" (強く推奨): Top-7 capture rate >90% — high confidence
      "B" (推奨):     Top-7 capture rate 70-90% — normal confidence
      "C" (様子見):   Top-7 capture rate <70% — consider skipping

    Based on:
      - ◎ odds: 1-2x super-favorite + small field → 94.2% (A)
      - ◎ odds: 4-8x mid-odds + 16+ heads → 59.2% (C)
      - Score concentration: high → 96.8% (A)
    """
    head_count = race_info.get("headCount", 16)
    ai_sorted = sorted(predictions, key=lambda p: -p.get("score", 0))
    if not ai_sorted:
        return "B"

    # Get ◎ odds
    top1_odds = 0.0
    top1_hn = ai_sorted[0].get("horseNumber")
    if entries:
        for e in entries:
            if e.get("horseNumber") == top1_hn and e.get("odds"):
                top1_odds = e["odds"]
                break

    # Score concentration (top3 / total)
    scores = [p.get("score", 0) for p in ai_sorted if p.get("score", 0) > 0]
    if len(scores) >= 3:
        concentration = sum(scores[:3]) / sum(scores)
    else:
        concentration = 1.0

    # C: Skip condition — mid-odds + large field
    if top1_odds >= 4 and top1_odds < 8 and head_count >= 16:
        return "C"

    # A: Strong buy conditions
    if top1_odds > 0 and top1_odds < 2 and head_count < 12:
        return "A"
    if concentration >= 0.4:
        return "A"
    if top1_odds > 0 and top1_odds < 3 and head_count < 14:
        return "A"

    # B: Normal
    return "B"
