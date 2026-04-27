"""Dynamic per-race betting optimizer (v8 — Full 8-type balanced strategy).

Generates all candidate bets across 8 JRA bet types, estimates hit probability
via softmax + Monte Carlo simulation, calculates EV, and selects the best
bets balancing hit rate and ROI.

Strategy: ROI anchor (3連単) + Hit anchor (ワイド/複勝) + EV fill.
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

# Shrinkage factor for estimated (non-real) odds
ESTIMATED_ODDS_SHRINKAGE = 0.75

# Minimum EV threshold — allow hit-rate types through while filtering junk
MIN_EV_THRESHOLD = -0.45

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

    # ── 馬連: top 6 pairs (was top 4 — expanded for 2-3着 coverage) ──
    for h1, h2 in combinations(top_horses[:6], 2):
        candidates.append({
            "type": "umaren", "typeLabel": "馬連",
            "horses": sorted([h1, h2]), "ordered": False,
        })

    # ── 馬単: top 5 ordered pairs (was top 4) ──
    for h1, h2 in permutations(top_horses[:5], 2):
        candidates.append({
            "type": "umatan", "typeLabel": "馬単",
            "horses": [h1, h2], "ordered": True,
        })

    # ── ワイド: top 6 pairs (was top 5) ──
    for h1, h2 in combinations(top_horses[:6], 2):
        candidates.append({
            "type": "wide", "typeLabel": "ワイド",
            "horses": sorted([h1, h2]), "ordered": False,
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
    """Main entry point: optimize betting selection for a single race.

    Balanced strategy: maximize both hit rate and ROI by selecting across
    all 8 bet types with guaranteed ROI anchor + hit rate anchor.

    Args:
        predictions: List of {horseNumber, score, ...} from scoring engine
        odds_data: Dict with bet type keys
        race_info: Dict with raceId, headCount, etc.
        max_bets: Maximum number of bets to return
        entries: Optional race entries (for 枠連 frame data)
        mc_samples: MC simulation samples. Default 5000; use 500 during
                    weight optimization for 10x speedup.
    """
    head_count = race_info.get("headCount", 16)
    if head_count < 3:
        return []

    probs = scores_to_probabilities(predictions, head_count)
    if len(probs) < 3:
        return []

    # Detect race pattern and recompute with adjusted temperature
    pattern = detect_race_pattern(probs)
    temp_multiplier = {
        "本命堅軸": 0.85,
        "混戦模様": 1.15,
        "2強対決": 0.92,
        "標準配置": 1.0,
        "少頭数": 1.0,
    }.get(pattern, 1.0)

    if temp_multiplier != 1.0:
        probs = scores_to_probabilities(predictions, head_count, temp_adjust=temp_multiplier)

    # Generate candidates from ALL 8 types
    candidates = generate_candidates(probs, top_n=min(7, len(probs)), entries=entries)

    # Monte Carlo simulation
    rng = random.Random(42)
    finishes = monte_carlo_finish(probs, mc_samples, rng=rng)
    candidates = estimate_hit_probabilities(finishes, candidates)

    # Calculate EV
    market_probs = {}
    for hn, prob in probs.items():
        for p in predictions:
            if p.get("horseNumber") == hn:
                odds_val = None
                for entry_data in (odds_data.get("tansho") or []):
                    if entry_data.get("horses") == [hn]:
                        odds_val = entry_data.get("odds")
                        break
                if odds_val and odds_val > 0:
                    market_probs[hn] = 1.0 / odds_val
                else:
                    market_probs[hn] = prob
                break

    real_count = 0
    for candidate in candidates:
        oi = find_odds_for_bet(candidate, odds_data)
        candidate["_oi"] = oi
        if oi:
            real_count += 1

    has_real = real_count > 5

    for candidate in candidates:
        odds_info = candidate.pop("_oi", None)
        ai_hit = candidate["hitProb"]

        if odds_info:
            candidate["odds"] = odds_info["odds"]
            candidate["payout"] = odds_info["payout"]
            candidate["hasRealOdds"] = True
            # Preserve range for ワイド/複勝 display
            if "oddsMin" in odds_info:
                candidate["oddsMin"] = odds_info["oddsMin"]
            if "oddsMax" in odds_info:
                candidate["oddsMax"] = odds_info["oddsMax"]
            base_ev = ai_hit * odds_info["odds"] - 1.0
            market_implied_hit = 1.0 / odds_info["odds"] if odds_info["odds"] > 0 else ai_hit
            edge = ai_hit - market_implied_hit
            edge_bonus = max(-0.15, min(0.30, edge * 0.20))
            candidate["ev"] = base_ev + edge_bonus
        else:
            est_odds = implied_fair_odds(ai_hit)
            candidate["odds"] = round(est_odds, 1)
            candidate["payout"] = int(est_odds * 100)
            candidate["hasRealOdds"] = False
            if has_real:
                candidate["ev"] = ai_hit * est_odds - 1.0 - 0.10
            else:
                type_roi_bonus = {
                    "sanrentan": 0.25,
                    "umatan": 0.10,
                    "wide": 0.05,
                    "tansho": 0.0,
                    "fukusho": -0.03,
                    "umaren": 0.0,
                    "sanrenpuku": -0.05,
                    "wakuren": -0.03,
                }
                bonus = type_roi_bonus.get(candidate["type"], 0)
                candidate["ev"] = ai_hit * est_odds - 1.0 + bonus

    # Clean up internal frame_map from candidates
    for c in candidates:
        c.pop("_frame_map", None)

    # Confidence-based bet count reduction
    top_evs = sorted([c["ev"] for c in candidates if c.get("ev") is not None], reverse=True)
    if top_evs:
        best_ev = top_evs[0]
        if best_ev < -0.35:
            max_bets = min(max_bets, 3)
        elif best_ev < -0.20:
            max_bets = min(max_bets, 4)

    return _diversify(candidates, max_bets)


def _diversify(candidates: List[Dict], max_bets: int) -> List[Dict]:
    """Balanced selection: maximize both hit rate and ROI.

    Two-phase selection:
      Phase 1 — Guaranteed anchors:
        - 1x ROI anchor: best 3連単 (profit engine, ROI 250%)
        - 1x Hit anchor: best from {ワイド, 複勝} (consistency, 20%+ hit rate)
      Phase 2 — Fill remaining by EV with type diversity limits.

    This ensures every race has at least one high-ROI bet and one
    high-hit-rate bet, then fills the rest optimally.
    """
    TYPE_LIMITS = {
        "wide": 3,         # Primary: double-hit=big win, user-preferred
        "sanrentan": 1,    # ROI engine (strict 1 slot only)
        "umatan": 1,       # Exacta middle ground
        "sanrenpuku": 1,   # Trio unordered
        "tansho": 1,       # Win bet
        "fukusho": 1,      # Place bet (not prioritized)
        "umaren": 0,       # BLOCKED: ROI 54% consistently worst
        "wakuren": 1,      # Frame pair
    }

    # EV sort bonuses — ワイド重視 + 3連単抑制
    TYPE_BONUS = {
        "wide": 0.18,       # Highest priority (user-preferred, stable ROI)
        "sanrentan": 0.08,  # Reduced from 0.15 (prevent over-selection)
        "umatan": 0.05,     # Middle ground
        "tansho": 0.03,     # Win bet
        "sanrenpuku": 0.02, # Trio
        "fukusho": 0.0,     # Not prioritized per user request
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

    # Phase 1a: ROI anchor — best 3連単
    for bet in viable:
        if bet["type"] == "sanrentan":
            _pick(bet)
            break

    # Phase 1b: Hit anchor — best ワイド (user-preferred, stable ROI)
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
