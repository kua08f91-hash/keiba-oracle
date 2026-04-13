"""Dynamic per-race betting optimizer.

For each race, generates all candidate bets, estimates hit probability
via softmax score-to-probability conversion + Monte Carlo simulation,
calculates expected value (EV) using available odds, and selects the
top 5 bets by EV.

Calibration targets (from scoring.py v4, 1,107 historical races):
- 単勝 top-1 accuracy: 32.8%
- ワイド top-pair hit rate: 55.0%
- 馬連 top-pair hit rate: 13.6%
- 三連複 top-trio hit rate: 7.0%
"""
from __future__ import annotations

import math
import random
from itertools import combinations, permutations
from typing import List, Dict, Optional, Any

# Softmax temperature — calibrated so rank-1 P(win) ≈ 0.328 on historical data.
# Lower T = more confident in top horse, higher T = more spread out.
TEMPERATURE = 9.5

# Monte Carlo samples for hit probability estimation
MC_SAMPLES = 5000

# JRA takeout rate (~25%)
JRA_TAKEOUT = 0.25

# Shrinkage factor for estimated (non-real) odds
ESTIMATED_ODDS_SHRINKAGE = 0.75

# Minimum EV threshold — only include bets with EV above this
MIN_EV_THRESHOLD = -0.60

# Maximum bets to return
MAX_BETS = 5

# Bet type labels
BET_TYPES = {
    "tansho": "単勝",
    "fukusho": "複勝",
    "umaren": "馬連",
    "wide": "ワイド",
    "sanrenpuku": "3連複",
    "sanrentan": "3連単",
}


def scores_to_probabilities(
    predictions: List[Dict], head_count: int, temp_adjust: float = 1.0
) -> Dict[int, float]:
    """Convert AI prediction scores to win probabilities via softmax.

    Args:
        predictions: List of {horseNumber, score, ...}
        head_count: Number of horses in the race
        temp_adjust: Multiplier for temperature (e.g. 0.85 for concentrated)

    Returns:
        Dict mapping horseNumber -> P(win)
    """
    scored = [p for p in predictions if p.get("score", 0) > 0 and not p.get("isScratched")]
    if not scored:
        return {}

    scores = {p["horseNumber"]: p["score"] for p in scored}

    # Field size adjustment: scale temperature slightly with field size
    # Larger fields = more uncertainty = higher effective temperature
    t = TEMPERATURE * (max(head_count, 3) / 14) ** 0.2 * temp_adjust

    # Softmax
    max_score = max(scores.values())
    exp_scores = {}
    for hn, s in scores.items():
        exp_scores[hn] = math.exp((s - max_score) / t)

    total = sum(exp_scores.values())
    probs = {hn: v / total for hn, v in exp_scores.items()}

    return probs


def monte_carlo_finish(
    probs: Dict[int, float], n_samples: int = MC_SAMPLES, rng: random.Random = None
) -> List[List[int]]:
    """Generate Monte Carlo finish order samples.

    Uses weighted random sampling without replacement to simulate
    race finishes based on win probabilities.

    Args:
        probs: Dict mapping horseNumber -> P(win)
        n_samples: Number of simulations
        rng: Optional local Random instance (avoids polluting global state)

    Returns:
        List of finish orders (each is a list of horse numbers, 1st to last)
    """
    if rng is None:
        rng = random.Random(42)
    horses = list(probs.keys())
    weights = [probs[h] for h in horses]
    finishes = []

    for _ in range(n_samples):
        remaining = list(zip(horses, weights))
        finish = []
        for _ in range(min(len(remaining), 3)):  # Only need top 3
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
    """Estimate hit probability for each candidate bet using MC finishes.

    Args:
        finishes: List of simulated finish orders (top 3)
        candidates: List of candidate bet dicts with 'type', 'horses', 'ordered'

    Returns:
        Same candidates with 'hitProb' added
    """
    n = len(finishes)
    if n == 0:
        for c in candidates:
            c["hitProb"] = 0.0
        return candidates

    for candidate in candidates:
        bet_type = candidate["type"]
        horses = candidate["horses"]
        hits = 0

        for finish in finishes:
            if bet_type == "tansho":
                # Horse wins (1st place)
                if len(finish) >= 1 and finish[0] == horses[0]:
                    hits += 1

            elif bet_type == "fukusho":
                # Horse in top 3
                if horses[0] in finish[:3]:
                    hits += 1

            elif bet_type == "umaren":
                # Both horses in top 2 (unordered)
                if len(finish) >= 2 and set(horses) == set(finish[:2]):
                    hits += 1

            elif bet_type == "wide":
                # Both horses in top 3 (unordered)
                if len(finish) >= 3 and set(horses).issubset(set(finish[:3])):
                    hits += 1

            elif bet_type == "sanrenpuku":
                # All 3 horses in top 3 (unordered)
                if len(finish) >= 3 and set(horses) == set(finish[:3]):
                    hits += 1

            elif bet_type == "sanrentan":
                # Exact order match
                if len(finish) >= 3 and finish[:3] == horses:
                    hits += 1

        candidate["hitProb"] = hits / n

    return candidates


def generate_candidates(
    probs: Dict[int, float],
    top_n: int = 5,
) -> List[Dict]:
    """Generate all candidate bets from top-ranked horses.

    Returns list of {type, typeLabel, horses, ordered, rank_info}
    """
    ranked = sorted(probs.items(), key=lambda x: -x[1])
    top_horses = [h for h, _ in ranked[:top_n]]
    candidates = []

    # 単勝: top 3
    for h in top_horses[:3]:
        candidates.append({
            "type": "tansho",
            "typeLabel": "単勝",
            "horses": [h],
            "ordered": False,
        })

    # 複勝: top 4
    for h in top_horses[:4]:
        candidates.append({
            "type": "fukusho",
            "typeLabel": "複勝",
            "horses": [h],
            "ordered": False,
        })

    # 馬連: top 4 pairs
    for h1, h2 in combinations(top_horses[:4], 2):
        candidates.append({
            "type": "umaren",
            "typeLabel": "馬連",
            "horses": sorted([h1, h2]),
            "ordered": False,
        })

    # ワイド: top 4 pairs
    for h1, h2 in combinations(top_horses[:4], 2):
        candidates.append({
            "type": "wide",
            "typeLabel": "ワイド",
            "horses": sorted([h1, h2]),
            "ordered": False,
        })

    # 三連複: top 4 triples
    for h1, h2, h3 in combinations(top_horses[:4], 3):
        candidates.append({
            "type": "sanrenpuku",
            "typeLabel": "3連複",
            "horses": sorted([h1, h2, h3]),
            "ordered": False,
        })

    # 三連単: top 3 permutations
    for perm in permutations(top_horses[:3], 3):
        candidates.append({
            "type": "sanrentan",
            "typeLabel": "3連単",
            "horses": list(perm),
            "ordered": True,
        })

    return candidates


def find_odds_for_bet(bet: Dict, odds_data: Dict) -> Optional[Dict]:
    """Look up odds for a specific bet from the odds data.

    Returns {odds, payout} or None.
    """
    bet_type = bet["type"]
    horses = bet["horses"]
    ordered = bet.get("ordered", False)

    if not odds_data or bet_type not in odds_data:
        return None

    entries = odds_data[bet_type]
    for entry in entries:
        entry_horses = entry.get("horses", [])
        if ordered or bet_type == "sanrentan":
            if entry_horses == horses:
                return {"odds": entry["odds"], "payout": entry["payout"]}
        else:
            if set(entry_horses) == set(horses):
                return {"odds": entry["odds"], "payout": entry["payout"]}

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
) -> List[Dict]:
    """Main entry point: optimize betting selection for a single race.

    Args:
        predictions: List of {horseNumber, score, ...} from scoring engine
        odds_data: Dict with keys tansho/fukusho/umaren/wide/sanrenpuku/sanrentan
        race_info: Dict with raceId, headCount, etc.
        max_bets: Maximum number of bets to return

    Returns:
        List of top bets sorted by EV, each with:
        {type, typeLabel, horses, ordered, odds, payout, ev, hitProb, hasRealOdds}
    """
    head_count = race_info.get("headCount", 16)
    if head_count < 3:
        return []

    # Step 1: Convert scores to probabilities (first pass for pattern detection)
    probs = scores_to_probabilities(predictions, head_count)
    if len(probs) < 3:
        return []

    # Step 1b: Detect race pattern and recompute with adjusted temperature
    pattern = detect_race_pattern(probs)
    temp_multiplier = {
        "本命堅軸": 0.85,   # Dominant favorite: more concentrated
        "混戦模様": 1.15,   # Competitive: more spread out
        "2強対決": 0.92,    # Two-horse race: slightly concentrated
        "標準配置": 1.0,    # Standard: no change
        "少頭数": 1.0,      # Small field: no change
    }.get(pattern, 1.0)

    if temp_multiplier != 1.0:
        probs = scores_to_probabilities(
            predictions, head_count, temp_adjust=temp_multiplier
        )

    # Step 2: Generate candidate bets
    candidates = generate_candidates(probs, top_n=min(5, len(probs)))

    # Step 3: Monte Carlo simulation for hit probabilities
    rng = random.Random(42)  # Local RNG to avoid polluting global state
    finishes = monte_carlo_finish(probs, MC_SAMPLES, rng=rng)
    candidates = estimate_hit_probabilities(finishes, candidates)

    # Step 4: Calculate EV with value edge consideration
    # Build market-implied probability map from odds
    market_probs = {}
    for hn, prob in probs.items():
        for p in predictions:
            if p.get("horseNumber") == hn:
                odds_val = None
                # Try to get individual horse odds from predictions factors
                for entry_data in (odds_data.get("tansho") or []):
                    if entry_data.get("horses") == [hn]:
                        odds_val = entry_data.get("odds")
                        break
                if odds_val and odds_val > 0:
                    market_probs[hn] = 1.0 / odds_val
                else:
                    market_probs[hn] = prob  # Fallback to AI prob
                break

    # First pass: check which have real odds
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
            # Core EV: AI probability * real odds - 1
            base_ev = ai_hit * odds_info["odds"] - 1.0

            # Value edge bonus: reward bets where AI probability > market-implied
            market_implied_hit = 1.0 / odds_info["odds"] if odds_info["odds"] > 0 else ai_hit
            edge = ai_hit - market_implied_hit
            # Add 20% of edge as bonus to EV (caps both directions)
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
                    "umaren": 0.05,
                    "wide": 0.03,
                    "tansho": 0.0,
                    "fukusho": -0.05,
                    "sanrenpuku": -0.10,
                }
                bonus = type_roi_bonus.get(candidate["type"], 0)
                candidate["ev"] = ai_hit * est_odds - 1.0 + bonus

    # Step 5: Filter and diversify
    return _diversify(candidates, max_bets)


def _diversify(candidates: List[Dict], max_bets: int) -> List[Dict]:
    """Filter low-EV/low-odds bets and diversify by type.

    Optimized based on live results (4/5-4/12):
    - ワイド: ROI 120.6% — max 2 per race (best performer)
    - 単勝:  ROI 120.2% — max 1 per race (high hit rate)
    - 3連単:  ROI 110.0% — max 1 per race (high variance)
    - 馬連:  ROI 81.4%  — max 1 per race (reduced)
    - 3連複:  ROI 89.1%  — max 1 per race (reduced)
    - 複勝:  ROI 95.4%  — max 1 per race (low payout)
    """
    MIN_ODDS = 2.0
    # Live-data-informed type limits
    TYPE_LIMITS = {
        "wide": 2,         # Best ROI in live
        "tansho": 1,       # High hit rate
        "sanrentan": 1,    # High variance, cap at 1
        "umaren": 1,       # Reduced from 2
        "sanrenpuku": 1,   # Reduced from 2
        "fukusho": 1,      # Low payout
    }

    viable = [
        c for c in candidates
        if c["ev"] > MIN_EV_THRESHOLD
        and c["hitProb"] > 0.001
        and c.get("odds", 0) > MIN_ODDS
    ]

    # Boost EV for types with proven live ROI
    TYPE_BONUS = {
        "wide": 0.05,      # Slight boost for best live performer
        "tansho": 0.03,
    }
    for c in viable:
        bonus = TYPE_BONUS.get(c["type"], 0)
        c["_sort_ev"] = c["ev"] + bonus

    viable.sort(key=lambda x: -x.get("_sort_ev", x["ev"]))

    selected = []
    selected_ids = set()
    type_counts = {}
    for bet in viable:
        t = bet["type"]
        limit = TYPE_LIMITS.get(t, 2)
        count = type_counts.get(t, 0)
        if count >= limit:
            continue
        selected.append(bet)
        selected_ids.add(id(bet))
        type_counts[t] = count + 1
        if len(selected) >= max_bets:
            break

    # Fill from remaining if needed
    if len(selected) < max_bets:
        for bet in viable:
            if id(bet) not in selected_ids:
                selected.append(bet)
                selected_ids.add(id(bet))
                if len(selected) >= max_bets:
                    break

    # Clean up temp sort key
    for bet in viable:
        bet.pop("_sort_ev", None)

    for i, bet in enumerate(selected):
        bet["rank"] = i + 1

    return selected


def pick_longshot(
    candidates: List[Dict],
    selected_bets: List[Dict],
    probs: Dict[int, float],
) -> Optional[Dict]:
    """Pick the best high-odds longshot bet not already in the top 5.

    Criteria (tightened based on live results):
    - Odds 20-100x (high payout, exclude moonshots)
    - hitProb >= 0.5% (realistic chance, was 0.3%)
    - EV > -0.3 (stricter value threshold, was -0.5)
    - Prefers bets with highest odds * hitProb (expected payout)
    - Not already in selected_bets
    """
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
        # Score: higher odds * higher hitProb = best expected payout
        longshot_score = c.get("odds", 0) * c.get("hitProb", 0)
        longshots.append((longshot_score, c))

    if not longshots:
        return None

    longshots.sort(key=lambda x: -x[0])
    best = longshots[0][1]
    best["rank"] = 0  # Special rank for longshot
    return best


def detect_race_pattern(probs: Dict[int, float]) -> str:
    """Detect the score distribution pattern for UI display.

    Returns a pattern label string.
    """
    ranked = sorted(probs.values(), reverse=True)
    if len(ranked) < 3:
        return "少頭数"

    gap_1_2 = ranked[0] - ranked[1]
    gap_2_3 = ranked[1] - ranked[2]
    spread = ranked[0] - ranked[2]

    if gap_1_2 > 0.10:
        return "本命堅軸"  # Dominant favorite
    elif spread < 0.06:
        return "混戦模様"  # Competitive field
    elif gap_2_3 > 0.08:
        return "2強対決"  # Two strong horses
    else:
        return "標準配置"  # Standard distribution
