"""EV-driven bet selector (Phase 3).

Selects bets based on calibrated probabilities vs market odds.
Only buys when Expected Value > threshold (positive edge).

This replaces the S8 value-range strategy when calibration is validated.

Usage:
    selector = EVSelector(calibrator=calibrator)
    bets = selector.select(predictions, odds_data, race_info, entries)
"""
from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional

from .bet_optimizer import (
    scores_to_probabilities, monte_carlo_finish, estimate_hit_probabilities,
    generate_candidates, find_odds_for_bet, detect_race_pattern,
    MC_SAMPLES, HITPROB_DEFLATION, MIN_ODDS_BY_TYPE,
)
from .calibration import IsotonicCalibrator

logger = logging.getLogger(__name__)

# EV threshold: only buy when EV > this value
# Must be > 0 to overcome estimation uncertainty
EV_THRESHOLD = 0.05

# Max bets per race (risk control)
MAX_BETS_PER_RACE = 10

# Max exposure per race (total bet amount)
MAX_EXPOSURE_PER_RACE = 5000  # yen

# Allowed bet types
ALLOWED_TYPES = {"umaren", "umatan", "wide", "sanrenpuku", "tansho", "fukusho"}


class EVSelector:
    """Select bets based on positive expected value."""

    def __init__(self, calibrator: Optional[IsotonicCalibrator] = None):
        self._calibrator = calibrator

    def select(
        self,
        predictions: List[Dict],
        odds_data: Dict,
        race_info: Dict,
        entries: Optional[List[Dict]] = None,
        bet_amount: int = 500,
    ) -> List[Dict]:
        """Select bets with positive EV.

        Returns list of bets sorted by EV (highest first).
        May return 0 bets if no positive EV found.
        """
        head_count = race_info.get("headCount", 16)
        if head_count < 3:
            return []

        probs = scores_to_probabilities(predictions, head_count)
        if len(probs) < 3:
            return []

        # Pattern adjustment
        pattern = detect_race_pattern(probs)
        temp = {"本命堅軸": 0.85, "混戦模様": 1.15, "2強対決": 0.92}.get(pattern, 1.0)
        if temp != 1.0:
            probs = scores_to_probabilities(predictions, head_count, temp_adjust=temp)

        # Generate candidates
        candidates = generate_candidates(probs, top_n=min(7, len(probs)), entries=entries)

        # MC simulation for hit probabilities
        rng = random.Random(42)
        finishes = monte_carlo_finish(probs, MC_SAMPLES, rng=rng)
        candidates = estimate_hit_probabilities(finishes, candidates)

        # Apply deflation
        for c in candidates:
            c["hitProb"] *= HITPROB_DEFLATION.get(c["type"], 1.0)

        # Calibrate if calibrator available
        if self._calibrator is not None:
            import numpy as np
            hit_probs = np.array([c["hitProb"] for c in candidates])
            calibrated = self._calibrator.transform(hit_probs)
            for c, cp in zip(candidates, calibrated):
                c["hitProb"] = float(cp)

        # Calculate EV for each candidate
        selected = []
        for c in candidates:
            if c["type"] not in ALLOWED_TYPES:
                continue

            oi = find_odds_for_bet(c, odds_data)
            if not oi or oi["odds"] <= 0:
                continue

            min_odds = MIN_ODDS_BY_TYPE.get(c["type"], 2.0)
            if oi["odds"] < min_odds:
                continue

            # EV = P(hit) × odds - 1
            ev = c["hitProb"] * oi["odds"] - 1.0

            if ev > EV_THRESHOLD:
                c["ev"] = ev
                c["odds"] = oi["odds"]
                c["payout"] = oi["payout"]
                c["hasRealOdds"] = True
                if "oddsMin" in oi:
                    c["oddsMin"] = oi["oddsMin"]
                if "oddsMax" in oi:
                    c["oddsMax"] = oi["oddsMax"]
                selected.append(c)

        # Sort by EV (highest first)
        selected.sort(key=lambda x: -x["ev"])

        # Limit by count and exposure
        final = []
        total_exposure = 0
        for c in selected:
            if len(final) >= MAX_BETS_PER_RACE:
                break
            if total_exposure + bet_amount > MAX_EXPOSURE_PER_RACE:
                break
            final.append(c)
            total_exposure += bet_amount

        # Clean up and rank
        for c in candidates:
            c.pop("_frame_map", None)
        for i, bet in enumerate(final):
            bet["rank"] = i + 1

        return final
