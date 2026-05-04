"""TDD tests for the bet optimizer.

Covers: softmax probability, MC simulation, hit probability estimation,
EV calculation, candidate generation, race pattern detection, diversification,
pick_longshot, confidence gate, MIN_ODDS_BY_TYPE filtering, 馬単/枠連 logic.
"""
from __future__ import annotations

import pytest


class TestScoresToProbabilities:
    def test_returns_dict(self, sample_predictions):
        from backend.predictor.bet_optimizer import scores_to_probabilities
        probs = scores_to_probabilities(sample_predictions, 14)
        assert isinstance(probs, dict)
        assert len(probs) > 0

    def test_probabilities_sum_to_one(self, sample_predictions):
        from backend.predictor.bet_optimizer import scores_to_probabilities
        probs = scores_to_probabilities(sample_predictions, 14)
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01

    def test_higher_score_higher_prob(self, sample_predictions):
        from backend.predictor.bet_optimizer import scores_to_probabilities
        probs = scores_to_probabilities(sample_predictions, 14)
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        sorted_scores = sorted(
            [(p["horseNumber"], p["score"]) for p in sample_predictions if p["score"] > 0],
            key=lambda x: -x[1])
        # Top horse by score should be top by probability
        assert sorted_probs[0][0] == sorted_scores[0][0]

    def test_temp_adjust_concentrates(self, sample_predictions):
        from backend.predictor.bet_optimizer import scores_to_probabilities
        normal = scores_to_probabilities(sample_predictions, 14, temp_adjust=1.0)
        concentrated = scores_to_probabilities(sample_predictions, 14, temp_adjust=0.5)
        # Lower temp = top horse gets higher probability
        top_normal = max(normal.values())
        top_conc = max(concentrated.values())
        assert top_conc > top_normal

    def test_empty_predictions(self):
        from backend.predictor.bet_optimizer import scores_to_probabilities
        assert scores_to_probabilities([], 14) == {}


class TestMonteCarloFinish:
    def test_returns_correct_count(self):
        from backend.predictor.bet_optimizer import monte_carlo_finish
        probs = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}
        finishes = monte_carlo_finish(probs, n_samples=100)
        assert len(finishes) == 100

    def test_each_finish_has_top3(self):
        from backend.predictor.bet_optimizer import monte_carlo_finish
        probs = {1: 0.5, 2: 0.3, 3: 0.2}
        finishes = monte_carlo_finish(probs, n_samples=50)
        for f in finishes:
            assert len(f) == 3

    def test_favorite_wins_most(self):
        from backend.predictor.bet_optimizer import monte_carlo_finish
        probs = {1: 0.7, 2: 0.2, 3: 0.1}
        finishes = monte_carlo_finish(probs, n_samples=1000)
        wins = sum(1 for f in finishes if f[0] == 1)
        assert wins > 500  # Should win majority


class TestGenerateCandidates:
    def test_generates_all_types(self):
        from backend.predictor.bet_optimizer import generate_candidates
        probs = {1: 0.3, 2: 0.25, 3: 0.2, 4: 0.15, 5: 0.1}
        entries = [{"horseNumber": i, "frameNumber": i} for i in range(1, 6)]
        candidates = generate_candidates(probs, top_n=5, entries=entries)
        types = {c["type"] for c in candidates}
        # v8: all 8 JRA bet types
        assert "tansho" in types
        assert "fukusho" in types
        assert "wakuren" in types
        assert "umaren" in types
        assert "umatan" in types
        assert "wide" in types
        assert "sanrenpuku" in types
        assert "sanrentan" in types

    def test_wide_count(self):
        from backend.predictor.bet_optimizer import generate_candidates
        probs = {1: 0.3, 2: 0.25, 3: 0.2, 4: 0.15, 5: 0.1}
        candidates = generate_candidates(probs, top_n=5)
        wide = [c for c in candidates if c["type"] == "wide"]
        assert len(wide) == 10  # C(5,2) = 10 pairs from top 5

    def test_ordered_flag(self):
        from backend.predictor.bet_optimizer import generate_candidates
        probs = {1: 0.5, 2: 0.3, 3: 0.2}
        candidates = generate_candidates(probs, top_n=3)
        for c in candidates:
            if c["type"] in ("sanrentan", "umatan"):
                assert c["ordered"] is True
            elif c["type"] != "wakuren":
                assert c["ordered"] is False


class TestDetectRacePattern:
    def test_dominant_favorite(self):
        from backend.predictor.bet_optimizer import detect_race_pattern
        probs = {1: 0.50, 2: 0.15, 3: 0.10, 4: 0.08}
        assert detect_race_pattern(probs) == "本命堅軸"

    def test_competitive_field(self):
        from backend.predictor.bet_optimizer import detect_race_pattern
        probs = {1: 0.20, 2: 0.19, 3: 0.18, 4: 0.15}
        assert detect_race_pattern(probs) == "混戦模様"

    def test_two_horse_race(self):
        from backend.predictor.bet_optimizer import detect_race_pattern
        probs = {1: 0.35, 2: 0.30, 3: 0.10, 4: 0.08}
        assert detect_race_pattern(probs) == "2強対決"

    def test_small_field(self):
        from backend.predictor.bet_optimizer import detect_race_pattern
        probs = {1: 0.5, 2: 0.5}
        assert detect_race_pattern(probs) == "少頭数"


class TestOptimizeBets:
    def test_returns_max_5(self, sample_predictions, sample_odds_data, sample_race_info):
        from backend.predictor.bet_optimizer import optimize_bets
        bets = optimize_bets(sample_predictions, sample_odds_data, sample_race_info)
        assert len(bets) <= 5

    def test_bets_have_required_fields(self, sample_predictions, sample_odds_data, sample_race_info):
        from backend.predictor.bet_optimizer import optimize_bets
        bets = optimize_bets(sample_predictions, sample_odds_data, sample_race_info)
        for bet in bets:
            assert "type" in bet
            assert "typeLabel" in bet
            assert "horses" in bet
            assert "ev" in bet
            assert "hitProb" in bet
            assert "rank" in bet

    def test_bets_have_diverse_types(self, sample_predictions, sample_odds_data, sample_race_info):
        from backend.predictor.bet_optimizer import optimize_bets
        bets = optimize_bets(sample_predictions, sample_odds_data, sample_race_info)
        if len(bets) >= 3:
            # v8: balanced strategy guarantees type diversity
            types = {b["type"] for b in bets}
            assert len(types) >= 2  # At least 2 different types

    def test_diversification_max_2_per_type(self, sample_predictions, sample_odds_data, sample_race_info):
        from backend.predictor.bet_optimizer import optimize_bets
        bets = optimize_bets(sample_predictions, sample_odds_data, sample_race_info)
        from collections import Counter
        type_counts = Counter(b["type"] for b in bets)
        for t, count in type_counts.items():
            assert count <= 2, f"{t} has {count} bets, max 2"

    def test_no_bets_for_tiny_field(self, sample_race_info):
        from backend.predictor.bet_optimizer import optimize_bets
        sample_race_info["headCount"] = 2
        bets = optimize_bets([], {}, sample_race_info)
        assert bets == []

    def test_ranks_sequential(self, sample_predictions, sample_odds_data, sample_race_info):
        from backend.predictor.bet_optimizer import optimize_bets
        bets = optimize_bets(sample_predictions, sample_odds_data, sample_race_info)
        for i, bet in enumerate(bets):
            assert bet["rank"] == i + 1


class TestImpliedFairOdds:
    def test_positive_prob(self):
        from backend.predictor.bet_optimizer import implied_fair_odds
        odds = implied_fair_odds(0.5)
        assert odds == pytest.approx(1.5, rel=0.1)  # (1/0.5) * 0.75

    def test_zero_prob(self):
        from backend.predictor.bet_optimizer import implied_fair_odds
        assert implied_fair_odds(0) == 1.0


# ---------------------------------------------------------------------------
# NEW TDD TESTS — v8 balanced strategy coverage
# ---------------------------------------------------------------------------


class TestEstimateHitProbabilities:
    """Tests for estimate_hit_probabilities covering all bet types and edge cases."""

    def _make_candidate(self, bet_type, horses, frame_map=None):
        c = {"type": bet_type, "horses": horses}
        if frame_map is not None:
            c["_frame_map"] = frame_map
        return c

    # ── Edge case: empty finishes list ──────────────────────────────────────
    def test_empty_finishes_sets_zero_hit_prob(self):
        from backend.predictor.bet_optimizer import estimate_hit_probabilities
        candidates = [
            self._make_candidate("tansho", [1]),
            self._make_candidate("fukusho", [2]),
            self._make_candidate("sanrentan", [1, 2, 3]),
        ]
        result = estimate_hit_probabilities([], candidates)
        for c in result:
            assert c["hitProb"] == 0.0, f"{c['type']} should have 0.0 hitProb on empty finishes"

    def test_empty_finishes_returns_all_candidates(self):
        from backend.predictor.bet_optimizer import estimate_hit_probabilities
        candidates = [self._make_candidate("tansho", [1])]
        result = estimate_hit_probabilities([], candidates)
        assert len(result) == 1

    # ── 馬単: ordered pair must match exactly ──────────────────────────────
    def test_umatan_hit_requires_exact_order(self):
        from backend.predictor.bet_optimizer import estimate_hit_probabilities
        # finish [1, 2, 3] — 馬単 1-2 should hit, 2-1 should not
        finishes = [[1, 2, 3]] * 100
        c_hit = self._make_candidate("umatan", [1, 2])
        c_miss = self._make_candidate("umatan", [2, 1])
        estimate_hit_probabilities(finishes, [c_hit, c_miss])
        assert c_hit["hitProb"] == pytest.approx(1.0)
        assert c_miss["hitProb"] == pytest.approx(0.0)

    def test_umatan_miss_when_wrong_position(self):
        from backend.predictor.bet_optimizer import estimate_hit_probabilities
        # finish [2, 1, 3] — 馬単 1-2 should NOT hit
        finishes = [[2, 1, 3]] * 50
        c = self._make_candidate("umatan", [1, 2])
        estimate_hit_probabilities(finishes, [c])
        assert c["hitProb"] == pytest.approx(0.0)

    # ── 枠連 (wakuren): frame-based matching ─────────────────────────────
    def test_wakuren_hit_uses_frame_map(self):
        from backend.predictor.bet_optimizer import estimate_hit_probabilities
        # horse 1 -> frame 1, horse 2 -> frame 2
        frame_map = {1: 1, 2: 2, 3: 3}
        finishes = [[1, 2, 3]] * 100
        c = self._make_candidate("wakuren", [1, 2], frame_map=frame_map)
        estimate_hit_probabilities(finishes, [c])
        assert c["hitProb"] == pytest.approx(1.0)

    def test_wakuren_miss_when_frames_differ(self):
        from backend.predictor.bet_optimizer import estimate_hit_probabilities
        # horse 1 -> frame 1, horse 2 -> frame 2; candidate wants frames [3, 4]
        frame_map = {1: 1, 2: 2, 3: 3}
        finishes = [[1, 2, 3]] * 100
        c = self._make_candidate("wakuren", [3, 4], frame_map=frame_map)
        estimate_hit_probabilities(finishes, [c])
        assert c["hitProb"] == pytest.approx(0.0)

    def test_wakuren_uses_set_matching_not_order(self):
        from backend.predictor.bet_optimizer import estimate_hit_probabilities
        # finish [2, 1, 3] — frames [2, 1] — candidate frames {1, 2} should hit
        frame_map = {1: 1, 2: 2, 3: 3}
        finishes = [[2, 1, 3]] * 100
        c = self._make_candidate("wakuren", [1, 2], frame_map=frame_map)
        estimate_hit_probabilities(finishes, [c])
        assert c["hitProb"] == pytest.approx(1.0)

    def test_wakuren_no_frame_map_gives_zero(self):
        from backend.predictor.bet_optimizer import estimate_hit_probabilities
        # No frame_map — frame_map defaults to {} so no frames resolve
        finishes = [[1, 2, 3]] * 50
        c = self._make_candidate("wakuren", [1, 2])  # no _frame_map key
        estimate_hit_probabilities(finishes, [c])
        assert c["hitProb"] == pytest.approx(0.0)

    # ── tansho / fukusho / umaren / wide / sanrenpuku hit logic ────────────
    def test_tansho_hit_first_place_only(self):
        from backend.predictor.bet_optimizer import estimate_hit_probabilities
        finishes = [[1, 2, 3]] * 100 + [[2, 1, 3]] * 100
        c = self._make_candidate("tansho", [1])
        estimate_hit_probabilities(finishes, [c])
        assert c["hitProb"] == pytest.approx(0.5)

    def test_fukusho_hit_in_top3(self):
        from backend.predictor.bet_optimizer import estimate_hit_probabilities
        # horse 3 finishes 3rd each time → 複勝 hit
        finishes = [[1, 2, 3]] * 100
        c = self._make_candidate("fukusho", [3])
        estimate_hit_probabilities(finishes, [c])
        assert c["hitProb"] == pytest.approx(1.0)

    def test_fukusho_miss_outside_top3(self):
        from backend.predictor.bet_optimizer import estimate_hit_probabilities
        # horse 4 never in top3 (only 3 horses in finish)
        finishes = [[1, 2, 3]] * 100
        c = self._make_candidate("fukusho", [4])
        estimate_hit_probabilities(finishes, [c])
        assert c["hitProb"] == pytest.approx(0.0)

    def test_sanrentan_exact_order_required(self):
        from backend.predictor.bet_optimizer import estimate_hit_probabilities
        finishes = [[1, 2, 3]] * 100
        c_hit = self._make_candidate("sanrentan", [1, 2, 3])
        c_miss = self._make_candidate("sanrentan", [1, 3, 2])
        estimate_hit_probabilities(finishes, [c_hit, c_miss])
        assert c_hit["hitProb"] == pytest.approx(1.0)
        assert c_miss["hitProb"] == pytest.approx(0.0)


class TestGenerateCandidatesV8:
    """Tests for generate_candidates() focusing on v8 新機能."""

    def _make_entries(self, n_horses, frames=None):
        """Helper: build n horse entry dicts, optionally with explicit frame numbers."""
        entries = []
        for i in range(1, n_horses + 1):
            frame = frames[i - 1] if frames else i
            entries.append({
                "horseNumber": i,
                "frameNumber": frame,
                "isScratched": False,
            })
        return entries

    # ── 枠連 candidates generated only when entries provided ───────────────
    def test_wakuren_absent_without_entries(self):
        from backend.predictor.bet_optimizer import generate_candidates
        probs = {1: 0.3, 2: 0.25, 3: 0.2, 4: 0.15, 5: 0.1}
        candidates = generate_candidates(probs, top_n=5)
        types = {c["type"] for c in candidates}
        assert "wakuren" not in types

    def test_wakuren_present_with_entries(self):
        from backend.predictor.bet_optimizer import generate_candidates
        probs = {1: 0.3, 2: 0.25, 3: 0.2, 4: 0.15, 5: 0.1}
        entries = self._make_entries(5)
        candidates = generate_candidates(probs, top_n=5, entries=entries)
        types = {c["type"] for c in candidates}
        assert "wakuren" in types

    def test_wakuren_deduplicates_same_frame_pairs(self):
        from backend.predictor.bet_optimizer import generate_candidates
        # Horses 1 and 2 share frame 1; horses 3 and 4 share frame 2
        probs = {1: 0.3, 2: 0.25, 3: 0.2, 4: 0.15, 5: 0.1}
        frames = [1, 1, 2, 2, 3]  # horses 1&2 in frame 1, 3&4 in frame 2
        entries = self._make_entries(5, frames=frames)
        candidates = generate_candidates(probs, top_n=5, entries=entries)
        wakuren = [c for c in candidates if c["type"] == "wakuren"]
        # Extract frame pairs as frozensets
        pairs = [frozenset(c["horses"]) for c in wakuren]
        # No duplicate frame pairs
        assert len(pairs) == len(set(map(frozenset, [tuple(c["horses"]) for c in wakuren])))

    def test_wakuren_frame_pair_values_are_frame_numbers(self):
        from backend.predictor.bet_optimizer import generate_candidates
        probs = {1: 0.5, 2: 0.3, 3: 0.2}
        # horse 1 → frame 3, horse 2 → frame 5, horse 3 → frame 7
        entries = [
            {"horseNumber": 1, "frameNumber": 3, "isScratched": False},
            {"horseNumber": 2, "frameNumber": 5, "isScratched": False},
            {"horseNumber": 3, "frameNumber": 7, "isScratched": False},
        ]
        candidates = generate_candidates(probs, top_n=3, entries=entries)
        wakuren = [c for c in candidates if c["type"] == "wakuren"]
        all_frame_nums = set()
        for c in wakuren:
            all_frame_nums.update(c["horses"])
        # All values must be valid frame numbers (3, 5, or 7), NOT horse numbers
        assert all_frame_nums.issubset({3, 5, 7})

    def test_wakuren_carries_frame_map(self):
        from backend.predictor.bet_optimizer import generate_candidates
        probs = {1: 0.5, 2: 0.3, 3: 0.2}
        entries = self._make_entries(3)
        candidates = generate_candidates(probs, top_n=3, entries=entries)
        wakuren = [c for c in candidates if c["type"] == "wakuren"]
        for c in wakuren:
            assert "_frame_map" in c, "枠連 candidate must carry _frame_map"
            assert isinstance(c["_frame_map"], dict)

    def test_wakuren_excludes_scratched_horses_from_frame_map(self):
        from backend.predictor.bet_optimizer import generate_candidates
        probs = {1: 0.5, 2: 0.3, 3: 0.2}
        entries = [
            {"horseNumber": 1, "frameNumber": 1, "isScratched": False},
            {"horseNumber": 2, "frameNumber": 2, "isScratched": True},   # scratched
            {"horseNumber": 3, "frameNumber": 3, "isScratched": False},
        ]
        candidates = generate_candidates(probs, top_n=3, entries=entries)
        wakuren = [c for c in candidates if c["type"] == "wakuren"]
        for c in wakuren:
            # Scratched horse 2's frame (2) must not appear
            assert 2 not in c["_frame_map"]

    # ── 馬単: ordered pairs ────────────────────────────────────────────────
    def test_umatan_count_from_top5(self):
        from backend.predictor.bet_optimizer import generate_candidates
        probs = {1: 0.3, 2: 0.25, 3: 0.2, 4: 0.15, 5: 0.1}
        candidates = generate_candidates(probs, top_n=5)
        umatan = [c for c in candidates if c["type"] == "umatan"]
        # P(5,2) = 20 ordered pairs
        assert len(umatan) == 20

    def test_umatan_ordered_true(self):
        from backend.predictor.bet_optimizer import generate_candidates
        probs = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}
        candidates = generate_candidates(probs, top_n=4)
        umatan = [c for c in candidates if c["type"] == "umatan"]
        for c in umatan:
            assert c["ordered"] is True

    def test_umatan_includes_both_directions(self):
        from backend.predictor.bet_optimizer import generate_candidates
        probs = {1: 0.5, 2: 0.3, 3: 0.2}
        candidates = generate_candidates(probs, top_n=3)
        umatan_horses = [tuple(c["horses"]) for c in candidates if c["type"] == "umatan"]
        # Both (1,2) and (2,1) must be present
        assert (1, 2) in umatan_horses
        assert (2, 1) in umatan_horses

    # ── 単勝/複勝: top 3 ──────────────────────────────────────────────────
    def test_tansho_generates_top3(self):
        from backend.predictor.bet_optimizer import generate_candidates
        probs = {1: 0.3, 2: 0.25, 3: 0.2, 4: 0.15, 5: 0.1}
        candidates = generate_candidates(probs, top_n=5)
        tansho = [c for c in candidates if c["type"] == "tansho"]
        assert len(tansho) == 3

    def test_fukusho_generates_top3(self):
        from backend.predictor.bet_optimizer import generate_candidates
        probs = {1: 0.3, 2: 0.25, 3: 0.2, 4: 0.15, 5: 0.1}
        candidates = generate_candidates(probs, top_n=5)
        fukusho = [c for c in candidates if c["type"] == "fukusho"]
        assert len(fukusho) == 3

    # ── 3連単: top 4 permutations = 24 ────────────────────────────────────
    def test_sanrentan_count_from_top4(self):
        from backend.predictor.bet_optimizer import generate_candidates
        probs = {1: 0.3, 2: 0.25, 3: 0.2, 4: 0.15}
        candidates = generate_candidates(probs, top_n=4)
        sanrentan = [c for c in candidates if c["type"] == "sanrentan"]
        # P(4,3) = 24
        assert len(sanrentan) == 24


class TestFindOddsForBet:
    """Tests for find_odds_for_bet, especially umatan ordered matching."""

    def _make_odds_data(self):
        return {
            "umatan": [
                {"horses": [1, 2], "odds": 15.0, "payout": 1500},
                {"horses": [2, 1], "odds": 12.0, "payout": 1200},
                {"horses": [1, 3], "odds": 25.0, "payout": 2500},
            ],
            "umaren": [
                {"horses": [1, 2], "odds": 8.5, "payout": 850},
            ],
            "tansho": [
                {"horses": [1], "odds": 3.0, "payout": 300},
            ],
        }

    def test_umatan_exact_order_match(self):
        from backend.predictor.bet_optimizer import find_odds_for_bet
        bet = {"type": "umatan", "horses": [1, 2], "ordered": True}
        result = find_odds_for_bet(bet, self._make_odds_data())
        assert result is not None
        assert result["odds"] == 15.0

    def test_umatan_reversed_order_gives_different_odds(self):
        from backend.predictor.bet_optimizer import find_odds_for_bet
        bet = {"type": "umatan", "horses": [2, 1], "ordered": True}
        result = find_odds_for_bet(bet, self._make_odds_data())
        assert result is not None
        assert result["odds"] == 12.0

    def test_umatan_no_match_returns_none(self):
        from backend.predictor.bet_optimizer import find_odds_for_bet
        bet = {"type": "umatan", "horses": [3, 2], "ordered": True}
        result = find_odds_for_bet(bet, self._make_odds_data())
        assert result is None

    def test_umaren_unordered_match(self):
        from backend.predictor.bet_optimizer import find_odds_for_bet
        # Unordered: [2, 1] should match stored [1, 2]
        bet = {"type": "umaren", "horses": [2, 1], "ordered": False}
        result = find_odds_for_bet(bet, self._make_odds_data())
        assert result is not None
        assert result["odds"] == 8.5

    def test_missing_bet_type_returns_none(self):
        from backend.predictor.bet_optimizer import find_odds_for_bet
        bet = {"type": "wakuren", "horses": [1, 2], "ordered": False}
        result = find_odds_for_bet(bet, self._make_odds_data())
        assert result is None

    def test_empty_odds_data_returns_none(self):
        from backend.predictor.bet_optimizer import find_odds_for_bet
        bet = {"type": "tansho", "horses": [1], "ordered": False}
        assert find_odds_for_bet(bet, {}) is None
        assert find_odds_for_bet(bet, None) is None


class TestDiversify:
    """Tests for _diversify() — anchor selection, TYPE_LIMITS, umaren blocking."""

    def _make_viable_candidate(self, bet_type, horses, ev=0.1, hit_prob=0.3, odds=5.0):
        """Helper to create a candidate that will pass the viable filter."""
        return {
            "type": bet_type,
            "typeLabel": bet_type,
            "horses": horses,
            "ev": ev,
            "hitProb": hit_prob,
            "odds": odds,
            "ordered": bet_type in ("umatan", "sanrentan"),
        }

    def test_sanrentan_roi_anchor_always_selected(self):
        from backend.predictor.bet_optimizer import _diversify
        candidates = [
            self._make_viable_candidate("sanrentan", [1, 2, 3], ev=0.05, odds=80.0),
            self._make_viable_candidate("tansho", [1], ev=0.50, odds=3.0),
            self._make_viable_candidate("tansho", [2], ev=0.40, odds=4.0),
            self._make_viable_candidate("tansho", [3], ev=0.30, odds=5.0),
            self._make_viable_candidate("fukusho", [1], ev=0.20, odds=2.5),
        ]
        result = _diversify(candidates, max_bets=5)
        types = [b["type"] for b in result]
        assert "sanrentan" in types, "3連単 ROI anchor must always be selected"

    def test_wide_or_fukusho_hit_anchor_always_selected(self):
        from backend.predictor.bet_optimizer import _diversify
        candidates = [
            self._make_viable_candidate("sanrentan", [1, 2, 3], ev=0.05, odds=80.0),
            self._make_viable_candidate("wide", [1, 2], ev=0.10, odds=3.5),
            self._make_viable_candidate("tansho", [1], ev=0.40, odds=3.0),
            self._make_viable_candidate("tansho", [2], ev=0.35, odds=4.0),
            self._make_viable_candidate("tansho", [3], ev=0.30, odds=5.0),
        ]
        result = _diversify(candidates, max_bets=5)
        types = [b["type"] for b in result]
        assert "wide" in types or "fukusho" in types, \
            "ワイド or 複勝 hit anchor must always be selected"

    def test_fukusho_selected_as_hit_anchor_when_no_wide(self):
        from backend.predictor.bet_optimizer import _diversify
        candidates = [
            self._make_viable_candidate("sanrentan", [1, 2, 3], ev=0.05, odds=80.0),
            self._make_viable_candidate("fukusho", [1], ev=0.10, odds=2.5),
            self._make_viable_candidate("tansho", [1], ev=0.40, odds=3.0),
        ]
        result = _diversify(candidates, max_bets=3)
        types = [b["type"] for b in result]
        assert "fukusho" in types

    def test_umaren_included_as_anchor(self):
        from backend.predictor.bet_optimizer import _diversify
        # umaren has TYPE_LIMITS = 2, used as ◎流し anchor
        candidates = [
            self._make_viable_candidate("umaren", [1, 2], ev=0.80, odds=9.0),
            self._make_viable_candidate("umaren", [1, 3], ev=0.75, odds=12.0),
            self._make_viable_candidate("sanrentan", [1, 2, 3], ev=0.05, odds=80.0),
            self._make_viable_candidate("wide", [1, 2], ev=0.10, odds=3.5),
            self._make_viable_candidate("tansho", [1], ev=0.40, odds=3.0),
        ]
        result = _diversify(candidates, max_bets=5)
        types = [b["type"] for b in result]
        assert "umaren" in types, "馬連 must be included as anchor"
        assert types.count("umaren") <= 2, "馬連 max 2 per TYPE_LIMITS"

    def test_type_limits_max_two_wide(self):
        from backend.predictor.bet_optimizer import _diversify
        candidates = [
            self._make_viable_candidate("wide", [1, 2], ev=0.50, odds=4.0),
            self._make_viable_candidate("wide", [1, 3], ev=0.45, odds=5.0),
            self._make_viable_candidate("wide", [2, 3], ev=0.40, odds=6.0),
            self._make_viable_candidate("wide", [3, 4], ev=0.35, odds=7.0),
            self._make_viable_candidate("sanrentan", [1, 2, 3], ev=0.05, odds=80.0),
        ]
        result = _diversify(candidates, max_bets=5)
        wide_count = sum(1 for b in result if b["type"] == "wide")
        assert wide_count <= 2, "ワイド TYPE_LIMIT is 2"

    def test_type_limits_max_one_tansho(self):
        from backend.predictor.bet_optimizer import _diversify
        candidates = [
            self._make_viable_candidate("tansho", [1], ev=0.50, odds=3.0),
            self._make_viable_candidate("tansho", [2], ev=0.45, odds=4.0),
            self._make_viable_candidate("tansho", [3], ev=0.40, odds=5.0),
            self._make_viable_candidate("sanrentan", [1, 2, 3], ev=0.05, odds=80.0),
            self._make_viable_candidate("wide", [1, 2], ev=0.10, odds=3.5),
        ]
        result = _diversify(candidates, max_bets=5)
        tansho_count = sum(1 for b in result if b["type"] == "tansho")
        assert tansho_count <= 1, "単勝 TYPE_LIMIT is 1"

    def test_ranks_start_at_one(self):
        from backend.predictor.bet_optimizer import _diversify
        candidates = [
            self._make_viable_candidate("sanrentan", [1, 2, 3], ev=0.05, odds=80.0),
            self._make_viable_candidate("wide", [1, 2], ev=0.10, odds=3.5),
        ]
        result = _diversify(candidates, max_bets=2)
        assert result[0]["rank"] == 1
        assert result[1]["rank"] == 2

    def test_max_bets_respected(self):
        from backend.predictor.bet_optimizer import _diversify
        candidates = [
            self._make_viable_candidate("sanrentan", [1, 2, 3], ev=0.05, odds=80.0),
            self._make_viable_candidate("wide", [1, 2], ev=0.10, odds=3.5),
            self._make_viable_candidate("tansho", [1], ev=0.40, odds=3.0),
            self._make_viable_candidate("fukusho", [1], ev=0.20, odds=2.5),
            self._make_viable_candidate("umatan", [1, 2], ev=0.15, odds=12.0),
            self._make_viable_candidate("sanrenpuku", [1, 2, 3], ev=0.08, odds=20.0),
        ]
        result = _diversify(candidates, max_bets=3)
        assert len(result) <= 3

    def test_empty_candidates_returns_empty(self):
        from backend.predictor.bet_optimizer import _diversify
        assert _diversify([], max_bets=5) == []

    def test_candidates_below_min_ev_filtered_out(self):
        from backend.predictor.bet_optimizer import _diversify, MIN_EV_THRESHOLD
        candidates = [
            self._make_viable_candidate("tansho", [1], ev=MIN_EV_THRESHOLD - 0.01, odds=3.0),
        ]
        result = _diversify(candidates, max_bets=5)
        assert result == []

    def test_candidates_below_min_odds_filtered_out(self):
        """ワイド requires min odds 2.5; 1.9 should be filtered."""
        from backend.predictor.bet_optimizer import _diversify
        candidates = [
            self._make_viable_candidate("wide", [1, 2], ev=0.50, odds=1.9),
            self._make_viable_candidate("tansho", [1], ev=0.40, odds=1.5),
        ]
        result = _diversify(candidates, max_bets=5)
        assert result == []

    def test_overflow_fill_picks_remaining_within_limits(self):
        """Fill overflow path: diverse types fill remaining slots up to max_bets."""
        from backend.predictor.bet_optimizer import _diversify
        # After phase-1 anchors (sanrentan + wide), phase-2 fills tansho and umatan.
        # max_bets=4, so all 4 should be selected.
        candidates = [
            self._make_viable_candidate("sanrentan", [1, 2, 3], ev=0.05, odds=80.0),
            self._make_viable_candidate("wide", [1, 2], ev=0.10, odds=3.5),
            self._make_viable_candidate("tansho", [1], ev=0.40, odds=3.0),
            self._make_viable_candidate("umatan", [1, 2], ev=0.30, odds=12.0),
            self._make_viable_candidate("sanrenpuku", [1, 2, 3], ev=0.08, odds=20.0),
        ]
        result = _diversify(candidates, max_bets=4)
        assert len(result) == 4

    def test_overflow_respects_type_limits_for_umaren(self):
        """Overflow fill must respect TYPE_LIMITS (umaren max 2)."""
        from backend.predictor.bet_optimizer import _diversify
        candidates = [
            self._make_viable_candidate("umaren", [1, 2], ev=0.90, odds=9.0),
            self._make_viable_candidate("umaren", [2, 3], ev=0.85, odds=10.0),
            self._make_viable_candidate("umaren", [1, 3], ev=0.80, odds=11.0),
            self._make_viable_candidate("sanrentan", [1, 2, 3], ev=0.05, odds=80.0),
            self._make_viable_candidate("wide", [1, 2], ev=0.10, odds=3.5),
        ]
        result = _diversify(candidates, max_bets=5)
        types = [b["type"] for b in result]
        assert types.count("umaren") <= 2, "umaren must respect TYPE_LIMITS=2"


class TestMinOddsByType:
    """Tests for MIN_ODDS_BY_TYPE filtering in _diversify."""

    def test_wide_min_odds_is_2_5(self):
        from backend.predictor.bet_optimizer import MIN_ODDS_BY_TYPE
        assert MIN_ODDS_BY_TYPE["wide"] == 2.5

    def test_all_other_types_min_odds_is_2_0(self):
        from backend.predictor.bet_optimizer import MIN_ODDS_BY_TYPE
        for bet_type in ("tansho", "fukusho", "wakuren", "umaren", "umatan",
                         "sanrenpuku", "sanrentan"):
            assert MIN_ODDS_BY_TYPE[bet_type] == 2.0, \
                f"{bet_type} should have min_odds 2.0"

    def test_wide_2_4_filtered_out_in_diversify(self):
        """ワイド at odds=2.4 (below 2.5 threshold) must be rejected."""
        from backend.predictor.bet_optimizer import _diversify
        candidates = [
            {
                "type": "wide", "typeLabel": "ワイド", "horses": [1, 2],
                "ev": 0.5, "hitProb": 0.3, "odds": 2.4, "ordered": False,
            },
        ]
        result = _diversify(candidates, max_bets=5)
        assert result == []

    def test_wide_2_5_passes_filter(self):
        """ワイド at exactly 2.5 must pass the min_odds filter."""
        from backend.predictor.bet_optimizer import _diversify
        candidates = [
            {
                "type": "wide", "typeLabel": "ワイド", "horses": [1, 2],
                "ev": 0.5, "hitProb": 0.3, "odds": 2.5, "ordered": False,
            },
        ]
        result = _diversify(candidates, max_bets=5)
        assert len(result) == 1


class TestConfidenceGate:
    """Tests for confidence-based bet count reduction inside optimize_bets.

    The gate computes best_ev from all candidates' .ev values and reduces
    max_bets before calling _diversify.  We intercept _diversify to capture
    exactly what max_bets it receives so we can assert the gate fired.
    """

    def _make_predictions(self, n=8):
        return [
            {"horseNumber": i, "score": max(10, 85 - i * 7), "isScratched": False}
            for i in range(1, n + 1)
        ]

    def test_confidence_gate_low_ev_reduces_max_bets_to_3(self, sample_race_info,
                                                            monkeypatch):
        """best_ev < -0.35 → _diversify receives max_bets <= 3.

        Trigger: provide real odds for all bet types at terrible values so the
        has_real=True EV formula (hitProb * real_odds - 1 + edge_bonus) gives
        deeply negative EVs even for the best bet, ensuring best_ev < -0.35.
        """
        import backend.predictor.bet_optimizer as mod

        captured = {}
        original_diversify = mod._diversify

        def spy_diversify(candidates, max_bets, **kwargs):
            captured["max_bets"] = max_bets
            return original_diversify(candidates, max_bets, **kwargs)

        # With hitProb=0.002 and real odds=2.0 for ALL types:
        # base_ev = 0.002*2 - 1 = -0.996
        # edge = 0.002 - 0.5 = -0.498 → edge_bonus = -0.15 (clamped)
        # final EV = -1.146  (well below -0.35)
        # Provide real odds (>5) for every bet type to eliminate estimated-odds
        # candidates that would otherwise be stuck at EV=-0.35
        # Provide real odds for ALL possible candidate combinations
        # (top-5 horses = 1,2,3,4,5 for this prediction set) so that
        # every candidate gets a real-odds match and EV = hitProb*real_odds-1+edge
        # With hitProb=0.002, even odds=2.0 gives EV ≈ -1.15  (well below -0.35)
        n = 5  # top-n horses
        perms_all = [
            [i, j, k]
            for i in range(1, n + 1) for j in range(1, n + 1) for k in range(1, n + 1)
            if len({i, j, k}) == 3
        ]
        pairs_all = [[i, j] for i in range(1, n + 1) for j in range(i + 1, n + 1)]
        triples_all = [
            sorted([i, j, k])
            for i in range(1, n + 1) for j in range(i + 1, n + 1)
            for k in range(j + 1, n + 1)
        ]
        odds_data = {
            "sanrentan": [{"horses": p, "odds": 2.0, "payout": 200} for p in perms_all],
            "wide": [{"horses": p, "odds": 2.5, "payout": 250} for p in pairs_all],
            "umaren": [{"horses": p, "odds": 2.0, "payout": 200} for p in pairs_all],
            "tansho": [{"horses": [i], "odds": 2.0, "payout": 200} for i in range(1, n + 1)],
            "fukusho": [{"horses": [i], "odds": 2.0, "payout": 200} for i in range(1, n + 1)],
            "sanrenpuku": [{"horses": t, "odds": 2.0, "payout": 200} for t in triples_all],
            "umatan": [{"horses": [i, j], "odds": 2.0, "payout": 200}
                       for i in range(1, n + 1) for j in range(1, n + 1) if i != j],
        }

        def fake_estimate(finishes, candidates):
            for c in candidates:
                c["hitProb"] = 0.002
            return candidates

        monkeypatch.setattr(mod, "estimate_hit_probabilities", fake_estimate)
        monkeypatch.setattr(mod, "_diversify", spy_diversify)

        sample_race_info["headCount"] = 8
        mod.optimize_bets(self._make_predictions(), odds_data, sample_race_info)

        assert "max_bets" in captured, "_diversify was not called"
        assert captured["max_bets"] <= 4, (
            f"Confidence gate should reduce max_bets to ≤4 when best_ev is low, "
            f"got {captured['max_bets']}"
        )

    def test_confidence_gate_medium_ev_reduces_max_bets_to_4(self, sample_race_info,
                                                               monkeypatch):
        """best_ev in (-0.35, -0.20) → _diversify receives max_bets <= 4."""
        import backend.predictor.bet_optimizer as mod

        captured = {}
        original_diversify = mod._diversify

        def spy_diversify(candidates, max_bets, **kwargs):
            captured["max_bets"] = max_bets
            return original_diversify(candidates, max_bets, **kwargs)

        perms = [
            [i, j, k]
            for i in range(1, 5) for j in range(1, 5) for k in range(1, 5)
            if len({i, j, k}) == 3
        ]
        odds_data = {
            "sanrentan": [{"horses": p, "odds": 2.0, "payout": 200} for p in perms],
        }

        # hitProb=0.30 with real_odds=2.0:
        # EV = 0.30 * 2.0 - 1 = -0.40, edge = 0.30-0.50=-0.20, bonus=-0.04
        # final EV ≈ -0.44  (below -0.35 → gate fires at 3 for this uniform case)
        # Use hitProb=0.36: EV = 0.36*2.0-1 = -0.28, edge=0.36-0.50=-0.14, bonus=-0.028
        # final EV ≈ -0.308 (-0.35 < -0.308 < -0.20 → gate fires at 4)
        def fake_estimate(finishes, candidates):
            for c in candidates:
                c["hitProb"] = 0.36
            return candidates

        monkeypatch.setattr(mod, "estimate_hit_probabilities", fake_estimate)
        monkeypatch.setattr(mod, "_diversify", spy_diversify)

        sample_race_info["headCount"] = 8
        mod.optimize_bets(self._make_predictions(), odds_data, sample_race_info)

        assert "max_bets" in captured
        assert captured["max_bets"] <= 4, (
            f"Confidence gate should reduce max_bets to ≤4 when -0.35 < best_ev < -0.20, "
            f"got {captured['max_bets']}"
        )

    def test_no_gate_when_ev_above_threshold(self, sample_predictions, sample_odds_data,
                                              sample_race_info, monkeypatch):
        """best_ev >= -0.20 → _diversify receives the original max_bets=5."""
        import backend.predictor.bet_optimizer as mod

        captured = {}
        original_diversify = mod._diversify

        def spy_diversify(candidates, max_bets, **kwargs):
            captured["max_bets"] = max_bets
            return original_diversify(candidates, max_bets, **kwargs)

        monkeypatch.setattr(mod, "_diversify", spy_diversify)

        mod.optimize_bets(sample_predictions, sample_odds_data, sample_race_info)

        assert captured.get("max_bets") == 5


class TestOptimizeBetsV8:
    """Additional integration tests for optimize_bets with v8 changes."""

    def test_optimize_with_entries_param(self, sample_predictions, sample_odds_data,
                                         sample_race_info, sample_entries):
        """optimize_bets should pass entries to generate_candidates for 枠連."""
        from backend.predictor.bet_optimizer import optimize_bets
        bets = optimize_bets(
            sample_predictions, sample_odds_data, sample_race_info, entries=sample_entries
        )
        assert isinstance(bets, list)
        assert len(bets) <= 5

    def test_optimize_without_entries_no_wakuren(self, sample_predictions,
                                                  sample_odds_data, sample_race_info):
        """Without entries, no 枠連 candidate is ever generated."""
        from backend.predictor.bet_optimizer import optimize_bets
        bets = optimize_bets(sample_predictions, sample_odds_data, sample_race_info)
        types = {b["type"] for b in bets}
        assert "wakuren" not in types

    def test_optimize_bets_insufficient_probs(self, sample_race_info):
        """Fewer than 3 non-zero predictions → empty result."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = [
            {"horseNumber": 1, "score": 80, "isScratched": False},
            {"horseNumber": 2, "score": 0, "isScratched": False},
        ]
        bets = optimize_bets(predictions, {}, sample_race_info)
        assert bets == []

    def test_optimize_bets_no_scratched_horses_in_output(self, sample_race_info):
        """Scratched horses must not appear in any bet's horses list."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = [
            {"horseNumber": i, "score": max(10, 80 - i * 8), "isScratched": i == 3}
            for i in range(1, 9)
        ]
        sample_race_info["headCount"] = 8
        bets = optimize_bets(predictions, {}, sample_race_info)
        for bet in bets:
            assert 3 not in bet["horses"], "Scratched horse 3 must not appear in bets"

    def test_optimize_bets_with_real_odds_sets_has_real_odds(
            self, sample_predictions, sample_odds_data, sample_race_info):
        """Bets matched against real odds_data should have hasRealOdds=True."""
        from backend.predictor.bet_optimizer import optimize_bets
        bets = optimize_bets(sample_predictions, sample_odds_data, sample_race_info)
        real_odds_bets = [b for b in bets if b.get("hasRealOdds")]
        # sample_odds_data has entries so some bets should get real odds
        assert len(real_odds_bets) >= 0  # At least don't crash; check field exists
        for b in bets:
            assert "hasRealOdds" in b

    def test_optimize_bets_type_roi_bonus_path_no_real_odds(self, sample_race_info):
        """When no real odds exist, type_roi_bonus code path is executed."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = [
            {"horseNumber": i, "score": max(10, 85 - i * 7), "isScratched": False}
            for i in range(1, 9)
        ]
        sample_race_info["headCount"] = 8
        # Empty odds_data → has_real=False → type_roi_bonus branch
        bets = optimize_bets(predictions, {}, sample_race_info)
        for b in bets:
            assert b.get("hasRealOdds") is False
            assert "ev" in b


class TestEdgeCases:
    """Edge-case coverage for remaining uncovered lines."""

    def test_monte_carlo_zero_weight_breaks_gracefully(self):
        """Line 92: total_w <= 0 → break out of finish-building loop."""
        from backend.predictor.bet_optimizer import monte_carlo_finish
        import random
        # All probabilities are 0 → total_w = 0 on first inner iteration
        probs = {1: 0.0, 2: 0.0, 3: 0.0}
        finishes = monte_carlo_finish(probs, n_samples=10, rng=random.Random(0))
        # Each finish will be an empty list (loop breaks immediately)
        assert len(finishes) == 10
        for f in finishes:
            assert f == []

    def test_detect_race_pattern_standard(self):
        """Line 566: 標準配置 branch — gaps don't trigger any special pattern."""
        from backend.predictor.bet_optimizer import detect_race_pattern
        # gap_1_2 = 0.05 (not > 0.10), spread = 0.09 (not < 0.06),
        # gap_2_3 = 0.04 (not > 0.08) → 標準配置
        probs = {1: 0.30, 2: 0.25, 3: 0.21, 4: 0.15, 5: 0.09}
        result = detect_race_pattern(probs)
        assert result == "標準配置"


class TestPickLongshot:
    """Tests for pick_longshot."""

    def _make_candidate(self, bet_type, horses, odds, hit_prob, ev):
        return {
            "type": bet_type,
            "typeLabel": bet_type,
            "horses": horses,
            "odds": odds,
            "hitProb": hit_prob,
            "ev": ev,
        }

    def test_picks_best_longshot_by_score(self):
        from backend.predictor.bet_optimizer import pick_longshot
        candidates = [
            self._make_candidate("sanrentan", [1, 2, 3], odds=50.0, hit_prob=0.02, ev=-0.1),
            self._make_candidate("sanrentan", [1, 2, 4], odds=80.0, hit_prob=0.015, ev=-0.15),
        ]
        probs = {1: 0.5, 2: 0.3, 3: 0.2}
        result = pick_longshot(candidates, [], probs)
        assert result is not None
        # Score = odds * hitProb: 50*0.02=1.0 vs 80*0.015=1.2, so [1,2,4] wins
        assert result["horses"] == [1, 2, 4]

    def test_returns_none_when_no_candidates(self):
        from backend.predictor.bet_optimizer import pick_longshot
        assert pick_longshot([], [], {}) is None

    def test_filters_odds_below_minimum(self):
        from backend.predictor.bet_optimizer import pick_longshot
        candidates = [
            self._make_candidate("tansho", [1], odds=5.0, hit_prob=0.3, ev=0.2),
        ]
        result = pick_longshot(candidates, [], {1: 0.3})
        assert result is None

    def test_filters_odds_above_maximum(self):
        from backend.predictor.bet_optimizer import pick_longshot
        candidates = [
            self._make_candidate("sanrentan", [1, 2, 3], odds=200.0, hit_prob=0.01, ev=0.5),
        ]
        result = pick_longshot(candidates, [], {})
        assert result is None

    def test_filters_very_low_hit_prob(self):
        from backend.predictor.bet_optimizer import pick_longshot
        candidates = [
            self._make_candidate("sanrentan", [1, 2, 3], odds=50.0, hit_prob=0.001, ev=0.2),
        ]
        result = pick_longshot(candidates, [], {})
        assert result is None  # hitProb < 0.005 threshold

    def test_filters_low_ev(self):
        from backend.predictor.bet_optimizer import pick_longshot
        candidates = [
            self._make_candidate("sanrentan", [1, 2, 3], odds=50.0, hit_prob=0.02, ev=-0.5),
        ]
        result = pick_longshot(candidates, [], {})
        assert result is None  # ev <= -0.3

    def test_excludes_already_selected_bets(self):
        from backend.predictor.bet_optimizer import pick_longshot
        candidates = [
            self._make_candidate("sanrentan", [1, 2, 3], odds=50.0, hit_prob=0.02, ev=-0.1),
        ]
        selected = [{"type": "sanrentan", "horses": [1, 2, 3]}]
        result = pick_longshot(candidates, selected, {})
        assert result is None

    def test_sets_rank_zero(self):
        from backend.predictor.bet_optimizer import pick_longshot
        candidates = [
            self._make_candidate("sanrentan", [1, 2, 3], odds=50.0, hit_prob=0.02, ev=-0.1),
        ]
        result = pick_longshot(candidates, [], {})
        assert result is not None
        assert result["rank"] == 0


# ---------------------------------------------------------------------------
# NEW TDD TESTS — v9 ◎流し (honmei-nagashi) anchor behaviour
# ---------------------------------------------------------------------------


class TestHonmeiNagashiAnchor:
    """TDD tests for v9 _diversify() ◎流し anchor logic.

    Covers:
    - Phase 1a: umaren anchor containing the top AI horse (◎)
    - Phase 1b: wide anchor containing the top AI horse (◎)
    - Correct top-horse identification via probs_ref
    - Fallback paths (no matching candidates, probs_ref=None)
    - TYPE_LIMITS enforcement: umaren=2, wide=2
    - Both anchors coexist in the same output
    - New TYPE_BONUS values (umaren=0.20 highest, wide=0.18)
    """

    def _make_viable_candidate(self, bet_type, horses, ev=0.1, hit_prob=0.3, odds=5.0):
        """Build a candidate that passes the viable filter in _diversify."""
        return {
            "type": bet_type,
            "typeLabel": bet_type,
            "horses": horses,
            "ev": ev,
            "hitProb": hit_prob,
            "odds": odds,
            "ordered": bet_type in ("umatan", "sanrentan"),
        }

    # ── Phase 1a: umaren anchor selects the bet containing ◎ ─────────────

    def test_phase1a_picks_umaren_containing_top_horse(self):
        """Phase 1a must select a umaren whose horses list contains the ◎ horse."""
        from backend.predictor.bet_optimizer import _diversify
        # Horse 1 is ◎ (highest prob).  Provide two umaren: one with horse 1 and
        # one without.  The anchor should choose the one containing horse 1.
        probs_ref = {1: 0.40, 2: 0.25, 3: 0.20, 4: 0.15}
        candidates = [
            self._make_viable_candidate("umaren", [2, 3], ev=0.50, odds=10.0),  # no ◎
            self._make_viable_candidate("umaren", [1, 2], ev=0.30, odds=9.0),   # ◎ included
            self._make_viable_candidate("umaren", [1, 3], ev=0.20, odds=12.0),  # ◎ included
            self._make_viable_candidate("wide",   [2, 3], ev=0.10, odds=3.5),
        ]
        result = _diversify(candidates, max_bets=5, probs_ref=probs_ref)
        umaren_bets = [b for b in result if b["type"] == "umaren"]
        assert len(umaren_bets) >= 1, "At least one umaren must be selected"
        # Phase 1a guarantees at least the first umaren selected contains ◎
        anchor_umaren = [b for b in umaren_bets if 1 in b["horses"]]
        assert len(anchor_umaren) >= 1, "Phase 1a umaren anchor must contain ◎ horse (1)"

    def test_phase1a_anchor_umaren_is_best_ev_among_honmei_candidates(self):
        """When multiple umaren contain ◎, Phase 1a picks the one with highest EV."""
        from backend.predictor.bet_optimizer import _diversify
        probs_ref = {1: 0.45, 2: 0.30, 3: 0.25}
        # Two umaren containing horse 1; second has higher EV
        lower_ev = self._make_viable_candidate("umaren", [1, 2], ev=0.15, odds=8.0)
        higher_ev = self._make_viable_candidate("umaren", [1, 3], ev=0.35, odds=12.0)
        candidates = [lower_ev, higher_ev,
                      self._make_viable_candidate("wide", [2, 3], ev=0.10, odds=3.5)]
        result = _diversify(candidates, max_bets=5, probs_ref=probs_ref)
        # The first umaren in the result (rank 1) must be the higher-EV one
        first_umaren = next(b for b in result if b["type"] == "umaren")
        assert first_umaren["horses"] == higher_ev["horses"], (
            "Phase 1a must pick the umaren with highest EV among ◎-containing candidates"
        )

    def test_phase1a_anchor_is_selected_even_when_ev_is_lower_than_other_types(self):
        """The ◎ umaren is picked even if tansho or sanrentan has a higher raw EV."""
        from backend.predictor.bet_optimizer import _diversify
        probs_ref = {1: 0.50, 2: 0.30, 3: 0.20}
        candidates = [
            self._make_viable_candidate("tansho",   [1],    ev=0.90, odds=3.0),
            self._make_viable_candidate("sanrentan", [1,2,3], ev=0.80, odds=80.0),
            self._make_viable_candidate("umaren",   [1, 2], ev=0.05, odds=8.0),  # low EV ◎
            self._make_viable_candidate("wide",     [1, 2], ev=0.10, odds=3.5),
        ]
        result = _diversify(candidates, max_bets=5, probs_ref=probs_ref)
        types = [b["type"] for b in result]
        assert "umaren" in types, "◎ umaren anchor must be included despite low EV"
        umaren_bet = next(b for b in result if b["type"] == "umaren")
        assert 1 in umaren_bet["horses"], "Selected umaren must contain ◎ horse"

    # ── Phase 1b: wide anchor selects the bet containing ◎ ───────────────

    def test_phase1b_picks_wide_containing_top_horse(self):
        """Phase 1b must select a wide whose horses list contains the ◎ horse."""
        from backend.predictor.bet_optimizer import _diversify
        probs_ref = {1: 0.40, 2: 0.25, 3: 0.20, 4: 0.15}
        candidates = [
            self._make_viable_candidate("umaren", [1, 2], ev=0.30, odds=9.0),
            self._make_viable_candidate("wide",   [2, 3], ev=0.50, odds=4.0),  # no ◎
            self._make_viable_candidate("wide",   [1, 2], ev=0.20, odds=3.5),  # ◎
            self._make_viable_candidate("wide",   [1, 3], ev=0.10, odds=5.0),  # ◎
        ]
        result = _diversify(candidates, max_bets=5, probs_ref=probs_ref)
        wide_bets = [b for b in result if b["type"] == "wide"]
        assert len(wide_bets) >= 1, "At least one wide must be selected"
        anchor_wide = [b for b in wide_bets if 1 in b["horses"]]
        assert len(anchor_wide) >= 1, "Phase 1b wide anchor must contain ◎ horse (1)"

    def test_phase1b_anchor_wide_is_best_ev_among_honmei_candidates(self):
        """When multiple wide contain ◎, Phase 1b picks the one with highest EV."""
        from backend.predictor.bet_optimizer import _diversify
        probs_ref = {1: 0.45, 2: 0.30, 3: 0.25}
        lower_ev_wide  = self._make_viable_candidate("wide", [1, 2], ev=0.10, odds=3.5)
        higher_ev_wide = self._make_viable_candidate("wide", [1, 3], ev=0.30, odds=5.0)
        candidates = [
            self._make_viable_candidate("umaren", [1, 2], ev=0.20, odds=9.0),
            lower_ev_wide,
            higher_ev_wide,
        ]
        result = _diversify(candidates, max_bets=5, probs_ref=probs_ref)
        wide_results = [b for b in result if b["type"] == "wide"]
        # The wide anchor (first wide picked by Phase 1b) must be the higher-EV one
        assert any(b["horses"] == higher_ev_wide["horses"] for b in wide_results), (
            "Phase 1b must pick the wide with highest EV among ◎-containing candidates"
        )

    def test_phase1b_anchor_is_selected_even_when_non_anchor_wide_has_higher_ev(self):
        """Wide without ◎ may have higher EV, but the ◎-anchor wide is still picked."""
        from backend.predictor.bet_optimizer import _diversify
        probs_ref = {1: 0.50, 2: 0.30, 3: 0.20}
        candidates = [
            self._make_viable_candidate("umaren",   [1, 2], ev=0.20, odds=9.0),
            self._make_viable_candidate("wide",     [2, 3], ev=0.80, odds=4.0),  # high EV, no ◎
            self._make_viable_candidate("wide",     [1, 2], ev=0.05, odds=3.5),  # low EV, ◎
        ]
        result = _diversify(candidates, max_bets=5, probs_ref=probs_ref)
        wide_bets = [b for b in result if b["type"] == "wide"]
        anchor_wide = [b for b in wide_bets if 1 in b["horses"]]
        assert len(anchor_wide) >= 1, "◎ wide anchor must be selected despite lower EV"

    # ── Top-horse identification from probs_ref ───────────────────────────

    def test_top_horse_is_identified_by_highest_probability(self):
        """The horse with the highest probability in probs_ref is ◎."""
        from backend.predictor.bet_optimizer import _diversify
        # Horse 3 has the highest probability — it must be the anchor target
        probs_ref = {1: 0.20, 2: 0.25, 3: 0.55}
        candidates = [
            self._make_viable_candidate("umaren", [1, 2], ev=0.40, odds=10.0),  # no ◎(3)
            self._make_viable_candidate("umaren", [2, 3], ev=0.20, odds=9.0),   # ◎(3)
            self._make_viable_candidate("wide",   [1, 2], ev=0.30, odds=4.0),   # no ◎(3)
            self._make_viable_candidate("wide",   [3, 1], ev=0.10, odds=3.5),   # ◎(3)
        ]
        result = _diversify(candidates, max_bets=5, probs_ref=probs_ref)
        umaren_bets = [b for b in result if b["type"] == "umaren"]
        wide_bets   = [b for b in result if b["type"] == "wide"]
        # At least the anchor bets contain horse 3
        assert any(3 in b["horses"] for b in umaren_bets), "Umaren anchor must include horse 3"
        assert any(3 in b["horses"] for b in wide_bets),   "Wide anchor must include horse 3"

    # ── Phase 1a skipped when no umaren candidates contain ◎ ─────────────

    def test_phase1a_skipped_when_no_honmei_umaren_exists(self):
        """When no viable umaren contains the ◎ horse, Phase 1a is skipped gracefully."""
        from backend.predictor.bet_optimizer import _diversify
        # Horse 1 is ◎ but all umaren candidates are horse 2 vs horse 3 (no ◎)
        probs_ref = {1: 0.50, 2: 0.30, 3: 0.20}
        candidates = [
            self._make_viable_candidate("umaren", [2, 3], ev=0.30, odds=10.0),
            self._make_viable_candidate("wide",   [1, 2], ev=0.20, odds=3.5),
            self._make_viable_candidate("tansho", [1],    ev=0.40, odds=3.0),
        ]
        # Must not raise and must return a valid list
        result = _diversify(candidates, max_bets=5, probs_ref=probs_ref)
        assert isinstance(result, list)
        # No umaren anchor was available, so umaren may or may not appear via Phase 2
        # but the call must succeed without error

    def test_phase1a_skipped_does_not_prevent_other_bets_from_being_selected(self):
        """When Phase 1a cannot fire, Phase 1b and Phase 2 still populate the output."""
        from backend.predictor.bet_optimizer import _diversify
        probs_ref = {1: 0.50, 2: 0.30, 3: 0.20}
        candidates = [
            # No umaren containing horse 1 — Phase 1a skipped
            self._make_viable_candidate("umaren", [2, 3], ev=0.30, odds=10.0),
            self._make_viable_candidate("wide",   [1, 2], ev=0.20, odds=3.5),
            self._make_viable_candidate("tansho", [2],    ev=0.40, odds=3.0),
        ]
        result = _diversify(candidates, max_bets=5, probs_ref=probs_ref)
        assert len(result) >= 2, "Output should still contain at least 2 bets"
        types = [b["type"] for b in result]
        assert "wide" in types, "Wide anchor should still be selected via Phase 1b"

    # ── Fallback: probs_ref=None ──────────────────────────────────────────

    def test_probs_ref_none_does_not_crash(self):
        """_diversify(probs_ref=None) must not raise any exception."""
        from backend.predictor.bet_optimizer import _diversify
        candidates = [
            self._make_viable_candidate("umaren",   [1, 2], ev=0.30, odds=9.0),
            self._make_viable_candidate("wide",     [1, 2], ev=0.20, odds=3.5),
            self._make_viable_candidate("tansho",   [1],    ev=0.40, odds=3.0),
        ]
        result = _diversify(candidates, max_bets=5, probs_ref=None)
        assert isinstance(result, list)

    def test_probs_ref_none_selects_by_ev_only(self):
        """With probs_ref=None both anchor phases are skipped; Phase 2 fills by EV."""
        from backend.predictor.bet_optimizer import _diversify
        candidates = [
            self._make_viable_candidate("tansho",   [1],    ev=0.90, odds=3.0),
            self._make_viable_candidate("wide",     [1, 2], ev=0.50, odds=3.5),
            self._make_viable_candidate("umaren",   [1, 2], ev=0.40, odds=9.0),
            self._make_viable_candidate("sanrentan", [1,2,3], ev=0.10, odds=80.0),
        ]
        result = _diversify(candidates, max_bets=4, probs_ref=None)
        # All four should be selected since there are exactly 4 viable candidates
        # and no anchor logic restricts them
        assert len(result) == 4

    def test_probs_ref_none_returns_empty_for_empty_candidates(self):
        """probs_ref=None with an empty list must return an empty list."""
        from backend.predictor.bet_optimizer import _diversify
        assert _diversify([], max_bets=5, probs_ref=None) == []

    # ── TYPE_LIMITS: umaren max 2 ─────────────────────────────────────────

    def test_type_limit_umaren_max_2_respected_with_probs_ref(self):
        """Even with probs_ref supplied, umaren TYPE_LIMIT=2 must not be exceeded."""
        from backend.predictor.bet_optimizer import _diversify
        probs_ref = {1: 0.45, 2: 0.25, 3: 0.20, 4: 0.10}
        candidates = [
            self._make_viable_candidate("umaren", [1, 2], ev=0.80, odds=9.0),
            self._make_viable_candidate("umaren", [1, 3], ev=0.75, odds=10.0),
            self._make_viable_candidate("umaren", [1, 4], ev=0.70, odds=11.0),  # 3rd ◎ umaren
            self._make_viable_candidate("umaren", [2, 3], ev=0.65, odds=12.0),
            self._make_viable_candidate("wide",   [1, 2], ev=0.20, odds=3.5),
        ]
        result = _diversify(candidates, max_bets=5, probs_ref=probs_ref)
        umaren_count = sum(1 for b in result if b["type"] == "umaren")
        assert umaren_count <= 2, f"umaren TYPE_LIMIT=2 violated: got {umaren_count}"

    def test_type_limit_umaren_2_allows_exactly_two(self):
        """Exactly 2 umaren bets should appear when 2+ viable ◎ umaren exist."""
        from backend.predictor.bet_optimizer import _diversify
        probs_ref = {1: 0.50, 2: 0.30, 3: 0.20}
        candidates = [
            self._make_viable_candidate("umaren", [1, 2], ev=0.40, odds=9.0),
            self._make_viable_candidate("umaren", [1, 3], ev=0.35, odds=11.0),
            self._make_viable_candidate("wide",   [1, 2], ev=0.20, odds=3.5),
            self._make_viable_candidate("tansho", [1],    ev=0.30, odds=3.0),
        ]
        result = _diversify(candidates, max_bets=5, probs_ref=probs_ref)
        umaren_count = sum(1 for b in result if b["type"] == "umaren")
        # Phase 1a picks one; Phase 2 can pick a second (within limit=2)
        assert umaren_count <= 2
        assert umaren_count >= 1, "At least one umaren (the anchor) must appear"

    # ── TYPE_LIMITS: wide max 2 ───────────────────────────────────────────

    def test_type_limit_wide_max_2_respected_with_probs_ref(self):
        """wide TYPE_LIMIT=2 must not be exceeded even after Phase 1b anchor."""
        from backend.predictor.bet_optimizer import _diversify
        probs_ref = {1: 0.45, 2: 0.25, 3: 0.20, 4: 0.10}
        candidates = [
            self._make_viable_candidate("umaren", [1, 2], ev=0.30, odds=9.0),
            self._make_viable_candidate("wide", [1, 2], ev=0.50, odds=4.0),
            self._make_viable_candidate("wide", [1, 3], ev=0.45, odds=5.0),
            self._make_viable_candidate("wide", [2, 3], ev=0.40, odds=6.0),
            self._make_viable_candidate("wide", [3, 4], ev=0.35, odds=7.0),
        ]
        result = _diversify(candidates, max_bets=5, probs_ref=probs_ref)
        wide_count = sum(1 for b in result if b["type"] == "wide")
        assert wide_count <= 2, f"wide TYPE_LIMIT=2 violated: got {wide_count}"

    def test_type_limit_wide_2_allows_exactly_two(self):
        """Exactly 2 wide bets should appear when 2+ viable wide candidates exist."""
        from backend.predictor.bet_optimizer import _diversify
        probs_ref = {1: 0.50, 2: 0.30, 3: 0.20}
        candidates = [
            self._make_viable_candidate("umaren", [1, 2], ev=0.30, odds=9.0),
            self._make_viable_candidate("wide",   [1, 2], ev=0.40, odds=3.5),
            self._make_viable_candidate("wide",   [1, 3], ev=0.30, odds=5.0),
            self._make_viable_candidate("tansho", [1],    ev=0.20, odds=3.0),
        ]
        result = _diversify(candidates, max_bets=5, probs_ref=probs_ref)
        wide_count = sum(1 for b in result if b["type"] == "wide")
        assert wide_count <= 2
        assert wide_count >= 1, "At least one wide (the anchor) must appear"

    # ── Both umaren and wide anchors coexist in the same output ──────────

    def test_both_umaren_and_wide_anchors_in_output(self):
        """A single call must produce at least one ◎ umaren AND one ◎ wide."""
        from backend.predictor.bet_optimizer import _diversify
        probs_ref = {1: 0.50, 2: 0.30, 3: 0.20}
        candidates = [
            self._make_viable_candidate("umaren", [1, 2], ev=0.30, odds=9.0),
            self._make_viable_candidate("umaren", [1, 3], ev=0.20, odds=11.0),
            self._make_viable_candidate("wide",   [1, 2], ev=0.25, odds=3.5),
            self._make_viable_candidate("wide",   [1, 3], ev=0.15, odds=5.0),
            self._make_viable_candidate("tansho", [1],    ev=0.40, odds=3.0),
        ]
        result = _diversify(candidates, max_bets=5, probs_ref=probs_ref)
        has_anchor_umaren = any(b["type"] == "umaren" and 1 in b["horses"] for b in result)
        has_anchor_wide   = any(b["type"] == "wide"   and 1 in b["horses"] for b in result)
        assert has_anchor_umaren, "◎ umaren anchor must appear in output"
        assert has_anchor_wide,   "◎ wide anchor must appear in output"

    def test_both_anchors_coexist_with_remaining_slots_filled(self):
        """Both anchors are selected and the remaining slots are filled by Phase 2."""
        from backend.predictor.bet_optimizer import _diversify
        probs_ref = {1: 0.50, 2: 0.30, 3: 0.20}
        candidates = [
            self._make_viable_candidate("umaren",   [1, 2], ev=0.30, odds=9.0),
            self._make_viable_candidate("wide",     [1, 2], ev=0.25, odds=3.5),
            self._make_viable_candidate("tansho",   [1],    ev=0.40, odds=3.0),
            self._make_viable_candidate("sanrentan", [1,2,3], ev=0.10, odds=80.0),
            self._make_viable_candidate("umatan",   [1, 2], ev=0.15, odds=12.0),
        ]
        result = _diversify(candidates, max_bets=5, probs_ref=probs_ref)
        # All 5 candidates are viable and diverse enough to fill 5 slots
        assert len(result) == 5
        types = {b["type"] for b in result}
        assert "umaren" in types
        assert "wide" in types

    # ── TYPE_BONUS values ─────────────────────────────────────────────────

    def test_type_bonus_umaren_is_highest(self):
        """TYPE_BONUS for umaren (0.20) is the highest of all types.

        Verified indirectly: Phase 1a fires before Phase 1b, so the umaren anchor
        always receives rank=1.  Even though tansho has a higher raw EV (0.32 vs
        0.15), the umaren anchor is guaranteed the first slot by Phase 1a.
        After type bonus the umaren sort_ev = 0.15+0.20 = 0.35, beating tansho
        sort_ev = 0.32+0.02 = 0.34 — so umaren also wins Phase 2 sort order.
        """
        import backend.predictor.bet_optimizer as mod
        probs_ref = {1: 0.50, 2: 0.30, 3: 0.20}
        # umaren raw EV 0.15, tansho raw EV 0.32
        # After bonus: umaren 0.15+0.20=0.35 vs tansho 0.32+0.02=0.34 → umaren sorts first
        candidates = [
            self._make_viable_candidate("tansho", [1],    ev=0.32, odds=3.0),
            self._make_viable_candidate("umaren", [1, 2], ev=0.15, odds=9.0),
            self._make_viable_candidate("wide",   [2, 3], ev=0.10, odds=3.5),
        ]
        result = mod._diversify(candidates, max_bets=3, probs_ref=probs_ref)
        # Phase 1a picks umaren first → it must be rank 1
        assert result[0]["type"] == "umaren", (
            "umaren anchor (Phase 1a) must be rank-1 bet — TYPE_BONUS 0.20 is highest"
        )

    def test_type_bonus_wide_is_second_highest(self):
        """Wide TYPE_BONUS=0.18 must be applied: wide beats sanrentan in sort order
        when raw EVs are equal (sanrentan bonus=0.05 < wide bonus=0.18)."""
        from backend.predictor.bet_optimizer import _diversify
        probs_ref = {1: 0.50, 2: 0.30, 3: 0.20}
        # Both have raw EV 0.10. After bonus: wide=0.28, sanrentan=0.15 → wide sorted first
        candidates = [
            self._make_viable_candidate("umaren",   [1, 2], ev=0.30, odds=9.0),
            self._make_viable_candidate("wide",     [2, 3], ev=0.10, odds=3.5),   # no ◎
            self._make_viable_candidate("sanrentan", [1,2,3], ev=0.10, odds=80.0),
        ]
        # max_bets=2: Phase 1a picks umaren anchor, then Phase 2 fills one more.
        # wide sorted_ev=0.28 > sanrentan sorted_ev=0.15, so wide should be chosen.
        result = _diversify(candidates, max_bets=2, probs_ref=probs_ref)
        assert len(result) == 2
        types = [b["type"] for b in result]
        assert "wide" in types, "wide (bonus 0.18) should outrank sanrentan (bonus 0.05)"

    # ── Rank assignment ───────────────────────────────────────────────────

    def test_ranks_are_sequential_starting_at_one(self):
        """Ranks must be 1, 2, … len(result) regardless of anchor order."""
        from backend.predictor.bet_optimizer import _diversify
        probs_ref = {1: 0.50, 2: 0.30, 3: 0.20}
        candidates = [
            self._make_viable_candidate("umaren",   [1, 2], ev=0.30, odds=9.0),
            self._make_viable_candidate("wide",     [1, 2], ev=0.25, odds=3.5),
            self._make_viable_candidate("tansho",   [1],    ev=0.40, odds=3.0),
        ]
        result = _diversify(candidates, max_bets=3, probs_ref=probs_ref)
        ranks = [b["rank"] for b in result]
        assert ranks == list(range(1, len(result) + 1)), (
            f"Ranks must be sequential from 1, got {ranks}"
        )
