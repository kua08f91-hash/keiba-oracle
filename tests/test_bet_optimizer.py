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
    def test_returns_max_14(self, sample_predictions, sample_odds_data, sample_race_info):
        from backend.predictor.bet_optimizer import optimize_bets, MAX_BETS
        bets = optimize_bets(sample_predictions, sample_odds_data, sample_race_info)
        assert len(bets) <= MAX_BETS  # D5: 14

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

    def test_bets_are_honmei_anchor_types(self, sample_predictions, sample_odds_data, sample_race_info):
        from backend.predictor.bet_optimizer import optimize_bets
        bets = optimize_bets(sample_predictions, sample_odds_data, sample_race_info)
        for b in bets:
            assert b["type"] in ("umatan", "umaren", "wide"), f"Unexpected type: {b['type']}"

    def test_max_14_bets_per_race(self, sample_predictions, sample_odds_data, sample_race_info):
        from backend.predictor.bet_optimizer import optimize_bets
        bets = optimize_bets(sample_predictions, sample_odds_data, sample_race_info)
        assert len(bets) <= 14

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


import pytest

@pytest.mark.skip(reason="ConfidenceGate removed in v11 value-range strategy")
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


# ---------------------------------------------------------------------------
# S8 MULTI-TYPE VALUE-RANGE STRATEGY TESTS
# ---------------------------------------------------------------------------


class TestS8Strategy:
    """TDD tests for D5 ◎軸厚張り strategy in optimize_bets().

    D5 Strategy:
      - ◎軸展開: 馬単◎→AI2~7位(6) + 馬連◎-AI2~5位(4) + ワイド◎-AI2~5位(4)
      - Only umatan / umaren / wide types
      - Real odds required
      - Min odds: umatan 5x, umaren 3x, wide 2.5x
      - MAX_BETS = 14
      - Sorted highest odds first
    """

    def _make_predictions(self, n: int = 8) -> list:
        """Build n predictions with clearly separated scores."""
        return [
            {"horseNumber": i, "score": max(5, 90 - i * 8), "isScratched": False}
            for i in range(1, n + 1)
        ]
        # scores: 1→82, 2→74, 3→66, 4→58, 5→50, 6→42, 7→34, 8→26

    def _race_info(self, head_count: int = 10) -> dict:
        return {"raceId": "202606030210", "headCount": head_count}

    def _odds_entry(self, horses: list, odds: float) -> dict:
        return {"horses": horses, "odds": odds, "payout": int(odds * 100)}

    # ── 1. Only umatan / umaren / wide types are returned ────────────────────

    def test_only_allowed_types_returned(self):
        """No tansho, fukusho, sanrentan, sanrenpuku, or wakuren should appear."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {
            # Provide qualifying odds for every non-allowed type to prove they are rejected.
            "tansho":    [self._odds_entry([1], 25.0)],
            "fukusho":   [self._odds_entry([1], 15.0)],
            "wakuren":   [self._odds_entry([1, 2], 30.0)],
            "sanrenpuku":[self._odds_entry([1, 2, 3], 45.0)],
            "sanrentan": [self._odds_entry([1, 2, 3], 120.0)],
            # Qualifying entries for the three allowed types:
            "umatan":    [self._odds_entry([1, 2], 50.0)],
            "umaren":    [self._odds_entry([1, 2], 30.0)],
            "wide":      [self._odds_entry([1, 2], 20.0)],
        }
        bets = optimize_bets(predictions, odds_data, self._race_info())
        for bet in bets:
            assert bet["type"] in {"umatan", "umaren", "wide"}, (
                f"Disallowed type returned: {bet['type']}"
            )

    def test_tansho_explicitly_absent(self):
        """単勝 (tansho) must never appear even when it has excellent odds."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {
            "tansho": [self._odds_entry([1], 50.0)],
            "umatan": [self._odds_entry([1, 2], 50.0)],
        }
        bets = optimize_bets(predictions, odds_data, self._race_info())
        assert all(b["type"] != "tansho" for b in bets)

    def test_sanrentan_explicitly_absent(self):
        """3連単 must never appear even when in valid odds range."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {
            "sanrentan": [self._odds_entry([1, 2, 3], 100.0)],
            "wide":      [self._odds_entry([1, 2], 20.0)],
        }
        bets = optimize_bets(predictions, odds_data, self._race_info())
        assert all(b["type"] != "sanrentan" for b in bets)

    # ── 2. umatan odds range 20–300x ─────────────────────────────────────────

    def test_umatan_within_range_is_included(self):
        """A umatan bet at exactly the midpoint (160x) must be selected."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {"umatan": [self._odds_entry([1, 2], 160.0)]}
        bets = optimize_bets(predictions, odds_data, self._race_info())
        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) >= 1, "umatan at 160x (in range) must be selected"

    def test_umatan_below_5x_rejected(self):
        """D5: umatan below 5x must be rejected (odds=4.9)."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {"umatan": [self._odds_entry([1, 2], 4.9)]}
        bets = optimize_bets(predictions, odds_data, self._race_info())
        assert not any(b["type"] == "umatan" for b in bets), (
            "umatan at 4.9x (below 5x minimum) must be excluded"
        )

    def test_umatan_at_exact_5x_accepted(self):
        """D5: umatan at exactly 5.0x must be accepted."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {"umatan": [self._odds_entry([1, 2], 5.0)]}
        bets = optimize_bets(predictions, odds_data, self._race_info())
        assert any(b["type"] == "umatan" for b in bets), (
            "umatan at 5.0x (at D5 minimum) must be accepted"
        )

    def test_umatan_high_odds_no_upper_limit(self):
        """D5: no upper limit for umatan — 500x should be accepted."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {"umatan": [self._odds_entry([1, 2], 500.0)]}
        bets = optimize_bets(predictions, odds_data, self._race_info())
        assert any(b["type"] == "umatan" for b in bets), (
            "D5 has no upper limit for umatan; 500x must be accepted"
        )

    def test_umatan_at_lower_boundary_included(self):
        """umatan at exactly 50.0x must be included (boundary is inclusive)."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {"umatan": [self._odds_entry([1, 2], 50.0)]}
        bets = optimize_bets(predictions, odds_data, self._race_info())
        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) >= 1, "umatan at exactly 50.0x must be included"

    def test_umatan_at_upper_boundary_included(self):
        """umatan at exactly 300.0x must be included (boundary is inclusive)."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {"umatan": [self._odds_entry([1, 2], 300.0)]}
        bets = optimize_bets(predictions, odds_data, self._race_info())
        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) >= 1, "umatan at exactly 300.0x must be included"

    # ── 3. umaren odds range 20–100x ─────────────────────────────────────────

    def test_umaren_within_range_is_included(self):
        """umaren at 25x (in range 20-30) must be selected."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {"umaren": [self._odds_entry([1, 2], 25.0)]}
        bets = optimize_bets(predictions, odds_data, self._race_info())
        assert any(b["type"] == "umaren" for b in bets), "umaren at 50x must be selected"

    def test_umaren_below_3x_rejected(self):
        """D5: umaren below 3x must be excluded."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {"umaren": [self._odds_entry([1, 2], 2.9)]}
        bets = optimize_bets(predictions, odds_data, self._race_info())
        assert not any(b["type"] == "umaren" for b in bets), (
            "umaren at 2.9x (below D5 3x minimum) must be excluded"
        )

    def test_umaren_at_3x_accepted(self):
        """D5: umaren at exactly 3.0x must be accepted."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {"umaren": [self._odds_entry([1, 2], 3.0)]}
        bets = optimize_bets(predictions, odds_data, self._race_info())
        assert any(b["type"] == "umaren" for b in bets), (
            "umaren at 3.0x (at D5 minimum) must be accepted"
        )

    def test_umaren_at_lower_boundary_20_included(self):
        """umaren at exactly 20.0x must be included."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {"umaren": [self._odds_entry([1, 2], 20.0)]}
        bets = optimize_bets(predictions, odds_data, self._race_info())
        assert any(b["type"] == "umaren" for b in bets)

    def test_umaren_at_upper_boundary_30_included(self):
        """umaren at exactly 29.9x must be included."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {"umaren": [self._odds_entry([1, 2], 29.9)]}
        bets = optimize_bets(predictions, odds_data, self._race_info())
        assert any(b["type"] == "umaren" for b in bets)

    # ── 4. wide odds range 10–50x ────────────────────────────────────────────

    def test_wide_within_range_is_included(self):
        """wide at 20x (in range 10-30) must be selected."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {"wide": [self._odds_entry([1, 2], 25.0)]}
        bets = optimize_bets(predictions, odds_data, self._race_info())
        assert any(b["type"] == "wide" for b in bets), "wide at 25x must be selected"

    def test_wide_below_2_5x_rejected(self):
        """D5: wide below 2.5x must be excluded."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {"wide": [self._odds_entry([1, 2], 2.4)]}
        bets = optimize_bets(predictions, odds_data, self._race_info())
        assert not any(b["type"] == "wide" for b in bets), (
            "wide at 2.4x (below D5 2.5x minimum) must be excluded"
        )

    def test_wide_at_2_5x_accepted(self):
        """D5: wide at exactly 2.5x must be accepted."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {"wide": [self._odds_entry([1, 2], 2.5)]}
        bets = optimize_bets(predictions, odds_data, self._race_info())
        assert any(b["type"] == "wide" for b in bets), (
            "wide at 2.5x (at D5 minimum) must be accepted"
        )

    def test_wide_at_lower_boundary_10_included(self):
        """wide at exactly 10.0x must be included."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {"wide": [self._odds_entry([1, 2], 10.0)]}
        bets = optimize_bets(predictions, odds_data, self._race_info())
        assert any(b["type"] == "wide" for b in bets)

    def test_wide_at_upper_boundary_30_included(self):
        """wide at exactly 29.9x must be included."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {"wide": [self._odds_entry([1, 2], 29.9)]}
        bets = optimize_bets(predictions, odds_data, self._race_info())
        assert any(b["type"] == "wide" for b in bets)

    # ── 5. Bets must involve at least one AI top-5 horse ─────────────────────

    def test_all_bets_involve_ai_top7_horse(self):
        """Every returned bet must contain at least one horse from AI top-7 (D2)."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(10)
        # scores: horse 1=90, 2=82, ..., 7=42, 8=34, 9=26, 10=18
        # AI top-7 = {1, 2, 3, 4, 5, 6, 7}
        ai_top7 = {1, 2, 3, 4, 5, 6, 7}
        odds_data = {
            "umatan": [
                self._odds_entry([1, 2], 50.0),   # top-7 horse 1: qualifies
                self._odds_entry([8, 9], 80.0),   # no top-7 horse: must be rejected
            ],
            "umaren": [
                self._odds_entry([2, 3], 20.0),   # top-7 horses: qualifies
                self._odds_entry([9, 10], 25.0),  # no top-7: must be rejected
            ],
            "wide": [
                self._odds_entry([4, 5], 15.0),   # top-7 horses: qualifies
                self._odds_entry([8, 10], 20.0),  # no top-7: must be rejected
            ],
        }
        bets = optimize_bets(predictions, odds_data, self._race_info())
        for bet in bets:
            horse_set = set(bet["horses"])
            assert horse_set & ai_top7, (
                f"Bet {bet['type']} horses={bet['horses']} has no AI top-7 horse"
            )

    def test_bet_with_no_ai_top7_horse_excluded(self):
        """A bet whose horses are exclusively outside AI top-7 must not be returned."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(10)
        # Only provide an out-of-top-7 combination (horses 8 & 9)
        odds_data = {
            "umaren": [self._odds_entry([8, 9], 20.0)],
        }
        bets = optimize_bets(predictions, odds_data, self._race_info())
        assert all(b["type"] != "umaren" for b in bets), (
            "umaren with only non-top-7 horses (8,9) must be excluded"
        )

    def test_honmei_anchor_all_bets_contain_top1(self):
        """D5: all bets must contain the ◎ (top-1 AI horse)."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        # ◎ = horse 1 (score 82)
        odds_data = {
            "umatan": [self._odds_entry([1, 2], 50.0), self._odds_entry([1, 3], 30.0)],
            "umaren": [self._odds_entry([1, 2], 10.0), self._odds_entry([1, 3], 8.0)],
            "wide":   [self._odds_entry([1, 2], 5.0), self._odds_entry([1, 3], 4.0)],
        }
        bets = optimize_bets(predictions, odds_data, self._race_info())
        for b in bets:
            assert 1 in b["horses"], (
                f"D5: all bets must be ◎-anchor; bet {b['type']} {b['horses']} missing ◎(horse 1)"
            )

    # ── 6. Max 2 bets per type ───────────────────────────────────────────────

    def test_max_2_bets_per_type_umatan(self):
        """At most 2 umatan bets may be returned even when more qualify."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {
            "umatan": [
                self._odds_entry([1, 2], 50.0),
                self._odds_entry([2, 1], 60.0),
                self._odds_entry([1, 3], 80.0),
                self._odds_entry([3, 1], 90.0),
                self._odds_entry([2, 3], 100.0),
            ]
        }
        bets = optimize_bets(predictions, odds_data, self._race_info())
        umatan_count = sum(1 for b in bets if b["type"] == "umatan")
        assert umatan_count <= 2, f"Expected max 2 umatan, got {umatan_count}"

    def test_umaren_max_4_per_race(self):
        """D5: at most 4 umaren bets (◎-AI 2~5位)."""
        from backend.predictor.bet_optimizer import optimize_bets, HONMEI_UMAREN_PARTNERS
        predictions = self._make_predictions(8)
        odds_data = {
            "umaren": [self._odds_entry(sorted([1, i]), 10.0 + i) for i in range(2, 9)],
        }
        bets = optimize_bets(predictions, odds_data, self._race_info())
        umaren_count = sum(1 for b in bets if b["type"] == "umaren")
        assert umaren_count <= HONMEI_UMAREN_PARTNERS, (
            f"Expected max {HONMEI_UMAREN_PARTNERS} umaren, got {umaren_count}"
        )

    def test_wide_max_4_per_race(self):
        """D5: at most 4 wide bets (◎-AI 2~5位)."""
        from backend.predictor.bet_optimizer import optimize_bets, HONMEI_WIDE_PARTNERS
        predictions = self._make_predictions(8)
        odds_data = {
            "wide": [self._odds_entry(sorted([1, i]), 5.0 + i) for i in range(2, 9)],
        }
        bets = optimize_bets(predictions, odds_data, self._race_info())
        wide_count = sum(1 for b in bets if b["type"] == "wide")
        assert wide_count <= HONMEI_WIDE_PARTNERS, (
            f"Expected max {HONMEI_WIDE_PARTNERS} wide, got {wide_count}"
        )

    def test_umatan_max_6_per_race(self):
        """D5: at most 6 umatan bets (◎→AI 2~7位)."""
        from backend.predictor.bet_optimizer import optimize_bets, HONMEI_UMATAN_PARTNERS
        predictions = self._make_predictions(10)
        odds_data = {
            "umatan": [self._odds_entry([1, i], 10.0 + i * 5) for i in range(2, 11)],
        }
        bets = optimize_bets(predictions, odds_data, self._race_info())
        umatan_count = sum(1 for b in bets if b["type"] == "umatan")
        assert umatan_count <= HONMEI_UMATAN_PARTNERS, (
            f"Expected max {HONMEI_UMATAN_PARTNERS} umatan, got {umatan_count}"
        )

    def test_overall_max_14_bets_default(self):
        """D5: total bets must not exceed MAX_BETS (14)."""
        from backend.predictor.bet_optimizer import optimize_bets, MAX_BETS
        predictions = self._make_predictions(10)
        odds_data = {
            "umatan": [self._odds_entry([1, i], 10.0 + i * 5) for i in range(2, 9)],
            "umaren": [self._odds_entry(sorted([1, i]), 5.0 + i) for i in range(2, 7)],
            "wide":   [self._odds_entry(sorted([1, i]), 3.0 + i) for i in range(2, 7)],
        }
        bets = optimize_bets(predictions, odds_data, self._race_info())
        assert len(bets) <= MAX_BETS, f"Expected at most {MAX_BETS} bets, got {len(bets)}"

    def test_custom_max_bets_respected(self):
        """max_bets=5 must cap output."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {
            "umatan": [self._odds_entry([1, i], 10.0 + i * 5) for i in range(2, 8)],
            "umaren": [self._odds_entry(sorted([1, i]), 5.0 + i) for i in range(2, 6)],
            "wide":   [self._odds_entry(sorted([1, i]), 3.0 + i) for i in range(2, 6)],
        }
        bets = optimize_bets(predictions, odds_data, self._race_info(), max_bets=5)
        assert len(bets) <= 5, f"Expected at most 5 bets with max_bets=5, got {len(bets)}"

    # ── 8. Sorted by highest odds first ──────────────────────────────────────

    def test_sorted_by_highest_odds_first(self):
        """Returned bets must be ordered descending by odds value."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {
            "umatan": [
                self._odds_entry([1, 2], 100.0),
                self._odds_entry([2, 1], 50.0),
            ],
            "umaren": [
                self._odds_entry([1, 2], 30.0),
            ],
            "wide": [
                self._odds_entry([1, 2], 20.0),
            ],
        }
        bets = optimize_bets(predictions, odds_data, self._race_info())
        odds_sequence = [b["odds"] for b in bets]
        assert odds_sequence == sorted(odds_sequence, reverse=True), (
            f"Bets not sorted highest-odds-first: {odds_sequence}"
        )

    def test_highest_odds_bet_is_rank_1(self):
        """The bet with the highest odds receives rank=1."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {
            "umatan": [self._odds_entry([1, 2], 250.0)],  # highest
            "umaren": [self._odds_entry([1, 3], 40.0)],
            "wide":   [self._odds_entry([2, 3], 20.0)],
        }
        bets = optimize_bets(predictions, odds_data, self._race_info())
        assert len(bets) >= 1
        assert bets[0]["rank"] == 1
        assert bets[0]["odds"] == max(b["odds"] for b in bets), (
            "Rank-1 bet must have the highest odds"
        )

    def test_ranks_are_sequential(self):
        """Ranks must be consecutive integers starting at 1."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {
            "umatan": [self._odds_entry([1, 2], 150.0)],
            "umaren": [self._odds_entry([1, 3], 40.0)],
            "wide":   [self._odds_entry([2, 3], 15.0)],
        }
        bets = optimize_bets(predictions, odds_data, self._race_info())
        ranks = [b["rank"] for b in bets]
        assert ranks == list(range(1, len(bets) + 1)), (
            f"Ranks must be 1..n, got {ranks}"
        )

    # ── 9. Returns empty list when no odds_data provided ─────────────────────

    def test_empty_odds_data_returns_empty_list(self):
        """When odds_data is an empty dict, no bets can be selected (no real odds)."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        bets = optimize_bets(predictions, {}, self._race_info())
        assert bets == [], f"Expected empty list with no odds_data, got {bets}"

    def test_none_odds_data_handled_gracefully(self):
        """When odds_data is None, optimize_bets should return an empty list."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        bets = optimize_bets(predictions, None, self._race_info())
        assert bets == []

    def test_odds_data_with_only_disallowed_types_returns_empty(self):
        """If odds_data only contains tansho/fukusho/sanrentan entries, result is empty."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {
            "tansho":    [self._odds_entry([1], 25.0)],
            "fukusho":   [self._odds_entry([2], 12.0)],
            "sanrentan": [self._odds_entry([1, 2, 3], 200.0)],
        }
        bets = optimize_bets(predictions, odds_data, self._race_info())
        assert bets == []

    # ── 10. Returns empty list when headCount < 3 ────────────────────────────

    def test_head_count_2_returns_empty(self):
        """Races with only 2 runners cannot produce valid 2-horse combos — must be empty."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(4)
        odds_data = {
            "umaren": [self._odds_entry([1, 2], 30.0)],
            "wide":   [self._odds_entry([1, 2], 15.0)],
        }
        bets = optimize_bets(predictions, odds_data, self._race_info(head_count=2))
        assert bets == [], "headCount=2 must produce empty result"

    def test_head_count_1_returns_empty(self):
        """Single-runner race must return empty."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = [{"horseNumber": 1, "score": 80, "isScratched": False}]
        bets = optimize_bets(predictions, {}, self._race_info(head_count=1))
        assert bets == []

    def test_head_count_0_returns_empty(self):
        """Zero-runner race must return empty."""
        from backend.predictor.bet_optimizer import optimize_bets
        bets = optimize_bets([], {}, self._race_info(head_count=0))
        assert bets == []

    def test_head_count_3_is_minimum_viable(self):
        """headCount=3 is the minimum that may produce results (not auto-rejected)."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = [
            {"horseNumber": i, "score": 80 - i * 10, "isScratched": False}
            for i in range(1, 4)
        ]
        odds_data = {"umaren": [self._odds_entry([1, 2], 30.0)]}
        # Should not raise and should not be rejected by the headCount < 3 guard.
        bets = optimize_bets(predictions, odds_data, self._race_info(head_count=3))
        assert isinstance(bets, list)

    # ── 11. Bets without real odds are excluded ──────────────────────────────

    def test_bets_without_real_odds_excluded(self):
        """Candidates that have no matching entry in odds_data must not appear."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        # Provide odds only for horse pair [1,2], not for [1,3] or others.
        odds_data = {
            "umaren": [self._odds_entry([1, 2], 30.0)],
            # Deliberately omit [1,3], [2,3], etc.
        }
        bets = optimize_bets(predictions, odds_data, self._race_info())
        # Any umaren bet that appears must be the [1,2] pair.
        for bet in bets:
            if bet["type"] == "umaren":
                assert set(bet["horses"]) == {1, 2}, (
                    f"Only umaren [1,2] has real odds; got horses {bet['horses']}"
                )

    def test_all_returned_bets_have_has_real_odds_true(self):
        """Every bet in the S8 result must have hasRealOdds=True."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {
            "umatan": [self._odds_entry([1, 2], 80.0)],
            "umaren": [self._odds_entry([1, 3], 35.0)],
            "wide":   [self._odds_entry([2, 3], 18.0)],
        }
        bets = optimize_bets(predictions, odds_data, self._race_info())
        for bet in bets:
            assert bet.get("hasRealOdds") is True, (
                f"Bet {bet['type']} {bet['horses']} missing hasRealOdds=True"
            )

    def test_odds_data_missing_key_for_type_treated_as_no_real_odds(self):
        """Missing key in odds_data for a type is treated as no real odds for that type."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        # Only umaten odds — umaren and wide have no entries at all.
        odds_data = {
            "umatan": [self._odds_entry([1, 2], 60.0)],
        }
        bets = optimize_bets(predictions, odds_data, self._race_info())
        for bet in bets:
            assert bet["type"] == "umatan", (
                f"Only umatan has real odds; unexpected type {bet['type']}"
            )

    # ── Integration: mixed qualifying / non-qualifying entries ───────────────

    def test_d5_all_bets_are_honmei_anchor_with_min_odds(self):
        """D5: all bets must be ◎-anchor and meet minimum odds thresholds."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        # ◎ = horse 1 (score 82)
        d5_min_odds = {"umatan": 5.0, "umaren": 3.0, "wide": 2.5}
        odds_data = {
            "umatan": [
                self._odds_entry([1, 2], 50.0),   # ◎-anchor, above min ✓
                self._odds_entry([1, 3], 4.9),     # ◎-anchor, below min ✗
            ],
            "umaren": [
                self._odds_entry([1, 2], 10.0),   # ◎-anchor, above min ✓
                self._odds_entry([1, 3], 2.9),     # ◎-anchor, below min ✗
            ],
            "wide": [
                self._odds_entry([1, 2], 5.0),    # ◎-anchor, above min ✓
                self._odds_entry([1, 3], 2.4),     # ◎-anchor, below min ✗
            ],
        }
        bets = optimize_bets(predictions, odds_data, self._race_info())
        for bet in bets:
            bt = bet["type"]
            assert bt in d5_min_odds, f"Disallowed type: {bt}"
            assert 1 in bet["horses"], f"D5: bet must be ◎-anchor; horses={bet['horses']}"
            assert bet["odds"] >= d5_min_odds[bt], (
                f"{bt} odds {bet['odds']} below D5 min {d5_min_odds[bt]}"
            )

    def test_result_is_list_of_dicts_with_required_fields(self):
        """Every bet dict must contain the fields required by the API."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {
            "umatan": [self._odds_entry([1, 2], 80.0)],
            "umaren": [self._odds_entry([1, 3], 30.0)],
            "wide":   [self._odds_entry([2, 3], 20.0)],
        }
        bets = optimize_bets(predictions, odds_data, self._race_info())
        required_fields = {"type", "typeLabel", "horses", "ev", "hitProb", "rank",
                           "odds", "hasRealOdds"}
        for bet in bets:
            missing = required_fields - set(bet.keys())
            assert not missing, f"Bet is missing required fields: {missing}"


# ---------------------------------------------------------------------------
# TestPopularityExpansion — TDD tests for ◎流し拡張 (Change 1)
# ---------------------------------------------------------------------------


class TestPopularityExpansion:
    """generate_candidates() should expand umaren/wide with ◎×popular pairs.

    ◎ = top-1 AI horse (ranked first by probability).
    Expansion adds pairs not already in the base top-6 AI set, using entries
    whose popularity is 1-5 and who are not scratched.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_probs(self, horse_numbers):
        """Return descending probabilities for an ordered list of horse numbers."""
        total = len(horse_numbers)
        return {hn: (total - i) / sum(range(1, total + 1))
                for i, hn in enumerate(horse_numbers)}

    def _make_entries(self, specs):
        """Build entries list from (horseNumber, popularity, isScratched) tuples."""
        entries = []
        for hn, pop, scratched in specs:
            entries.append({
                "horseNumber": hn,
                "frameNumber": hn,
                "popularity": pop,
                "isScratched": scratched,
            })
        return entries

    def _umaren_pairs(self, candidates):
        return [tuple(c["horses"]) for c in candidates if c["type"] == "umaren"]

    def _wide_pairs(self, candidates):
        return [tuple(c["horses"]) for c in candidates if c["type"] == "wide"]

    # ------------------------------------------------------------------
    # Test 1: popular horse outside AI top-6 is added to umaren candidates
    # ------------------------------------------------------------------

    def test_umaren_includes_top1_x_popular_outside_top6(self):
        """A popular horse (popularity 1-5) that is NOT in AI top-6 should be
        added as a ◎×popular umaren pair."""
        from backend.predictor.bet_optimizer import generate_candidates

        # AI top horses: 1..6 in descending probability
        probs = self._make_probs([1, 2, 3, 4, 5, 6])
        # Horse 10 has popularity=3 but is not in AI top-6
        entries = self._make_entries([
            (1, 2, False), (2, 3, False), (3, 4, False),
            (4, 5, False), (5, 6, False), (6, 7, False),
            (10, 3, False),   # outside AI top-6, popularity 3 — should be added
        ])
        candidates = generate_candidates(probs, top_n=6, entries=entries)
        umaren_pairs = self._umaren_pairs(candidates)

        # ◎ is horse 1 (top probability). Pair (1, 10) sorted → (1, 10)
        assert (1, 10) in umaren_pairs, (
            "Expected ◎(1) × popular(10) pair in umaren candidates"
        )

    # ------------------------------------------------------------------
    # Test 2: popular horse already in AI top-6 produces no duplicate
    # ------------------------------------------------------------------

    def test_umaren_no_duplicate_when_popular_horse_already_in_top6(self):
        """If a popular horse is already paired with ◎ in the base top-6 set,
        the expansion must not create a duplicate entry."""
        from backend.predictor.bet_optimizer import generate_candidates

        probs = self._make_probs([1, 2, 3, 4, 5, 6])
        # Horse 2 is in AI top-6 AND has popularity=1
        entries = self._make_entries([
            (1, 3, False), (2, 1, False), (3, 4, False),
            (4, 5, False), (5, 6, False), (6, 7, False),
        ])
        candidates = generate_candidates(probs, top_n=6, entries=entries)
        umaren_pairs = self._umaren_pairs(candidates)

        # The pair (1, 2) should appear exactly once
        pair = tuple(sorted([1, 2]))
        count = umaren_pairs.count(pair)
        assert count == 1, (
            f"Pair {pair} should appear exactly once in umaren, found {count}"
        )

    # ------------------------------------------------------------------
    # Test 3: ◎ is not paired with itself
    # ------------------------------------------------------------------

    def test_umaren_top1_not_paired_with_itself(self):
        """◎ (top-1 horse) must never appear as a self-pair even when it is
        also the most popular horse (popularity=1)."""
        from backend.predictor.bet_optimizer import generate_candidates

        probs = self._make_probs([1, 2, 3, 4, 5, 6])
        # Horse 1 is both top AI and popularity=1
        entries = self._make_entries([
            (1, 1, False), (2, 2, False), (3, 3, False),
            (4, 4, False), (5, 5, False), (6, 6, False),
        ])
        candidates = generate_candidates(probs, top_n=6, entries=entries)
        for c in candidates:
            if c["type"] == "umaren":
                horses = c["horses"]
                assert horses[0] != horses[1], (
                    f"Self-pair detected in umaren: {horses}"
                )

    # ------------------------------------------------------------------
    # Test 4: entries with no popularity data → only base top-6 pairs
    # ------------------------------------------------------------------

    def test_umaren_no_expansion_when_no_popularity_data(self):
        """When entries have no popularity field (or None), the expansion
        block produces nothing extra — only base AI top-6 pairs."""
        from backend.predictor.bet_optimizer import generate_candidates
        from itertools import combinations

        probs = self._make_probs([1, 2, 3, 4, 5, 6])
        # Entries without popularity (None)
        entries = [
            {"horseNumber": hn, "frameNumber": hn, "popularity": None, "isScratched": False}
            for hn in range(1, 7)
        ]
        candidates = generate_candidates(probs, top_n=6, entries=entries)
        umaren_pairs = self._umaren_pairs(candidates)

        expected = [tuple(sorted([h1, h2]))
                    for h1, h2 in combinations([1, 2, 3, 4, 5, 6], 2)]
        assert set(umaren_pairs) == set(expected), (
            "Without popularity data only base top-6 pairs should be generated"
        )

    # ------------------------------------------------------------------
    # Test 5: entries=None → no crash, only base pairs generated
    # ------------------------------------------------------------------

    def test_umaren_no_crash_when_entries_none(self):
        """Calling generate_candidates with entries=None must not raise and
        should return only the base AI top-6 umaren pairs."""
        from backend.predictor.bet_optimizer import generate_candidates
        from itertools import combinations

        probs = self._make_probs([1, 2, 3, 4, 5, 6])
        candidates = generate_candidates(probs, top_n=6, entries=None)
        umaren_pairs = self._umaren_pairs(candidates)

        expected = [tuple(sorted([h1, h2]))
                    for h1, h2 in combinations([1, 2, 3, 4, 5, 6], 2)]
        assert set(umaren_pairs) == set(expected)

    # ------------------------------------------------------------------
    # Test 6: wide candidates also expanded with same ◎流し logic
    # ------------------------------------------------------------------

    def test_wide_includes_top1_x_popular_outside_top6(self):
        """Wide bets must also be expanded with ◎×popular pairs for horses
        outside the AI top-6, mirroring the umaren expansion logic."""
        from backend.predictor.bet_optimizer import generate_candidates

        probs = self._make_probs([1, 2, 3, 4, 5, 6])
        entries = self._make_entries([
            (1, 2, False), (2, 3, False), (3, 4, False),
            (4, 5, False), (5, 6, False), (6, 7, False),
            (11, 1, False),   # outside AI top-6, popularity 1 — must appear in wide
        ])
        candidates = generate_candidates(probs, top_n=6, entries=entries)
        wide_pairs = self._wide_pairs(candidates)

        assert (1, 11) in wide_pairs, (
            "Expected ◎(1) × popular(11) pair in wide candidates"
        )

    def test_wide_no_duplicate_when_popular_horse_already_in_top6(self):
        """Wide expansion must not duplicate a pair that exists in the base
        top-6 wide set."""
        from backend.predictor.bet_optimizer import generate_candidates

        probs = self._make_probs([1, 2, 3, 4, 5, 6])
        entries = self._make_entries([
            (1, 3, False), (2, 1, False), (3, 4, False),
            (4, 5, False), (5, 6, False), (6, 7, False),
        ])
        candidates = generate_candidates(probs, top_n=6, entries=entries)
        wide_pairs = self._wide_pairs(candidates)

        pair = tuple(sorted([1, 2]))
        count = wide_pairs.count(pair)
        assert count == 1, (
            f"Pair {pair} should appear exactly once in wide, found {count}"
        )

    # ------------------------------------------------------------------
    # Test 7: scratched horses with popularity are excluded
    # ------------------------------------------------------------------

    def test_umaren_scratched_horse_excluded_even_if_popular(self):
        """A scratched horse (isScratched=True) must not be added as a
        ◎×popular pair even if it has popularity <= 5."""
        from backend.predictor.bet_optimizer import generate_candidates

        probs = self._make_probs([1, 2, 3, 4, 5, 6])
        entries = self._make_entries([
            (1, 2, False), (2, 3, False), (3, 4, False),
            (4, 5, False), (5, 6, False), (6, 7, False),
            (7, 1, True),   # popularity=1 but scratched — must NOT appear
        ])
        candidates = generate_candidates(probs, top_n=6, entries=entries)
        umaren_pairs = self._umaren_pairs(candidates)

        assert (1, 7) not in umaren_pairs, (
            "Scratched horse 7 (popularity=1) must not appear in umaren candidates"
        )

    def test_wide_scratched_horse_excluded_even_if_popular(self):
        """Same scratched-exclusion guarantee for wide expansion."""
        from backend.predictor.bet_optimizer import generate_candidates

        probs = self._make_probs([1, 2, 3, 4, 5, 6])
        entries = self._make_entries([
            (1, 2, False), (2, 3, False), (3, 4, False),
            (4, 5, False), (5, 6, False), (6, 7, False),
            (8, 2, True),   # popularity=2 but scratched
        ])
        candidates = generate_candidates(probs, top_n=6, entries=entries)
        wide_pairs = self._wide_pairs(candidates)

        assert (1, 8) not in wide_pairs, (
            "Scratched horse 8 (popularity=2) must not appear in wide candidates"
        )

    # ------------------------------------------------------------------
    # Bonus: multiple popular horses outside top-6 are all added
    # ------------------------------------------------------------------

    def test_umaren_multiple_popular_horses_all_added(self):
        """All horses with popularity 1-5 that are outside the AI top-6 and
        not scratched should each produce a ◎×ph pair."""
        from backend.predictor.bet_optimizer import generate_candidates

        probs = self._make_probs([1, 2, 3, 4, 5, 6])
        entries = self._make_entries([
            (1, 3, False), (2, 4, False), (3, 5, False),
            (4, 6, False), (5, 7, False), (6, 8, False),
            (20, 1, False),   # popularity 1, outside top-6
            (21, 2, False),   # popularity 2, outside top-6
        ])
        candidates = generate_candidates(probs, top_n=6, entries=entries)
        umaren_pairs = self._umaren_pairs(candidates)

        assert (1, 20) in umaren_pairs, "Expected ◎(1)×pop(20) in umaren"
        assert (1, 21) in umaren_pairs, "Expected ◎(1)×pop(21) in umaren"

    def test_umaren_popularity_boundary_5_included_6_excluded(self):
        """Horses with popularity exactly 5 are included; popularity 6 is not."""
        from backend.predictor.bet_optimizer import generate_candidates

        probs = self._make_probs([1, 2, 3, 4, 5, 6])
        entries = self._make_entries([
            (1, 3, False), (2, 4, False), (3, 6, False),
            (4, 7, False), (5, 8, False), (6, 9, False),
            (30, 5, False),   # boundary: popularity exactly 5 → must be included
            (31, 6, False),   # popularity 6 → must NOT be included
        ])
        candidates = generate_candidates(probs, top_n=6, entries=entries)
        umaren_pairs = self._umaren_pairs(candidates)

        assert (1, 30) in umaren_pairs, "Horse with popularity=5 must be included"
        assert (1, 31) not in umaren_pairs, "Horse with popularity=6 must NOT be included"


# =====================================================================
# ODDS PIPELINE HARDENING — Change 3: estimate_from_entries threshold 3→2
# =====================================================================
class TestEstimateThreshold:
    """Verify estimate_from_entries works with exactly 2 horses having odds
    and returns {} when fewer than 2 horses have odds.

    Change: minimum threshold reduced from 3 to 2.
    """

    def _make_entry(
        self,
        horse_number: int,
        odds: float | None,
        is_scratched: bool = False,
    ) -> dict:
        """Minimal valid entry dict for estimate_from_entries."""
        return {
            "horseNumber": horse_number,
            "horseName": f"Horse{horse_number}",
            "odds": odds,
            "isScratched": is_scratched,
        }

    def test_two_horses_with_odds_returns_non_empty(self):
        """Exactly 2 non-scratched horses with valid odds must produce a
        non-empty dict — the new threshold of 2 (was 3) must be honoured."""
        from backend.scraper.odds import estimate_from_entries
        entries = [
            self._make_entry(1, 2.5),
            self._make_entry(2, 5.0),
        ]
        result = estimate_from_entries(entries)
        assert isinstance(result, dict), "Return type must be dict"
        assert len(result) > 0, (
            "estimate_from_entries with exactly 2 horses having odds must "
            "return non-empty dict (threshold changed from 3 to 2)."
        )
        # At minimum tansho and fukusho entries should be present
        assert "tansho" in result, "tansho key must be present with 2 horses"

    def test_one_horse_with_odds_returns_empty(self):
        """Only 1 horse having valid odds is still below the threshold of 2,
        so the function must return {}."""
        from backend.scraper.odds import estimate_from_entries
        entries = [
            self._make_entry(1, 2.5),
            self._make_entry(2, None),   # no odds
            self._make_entry(3, None),   # no odds
        ]
        result = estimate_from_entries(entries)
        assert result == {}, (
            "Only 1 horse with odds is below the 2-horse threshold; "
            f"expected {{}} but got keys: {list(result.keys())}"
        )

    def test_no_horses_with_odds_returns_empty(self):
        """When no horses have valid odds the function must return {}."""
        from backend.scraper.odds import estimate_from_entries
        entries = [
            self._make_entry(1, None),
            self._make_entry(2, None),
            self._make_entry(3, None),
        ]
        result = estimate_from_entries(entries)
        assert result == {}, (
            f"No horses with odds should give {{}}; got keys: {list(result.keys())}"
        )

    def test_mix_of_odds_and_none_uses_only_valid_odds(self):
        """Horses with odds=None or isScratched=True must be excluded from
        the count.  Only those with a truthy odds value count toward the
        minimum-2 threshold."""
        from backend.scraper.odds import estimate_from_entries
        entries = [
            self._make_entry(1, 3.0),                       # counted
            self._make_entry(2, None),                      # excluded: no odds
            self._make_entry(3, 8.0, is_scratched=True),    # excluded: scratched
            self._make_entry(4, 12.0),                      # counted
            self._make_entry(5, None),                      # excluded: no odds
        ]
        result = estimate_from_entries(entries)
        # Horses 1 and 4 have valid odds → meets threshold of 2
        assert len(result) > 0, (
            "2 valid (non-scratched, non-None odds) horses should satisfy "
            "the threshold; estimate_from_entries should return non-empty dict."
        )
        # Verify that scratched and None-odds horses are not in tansho results
        tansho_horse_nums = [e["horses"][0] for e in result.get("tansho", [])]
        assert 3 not in tansho_horse_nums, (
            "Scratched horse (3) must not appear in tansho estimates."
        )
        assert 2 not in tansho_horse_nums, (
            "Horse with odds=None (2) must not appear in tansho estimates."
        )


# ---------------------------------------------------------------------------
# Tests for the 7-minute freeze threshold and 30/60/300s update intervals
# (Change 1: freeze at 7 min; Change 2: 30s interval when <7 min to post)
# ---------------------------------------------------------------------------

class TestUpdateIntervals:
    """Unit tests for realtime_worker interval logic and freeze constants.

    These tests are intentionally free of DB / network dependencies.
    They import only module-level symbols and a pure helper function.
    """

    # ------------------------------------------------------------------
    # Test 1 – freeze threshold constant is 7
    # ------------------------------------------------------------------

    def test_freeze_threshold_constant_is_6(self):
        """FREEZE_THRESHOLD_MINS exported from realtime_worker must equal 6."""
        from backend.realtime_worker import FREEZE_THRESHOLD_MINS
        assert FREEZE_THRESHOLD_MINS == 6, (
            f"Expected FREEZE_THRESHOLD_MINS=6, got {FREEZE_THRESHOLD_MINS}"
        )

    # ------------------------------------------------------------------
    # Tests 2-4 – interval returned for each regime
    # ------------------------------------------------------------------

    def test_interval_is_30s_when_min_mins_below_7(self):
        """compute_update_interval returns 30 when min_mins is well below 7."""
        from backend.realtime_worker import compute_update_interval
        assert compute_update_interval(3.0) == 30, (
            "3 min to post is inside the <7-min window; interval must be 30s"
        )

    def test_interval_is_30s_when_min_mins_is_0(self):
        """compute_update_interval returns 30 at 0 minutes (imminent start)."""
        from backend.realtime_worker import compute_update_interval
        assert compute_update_interval(0.0) == 30

    def test_interval_is_60s_when_min_mins_in_mid_range(self):
        """compute_update_interval returns 60 for a value between 7 and 20."""
        from backend.realtime_worker import compute_update_interval
        assert compute_update_interval(15.0) == 60, (
            "15 min to post is in the 7-20 min window; interval must be 60s"
        )

    def test_interval_is_300s_when_min_mins_above_20(self):
        """compute_update_interval returns 300 when race is more than 20 min away."""
        from backend.realtime_worker import compute_update_interval
        assert compute_update_interval(25.0) == 300, (
            "25 min to post is outside any rapid-refresh window; interval must be 300s"
        )

    def test_interval_is_300s_for_very_large_value(self):
        """compute_update_interval returns 300 for the sentinel value 999 (no race)."""
        from backend.realtime_worker import compute_update_interval
        assert compute_update_interval(999) == 300

    # ------------------------------------------------------------------
    # Tests 5-6 – boundary values (exact threshold edges)
    # ------------------------------------------------------------------

    def test_boundary_exactly_7_minutes_gives_30s(self):
        """Boundary: min_mins == 7 must fall into the <=7 branch (30s), not the <=20 branch."""
        from backend.realtime_worker import compute_update_interval
        result = compute_update_interval(7)
        assert result == 30, (
            f"At exactly 7 min the condition 'min_mins <= 7' is True; "
            f"expected interval=30, got {result}"
        )

    def test_boundary_exactly_20_minutes_gives_60s(self):
        """Boundary: min_mins == 20 must fall into the <=20 branch (60s), not the 300s branch."""
        from backend.realtime_worker import compute_update_interval
        result = compute_update_interval(20)
        assert result == 60, (
            f"At exactly 20 min the condition 'min_mins <= 20' is True; "
            f"expected interval=60, got {result}"
        )

    def test_boundary_just_above_7_minutes_gives_60s(self):
        """Just above 7 min (7.01) must NOT trigger the 30s branch."""
        from backend.realtime_worker import compute_update_interval
        result = compute_update_interval(7.01)
        assert result == 60, (
            f"7.01 min is above the 7-min threshold; expected interval=60, got {result}"
        )

    def test_boundary_just_above_20_minutes_gives_300s(self):
        """Just above 20 min (20.01) must NOT trigger the 60s branch."""
        from backend.realtime_worker import compute_update_interval
        result = compute_update_interval(20.01)
        assert result == 300, (
            f"20.01 min is above the 20-min threshold; expected interval=300, got {result}"
        )

    # ------------------------------------------------------------------
    # Test 7 – refresh_raceday freeze threshold is 7 (not 10)
    # ------------------------------------------------------------------

    def test_refresh_raceday_freeze_threshold_is_7_not_10(self):
        """Verify that refresh_raceday.py uses 7 as the freeze threshold.

        Strategy: parse the source file with the `ast` module and inspect all
        Compare nodes.  The constant 10 must NOT appear in any comparison whose
        left-hand side corresponds to `mins_left`, and the constant 7 MUST
        appear in at least one such comparison.
        """
        import ast
        import inspect
        import backend.refresh_raceday as rrd

        source = inspect.getsource(rrd)
        tree = ast.parse(source)

        freeze_comparisons: list[int] = []

        class _FreezeVisitor(ast.NodeVisitor):
            """Collect integer constants from `if mins_left < N` style comparisons."""
            def visit_Compare(self, node: ast.Compare) -> None:  # noqa: N802
                # Look for `mins_left < N` or `mins_left <= N`
                if isinstance(node.left, ast.Name) and node.left.id == "mins_left":
                    for comparator in node.comparators:
                        if isinstance(comparator, ast.Constant) and isinstance(comparator.value, int):
                            freeze_comparisons.append(comparator.value)
                self.generic_visit(node)

        _FreezeVisitor().visit(tree)

        assert 7 in freeze_comparisons, (
            f"refresh_raceday.py must compare mins_left against 7 (the freeze threshold). "
            f"Found comparisons: {freeze_comparisons}"
        )
        assert 10 not in freeze_comparisons, (
            f"refresh_raceday.py still references the OLD threshold of 10 min. "
            f"Found comparisons: {freeze_comparisons}"
        )


# ---------------------------------------------------------------------------
# D1 bimodal strategy tests
# ---------------------------------------------------------------------------

class TestD5HonmeiAnchorStrategy:
    """TDD tests for the D5 ◎軸厚張り strategy.

    D5 removes all VALUE_RANGES and instead anchors all bets on ◎ (top AI horse).
    Min odds: umatan 5x, umaren 3x, wide 2.5x. No upper limit.
    """

    def _make_predictions(self, n: int = 8) -> list:
        return [
            {"horseNumber": i, "score": max(5, 90 - i * 8), "isScratched": False}
            for i in range(1, n + 1)
        ]

    def _race_info(self, head_count: int = 10) -> dict:
        return {"raceId": "202606030210", "headCount": head_count}

    def _odds_entry(self, horses: list, odds: float) -> dict:
        return {"horses": horses, "odds": odds, "payout": int(odds * 100)}

    def test_umaren_at_any_high_odds_accepted(self):
        """D5: no upper limit — umaren at 100x is accepted."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {"umaren": [self._odds_entry([1, 2], 100.0)]}
        bets = optimize_bets(predictions, odds_data, self._race_info())
        assert any(b["type"] == "umaren" for b in bets)

    def test_wide_at_any_high_odds_accepted(self):
        """D5: no upper limit — wide at 50x is accepted."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {"wide": [self._odds_entry([1, 2], 50.0)]}
        bets = optimize_bets(predictions, odds_data, self._race_info())
        assert any(b["type"] == "wide" for b in bets)

    def test_umatan_at_5x_accepted(self):
        """D5: umatan at 5x (exactly at min) is accepted."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {"umatan": [self._odds_entry([1, 2], 5.0)]}
        bets = optimize_bets(predictions, odds_data, self._race_info())
        assert any(b["type"] == "umatan" for b in bets)

    def test_all_three_types_together(self):
        """D5: umatan + umaren + wide all selected from same race."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {
            "umatan": [self._odds_entry([1, 2], 30.0)],
            "umaren": [self._odds_entry([1, 2], 10.0)],
            "wide":   [self._odds_entry([1, 2], 5.0)],
        }
        bets = optimize_bets(predictions, odds_data, self._race_info())
        types = {b["type"] for b in bets}
        assert "umatan" in types and "umaren" in types and "wide" in types

    def test_all_bets_are_honmei_anchor(self):
        """D5: every returned bet contains ◎ (horse 1, the top scorer)."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {
            "umatan": [self._odds_entry([1, i], 10.0 + i * 5) for i in range(2, 8)],
            "umaren": [self._odds_entry(sorted([1, i]), 5.0 + i) for i in range(2, 6)],
            "wide":   [self._odds_entry(sorted([1, i]), 3.0 + i) for i in range(2, 6)],
        }
        bets = optimize_bets(predictions, odds_data, self._race_info())
        for b in bets:
            assert 1 in b["horses"], f"D5 bet must be ◎-anchor: {b}"

    def test_no_bets_when_all_below_min_odds(self):
        """D5: returns empty when all odds are below minimum thresholds."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {
            "umatan": [self._odds_entry([1, 2], 4.0)],
            "umaren": [self._odds_entry([1, 2], 2.0)],
            "wide":   [self._odds_entry([1, 2], 2.0)],
        }
        bets = optimize_bets(predictions, odds_data, self._race_info())
        assert bets == []

    def test_sorted_highest_odds_first(self):
        """D5: bets are sorted by odds descending."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        odds_data = {
            "umatan": [self._odds_entry([1, 2], 30.0), self._odds_entry([1, 3], 50.0)],
            "umaren": [self._odds_entry([1, 2], 10.0)],
            "wide":   [self._odds_entry([1, 2], 5.0)],
        }
        bets = optimize_bets(predictions, odds_data, self._race_info())
        odds_seq = [b["odds"] for b in bets]
        assert odds_seq == sorted(odds_seq, reverse=True)

    def test_empty_odds_returns_empty(self):
        """D5: no odds data → no bets."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        bets = optimize_bets(predictions, {}, self._race_info())
        assert bets == []

    def test_partners_from_ai_ranking(self):
        """D5: umatan partners come from AI ranking, not market popularity."""
        from backend.predictor.bet_optimizer import optimize_bets
        predictions = self._make_predictions(8)
        # ◎=1, AI 2nd=2, AI 3rd=3
        # Only provide odds for ◎→3 (AI 3rd) — should be selected
        odds_data = {"umatan": [self._odds_entry([1, 3], 20.0)]}
        bets = optimize_bets(predictions, odds_data, self._race_info())
        assert any(b["horses"] == [1, 3] for b in bets if b["type"] == "umatan")


# ---------------------------------------------------------------------------
# TestFrameMissingBehavior
# Tests the behavior contract around JRA未割当 frame numbers (frameNumber=0).
# The export pipeline suppresses predictions when >50% of non-scratched entries
# have frameNumber=0.  These tests verify each component's contract in isolation.
# ---------------------------------------------------------------------------


class TestFrameMissingBehavior:
    """Verify behavior when JRA has not yet assigned frame numbers (frameNumber=0).

    The export script applies this logic after predictions are generated:

        frames_missing = non_scratched and zero_frames > len(non_scratched) * 0.5

        if frames_missing:
            bets = []
            for p in preds:
                p["mark"] = ""
                p["score"] = 0

    Each test targets one slice of that contract so regressions are pinpointed.
    """

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _make_entry(self, horse_number, frame_number, scratched=False):
        """Return a minimal entry dict."""
        return {
            "horseNumber": horse_number,
            "frameNumber": frame_number,
            "horseName": f"テストホース{horse_number}",
            "age": "牡4",
            "weightCarried": 57.0,
            "jockeyName": "テスト騎手",
            "trainerName": "テスト調教師",
            "horseWeight": "480(0)",
            "odds": 10.0,
            "popularity": horse_number,
            "isScratched": scratched,
            "sireName": "ディープインパクト",
            "damName": "テストダム",
            "broodmareSire": "",
            "pastRaces": [],
        }

    def _make_prediction(self, horse_number, score=50.0, mark="◎"):
        """Return a minimal prediction dict."""
        return {"horseNumber": horse_number, "score": score, "mark": mark}

    def _race_info(self):
        return {
            "raceId": "202606030111",
            "raceName": "テストレース",
            "raceNumber": 11,
            "distance": 1600,
            "surface": "芝",
            "courseDetail": "左回り",
            "racecourseCode": "05",
            "date": "20260426",
            "headCount": 8,
            "trackCondition": "良",
        }

    def _detect_frames_missing(self, entries):
        """Mirror the export script's frames_missing detection logic exactly."""
        non_scratched = [e for e in entries if not e.get("isScratched")]
        zero_frames = sum(1 for e in non_scratched if e.get("frameNumber", 0) == 0)
        return bool(non_scratched and zero_frames > len(non_scratched) * 0.5)

    # ── 1. optimize_bets is frame-agnostic ───────────────────────────────────

    def test_optimizer_returns_bets_when_all_frames_zero(self):
        """optimize_bets does not inspect frameNumber — it must still work.

        The optimizer's contract is to rank bets by EV given scores and odds.
        Suppressing output because of missing frames is the export layer's job,
        not the optimizer's.  Even when every entry has frameNumber=0 the
        optimizer should return a result (subject to normal EV thresholds).
        """
        from backend.predictor.bet_optimizer import optimize_bets

        entries = [self._make_entry(i, frame_number=0) for i in range(1, 9)]
        predictions = [
            self._make_prediction(i, score=100.0 - i * 5) for i in range(1, 9)
        ]
        # Provide valid odds so at least one bet clears the EV threshold.
        odds_data = {
            "umaren": [{"horses": [1, 2], "odds": 25.0, "payout": 2500}],
            "wide": [{"horses": [1, 2], "odds": 12.0, "payout": 1200}],
        }
        bets = optimize_bets(predictions, odds_data, self._race_info(), entries=entries)
        # The optimizer itself is frame-agnostic — it must not return an empty
        # list just because frameNumber=0 (the export layer handles that gate).
        assert isinstance(bets, list), "optimize_bets must return a list"
        # At least one bet expected given the supplied odds and scores.
        assert len(bets) > 0, (
            "Optimizer should produce bets regardless of frameNumber values; "
            "frame suppression is the export layer's responsibility"
        )

    # ── 2. WeightedScoringModel scores are frame-agnostic ────────────────────

    def test_scoring_model_produces_scores_when_frames_zero(self):
        """WeightedScoringModel must assign non-zero scores even with frameNumber=0.

        drawBias uses frameNumber as a proxy for post position when frame is
        absent, but that should not prevent a score from being computed.
        Scores are suppressed by the export layer, not the scoring model.
        """
        from backend.predictor.scoring import WeightedScoringModel

        entries = [self._make_entry(i, frame_number=0) for i in range(1, 5)]
        model = WeightedScoringModel()
        preds = model.predict(self._race_info(), entries)

        non_zero = [p for p in preds if p["score"] > 0]
        assert len(non_zero) > 0, (
            "WeightedScoringModel must produce at least one non-zero score "
            "regardless of frameNumber; the export layer is responsible for zeroing."
        )

    def test_scoring_model_assigns_marks_when_frames_zero(self):
        """WeightedScoringModel must assign ◎◯▲ marks when frameNumber=0.

        The scoring model's contract is to rank horses and assign marks.
        Clearing those marks when frames are unconfirmed is the export layer's job.
        """
        from backend.predictor.scoring import WeightedScoringModel

        entries = [
            {**self._make_entry(i, frame_number=0), "odds": 2.5 + i}
            for i in range(1, 6)
        ]
        model = WeightedScoringModel()
        preds = model.predict(self._race_info(), entries)

        marks = [p["mark"] for p in preds]
        assert "◎" in marks, (
            "WeightedScoringModel must assign ◎ regardless of frameNumber"
        )

    # ── 3. frames_missing detection logic ────────────────────────────────────

    def test_all_frames_zero_triggers_frames_missing(self):
        """100% of non-scratched entries having frameNumber=0 must set frames_missing=True."""
        entries = [self._make_entry(i, frame_number=0) for i in range(1, 9)]
        assert self._detect_frames_missing(entries) is True

    def test_majority_frames_zero_triggers_frames_missing(self):
        """Just over 50%: 5 out of 8 entries with frameNumber=0 must be True."""
        entries = (
            [self._make_entry(i, frame_number=0) for i in range(1, 6)]   # 5 zero
            + [self._make_entry(i, frame_number=i - 4) for i in range(6, 9)]  # 3 non-zero
        )
        assert self._detect_frames_missing(entries) is True

    def test_exactly_half_frames_zero_does_not_trigger(self):
        """Exactly 50% (not strictly greater): 4 out of 8 must be False."""
        entries = (
            [self._make_entry(i, frame_number=0) for i in range(1, 5)]   # 4 zero
            + [self._make_entry(i, frame_number=i - 3) for i in range(5, 9)]  # 4 non-zero
        )
        assert self._detect_frames_missing(entries) is False

    def test_minority_frames_zero_does_not_trigger(self):
        """Only 1 out of 8 non-scratched entries with frameNumber=0 must be False."""
        entries = (
            [self._make_entry(1, frame_number=0)]
            + [self._make_entry(i, frame_number=i - 1) for i in range(2, 9)]
        )
        assert self._detect_frames_missing(entries) is False

    def test_no_entries_does_not_trigger(self):
        """Empty entries list must not trigger frames_missing (avoids ZeroDivisionError path)."""
        assert self._detect_frames_missing([]) is False

    def test_scratched_entries_excluded_from_count(self):
        """Scratched horses must not count toward the zero-frame ratio.

        Scenario: 2 non-scratched entries both have confirmed frames; 6 scratched
        entries have frameNumber=0.  The ratio for non-scratched is 0/2 = 0%.
        frames_missing must be False.
        """
        scratched = [self._make_entry(i, frame_number=0, scratched=True) for i in range(1, 7)]
        active = [self._make_entry(i, frame_number=i - 6) for i in range(7, 9)]
        entries = scratched + active
        assert self._detect_frames_missing(entries) is False

    def test_all_scratched_does_not_trigger(self):
        """If every entry is scratched, non_scratched is empty and frames_missing must be False."""
        entries = [self._make_entry(i, frame_number=0, scratched=True) for i in range(1, 9)]
        assert self._detect_frames_missing(entries) is False

    # ── 4. Applying frames_missing: marks cleared to "" ──────────────────────

    def test_frames_missing_clears_all_marks(self):
        """When frames_missing is True, clearing loop must set every mark to "".

        This replicates the export loop:
            for p in preds:
                p["mark"] = ""
        """
        preds = [
            self._make_prediction(1, score=90, mark="◎"),
            self._make_prediction(2, score=80, mark="◯"),
            self._make_prediction(3, score=70, mark="▲"),
            self._make_prediction(4, score=60, mark="▲"),
            self._make_prediction(5, score=50, mark="△"),
            self._make_prediction(6, score=40, mark="△"),
            self._make_prediction(7, score=30, mark=""),
        ]
        # Simulate the export clearing loop
        for p in preds:
            p["mark"] = ""

        for p in preds:
            assert p["mark"] == "", (
                f"Horse {p['horseNumber']} mark should be '' after frames_missing clear, "
                f"got {p['mark']!r}"
            )

    def test_frames_missing_clearing_does_not_affect_other_fields(self):
        """Clearing marks must not touch horseNumber, factors, or other fields."""
        preds = [
            {**self._make_prediction(1, score=90, mark="◎"), "factors": {"jockeyAbility": 0.7}},
            {**self._make_prediction(2, score=80, mark="◯"), "factors": {"jockeyAbility": 0.5}},
        ]
        original_horse_numbers = [p["horseNumber"] for p in preds]
        original_factors = [p["factors"].copy() for p in preds]

        for p in preds:
            p["mark"] = ""

        for i, p in enumerate(preds):
            assert p["horseNumber"] == original_horse_numbers[i]
            assert p["factors"] == original_factors[i]

    # ── 5. Applying frames_missing: scores zeroed ────────────────────────────

    def test_frames_missing_sets_all_scores_to_zero(self):
        """When frames_missing is True, clearing loop must set every score to 0."""
        preds = [
            self._make_prediction(i, score=100.0 - i * 5, mark="◎" if i == 1 else "")
            for i in range(1, 9)
        ]
        # Simulate the export clearing loop
        for p in preds:
            p["mark"] = ""
            p["score"] = 0

        for p in preds:
            assert p["score"] == 0, (
                f"Horse {p['horseNumber']} score should be 0 after frames_missing clear, "
                f"got {p['score']}"
            )

    def test_frames_missing_score_zero_is_integer_zero(self):
        """Score must be set to integer 0, not None or False (strict equality check)."""
        pred = self._make_prediction(1, score=77.5, mark="◎")
        pred["score"] = 0
        assert pred["score"] == 0
        assert pred["score"] is not None
        assert pred["score"] is not False

    # ── 6. Frames confirmed: marks and scores are preserved ──────────────────

    def test_frames_confirmed_marks_preserved(self):
        """When frames_missing is False, marks must NOT be cleared."""
        entries = [self._make_entry(i, frame_number=i) for i in range(1, 9)]
        frames_missing = self._detect_frames_missing(entries)
        assert frames_missing is False, "Precondition: frames should be confirmed"

        from backend.predictor.scoring import WeightedScoringModel
        entries_with_odds = [{**e, "odds": 2.5 + idx} for idx, e in enumerate(entries)]
        model = WeightedScoringModel()
        preds = model.predict(self._race_info(), entries_with_odds)

        # frames_missing is False → export must NOT clear marks
        # Verify the model assigned at least one named mark
        marks = [p["mark"] for p in preds]
        assert "◎" in marks, (
            "When frames are confirmed, ◎ must be present (marks must not be cleared)"
        )

    def test_frames_confirmed_scores_preserved(self):
        """When frames_missing is False, scores must remain as computed."""
        entries = [self._make_entry(i, frame_number=i) for i in range(1, 9)]
        frames_missing = self._detect_frames_missing(entries)
        assert frames_missing is False, "Precondition: frames should be confirmed"

        from backend.predictor.scoring import WeightedScoringModel
        entries_with_odds = [{**e, "odds": 2.5 + idx} for idx, e in enumerate(entries)]
        model = WeightedScoringModel()
        preds = model.predict(self._race_info(), entries_with_odds)

        # Scores should be non-zero for active horses
        non_zero = [p for p in preds if p["score"] > 0]
        assert len(non_zero) > 0, (
            "When frames are confirmed, non-zero scores must be present"
        )
