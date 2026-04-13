"""TDD tests for the bet optimizer.

Covers: softmax probability, MC simulation, hit probability estimation,
EV calculation, candidate generation, race pattern detection, diversification.
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
        candidates = generate_candidates(probs, top_n=5)
        types = {c["type"] for c in candidates}
        assert "tansho" in types
        assert "fukusho" in types
        assert "umaren" in types
        assert "wide" in types
        assert "sanrenpuku" in types
        assert "sanrentan" in types

    def test_tansho_count(self):
        from backend.predictor.bet_optimizer import generate_candidates
        probs = {1: 0.3, 2: 0.25, 3: 0.2, 4: 0.15, 5: 0.1}
        candidates = generate_candidates(probs, top_n=5)
        tansho = [c for c in candidates if c["type"] == "tansho"]
        assert len(tansho) == 3  # top 3

    def test_ordered_flag(self):
        from backend.predictor.bet_optimizer import generate_candidates
        probs = {1: 0.5, 2: 0.3, 3: 0.2}
        candidates = generate_candidates(probs, top_n=3)
        for c in candidates:
            if c["type"] == "sanrentan":
                assert c["ordered"] is True
            elif c["type"] == "tansho":
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

    def test_bets_sorted_by_ev(self, sample_predictions, sample_odds_data, sample_race_info):
        from backend.predictor.bet_optimizer import optimize_bets
        bets = optimize_bets(sample_predictions, sample_odds_data, sample_race_info)
        if len(bets) >= 2:
            for i in range(len(bets) - 1):
                assert bets[i]["ev"] >= bets[i + 1]["ev"]

    def test_diversification_max_2_per_type(self, sample_predictions, sample_odds_data, sample_race_info):
        from backend.predictor.bet_optimizer import optimize_bets
        bets = optimize_bets(sample_predictions, sample_odds_data, sample_race_info)
        from collections import Counter
        type_counts = Counter(b["type"] for b in bets)
        for count in type_counts.values():
            assert count <= 2

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
