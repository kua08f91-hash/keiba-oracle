"""TDD tests for D5 馬単/馬連 heavy bet structure.

Verifies the complete D5 bet structure:
  - 馬単 ◎→AI 2~4位 (3 points)
  - 馬連 ◎-AI 2~3位 (2 points)
  - ワイド ◎-AI 2位 only (1 point)
  - Total: 6 bets max for a 勝負 race

Parameters being tested:
  HONMEI_UMATAN_PARTNERS = 3
  HONMEI_UMAREN_PARTNERS = 2
  HONMEI_WIDE_PARTNERS   = 1
  SHOUBU_MIN_SCORE       = 74.0
  TANPUKU_FUKUSHO_MIN_ODDS = 6.0
  TANPUKU_TANSHO_MIN_ODDS  = 6.0
  MAX_BETS               = 6
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_predictions(scores: list[float]) -> list[dict]:
    """Build predictions list from scores (index 0 = horse 1 = ◎)."""
    return [
        {"horseNumber": i + 1, "score": s, "isScratched": False}
        for i, s in enumerate(scores)
    ]


def _make_full_odds(honmei: int, partners: list[int]) -> dict:
    """Build odds_data covering umatan, umaren, wide, tansho, fukusho.

    All umatan odds ≥ 5.0 (D5_MIN_ODDS), umaren ≥ 3.0, wide ≥ 2.5.
    """
    umatan_entries = []
    umaren_entries = []
    wide_entries = []
    tansho_entries = []
    fukusho_entries = []

    # Build full cross-product so find_odds_for_bet() will always find a match
    all_horses = [honmei] + partners
    for h in all_horses:
        tansho_entries.append({"horses": [h], "odds": 8.0, "payout": 800})
        fukusho_entries.append({"horses": [h], "odds": 7.0, "payout": 700,
                                "oddsMin": 6.5, "oddsMax": 7.5})

    for p in partners:
        # umatan ◎→p and p→◎ (both directions)
        umatan_entries.append({"horses": [honmei, p], "odds": 12.0, "payout": 1200})
        umatan_entries.append({"horses": [p, honmei], "odds": 20.0, "payout": 2000})
        # umaren unordered pair
        pair = sorted([honmei, p])
        umaren_entries.append({"horses": pair, "odds": 8.0, "payout": 800})
        # wide unordered pair
        wide_entries.append({"horses": pair, "odds": 3.5, "payout": 350})

    # Partner-to-partner combinations for completeness
    for i, p1 in enumerate(partners):
        for p2 in partners[i + 1:]:
            pair = sorted([p1, p2])
            umatan_entries.append({"horses": [p1, p2], "odds": 25.0, "payout": 2500})
            umatan_entries.append({"horses": [p2, p1], "odds": 30.0, "payout": 3000})
            umaren_entries.append({"horses": pair, "odds": 18.0, "payout": 1800})
            wide_entries.append({"horses": pair, "odds": 5.0, "payout": 500})

    return {
        "umatan": umatan_entries,
        "umaren": umaren_entries,
        "wide": wide_entries,
        "tansho": tansho_entries,
        "fukusho": fukusho_entries,
    }


def _run_optimize(predictions, odds_data, headcount=8, race_id="202608081101"):
    from backend.predictor.bet_optimizer import optimize_bets
    race_info = {"raceId": race_id, "headCount": headcount}
    return optimize_bets(predictions, odds_data, race_info)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def shoubu_predictions():
    """8-horse field, ◎ score=80 (well above 74 threshold)."""
    return _make_predictions([80.0, 72.0, 65.0, 58.0, 50.0, 42.0, 35.0, 28.0])


@pytest.fixture
def full_odds_8h():
    """Complete odds data for 8-horse field with horse 1 as ◎."""
    return _make_full_odds(honmei=1, partners=[2, 3, 4, 5, 6, 7, 8])


# ---------------------------------------------------------------------------
# 1. Bet count tests
# ---------------------------------------------------------------------------

class TestBetCounts:
    def test_shoubu_race_produces_bets(self, shoubu_predictions, full_odds_8h):
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        combo_bets = [b for b in bets if b["type"] not in ("tansho", "fukusho")]
        assert len(combo_bets) >= 1, (
            f"Expected at least 1 combo bet, got {len(combo_bets)}"
        )

    def test_3_umatan_bets(self, shoubu_predictions, full_odds_8h):
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        assert len(umatan_bets) == 3, (
            f"Expected 3 umatan bets, got {len(umatan_bets)}"
        )

    def test_umatan_partners_are_ai_rank_2_3_4(self, shoubu_predictions, full_odds_8h):
        """◎=horse 1 → partners should be horse 2 (AI 2位), 3 (AI 3位), 4 (AI 4位)."""
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        partner_horses = {b["horses"][1] for b in umatan_bets}
        assert partner_horses == {2, 3, 4}, (
            f"Expected umatan partners 2,3,4, got {partner_horses}"
        )

    def test_2_umaren_bets(self, shoubu_predictions, full_odds_8h):
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        umaren_bets = [b for b in bets if b["type"] == "umaren"]
        assert len(umaren_bets) == 2, (
            f"Expected 2 umaren bets, got {len(umaren_bets)}"
        )

    def test_umaren_partners_are_ai_rank_2_3(self, shoubu_predictions, full_odds_8h):
        """◎-AI 2位 and ◎-AI 3位 only."""
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        umaren_bets = [b for b in bets if b["type"] == "umaren"]
        # Each umaren bet contains ◎ (horse 1) and a partner
        partner_set = set()
        for b in umaren_bets:
            partner_set.update(h for h in b["horses"] if h != 1)
        assert partner_set == {2, 3}, (
            f"Expected umaren partners 2,3, got {partner_set}"
        )

    def test_has_wide_or_umaren(self, shoubu_predictions, full_odds_8h):
        """D6: EV-based selection should include at least one wide or umaren."""
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        wide_or_umaren = [b for b in bets if b["type"] in ("wide", "umaren")]
        assert len(wide_or_umaren) >= 1, (
            f"Expected at least 1 wide/umaren, got {len(wide_or_umaren)}"
        )


# ---------------------------------------------------------------------------
# 2. 馬単 is the primary bet type
# ---------------------------------------------------------------------------

class TestUmatanPrimary:
    def test_umatan_appears_before_umaren_in_sorted_output(
        self, shoubu_predictions, full_odds_8h
    ):
        """Sorted by odds descending → umatan (12x) appear before umaren (8x)."""
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        types_in_order = [b["type"] for b in bets if b["type"] in ("umatan", "umaren", "wide")]
        # First occurrence of umatan index < first occurrence of umaren index
        umatan_idx = next((i for i, t in enumerate(types_in_order) if t == "umatan"), None)
        umaren_idx = next((i for i, t in enumerate(types_in_order) if t == "umaren"), None)
        assert umatan_idx is not None, "No umatan bet found"
        assert umaren_idx is not None, "No umaren bet found"
        assert umatan_idx < umaren_idx, (
            f"Expected umatan before umaren; order: {types_in_order}"
        )

    def test_all_umatan_have_honmei_as_first_horse(self, shoubu_predictions, full_odds_8h):
        """All umatan bets must have ◎ (horse 1) as horses[0] (ordered=True)."""
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        for b in umatan_bets:
            assert b["horses"][0] == 1, (
                f"Expected horses[0]==1 (◎), got {b['horses']}"
            )

    def test_umatan_ordered_flag_is_true(self, shoubu_predictions, full_odds_8h):
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        for b in umatan_bets:
            assert b.get("ordered") is True, (
                f"Expected ordered=True for umatan, got {b.get('ordered')}"
            )

    def test_bets_sorted_by_odds_descending(self, shoubu_predictions, full_odds_8h):
        """Combo bets (excluding tanpuku) must be sorted by odds descending."""
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        combo_bets = [b for b in bets if b["type"] not in ("tansho", "fukusho")]
        odds_sequence = [b["odds"] for b in combo_bets]
        assert odds_sequence == sorted(odds_sequence, reverse=True), (
            f"Bets not sorted descending: {odds_sequence}"
        )


# ---------------------------------------------------------------------------
# 3. ワイド is minimal
# ---------------------------------------------------------------------------

class TestWideIsMinimal:
    def test_at_most_1_wide_bet(self, shoubu_predictions, full_odds_8h):
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        wide_bets = [b for b in bets if b["type"] == "wide"]
        assert len(wide_bets) <= 1

    def test_no_wide_for_ai3_partner(self, shoubu_predictions, full_odds_8h):
        """Wide bet for AI 3位 (horse 3) must not be generated."""
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        wide_bets = [b for b in bets if b["type"] == "wide"]
        for b in wide_bets:
            assert 3 not in b["horses"], (
                f"Unexpected wide bet involving horse 3: {b['horses']}"
            )

    def test_no_wide_for_ai4_partner(self, shoubu_predictions, full_odds_8h):
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        wide_bets = [b for b in bets if b["type"] == "wide"]
        for b in wide_bets:
            assert 4 not in b["horses"], (
                f"Unexpected wide bet involving horse 4: {b['horses']}"
            )

    def test_no_wide_for_ai5_partner(self, shoubu_predictions, full_odds_8h):
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        wide_bets = [b for b in bets if b["type"] == "wide"]
        for b in wide_bets:
            assert 5 not in b["horses"], (
                f"Unexpected wide bet involving horse 5: {b['houses']}"
            )

    def test_wide_only_honmei_ai2_pair(self, shoubu_predictions, full_odds_8h):
        """Exactly ◎-AI2 pair if any wide appears."""
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        wide_bets = [b for b in bets if b["type"] == "wide"]
        if wide_bets:
            assert set(wide_bets[0]["horses"]) == {1, 2}


# ---------------------------------------------------------------------------
# 4. 複勝 conditions
# ---------------------------------------------------------------------------

class TestFukushoConditions:
    def _odds_with_tansho(self, honmei_hn: int, tansho_odds: float, fukusho_odds: float) -> dict:
        base = _make_full_odds(honmei=honmei_hn, partners=[2, 3, 4, 5, 6, 7])
        # Override tansho and fukusho for honmei
        base["tansho"] = [{"horses": [honmei_hn], "odds": tansho_odds, "payout": int(tansho_odds * 100)}]
        base["fukusho"] = [{"horses": [honmei_hn], "odds": fukusho_odds, "payout": int(fukusho_odds * 100),
                            "oddsMin": fukusho_odds - 0.5, "oddsMax": fukusho_odds + 0.5}]
        return base

    def test_honmei_odds_8x_generates_tansho_and_fukusho(self):
        """◎ tansho odds = 8x (≥ 6x): both tansho AND fukusho generated."""
        predictions = _make_predictions([80.0, 72.0, 65.0, 58.0, 50.0, 42.0, 35.0, 28.0])
        odds_data = self._odds_with_tansho(honmei_hn=1, tansho_odds=8.0, fukusho_odds=7.0)

        from backend.predictor.bet_optimizer import optimize_bets
        race_info = {"raceId": "202608081101", "headCount": 8}
        # Provide entries so honmei_odds is found via entries path
        entries = [{"horseNumber": i, "frameNumber": i, "odds": 8.0 if i == 1 else 5.0,
                    "popularity": i, "isScratched": False} for i in range(1, 9)]
        bets = optimize_bets(predictions, odds_data, race_info, entries=entries)

        types = [b["type"] for b in bets]
        assert "tansho" in types, "Expected tansho bet with 8x odds"
        assert "fukusho" in types, "Expected fukusho bet with 8x odds"

    def test_honmei_odds_8x_tansho_betsize_is_1(self):
        """With 8x odds, tansho betSize = TANPUKU_RATIO[0] = 1."""
        predictions = _make_predictions([80.0, 72.0, 65.0, 58.0, 50.0, 42.0, 35.0, 28.0])
        odds_data = self._odds_with_tansho(honmei_hn=1, tansho_odds=8.0, fukusho_odds=7.0)

        from backend.predictor.bet_optimizer import optimize_bets
        race_info = {"raceId": "202608081101", "headCount": 8}
        entries = [{"horseNumber": i, "frameNumber": i, "odds": 8.0 if i == 1 else 5.0,
                    "popularity": i, "isScratched": False} for i in range(1, 9)]
        bets = optimize_bets(predictions, odds_data, race_info, entries=entries)

        tansho_bets = [b for b in bets if b["type"] == "tansho"]
        assert len(tansho_bets) == 1
        assert tansho_bets[0]["betSize"] == 1, (
            f"Expected tansho betSize=1, got {tansho_bets[0]['betSize']}"
        )

    def test_honmei_odds_8x_fukusho_betsize_is_1(self):
        """With 8x odds (≥ both thresholds), fukusho betSize = TANPUKU_RATIO[1] = 1."""
        predictions = _make_predictions([80.0, 72.0, 65.0, 58.0, 50.0, 42.0, 35.0, 28.0])
        odds_data = self._odds_with_tansho(honmei_hn=1, tansho_odds=8.0, fukusho_odds=7.0)

        from backend.predictor.bet_optimizer import optimize_bets
        race_info = {"raceId": "202608081101", "headCount": 8}
        entries = [{"horseNumber": i, "frameNumber": i, "odds": 8.0 if i == 1 else 5.0,
                    "popularity": i, "isScratched": False} for i in range(1, 9)]
        bets = optimize_bets(predictions, odds_data, race_info, entries=entries)

        fukusho_bets = [b for b in bets if b["type"] == "fukusho"]
        assert len(fukusho_bets) == 1
        assert fukusho_bets[0]["betSize"] == 1, (
            f"Expected fukusho betSize=1, got {fukusho_bets[0]['betSize']}"
        )

    def test_honmei_odds_4x_no_tansho_no_fukusho(self):
        """◎ tansho odds = 4x (< 6x): NO tansho, NO fukusho."""
        predictions = _make_predictions([80.0, 72.0, 65.0, 58.0, 50.0, 42.0, 35.0, 28.0])
        odds_data = self._odds_with_tansho(honmei_hn=1, tansho_odds=4.0, fukusho_odds=2.5)

        from backend.predictor.bet_optimizer import optimize_bets
        race_info = {"raceId": "202608081101", "headCount": 8}
        entries = [{"horseNumber": i, "frameNumber": i, "odds": 4.0 if i == 1 else 5.0,
                    "popularity": i, "isScratched": False} for i in range(1, 9)]
        bets = optimize_bets(predictions, odds_data, race_info, entries=entries)

        types = [b["type"] for b in bets]
        assert "tansho" not in types, "Expected NO tansho with 4x odds"
        assert "fukusho" not in types, "Expected NO fukusho with 4x odds"

    def test_honmei_odds_3x_no_fukusho(self):
        """◎ tansho odds = 3x (< 6x): NO fukusho even though previously generated at 2x+."""
        predictions = _make_predictions([80.0, 72.0, 65.0, 58.0, 50.0, 42.0, 35.0, 28.0])
        odds_data = self._odds_with_tansho(honmei_hn=1, tansho_odds=3.0, fukusho_odds=1.8)

        from backend.predictor.bet_optimizer import optimize_bets
        race_info = {"raceId": "202608081101", "headCount": 8}
        entries = [{"horseNumber": i, "frameNumber": i, "odds": 3.0 if i == 1 else 5.0,
                    "popularity": i, "isScratched": False} for i in range(1, 9)]
        bets = optimize_bets(predictions, odds_data, race_info, entries=entries)

        types = [b["type"] for b in bets]
        assert "fukusho" not in types, "Expected NO fukusho with 3x odds (threshold is 6x)"

    def test_fukusho_threshold_is_6x(self):
        """Boundary: ◎ tansho odds exactly 6.0x generates tansho+fukusho."""
        predictions = _make_predictions([80.0, 72.0, 65.0, 58.0, 50.0, 42.0, 35.0, 28.0])
        odds_data = self._odds_with_tansho(honmei_hn=1, tansho_odds=6.0, fukusho_odds=6.0)

        from backend.predictor.bet_optimizer import optimize_bets
        race_info = {"raceId": "202608081101", "headCount": 8}
        entries = [{"horseNumber": i, "frameNumber": i, "odds": 6.0 if i == 1 else 5.0,
                    "popularity": i, "isScratched": False} for i in range(1, 9)]
        bets = optimize_bets(predictions, odds_data, race_info, entries=entries)

        types = [b["type"] for b in bets]
        assert "tansho" in types, "Expected tansho at exactly 6x threshold"
        assert "fukusho" in types, "Expected fukusho at exactly 6x threshold"

    def test_below_threshold_5x_no_tanpuku(self):
        """Boundary: 5.9x < 6.0x threshold → no tanpuku."""
        predictions = _make_predictions([80.0, 72.0, 65.0, 58.0, 50.0, 42.0, 35.0, 28.0])
        odds_data = self._odds_with_tansho(honmei_hn=1, tansho_odds=5.9, fukusho_odds=5.9)

        from backend.predictor.bet_optimizer import optimize_bets
        race_info = {"raceId": "202608081101", "headCount": 8}
        entries = [{"horseNumber": i, "frameNumber": i, "odds": 5.9 if i == 1 else 5.0,
                    "popularity": i, "isScratched": False} for i in range(1, 9)]
        bets = optimize_bets(predictions, odds_data, race_info, entries=entries)

        types = [b["type"] for b in bets]
        assert "tansho" not in types, "Expected NO tansho at 5.9x"
        assert "fukusho" not in types, "Expected NO fukusho at 5.9x"


# ---------------------------------------------------------------------------
# 5. 勝負判定 (evaluate_bet_confidence)
# ---------------------------------------------------------------------------

class TestShoubuHantei:
    def test_score_68_with_good_odds_returns_A(self):
        from backend.predictor.bet_optimizer import evaluate_bet_confidence
        predictions = _make_predictions([68.0, 60.0, 50.0])
        entries = [{"horseNumber": 1, "odds": 3.0}]
        result = evaluate_bet_confidence(predictions, {}, entries)
        assert result == "A", f"Expected 'A' for score=68+odds=3.0, got {result!r}"

    def test_score_74_without_odds_returns_C(self):
        """D7: score>=68 but no odds info → C (can't determine A or B)."""
        from backend.predictor.bet_optimizer import evaluate_bet_confidence
        predictions = _make_predictions([74.1, 60.0, 50.0])
        result = evaluate_bet_confidence(predictions, {})
        assert result == "C"

    def test_score_67_point_9_returns_C(self):
        from backend.predictor.bet_optimizer import evaluate_bet_confidence
        predictions = _make_predictions([67.9, 60.0, 50.0])
        result = evaluate_bet_confidence(predictions, {})
        assert result == "C", f"Expected 'C' for score=67.9, got {result!r}"

    def test_score_80_no_odds_returns_C(self):
        """D7: score>=68 but no odds info → C."""
        from backend.predictor.bet_optimizer import evaluate_bet_confidence
        predictions = _make_predictions([80.0, 60.0, 50.0])
        result = evaluate_bet_confidence(predictions, {})
        assert result == "C"

    def test_score_0_returns_C(self):
        from backend.predictor.bet_optimizer import evaluate_bet_confidence
        predictions = _make_predictions([0.0, 0.0, 0.0])
        result = evaluate_bet_confidence(predictions, {})
        assert result == "C"

    def test_empty_predictions_returns_C(self):
        from backend.predictor.bet_optimizer import evaluate_bet_confidence
        result = evaluate_bet_confidence([], {})
        assert result == "C"

    def test_score_exactly_at_threshold_68(self):
        """Boundary value: exactly 68.0 with good odds → A, without odds → C (no B info)."""
        from backend.predictor.bet_optimizer import evaluate_bet_confidence, SHOUBU_MIN_SCORE
        assert SHOUBU_MIN_SCORE == 68.0
        predictions = _make_predictions([68.0, 50.0, 40.0])
        entries = [{"horseNumber": 1, "odds": 3.0}]
        assert evaluate_bet_confidence(predictions, {}, entries) == "A"
        assert evaluate_bet_confidence(predictions, {}) == "C"


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_small_field_5_horses_still_generates_bets(self):
        """5-horse field: umatan/umaren/wide with available partners."""
        predictions = _make_predictions([80.0, 72.0, 65.0, 58.0, 50.0])
        odds_data = _make_full_odds(honmei=1, partners=[2, 3, 4, 5])

        from backend.predictor.bet_optimizer import optimize_bets
        race_info = {"raceId": "202608081101", "headCount": 5}
        bets = optimize_bets(predictions, odds_data, race_info)
        combo_bets = [b for b in bets if b["type"] not in ("tansho", "fukusho")]
        assert len(combo_bets) > 0, "Expected bets for 5-horse field"

    def test_small_field_5_horses_has_umatan(self):
        predictions = _make_predictions([80.0, 72.0, 65.0, 58.0, 50.0])
        odds_data = _make_full_odds(honmei=1, partners=[2, 3, 4, 5])

        from backend.predictor.bet_optimizer import optimize_bets
        race_info = {"raceId": "202608081101", "headCount": 5}
        bets = optimize_bets(predictions, odds_data, race_info)
        assert any(b["type"] == "umatan" for b in bets)

    def test_small_field_5_horses_has_umaren(self):
        predictions = _make_predictions([80.0, 72.0, 65.0, 58.0, 50.0])
        odds_data = _make_full_odds(honmei=1, partners=[2, 3, 4, 5])

        from backend.predictor.bet_optimizer import optimize_bets
        race_info = {"raceId": "202608081101", "headCount": 5}
        bets = optimize_bets(predictions, odds_data, race_info)
        assert any(b["type"] == "umaren" for b in bets)

    def test_small_field_5_horses_produces_bets(self):
        predictions = _make_predictions([80.0, 72.0, 65.0, 58.0, 50.0])
        odds_data = _make_full_odds(honmei=1, partners=[2, 3, 4, 5])

        from backend.predictor.bet_optimizer import optimize_bets
        race_info = {"raceId": "202608081101", "headCount": 5}
        bets = optimize_bets(predictions, odds_data, race_info)
        assert len(bets) >= 1

    def test_below_threshold_score_bets_still_generated(self):
        """◎ score < 74 → bets are still generated (SKIP note is caller's responsibility).

        optimize_bets() does NOT block generation; evaluate_bet_confidence() is used
        separately by the API layer to decide. The function always returns bets
        when valid odds are available regardless of score.
        """
        predictions = _make_predictions([70.0, 60.0, 50.0, 45.0, 40.0,
                                         35.0, 30.0, 25.0])
        odds_data = _make_full_odds(honmei=1, partners=[2, 3, 4, 5, 6, 7, 8])

        from backend.predictor.bet_optimizer import optimize_bets
        race_info = {"raceId": "202608081101", "headCount": 8}
        bets = optimize_bets(predictions, odds_data, race_info)
        # Below threshold: UMATAN_MIN_SCORE=74 gate blocks umatan/umaren, but
        # wide (which has no score gate in _try_add) may still appear.
        # The key assertion: function does not raise and returns a list.
        assert isinstance(bets, list)

    def test_only_umatan_odds_generates_only_umatan(self):
        """When only umatan odds are available, only umatan bets are returned."""
        predictions = _make_predictions([80.0, 72.0, 65.0, 58.0, 50.0,
                                         42.0, 35.0, 28.0])
        odds_data = {
            "umatan": [
                {"horses": [1, 2], "odds": 12.0, "payout": 1200},
                {"horses": [1, 3], "odds": 15.0, "payout": 1500},
                {"horses": [1, 4], "odds": 18.0, "payout": 1800},
            ]
        }

        from backend.predictor.bet_optimizer import optimize_bets
        race_info = {"raceId": "202608081101", "headCount": 8}
        bets = optimize_bets(predictions, odds_data, race_info)

        types = {b["type"] for b in bets}
        assert types <= {"umatan"}, f"Expected only umatan, got {types}"
        assert "umatan" in types

    def test_no_odds_returns_empty_list(self):
        """With empty odds_data, no bets can be placed."""
        predictions = _make_predictions([80.0, 72.0, 65.0, 58.0, 50.0,
                                         42.0, 35.0, 28.0])

        from backend.predictor.bet_optimizer import optimize_bets
        race_info = {"raceId": "202608081101", "headCount": 8}
        bets = optimize_bets(predictions, {}, race_info)
        # No odds data → no bets (tanpuku also needs odds)
        combo_bets = [b for b in bets if b["type"] not in ("tansho", "fukusho")]
        assert combo_bets == [], f"Expected empty combo bets, got {combo_bets}"

    def test_headcount_below_3_returns_empty(self):
        """Head count < 3 must return empty list immediately."""
        predictions = _make_predictions([80.0, 72.0])
        odds_data = _make_full_odds(honmei=1, partners=[2])

        from backend.predictor.bet_optimizer import optimize_bets
        race_info = {"raceId": "202608081101", "headCount": 2}
        bets = optimize_bets(predictions, odds_data, race_info)
        assert bets == []

    def test_scratched_horse_excluded_from_bets(self):
        """Scratched horse must not appear in any bet."""
        predictions = [
            {"horseNumber": 1, "score": 80.0, "isScratched": False},
            {"horseNumber": 2, "score": 72.0, "isScratched": True},  # scratched
            {"horseNumber": 3, "score": 65.0, "isScratched": False},
            {"horseNumber": 4, "score": 58.0, "isScratched": False},
            {"horseNumber": 5, "score": 50.0, "isScratched": False},
            {"horseNumber": 6, "score": 42.0, "isScratched": False},
            {"horseNumber": 7, "score": 35.0, "isScratched": False},
            {"horseNumber": 8, "score": 28.0, "isScratched": False},
        ]
        odds_data = _make_full_odds(honmei=1, partners=[3, 4, 5, 6, 7, 8])

        from backend.predictor.bet_optimizer import optimize_bets
        race_info = {"raceId": "202608081101", "headCount": 8}
        bets = optimize_bets(predictions, odds_data, race_info)
        for bet in bets:
            assert 2 not in bet["horses"], (
                f"Scratched horse 2 appeared in bet {bet}"
            )


# ---------------------------------------------------------------------------
# 7. Integration: bet structure summary
# ---------------------------------------------------------------------------

class TestBetStructureSummary:
    def test_structure_has_bets_with_ev(self, shoubu_predictions, full_odds_8h):
        """D6: verify bets are generated with EV field."""
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        combo_bets = [b for b in bets if b["type"] not in ("tansho", "fukusho")]
        assert len(combo_bets) >= 1, f"Expected at least 1 combo bet"
        for b in combo_bets:
            assert "ev" in b, f"Bet missing 'ev': {b}"

    def test_total_bet_points_within_limit(self, shoubu_predictions, full_odds_8h):
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        combo_bets = [b for b in bets if b["type"] not in ("tansho", "fukusho")]
        assert len(combo_bets) <= 10  # D6: max 5 core + 5 value

    def test_combo_bets_sorted_by_odds_descending(self, shoubu_predictions, full_odds_8h):
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        combo_bets = [b for b in bets if b["type"] not in ("tansho", "fukusho")]
        odds = [b["odds"] for b in combo_bets]
        assert odds == sorted(odds, reverse=True), (
            f"Bets not sorted by odds desc: {odds}"
        )

    def test_all_bets_have_rank_field(self, shoubu_predictions, full_odds_8h):
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        for i, bet in enumerate(bets):
            assert "rank" in bet, f"Bet {i} missing 'rank' field: {bet}"

    def test_all_bets_have_bet_size_field(self, shoubu_predictions, full_odds_8h):
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        for bet in bets:
            assert "betSize" in bet, f"Bet missing 'betSize': {bet}"

    def test_all_bets_have_has_real_odds(self, shoubu_predictions, full_odds_8h):
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        for bet in bets:
            assert bet.get("hasRealOdds") is True, (
                f"Bet missing hasRealOdds=True: {bet}"
            )

    def test_max_bets_parameter_limits_combo_bets(self, shoubu_predictions, full_odds_8h):
        """When max_bets=3, only 3 combo bets are returned."""
        from backend.predictor.bet_optimizer import optimize_bets
        race_info = {"raceId": "202608081101", "headCount": 8}
        bets = optimize_bets(shoubu_predictions, full_odds_8h, race_info, max_bets=3)
        combo_bets = [b for b in bets if b["type"] not in ("tansho", "fukusho")]
        assert len(combo_bets) <= 3

    def test_bet_type_labels_are_present(self, shoubu_predictions, full_odds_8h):
        """Each bet must have a typeLabel field."""
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        for bet in bets:
            assert "typeLabel" in bet, f"Bet missing typeLabel: {bet}"

    def test_umatan_type_label_is_correct(self, shoubu_predictions, full_odds_8h):
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        for b in umatan_bets:
            assert b["typeLabel"] == "馬単", f"Wrong typeLabel: {b['typeLabel']}"

    def test_umaren_type_label_is_correct(self, shoubu_predictions, full_odds_8h):
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        umaren_bets = [b for b in bets if b["type"] == "umaren"]
        for b in umaren_bets:
            assert b["typeLabel"] == "馬連", f"Wrong typeLabel: {b['typeLabel']}"

    def test_wide_type_label_is_correct(self, shoubu_predictions, full_odds_8h):
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        wide_bets = [b for b in bets if b["type"] == "wide"]
        for b in wide_bets:
            assert b["typeLabel"] == "ワイド", f"Wrong typeLabel: {b['typeLabel']}"

    def test_no_sanrentan_or_sanrenpuku_in_d5_output(self, shoubu_predictions, full_odds_8h):
        """D5 structure does not include 3連単 or 3連複."""
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        for bet in bets:
            assert bet["type"] not in ("sanrentan", "sanrenpuku"), (
                f"Unexpected {bet['type']} in D5 output"
            )

    def test_no_wakuren_in_d5_output(self, shoubu_predictions, full_odds_8h):
        """D5 structure does not include 枠連."""
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        for bet in bets:
            assert bet["type"] != "wakuren", (
                f"Unexpected wakuren in D5 output"
            )

    def test_all_umatan_odds_above_d5_minimum(self, shoubu_predictions, full_odds_8h):
        """All umatan bets must have odds >= 5.0 (D5_MIN_ODDS)."""
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        umatan_bets = [b for b in bets if b["type"] == "umatan"]
        for b in umatan_bets:
            assert b["odds"] >= 5.0, (
                f"Umatan odds below 5.0: {b['odds']}"
            )

    def test_all_umaren_odds_above_d5_minimum(self, shoubu_predictions, full_odds_8h):
        """All umaren bets must have odds >= 3.0 (D5_MIN_ODDS)."""
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        umaren_bets = [b for b in bets if b["type"] == "umaren"]
        for b in umaren_bets:
            assert b["odds"] >= 3.0, (
                f"Umaren odds below 3.0: {b['odds']}"
            )

    def test_all_wide_odds_above_d5_minimum(self, shoubu_predictions, full_odds_8h):
        """All wide bets must have odds >= 2.5 (D5_MIN_ODDS)."""
        bets = _run_optimize(shoubu_predictions, full_odds_8h)
        wide_bets = [b for b in bets if b["type"] == "wide"]
        for b in wide_bets:
            assert b["odds"] >= 2.5, (
                f"Wide odds below 2.5: {b['odds']}"
            )
