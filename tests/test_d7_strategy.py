"""TDD tests for D7 EV Focus betting strategy.

D7 two-layer structure:
  BUY layer (core_bets): ◎単勝 ONLY when score>=68 AND honmei odds in [2.0, 4.0)
  INFO layer (value_bets): All bet types where EV>0, capped at 5 bets

Key constants under test:
  SHOUBU_MIN_SCORE      = 68.0
  BUY_HONMEI_ODDS_MIN   = 2.0
  BUY_HONMEI_ODDS_MAX   = 4.0
  INFO_BET_TYPES        = {tansho, umaren, umatan, wide, sanrenpuku, sanrentan}
  INFO_MAX_BETS         = 5
  INFO_MIN_ODDS         = 2.0
  MAX_ODDS_CAP          per type (e.g., tansho:200, umatan:1000 ...)

Return schema of optimize_bets_dual():
  {
    "core_bets": list,       # 0 or 1 tansho BUY bets
    "value_bets": list,      # 0-5 EV>0 INFO bets
    "longshot": dict | None,
    "pattern": str,
    "layer1_active": bool,
    "honmei_odds": float,
  }
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_predictions(scores: list) -> list:
    """Build predictions list from a list of scores (index 0 = horse 1 = ◎)."""
    return [
        {"horseNumber": i + 1, "score": float(s), "isScratched": False}
        for i, s in enumerate(scores)
    ]


def _make_full_odds(
    honmei: int,
    partners: list,
    honmei_tansho_odds: float = 3.0,
) -> dict:
    """Build odds_data covering all INFO_BET_TYPES plus tansho/fukusho.

    honmei_tansho_odds controls the ◎ tansho odds (used for BUY layer gate).
    All combo odds are set to values that produce positive EV for test clarity.
    """
    tansho_entries = []
    fukusho_entries = []
    umaren_entries = []
    umatan_entries = []
    wide_entries = []
    sanrenpuku_entries = []
    sanrentan_entries = []

    all_horses = [honmei] + partners

    # ── Tansho / Fukusho ──
    for h in all_horses:
        odds = honmei_tansho_odds if h == honmei else 8.0
        tansho_entries.append({"horses": [h], "odds": odds, "payout": int(odds * 100)})
        fukusho_entries.append({
            "horses": [h], "odds": 2.5, "payout": 250,
            "oddsMin": 2.0, "oddsMax": 3.0,
        })

    # ── Umaren / Umatan / Wide ──
    for p in partners:
        pair = sorted([honmei, p])
        umaren_entries.append({"horses": pair, "odds": 10.0, "payout": 1000})
        umatan_entries.append({"horses": [honmei, p], "odds": 18.0, "payout": 1800})
        umatan_entries.append({"horses": [p, honmei], "odds": 25.0, "payout": 2500})
        wide_entries.append({"horses": pair, "odds": 4.0, "payout": 400})

    for i, p1 in enumerate(partners):
        for p2 in partners[i + 1:]:
            pair = sorted([p1, p2])
            umaren_entries.append({"horses": pair, "odds": 20.0, "payout": 2000})
            umatan_entries.append({"horses": [p1, p2], "odds": 30.0, "payout": 3000})
            umatan_entries.append({"horses": [p2, p1], "odds": 35.0, "payout": 3500})
            wide_entries.append({"horses": pair, "odds": 6.0, "payout": 600})

    # ── Sanrenpuku / Sanrentan ──
    for i, p1 in enumerate(partners):
        for j, p2 in enumerate(partners):
            if j <= i:
                continue
            trio = sorted([honmei, p1, p2])
            sanrenpuku_entries.append({"horses": trio, "odds": 50.0, "payout": 5000})
            # A few sanrentan permutations for honmei at front
            sanrentan_entries.append({"horses": [honmei, p1, p2], "odds": 150.0, "payout": 15000})
            sanrentan_entries.append({"horses": [honmei, p2, p1], "odds": 180.0, "payout": 18000})

    return {
        "tansho": tansho_entries,
        "fukusho": fukusho_entries,
        "umaren": umaren_entries,
        "umatan": umatan_entries,
        "wide": wide_entries,
        "sanrenpuku": sanrenpuku_entries,
        "sanrentan": sanrentan_entries,
    }


def _make_entries_with_odds(num_horses: int, honmei_odds: float) -> list:
    """Build entries list so honmei_odds is found via entries path."""
    return [
        {
            "horseNumber": i,
            "frameNumber": i,
            "odds": honmei_odds if i == 1 else 8.0,
            "popularity": i,
            "isScratched": False,
        }
        for i in range(1, num_horses + 1)
    ]


def _run_dual(
    predictions,
    odds_data,
    honmei_odds=3.0,
    headcount=8,
    race_id="202608231101",
    entries=None,
    mc_samples=200,  # small for test speed
):
    """Thin wrapper around optimize_bets_dual()."""
    from backend.predictor.bet_optimizer import optimize_bets_dual
    race_info = {"raceId": race_id, "headCount": headcount}
    if entries is None:
        entries = _make_entries_with_odds(headcount, honmei_odds)
    return optimize_bets_dual(
        predictions, odds_data, race_info, entries=entries, mc_samples=mc_samples
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def confident_predictions():
    """◎ score=75 — above SHOUBU_MIN_SCORE=68."""
    return _make_predictions([75.0, 65.0, 55.0, 48.0, 40.0, 33.0, 25.0, 18.0])


@pytest.fixture
def weak_predictions():
    """◎ score=60 — below SHOUBU_MIN_SCORE=68."""
    return _make_predictions([60.0, 55.0, 50.0, 45.0, 40.0, 35.0, 30.0, 25.0])


@pytest.fixture
def full_odds_buy_range(confident_predictions):
    """Odds data with ◎ tansho odds=3.0 (inside BUY range 2-4x)."""
    return _make_full_odds(honmei=1, partners=[2, 3, 4, 5, 6, 7, 8], honmei_tansho_odds=3.0)


@pytest.fixture
def full_odds_outside_range(confident_predictions):
    """Odds data with ◎ tansho odds=5.0 (outside BUY range)."""
    return _make_full_odds(honmei=1, partners=[2, 3, 4, 5, 6, 7, 8], honmei_tansho_odds=5.0)


# ---------------------------------------------------------------------------
# 1. Return schema
# ---------------------------------------------------------------------------

class TestReturnSchema:
    """optimize_bets_dual() must always return the correct keys."""

    def test_returns_dict(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        assert isinstance(result, dict)

    def test_has_core_bets_key(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        assert "core_bets" in result

    def test_has_value_bets_key(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        assert "value_bets" in result

    def test_has_longshot_key(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        assert "longshot" in result

    def test_has_pattern_key(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        assert "pattern" in result

    def test_has_layer1_active_key(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        assert "layer1_active" in result

    def test_has_honmei_odds_key(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        assert "honmei_odds" in result

    def test_core_bets_is_list(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        assert isinstance(result["core_bets"], list)

    def test_value_bets_is_list(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        assert isinstance(result["value_bets"], list)

    def test_layer1_active_is_bool(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        assert isinstance(result["layer1_active"], bool)

    def test_pattern_is_str(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        assert isinstance(result["pattern"], str)

    def test_honmei_odds_is_float(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        assert isinstance(result["honmei_odds"], float)


# ---------------------------------------------------------------------------
# 2. BUY layer activation gates
# ---------------------------------------------------------------------------

class TestBuyLayerActivation:
    """layer1_active is True ONLY when BOTH gates pass: score>=68 AND odds in [2.0, 4.0)."""

    def test_active_when_score_68_and_odds_3x(self):
        """score=68 (boundary), odds=3.0 → active."""
        preds = _make_predictions([68.0, 55.0, 45.0, 38.0, 30.0, 22.0, 15.0, 8.0])
        odds = _make_full_odds(1, [2, 3, 4, 5, 6, 7, 8], honmei_tansho_odds=3.0)
        result = _run_dual(preds, odds, honmei_odds=3.0)
        assert result["layer1_active"] is True, (
            f"Expected layer1_active=True for score=68, odds=3.0; got {result['layer1_active']}"
        )

    def test_active_when_score_75_and_odds_2x_boundary(self):
        """odds=2.0 (min boundary, inclusive) → active."""
        preds = _make_predictions([75.0, 60.0, 50.0, 42.0, 35.0, 28.0, 20.0, 12.0])
        odds = _make_full_odds(1, [2, 3, 4, 5, 6, 7, 8], honmei_tansho_odds=2.0)
        result = _run_dual(preds, odds, honmei_odds=2.0)
        assert result["layer1_active"] is True, (
            f"Expected layer1_active=True for odds=2.0 (min boundary); got {result['layer1_active']}"
        )

    def test_inactive_when_odds_exactly_4x(self):
        """odds=4.0 is the exclusive upper bound → NOT active."""
        preds = _make_predictions([75.0, 60.0, 50.0, 42.0, 35.0, 28.0, 20.0, 12.0])
        odds = _make_full_odds(1, [2, 3, 4, 5, 6, 7, 8], honmei_tansho_odds=4.0)
        result = _run_dual(preds, odds, honmei_odds=4.0)
        assert result["layer1_active"] is False, (
            f"Expected layer1_active=False for odds=4.0 (exclusive max); got {result['layer1_active']}"
        )

    def test_inactive_when_odds_above_range(self):
        """odds=5.0 > 4.0 → NOT active even with high score."""
        preds = _make_predictions([80.0, 65.0, 55.0, 48.0, 40.0, 33.0, 25.0, 18.0])
        odds = _make_full_odds(1, [2, 3, 4, 5, 6, 7, 8], honmei_tansho_odds=5.0)
        result = _run_dual(preds, odds, honmei_odds=5.0)
        assert result["layer1_active"] is False

    def test_inactive_when_odds_below_range(self):
        """odds=1.5 < 2.0 → NOT active (too short for value)."""
        preds = _make_predictions([80.0, 65.0, 55.0, 48.0, 40.0, 33.0, 25.0, 18.0])
        odds = _make_full_odds(1, [2, 3, 4, 5, 6, 7, 8], honmei_tansho_odds=1.5)
        result = _run_dual(preds, odds, honmei_odds=1.5)
        assert result["layer1_active"] is False

    def test_inactive_when_score_below_68(self):
        """score=67.9 < 68 → NOT active even with perfect odds."""
        preds = _make_predictions([67.9, 60.0, 50.0, 43.0, 36.0, 29.0, 22.0, 15.0])
        odds = _make_full_odds(1, [2, 3, 4, 5, 6, 7, 8], honmei_tansho_odds=3.0)
        result = _run_dual(preds, odds, honmei_odds=3.0)
        assert result["layer1_active"] is False, (
            f"Expected False for score=67.9; got {result['layer1_active']}"
        )

    def test_inactive_when_score_zero(self):
        """score=0 → NOT active."""
        preds = _make_predictions([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        odds = _make_full_odds(1, [2, 3, 4, 5, 6, 7, 8], honmei_tansho_odds=3.0)
        result = _run_dual(preds, odds, honmei_odds=3.0)
        assert result["layer1_active"] is False

    def test_inactive_score_gate_fails_odds_ok(self):
        """score=50 (weak), odds=3.0 (ok) → NOT active (score gate fails)."""
        preds = _make_predictions([50.0, 45.0, 40.0, 35.0, 30.0, 25.0, 20.0, 15.0])
        odds = _make_full_odds(1, [2, 3, 4, 5, 6, 7, 8], honmei_tansho_odds=3.0)
        result = _run_dual(preds, odds, honmei_odds=3.0)
        assert result["layer1_active"] is False

    def test_active_high_score_mid_range_odds(self):
        """score=90, odds=3.5 → active."""
        preds = _make_predictions([90.0, 65.0, 55.0, 48.0, 40.0, 33.0, 25.0, 18.0])
        odds = _make_full_odds(1, [2, 3, 4, 5, 6, 7, 8], honmei_tansho_odds=3.5)
        result = _run_dual(preds, odds, honmei_odds=3.5)
        assert result["layer1_active"] is True


# ---------------------------------------------------------------------------
# 3. BUY layer bet count and type
# ---------------------------------------------------------------------------

class TestBuyLayerBetCount:
    """BUY layer must produce exactly 0 or 1 tansho bet."""

    def test_buy_layer_produces_exactly_1_bet_when_active(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range, honmei_odds=3.0)
        assert result["layer1_active"] is True
        assert len(result["core_bets"]) == 1, (
            f"Expected exactly 1 core bet, got {len(result['core_bets'])}"
        )

    def test_buy_layer_bet_is_tansho_type(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range, honmei_odds=3.0)
        assert result["layer1_active"] is True
        bet = result["core_bets"][0]
        assert bet["type"] == "tansho", (
            f"Expected BUY bet type='tansho', got {bet['type']!r}"
        )

    def test_buy_layer_bet_is_for_honmei(self, confident_predictions, full_odds_buy_range):
        """The single BUY bet must be on horse 1 (◎, highest score)."""
        result = _run_dual(confident_predictions, full_odds_buy_range, honmei_odds=3.0)
        assert result["layer1_active"] is True
        bet = result["core_bets"][0]
        assert bet["horses"] == [1], (
            f"Expected honmei horse=[1], got {bet['horses']}"
        )

    def test_buy_layer_empty_when_inactive(self, confident_predictions, full_odds_outside_range):
        """When layer1_active=False, core_bets must be empty."""
        result = _run_dual(confident_predictions, full_odds_outside_range, honmei_odds=5.0)
        assert result["layer1_active"] is False
        assert result["core_bets"] == [], (
            f"Expected empty core_bets when inactive, got {result['core_bets']}"
        )

    def test_buy_layer_empty_for_weak_predictions(self, weak_predictions, full_odds_buy_range):
        """Weak ◎ (score<68) → core_bets empty even if odds are in range."""
        result = _run_dual(weak_predictions, full_odds_buy_range, honmei_odds=3.0)
        assert result["core_bets"] == [], (
            f"Expected empty core_bets for low score, got {result['core_bets']}"
        )

    def test_buy_layer_never_exceeds_1_bet(self, confident_predictions, full_odds_buy_range):
        """core_bets length is always 0 or 1, never more."""
        result = _run_dual(confident_predictions, full_odds_buy_range, honmei_odds=3.0)
        assert len(result["core_bets"]) <= 1, (
            f"BUY layer must not exceed 1 bet, got {len(result['core_bets'])}"
        )


# ---------------------------------------------------------------------------
# 4. BUY layer required bet fields
# ---------------------------------------------------------------------------

class TestBuyLayerBetFields:
    """Each BUY layer bet must have all required display fields."""

    def _get_buy_bet(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range, honmei_odds=3.0)
        assert result["layer1_active"] and result["core_bets"], "Precondition: BUY layer must be active"
        return result["core_bets"][0]

    def test_buy_bet_has_odds(self, confident_predictions, full_odds_buy_range):
        bet = self._get_buy_bet(confident_predictions, full_odds_buy_range)
        assert "odds" in bet

    def test_buy_bet_odds_is_in_range(self, confident_predictions, full_odds_buy_range):
        bet = self._get_buy_bet(confident_predictions, full_odds_buy_range)
        assert 2.0 <= bet["odds"] < 4.0, (
            f"BUY odds must be in [2.0, 4.0), got {bet['odds']}"
        )

    def test_buy_bet_has_ev(self, confident_predictions, full_odds_buy_range):
        bet = self._get_buy_bet(confident_predictions, full_odds_buy_range)
        assert "ev" in bet

    def test_buy_bet_has_bet_size(self, confident_predictions, full_odds_buy_range):
        bet = self._get_buy_bet(confident_predictions, full_odds_buy_range)
        assert "betSize" in bet

    def test_buy_bet_size_is_positive_int(self, confident_predictions, full_odds_buy_range):
        bet = self._get_buy_bet(confident_predictions, full_odds_buy_range)
        assert isinstance(bet["betSize"], int)
        assert bet["betSize"] >= 1

    def test_buy_bet_has_rank_1(self, confident_predictions, full_odds_buy_range):
        bet = self._get_buy_bet(confident_predictions, full_odds_buy_range)
        assert bet["rank"] == 1, f"Expected rank=1 for BUY bet, got {bet['rank']}"

    def test_buy_bet_has_real_odds_true(self, confident_predictions, full_odds_buy_range):
        bet = self._get_buy_bet(confident_predictions, full_odds_buy_range)
        assert bet.get("hasRealOdds") is True, (
            f"Expected hasRealOdds=True for BUY bet, got {bet.get('hasRealOdds')}"
        )

    def test_buy_bet_has_hit_prob(self, confident_predictions, full_odds_buy_range):
        bet = self._get_buy_bet(confident_predictions, full_odds_buy_range)
        assert "hitProb" in bet
        assert 0.0 <= bet["hitProb"] <= 1.0


# ---------------------------------------------------------------------------
# 5. Kelly bet sizing on BUY bets
# ---------------------------------------------------------------------------

class TestKellyBetSizing:
    """BUY layer uses Kelly sizing (half-Kelly) for betSize."""

    def test_kelly_bet_size_function_with_positive_ev(self):
        """kelly_bet_size returns >= 1 for positive EV bets."""
        from backend.predictor.bet_optimizer import kelly_bet_size
        # hitProb=0.35, odds=3.0 → EV = 0.35*3.0 - 1 = 0.05 (positive)
        size = kelly_bet_size(0.35, 3.0)
        assert size >= 1

    def test_kelly_bet_size_minimum_is_1(self):
        """Even near-zero EV gives minimum betSize=1."""
        from backend.predictor.bet_optimizer import kelly_bet_size
        # Near-zero hit prob
        size = kelly_bet_size(0.001, 3.0)
        assert size >= 1, f"Kelly minimum must be 1, got {size}"

    def test_kelly_bet_size_scales_with_higher_prob(self):
        """Higher hit probability yields higher bet size."""
        from backend.predictor.bet_optimizer import kelly_bet_size
        low_prob_size = kelly_bet_size(0.20, 3.0)
        high_prob_size = kelly_bet_size(0.45, 3.0)
        assert high_prob_size >= low_prob_size, (
            f"Higher prob should give >= bet size: low={low_prob_size}, high={high_prob_size}"
        )

    def test_kelly_bet_size_capped_at_10(self):
        """betSize must never exceed 10 (KELLY_MAX_BET protection)."""
        from backend.predictor.bet_optimizer import kelly_bet_size
        size = kelly_bet_size(0.99, 100.0)
        assert size <= 10, f"Kelly size must be capped at 10, got {size}"

    def test_buy_layer_applies_kelly_sizing(self, confident_predictions, full_odds_buy_range):
        """BUY layer bet has betSize computed via Kelly (>= 1)."""
        result = _run_dual(confident_predictions, full_odds_buy_range, honmei_odds=3.0)
        assert result["layer1_active"]
        bet = result["core_bets"][0]
        assert bet["betSize"] >= 1

    def test_kelly_zero_or_neg_hit_prob_returns_1(self):
        from backend.predictor.bet_optimizer import kelly_bet_size
        assert kelly_bet_size(0.0, 3.0) == 1
        assert kelly_bet_size(-0.1, 3.0) == 1

    def test_kelly_odds_at_1_returns_1(self):
        """odds <= 1 is degenerate → return 1."""
        from backend.predictor.bet_optimizer import kelly_bet_size
        assert kelly_bet_size(0.5, 1.0) == 1
        assert kelly_bet_size(0.5, 0.5) == 1


# ---------------------------------------------------------------------------
# 6. INFO layer EV gate
# ---------------------------------------------------------------------------

class TestInfoLayerEvGate:
    """INFO layer (value_bets) must contain ONLY bets with EV > 0."""

    def test_all_value_bets_have_positive_ev(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        for bet in result["value_bets"]:
            assert bet.get("ev", -1) > 0, (
                f"INFO bet with non-positive EV found: ev={bet.get('ev')}, bet={bet}"
            )

    def test_value_bets_ev_field_present(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        for bet in result["value_bets"]:
            assert "ev" in bet, f"INFO bet missing 'ev' field: {bet}"

    def test_value_bets_sorted_by_ev_descending(self, confident_predictions, full_odds_buy_range):
        """INFO bets must be ordered by EV descending (best value first)."""
        result = _run_dual(confident_predictions, full_odds_buy_range)
        evs = [b["ev"] for b in result["value_bets"]]
        assert evs == sorted(evs, reverse=True), (
            f"INFO bets not sorted by EV desc: {evs}"
        )

    def test_no_zero_ev_in_value_bets(self, confident_predictions, full_odds_buy_range):
        """EV must be strictly > 0, not == 0."""
        result = _run_dual(confident_predictions, full_odds_buy_range)
        for bet in result["value_bets"]:
            assert bet["ev"] > 0, f"EV=0 bet should not appear in INFO layer: {bet}"


# ---------------------------------------------------------------------------
# 7. INFO layer cap at 5 bets
# ---------------------------------------------------------------------------

class TestInfoLayerMaxBets:
    """INFO layer must never exceed INFO_MAX_BETS=5 bets."""

    def test_value_bets_max_5_typical_race(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        assert len(result["value_bets"]) <= 5, (
            f"INFO layer exceeded 5 bets: got {len(result['value_bets'])}"
        )

    def test_value_bets_max_5_large_field(self):
        """16-horse race provides many EV+ candidates; still capped at 5."""
        preds = _make_predictions([75.0, 68.0, 60.0, 55.0, 50.0, 45.0,
                                    40.0, 35.0, 30.0, 25.0, 20.0, 15.0,
                                    12.0, 10.0, 8.0, 5.0])
        partners = list(range(2, 17))
        odds = _make_full_odds(1, partners, honmei_tansho_odds=3.0)
        entries = _make_entries_with_odds(16, 3.0)
        result = _run_dual(preds, odds, honmei_odds=3.0, headcount=16, entries=entries)
        assert len(result["value_bets"]) <= 5, (
            f"INFO layer must cap at 5, got {len(result['value_bets'])}"
        )

    def test_info_max_bets_constant_is_5(self):
        from backend.predictor.bet_optimizer import INFO_MAX_BETS
        assert INFO_MAX_BETS == 5

    def test_value_bets_can_be_empty_with_no_ev_plus(self):
        """When no bets have EV>0, value_bets is empty (never negative count)."""
        preds = _make_predictions([75.0, 65.0, 55.0, 48.0, 40.0, 33.0, 25.0, 18.0])
        # Very low odds = no EV+ bets possible
        odds = {
            "tansho": [{"horses": [1], "odds": 1.1, "payout": 110}],
            "umaren": [{"horses": [1, 2], "odds": 1.5, "payout": 150}],
        }
        result = _run_dual(preds, odds, honmei_odds=1.1, mc_samples=100)
        assert isinstance(result["value_bets"], list)
        # All odds are < INFO_MIN_ODDS=2.0, so value_bets should be empty
        assert len(result["value_bets"]) == 0


# ---------------------------------------------------------------------------
# 8. INFO layer bet types
# ---------------------------------------------------------------------------

class TestInfoLayerBetTypes:
    """INFO layer allows all INFO_BET_TYPES; never fukusho/wakuren."""

    def test_info_bet_types_constant(self):
        from backend.predictor.bet_optimizer import INFO_BET_TYPES
        expected = {"tansho", "umaren", "umatan", "wide", "sanrenpuku", "sanrentan"}
        assert INFO_BET_TYPES == expected

    def test_value_bets_only_from_allowed_types(self, confident_predictions, full_odds_buy_range):
        from backend.predictor.bet_optimizer import INFO_BET_TYPES
        result = _run_dual(confident_predictions, full_odds_buy_range)
        for bet in result["value_bets"]:
            assert bet["type"] in INFO_BET_TYPES, (
                f"INFO bet has disallowed type {bet['type']!r}"
            )

    def test_value_bets_no_fukusho(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        types = [b["type"] for b in result["value_bets"]]
        assert "fukusho" not in types, f"fukusho must not appear in INFO layer: {types}"

    def test_value_bets_no_wakuren(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        types = [b["type"] for b in result["value_bets"]]
        assert "wakuren" not in types, f"wakuren must not appear in INFO layer: {types}"

    def test_core_bet_is_not_duplicated_in_value_bets(self, confident_predictions, full_odds_buy_range):
        """The ◎ tansho BUY bet must not appear again in value_bets."""
        result = _run_dual(confident_predictions, full_odds_buy_range, honmei_odds=3.0)
        assert result["layer1_active"]
        core_keys = {(b["type"], tuple(b["horses"])) for b in result["core_bets"]}
        for vb in result["value_bets"]:
            key = (vb["type"], tuple(vb["horses"]))
            assert key not in core_keys, (
                f"Core bet appears in value_bets: {vb}"
            )


# ---------------------------------------------------------------------------
# 9. INFO layer minimum odds filter
# ---------------------------------------------------------------------------

class TestInfoLayerMinOdds:
    """INFO layer must respect INFO_MIN_ODDS=2.0 — no 1.x odds."""

    def test_info_min_odds_constant_is_2(self):
        from backend.predictor.bet_optimizer import INFO_MIN_ODDS
        assert INFO_MIN_ODDS == 2.0

    def test_value_bets_all_odds_above_2(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        for bet in result["value_bets"]:
            assert bet["odds"] >= 2.0, (
                f"INFO bet odds {bet['odds']} is below INFO_MIN_ODDS=2.0: {bet}"
            )

    def test_value_bets_exclude_sub_2_odds_entries(self):
        """Odds data with tansho at 1.8x must not enter INFO layer."""
        preds = _make_predictions([75.0, 65.0, 55.0, 48.0, 40.0, 33.0, 25.0, 18.0])
        odds = {
            "tansho": [{"horses": [1], "odds": 1.8, "payout": 180}],
            "umaren": [
                {"horses": [1, 2], "odds": 10.0, "payout": 1000},
            ],
        }
        result = _run_dual(preds, odds, honmei_odds=1.8, mc_samples=100)
        # 1.8x tansho should be filtered out
        tansho_bets = [b for b in result["value_bets"] if b["type"] == "tansho"]
        for tb in tansho_bets:
            assert tb["odds"] >= 2.0


# ---------------------------------------------------------------------------
# 10. MAX_ODDS_CAP filtering
# ---------------------------------------------------------------------------

class TestMaxOddsCap:
    """Bets with odds exceeding MAX_ODDS_CAP must be excluded."""

    def test_max_odds_cap_constants_exist(self):
        from backend.predictor.bet_optimizer import MAX_ODDS_CAP
        assert "tansho" in MAX_ODDS_CAP
        assert "umaren" in MAX_ODDS_CAP
        assert "umatan" in MAX_ODDS_CAP
        assert "wide" in MAX_ODDS_CAP
        assert "sanrenpuku" in MAX_ODDS_CAP
        assert "sanrentan" in MAX_ODDS_CAP

    def test_tansho_cap_is_200(self):
        from backend.predictor.bet_optimizer import MAX_ODDS_CAP
        assert MAX_ODDS_CAP["tansho"] == 200.0

    def test_umatan_cap_is_1000(self):
        from backend.predictor.bet_optimizer import MAX_ODDS_CAP
        assert MAX_ODDS_CAP["umatan"] == 1000.0

    def test_sanrentan_cap_is_500(self):
        from backend.predictor.bet_optimizer import MAX_ODDS_CAP
        assert MAX_ODDS_CAP["sanrentan"] == 500.0

    def test_value_bets_exclude_over_cap_tansho(self):
        """Tansho at 99999x is beyond MAX_ODDS_CAP[tansho]=200 → excluded."""
        preds = _make_predictions([75.0, 65.0, 55.0, 48.0, 40.0, 33.0, 25.0, 18.0])
        odds = {
            "tansho": [
                {"horses": [1], "odds": 99999.0, "payout": 9999900},
                {"horses": [2], "odds": 8.0, "payout": 800},
            ],
            "umaren": [
                {"horses": [1, 2], "odds": 10.0, "payout": 1000},
            ],
        }
        result = _run_dual(preds, odds, honmei_odds=99999.0, mc_samples=100)
        tansho_bets = [b for b in result["value_bets"] if b["type"] == "tansho"
                       and b["horses"] == [1]]
        # The 99999x tansho for horse 1 should not appear
        for tb in tansho_bets:
            assert tb["odds"] <= 200.0, (
                f"Tansho with odds {tb['odds']} exceeds cap 200"
            )

    def test_value_bets_exclude_over_cap_umatan(self):
        """Umatan at 1001x exceeds cap → excluded."""
        preds = _make_predictions([75.0, 65.0, 55.0, 48.0, 40.0, 33.0, 25.0, 18.0])
        odds = {
            "tansho": [{"horses": [1], "odds": 3.0, "payout": 300}],
            "umatan": [
                {"horses": [1, 2], "odds": 1001.0, "payout": 100100},  # over cap
                {"horses": [1, 3], "odds": 50.0, "payout": 5000},      # under cap
            ],
        }
        result = _run_dual(preds, odds, honmei_odds=3.0, mc_samples=100)
        for bet in result["value_bets"]:
            if bet["type"] == "umatan":
                assert bet["odds"] <= 1000.0, (
                    f"Umatan odds {bet['odds']} exceeds cap 1000"
                )

    def test_no_99999x_odds_in_any_layer(self, confident_predictions, full_odds_buy_range):
        """After normal processing, no bet should ever have 99999x odds."""
        result = _run_dual(confident_predictions, full_odds_buy_range)
        all_bets = result["core_bets"] + result["value_bets"]
        for bet in all_bets:
            assert bet.get("odds", 0) < 9999, (
                f"Unrealistic odds found: {bet.get('odds')} in {bet}"
            )


# ---------------------------------------------------------------------------
# 11. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Boundary conditions and degenerate inputs."""

    def test_empty_predictions_returns_safe_dict(self):
        result = _run_dual([], {}, honmei_odds=3.0, mc_samples=50)
        assert result["core_bets"] == []
        assert result["value_bets"] == []
        assert result["layer1_active"] is False

    def test_single_horse_returns_safe_dict(self):
        preds = _make_predictions([80.0])
        odds = _make_full_odds(1, [], honmei_tansho_odds=3.0)
        result = _run_dual(preds, odds, honmei_odds=3.0, headcount=1,
                           entries=[{"horseNumber": 1, "frameNumber": 1,
                                     "odds": 3.0, "popularity": 1, "isScratched": False}],
                           mc_samples=50)
        assert result["core_bets"] == []
        assert result["value_bets"] == []

    def test_headcount_2_returns_safe_dict(self):
        """headCount < 3 → immediate safe return."""
        preds = _make_predictions([80.0, 60.0])
        odds = _make_full_odds(1, [2], honmei_tansho_odds=3.0)
        from backend.predictor.bet_optimizer import optimize_bets_dual
        race_info = {"raceId": "202608231101", "headCount": 2}
        result = optimize_bets_dual(preds, odds, race_info, mc_samples=50)
        assert result["core_bets"] == []
        assert result["value_bets"] == []
        assert result["layer1_active"] is False

    def test_no_odds_data_returns_empty_layers(self):
        """Empty odds_data → no bets possible."""
        preds = _make_predictions([75.0, 65.0, 55.0, 48.0, 40.0, 33.0, 25.0, 18.0])
        result = _run_dual(preds, {}, honmei_odds=3.0, mc_samples=100)
        assert result["core_bets"] == []
        assert result["value_bets"] == []

    def test_scratched_honmei_second_horse_becomes_honmei(self):
        """If horse 1 is scratched, horse 2 (next highest score) is ◎."""
        preds = [
            {"horseNumber": 1, "score": 80.0, "isScratched": True},   # scratched
            {"horseNumber": 2, "score": 70.0, "isScratched": False},  # becomes ◎
            {"horseNumber": 3, "score": 60.0, "isScratched": False},
            {"horseNumber": 4, "score": 50.0, "isScratched": False},
            {"horseNumber": 5, "score": 40.0, "isScratched": False},
            {"horseNumber": 6, "score": 30.0, "isScratched": False},
        ]
        odds = _make_full_odds(2, [3, 4, 5, 6], honmei_tansho_odds=3.0)
        entries = [
            {"horseNumber": 1, "frameNumber": 1, "odds": None, "popularity": None, "isScratched": True},
            *[{"horseNumber": i, "frameNumber": i, "odds": 3.0 if i == 2 else 8.0,
               "popularity": i - 1, "isScratched": False} for i in range(2, 7)]
        ]
        result = _run_dual(preds, odds, honmei_odds=3.0, headcount=6, entries=entries, mc_samples=100)
        # layer1_active depends on horse 2 as ◎ — score=70>=68, odds=3.0 in [2,4)
        # Whatever the result, scratched horse 1 must not appear in any bet
        for bet in result["core_bets"] + result["value_bets"]:
            assert 1 not in bet["horses"], (
                f"Scratched horse 1 appeared in bet: {bet}"
            )

    def test_small_field_5_horses_returns_valid_structure(self):
        """5-horse field: all return keys present, counts within bounds."""
        preds = _make_predictions([75.0, 65.0, 55.0, 48.0, 40.0])
        odds = _make_full_odds(1, [2, 3, 4, 5], honmei_tansho_odds=3.0)
        result = _run_dual(preds, odds, honmei_odds=3.0, headcount=5,
                           entries=_make_entries_with_odds(5, 3.0), mc_samples=100)
        assert isinstance(result["core_bets"], list)
        assert isinstance(result["value_bets"], list)
        assert len(result["core_bets"]) <= 1
        assert len(result["value_bets"]) <= 5

    def test_all_predictions_same_score_returns_valid_structure(self):
        """Equal scores → valid output, no crash."""
        preds = _make_predictions([50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
        odds = _make_full_odds(1, [2, 3, 4, 5, 6, 7, 8], honmei_tansho_odds=3.0)
        result = _run_dual(preds, odds, mc_samples=100)
        assert isinstance(result, dict)
        assert len(result["core_bets"]) <= 1
        assert len(result["value_bets"]) <= 5

    def test_very_high_score_honmei_no_crash(self):
        """score=100 (max) should not cause divide-by-zero or overflow."""
        preds = _make_predictions([100.0, 50.0, 40.0, 30.0, 20.0, 15.0, 10.0, 5.0])
        odds = _make_full_odds(1, [2, 3, 4, 5, 6, 7, 8], honmei_tansho_odds=2.5)
        result = _run_dual(preds, odds, honmei_odds=2.5, mc_samples=100)
        assert isinstance(result, dict)

    def test_none_entries_does_not_crash(self, confident_predictions, full_odds_buy_range):
        """entries=None should fall back to tansho odds lookup gracefully."""
        from backend.predictor.bet_optimizer import optimize_bets_dual
        race_info = {"raceId": "202608231101", "headCount": 8}
        result = optimize_bets_dual(
            confident_predictions, full_odds_buy_range, race_info,
            entries=None, mc_samples=100
        )
        assert isinstance(result, dict)

    def test_missing_tansho_odds_no_core_bet(self):
        """If ◎ tansho odds are absent, BUY layer cannot activate."""
        preds = _make_predictions([75.0, 65.0, 55.0, 48.0, 40.0, 33.0, 25.0, 18.0])
        odds = {
            "umaren": [{"horses": [1, 2], "odds": 10.0, "payout": 1000}],
            # No tansho entry
        }
        result = _run_dual(preds, odds, honmei_odds=0.0, mc_samples=100)
        # No tansho odds → core_bets must be empty
        assert result["core_bets"] == []


# ---------------------------------------------------------------------------
# 12. layer1_active flag correctness
# ---------------------------------------------------------------------------

class TestLayer1ActiveFlag:
    """layer1_active must mirror the actual BUY layer state exactly."""

    def test_flag_true_implies_core_bets_not_empty(self, confident_predictions, full_odds_buy_range):
        """layer1_active=True → core_bets has exactly 1 bet."""
        result = _run_dual(confident_predictions, full_odds_buy_range, honmei_odds=3.0)
        if result["layer1_active"]:
            assert len(result["core_bets"]) == 1

    def test_flag_false_implies_core_bets_empty(self, confident_predictions, full_odds_outside_range):
        """layer1_active=False → core_bets is empty."""
        result = _run_dual(confident_predictions, full_odds_outside_range, honmei_odds=5.0)
        if not result["layer1_active"]:
            assert result["core_bets"] == []

    def test_honmei_odds_matches_returned_field(self):
        """honmei_odds in return dict should match the actual ◎ odds used."""
        preds = _make_predictions([75.0, 65.0, 55.0, 48.0, 40.0, 33.0, 25.0, 18.0])
        odds = _make_full_odds(1, [2, 3, 4, 5, 6, 7, 8], honmei_tansho_odds=3.5)
        entries = _make_entries_with_odds(8, 3.5)
        result = _run_dual(preds, odds, honmei_odds=3.5, entries=entries)
        assert result["honmei_odds"] == pytest.approx(3.5, abs=0.05), (
            f"Expected honmei_odds≈3.5, got {result['honmei_odds']}"
        )

    def test_odds_exactly_4_gives_false(self):
        """Boundary: 4.0 is exclusive upper bound → layer1_active must be False."""
        preds = _make_predictions([75.0, 65.0, 55.0, 48.0, 40.0, 33.0, 25.0, 18.0])
        odds = _make_full_odds(1, [2, 3, 4, 5, 6, 7, 8], honmei_tansho_odds=4.0)
        entries = _make_entries_with_odds(8, 4.0)
        result = _run_dual(preds, odds, honmei_odds=4.0, entries=entries)
        assert result["layer1_active"] is False
        assert result["core_bets"] == []

    def test_odds_3_99_is_in_range(self):
        """3.99 is inside [2.0, 4.0) → active."""
        preds = _make_predictions([75.0, 65.0, 55.0, 48.0, 40.0, 33.0, 25.0, 18.0])
        odds = _make_full_odds(1, [2, 3, 4, 5, 6, 7, 8], honmei_tansho_odds=3.99)
        entries = _make_entries_with_odds(8, 3.99)
        result = _run_dual(preds, odds, honmei_odds=3.99, entries=entries)
        assert result["layer1_active"] is True

    def test_score_boundary_67_9_gives_false(self):
        """score=67.9 is just below 68 → layer1_active=False."""
        preds = _make_predictions([67.9, 60.0, 50.0, 43.0, 36.0, 29.0, 22.0, 15.0])
        odds = _make_full_odds(1, [2, 3, 4, 5, 6, 7, 8], honmei_tansho_odds=3.0)
        result = _run_dual(preds, odds, honmei_odds=3.0)
        assert result["layer1_active"] is False

    def test_score_boundary_68_gives_true(self):
        """score=68.0 is exactly at threshold → layer1_active=True."""
        preds = _make_predictions([68.0, 60.0, 50.0, 43.0, 36.0, 29.0, 22.0, 15.0])
        odds = _make_full_odds(1, [2, 3, 4, 5, 6, 7, 8], honmei_tansho_odds=3.0)
        result = _run_dual(preds, odds, honmei_odds=3.0)
        assert result["layer1_active"] is True


# ---------------------------------------------------------------------------
# 13. INFO layer required bet fields
# ---------------------------------------------------------------------------

class TestInfoLayerBetFields:
    """Every bet in value_bets must have all standard display fields."""

    def test_all_value_bets_have_type(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        for bet in result["value_bets"]:
            assert "type" in bet

    def test_all_value_bets_have_horses(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        for bet in result["value_bets"]:
            assert "horses" in bet
            assert len(bet["horses"]) >= 1

    def test_all_value_bets_have_odds(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        for bet in result["value_bets"]:
            assert "odds" in bet, f"Missing odds in INFO bet: {bet}"

    def test_all_value_bets_have_bet_size(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        for bet in result["value_bets"]:
            assert "betSize" in bet, f"Missing betSize in INFO bet: {bet}"
            assert bet["betSize"] >= 1

    def test_all_value_bets_have_rank(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        for i, bet in enumerate(result["value_bets"]):
            assert "rank" in bet, f"Bet {i} missing 'rank': {bet}"

    def test_all_value_bets_have_hit_prob(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        for bet in result["value_bets"]:
            assert "hitProb" in bet
            assert 0.0 <= bet["hitProb"] <= 1.0

    def test_all_value_bets_have_type_label(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        for bet in result["value_bets"]:
            assert "typeLabel" in bet, f"Missing typeLabel in INFO bet: {bet}"

    def test_value_bets_tansho_label_is_correct(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        for bet in result["value_bets"]:
            if bet["type"] == "tansho":
                assert bet["typeLabel"] == "単勝"

    def test_value_bets_umaren_label_is_correct(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        for bet in result["value_bets"]:
            if bet["type"] == "umaren":
                assert bet["typeLabel"] == "馬連"

    def test_value_bets_wide_label_is_correct(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        for bet in result["value_bets"]:
            if bet["type"] == "wide":
                assert bet["typeLabel"] == "ワイド"

    def test_value_bets_sanrenpuku_label_is_correct(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        for bet in result["value_bets"]:
            if bet["type"] == "sanrenpuku":
                assert bet["typeLabel"] == "3連複"

    def test_value_bets_sanrentan_label_is_correct(self, confident_predictions, full_odds_buy_range):
        result = _run_dual(confident_predictions, full_odds_buy_range)
        for bet in result["value_bets"]:
            if bet["type"] == "sanrentan":
                assert bet["typeLabel"] == "3連単"


# ---------------------------------------------------------------------------
# 14. Constants validation
# ---------------------------------------------------------------------------

class TestD7Constants:
    """All D7 strategy constants must have the correct values."""

    def test_shoubu_min_score_is_68(self):
        from backend.predictor.bet_optimizer import SHOUBU_MIN_SCORE
        assert SHOUBU_MIN_SCORE == 68.0

    def test_buy_honmei_odds_min_is_2(self):
        from backend.predictor.bet_optimizer import BUY_HONMEI_ODDS_MIN
        assert BUY_HONMEI_ODDS_MIN == 2.0

    def test_buy_honmei_odds_max_is_4(self):
        from backend.predictor.bet_optimizer import BUY_HONMEI_ODDS_MAX
        assert BUY_HONMEI_ODDS_MAX == 4.0

    def test_info_max_bets_is_5(self):
        from backend.predictor.bet_optimizer import INFO_MAX_BETS
        assert INFO_MAX_BETS == 5

    def test_info_min_odds_is_2(self):
        from backend.predictor.bet_optimizer import INFO_MIN_ODDS
        assert INFO_MIN_ODDS == 2.0

    def test_info_bet_types_has_all_6_types(self):
        from backend.predictor.bet_optimizer import INFO_BET_TYPES
        assert len(INFO_BET_TYPES) == 6

    def test_info_bet_types_includes_sanrentan(self):
        from backend.predictor.bet_optimizer import INFO_BET_TYPES
        assert "sanrentan" in INFO_BET_TYPES

    def test_info_bet_types_includes_sanrenpuku(self):
        from backend.predictor.bet_optimizer import INFO_BET_TYPES
        assert "sanrenpuku" in INFO_BET_TYPES

    def test_info_bet_types_excludes_fukusho(self):
        from backend.predictor.bet_optimizer import INFO_BET_TYPES
        assert "fukusho" not in INFO_BET_TYPES

    def test_info_bet_types_excludes_wakuren(self):
        from backend.predictor.bet_optimizer import INFO_BET_TYPES
        assert "wakuren" not in INFO_BET_TYPES

    def test_kelly_fraction_is_half(self):
        from backend.predictor.bet_optimizer import KELLY_FRACTION
        assert KELLY_FRACTION == 0.5


# ---------------------------------------------------------------------------
# 15. Integration: full race simulation
# ---------------------------------------------------------------------------

class TestIntegration:
    """Full-race integration tests verifying D7 strategy end-to-end."""

    def test_typical_shoubu_race_structure(self, confident_predictions, full_odds_buy_range):
        """Full typical race: layer1 active, 1 core bet, up to 5 info bets."""
        result = _run_dual(confident_predictions, full_odds_buy_range, honmei_odds=3.0)
        assert result["layer1_active"] is True
        assert len(result["core_bets"]) == 1
        assert result["core_bets"][0]["type"] == "tansho"
        assert len(result["value_bets"]) <= 5
        assert all(b["ev"] > 0 for b in result["value_bets"])

    def test_weak_race_no_buy_but_has_info(self, weak_predictions, full_odds_buy_range):
        """Weak ◎ (score<68): BUY layer silent, INFO layer still provides picks."""
        result = _run_dual(weak_predictions, full_odds_buy_range, honmei_odds=3.0)
        assert result["layer1_active"] is False
        assert result["core_bets"] == []
        # INFO layer still runs regardless of BUY gate
        assert isinstance(result["value_bets"], list)

    def test_high_odds_race_no_buy_but_has_info(self, confident_predictions, full_odds_outside_range):
        """High odds ◎ (odds>4x): BUY layer silent, INFO layer still runs."""
        result = _run_dual(confident_predictions, full_odds_outside_range, honmei_odds=5.0)
        assert result["layer1_active"] is False
        assert result["core_bets"] == []
        assert isinstance(result["value_bets"], list)

    def test_total_bets_never_exceed_6(self, confident_predictions, full_odds_buy_range):
        """core_bets (max 1) + value_bets (max 5) = max 6 total."""
        result = _run_dual(confident_predictions, full_odds_buy_range)
        total = len(result["core_bets"]) + len(result["value_bets"])
        assert total <= 6, f"Total bets exceeded 6: {total}"

    def test_pattern_is_one_of_valid_values(self, confident_predictions, full_odds_buy_range):
        """Pattern must be one of the defined race patterns."""
        valid_patterns = {"本命堅軸", "混戦模様", "2強対決", "標準配置", "少頭数", ""}
        result = _run_dual(confident_predictions, full_odds_buy_range)
        assert result["pattern"] in valid_patterns, (
            f"Unexpected pattern value: {result['pattern']!r}"
        )

    def test_longshot_is_none_or_dict(self, confident_predictions, full_odds_buy_range):
        """Longshot must be None or a dict (never some other type)."""
        result = _run_dual(confident_predictions, full_odds_buy_range)
        assert result["longshot"] is None or isinstance(result["longshot"], dict)

    def test_value_bets_do_not_include_impossible_odds(self, confident_predictions, full_odds_buy_range):
        """No value_bet should have odds=0 or negative."""
        result = _run_dual(confident_predictions, full_odds_buy_range)
        for bet in result["value_bets"]:
            assert bet["odds"] > 0

    def test_repeated_calls_are_deterministic(self, confident_predictions, full_odds_buy_range):
        """Same inputs → same output (seeded RNG)."""
        r1 = _run_dual(confident_predictions, full_odds_buy_range, mc_samples=500)
        r2 = _run_dual(confident_predictions, full_odds_buy_range, mc_samples=500)
        assert r1["layer1_active"] == r2["layer1_active"]
        assert len(r1["core_bets"]) == len(r2["core_bets"])
        assert len(r1["value_bets"]) == len(r2["value_bets"])


# ---------------------------------------------------------------------------
# 16. Unit tests for helper functions (coverage gap closers)
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    """Direct unit tests for lower-level helper functions to boost coverage."""

    # ── monte_carlo_finish ──

    def test_monte_carlo_finish_returns_n_samples(self):
        from backend.predictor.bet_optimizer import monte_carlo_finish
        probs = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}
        finishes = monte_carlo_finish(probs, n_samples=100)
        assert len(finishes) == 100

    def test_monte_carlo_finish_each_sample_max_3_horses(self):
        from backend.predictor.bet_optimizer import monte_carlo_finish
        probs = {1: 0.5, 2: 0.3, 3: 0.2}
        finishes = monte_carlo_finish(probs, n_samples=50)
        for f in finishes:
            assert len(f) <= 3

    def test_monte_carlo_finish_only_valid_horse_numbers(self):
        from backend.predictor.bet_optimizer import monte_carlo_finish
        probs = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}
        finishes = monte_carlo_finish(probs, n_samples=50)
        valid = set(probs.keys())
        for f in finishes:
            for h in f:
                assert h in valid

    def test_monte_carlo_finish_no_duplicates_in_sample(self):
        from backend.predictor.bet_optimizer import monte_carlo_finish
        probs = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}
        finishes = monte_carlo_finish(probs, n_samples=50)
        for f in finishes:
            assert len(f) == len(set(f)), f"Duplicate horses in finish: {f}"

    def test_monte_carlo_uses_provided_rng(self):
        """Providing the same seeded rng gives deterministic output."""
        import random
        from backend.predictor.bet_optimizer import monte_carlo_finish
        probs = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}
        rng1 = random.Random(99)
        rng2 = random.Random(99)
        f1 = monte_carlo_finish(probs, n_samples=30, rng=rng1)
        f2 = monte_carlo_finish(probs, n_samples=30, rng=rng2)
        assert f1 == f2

    def test_monte_carlo_2_horse_field(self):
        """2-horse field: each finish has at most 2 horses."""
        from backend.predictor.bet_optimizer import monte_carlo_finish
        probs = {1: 0.6, 2: 0.4}
        finishes = monte_carlo_finish(probs, n_samples=20)
        for f in finishes:
            assert len(f) <= 2

    # ── estimate_hit_probabilities ──

    def test_estimate_hit_prob_empty_finishes_sets_zero(self):
        """Empty finishes list → hitProb=0.0 for all candidates."""
        from backend.predictor.bet_optimizer import estimate_hit_probabilities
        candidates = [
            {"type": "tansho", "horses": [1], "ordered": False},
            {"type": "umaren", "horses": [1, 2], "ordered": False},
        ]
        result = estimate_hit_probabilities([], candidates)
        for c in result:
            assert c["hitProb"] == 0.0

    def test_estimate_hit_prob_tansho_exact_winner(self):
        """Tansho hitProb ≈ 1.0 when horse always wins."""
        from backend.predictor.bet_optimizer import estimate_hit_probabilities
        finishes = [[1, 2, 3]] * 100
        candidate = {"type": "tansho", "horses": [1], "ordered": False}
        result = estimate_hit_probabilities(finishes, [candidate])
        assert result[0]["hitProb"] == pytest.approx(1.0)

    def test_estimate_hit_prob_tansho_never_wins(self):
        """Tansho hitProb = 0.0 when horse never wins."""
        from backend.predictor.bet_optimizer import estimate_hit_probabilities
        finishes = [[2, 1, 3]] * 100
        candidate = {"type": "tansho", "horses": [1], "ordered": False}
        result = estimate_hit_probabilities(finishes, [candidate])
        assert result[0]["hitProb"] == 0.0

    def test_estimate_hit_prob_fukusho_top3(self):
        """Fukusho: horse in top 3 counts as hit."""
        from backend.predictor.bet_optimizer import estimate_hit_probabilities
        # Horse 1 always in position 3 (index 2)
        finishes = [[2, 3, 1]] * 100
        candidate = {"type": "fukusho", "horses": [1], "ordered": False}
        result = estimate_hit_probabilities(finishes, [candidate])
        assert result[0]["hitProb"] == pytest.approx(1.0)

    def test_estimate_hit_prob_umaren_set_match(self):
        """Umaren: unordered pair must be in top 2."""
        from backend.predictor.bet_optimizer import estimate_hit_probabilities
        # Finish [1,2,...] → set([1,2]) == set([1,2])
        finishes = [[1, 2, 3]] * 50 + [[2, 1, 3]] * 50
        candidate = {"type": "umaren", "horses": [1, 2], "ordered": False}
        result = estimate_hit_probabilities(finishes, [candidate])
        assert result[0]["hitProb"] == pytest.approx(1.0)

    def test_estimate_hit_prob_umatan_ordered(self):
        """Umatan: ordered pair [1,2] only hits when finish[:2]==[1,2]."""
        from backend.predictor.bet_optimizer import estimate_hit_probabilities
        finishes = [[1, 2, 3]] * 60 + [[2, 1, 3]] * 40
        candidate = {"type": "umatan", "horses": [1, 2], "ordered": True}
        result = estimate_hit_probabilities(finishes, [candidate])
        assert result[0]["hitProb"] == pytest.approx(0.60)

    def test_estimate_hit_prob_wide_top3_subset(self):
        """Wide: pair must both be in top 3."""
        from backend.predictor.bet_optimizer import estimate_hit_probabilities
        finishes = [[1, 2, 3]] * 100
        candidate = {"type": "wide", "horses": [1, 3], "ordered": False}
        result = estimate_hit_probabilities(finishes, [candidate])
        assert result[0]["hitProb"] == pytest.approx(1.0)

    def test_estimate_hit_prob_sanrenpuku_set_match(self):
        """Sanrenpuku: set of 3 must match top 3."""
        from backend.predictor.bet_optimizer import estimate_hit_probabilities
        finishes = [[1, 2, 3]] * 100
        candidate = {"type": "sanrenpuku", "horses": [3, 1, 2], "ordered": False}
        result = estimate_hit_probabilities(finishes, [candidate])
        assert result[0]["hitProb"] == pytest.approx(1.0)

    def test_estimate_hit_prob_sanrentan_ordered(self):
        """Sanrentan: exact ordered triple in top 3."""
        from backend.predictor.bet_optimizer import estimate_hit_probabilities
        finishes = [[1, 2, 3]] * 70 + [[1, 3, 2]] * 30
        candidate = {"type": "sanrentan", "horses": [1, 2, 3], "ordered": True}
        result = estimate_hit_probabilities(finishes, [candidate])
        assert result[0]["hitProb"] == pytest.approx(0.70)

    # ── implied_fair_odds ──

    def test_implied_fair_odds_positive_prob(self):
        from backend.predictor.bet_optimizer import implied_fair_odds
        # fair = (1/0.5) * (1 - 0.25) = 2.0 * 0.75 = 1.5
        result = implied_fair_odds(0.5)
        assert result == pytest.approx(1.5)

    def test_implied_fair_odds_zero_prob_returns_1(self):
        from backend.predictor.bet_optimizer import implied_fair_odds
        result = implied_fair_odds(0.0)
        assert result == 1.0

    def test_implied_fair_odds_negative_prob_returns_1(self):
        from backend.predictor.bet_optimizer import implied_fair_odds
        result = implied_fair_odds(-0.1)
        assert result == 1.0

    # ── scores_to_probabilities ──

    def test_scores_to_probs_sum_to_1(self):
        from backend.predictor.bet_optimizer import scores_to_probabilities
        preds = _make_predictions([80.0, 60.0, 50.0, 40.0, 30.0])
        probs = scores_to_probabilities(preds, head_count=5)
        assert sum(probs.values()) == pytest.approx(1.0, abs=0.001)

    def test_scores_to_probs_highest_score_gets_highest_prob(self):
        from backend.predictor.bet_optimizer import scores_to_probabilities
        preds = _make_predictions([80.0, 60.0, 50.0, 40.0, 30.0])
        probs = scores_to_probabilities(preds, head_count=5)
        assert probs[1] == max(probs.values())

    def test_scores_to_probs_excludes_scratched(self):
        from backend.predictor.bet_optimizer import scores_to_probabilities
        preds = [
            {"horseNumber": 1, "score": 80.0, "isScratched": False},
            {"horseNumber": 2, "score": 60.0, "isScratched": True},
            {"horseNumber": 3, "score": 50.0, "isScratched": False},
        ]
        probs = scores_to_probabilities(preds, head_count=3)
        assert 2 not in probs

    def test_scores_to_probs_empty_returns_empty(self):
        from backend.predictor.bet_optimizer import scores_to_probabilities
        probs = scores_to_probabilities([], head_count=8)
        assert probs == {}

    def test_scores_to_probs_all_zero_returns_empty(self):
        from backend.predictor.bet_optimizer import scores_to_probabilities
        preds = _make_predictions([0.0, 0.0, 0.0])
        probs = scores_to_probabilities(preds, head_count=3)
        assert probs == {}

    # ── pick_longshot ──

    def test_pick_longshot_returns_none_when_no_candidates(self):
        from backend.predictor.bet_optimizer import pick_longshot
        result = pick_longshot([], [], {1: 0.4, 2: 0.3, 3: 0.3})
        assert result is None

    def test_pick_longshot_ignores_already_selected_bets(self):
        from backend.predictor.bet_optimizer import pick_longshot
        cand = {
            "type": "umatan", "horses": [1, 2], "odds": 50.0,
            "hitProb": 0.03, "ev": 0.5,
        }
        already_selected = [{"type": "umatan", "horses": [1, 2]}]
        result = pick_longshot([cand], already_selected, {1: 0.4, 2: 0.3, 3: 0.3})
        # Already selected → must not be returned as longshot
        assert result is None

    def test_pick_longshot_skips_low_odds(self):
        """Odds below MIN_LONGSHOT_ODDS=20 should be ignored."""
        from backend.predictor.bet_optimizer import pick_longshot
        cand = {
            "type": "umaren", "horses": [1, 2], "odds": 15.0,
            "hitProb": 0.10, "ev": 0.5,
        }
        result = pick_longshot([cand], [], {1: 0.4, 2: 0.4, 3: 0.2})
        assert result is None

    def test_pick_longshot_skips_very_low_hit_prob(self):
        """hitProb < 0.005 must be excluded."""
        from backend.predictor.bet_optimizer import pick_longshot
        cand = {
            "type": "sanrentan", "horses": [1, 2, 3], "odds": 80.0,
            "hitProb": 0.001, "ev": -0.92,
        }
        result = pick_longshot([cand], [], {1: 0.4, 2: 0.3, 3: 0.3})
        assert result is None

    # ── detect_race_pattern ──

    def test_detect_race_pattern_honmei_strong(self):
        """Large gap between 1st and 2nd → 本命堅軸."""
        from backend.predictor.bet_optimizer import detect_race_pattern
        # gap_1_2 > 0.10
        probs = {1: 0.60, 2: 0.20, 3: 0.10, 4: 0.10}
        assert detect_race_pattern(probs) == "本命堅軸"

    def test_detect_race_pattern_chaotic(self):
        """Tight spread → 混戦模様."""
        from backend.predictor.bet_optimizer import detect_race_pattern
        # spread < 0.06
        probs = {1: 0.34, 2: 0.33, 3: 0.33}
        assert detect_race_pattern(probs) == "混戦模様"

    def test_detect_race_pattern_small_field(self):
        """Field with fewer than 3 horses → 少頭数."""
        from backend.predictor.bet_optimizer import detect_race_pattern
        probs = {1: 0.6, 2: 0.4}
        assert detect_race_pattern(probs) == "少頭数"

    def test_detect_race_pattern_returns_string(self):
        from backend.predictor.bet_optimizer import detect_race_pattern
        probs = {1: 0.35, 2: 0.30, 3: 0.20, 4: 0.15}
        result = detect_race_pattern(probs)
        assert isinstance(result, str)
        assert len(result) > 0

    # ── find_odds_for_bet ──

    def test_find_odds_for_bet_returns_none_when_no_match(self):
        from backend.predictor.bet_optimizer import find_odds_for_bet
        bet = {"type": "umaren", "horses": [1, 2], "ordered": False}
        result = find_odds_for_bet(bet, {"umaren": [{"horses": [3, 4], "odds": 10.0, "payout": 1000}]})
        assert result is None

    def test_find_odds_for_bet_returns_none_for_empty_odds(self):
        from backend.predictor.bet_optimizer import find_odds_for_bet
        bet = {"type": "umaren", "horses": [1, 2], "ordered": False}
        assert find_odds_for_bet(bet, {}) is None

    def test_find_odds_for_bet_ordered_match(self):
        from backend.predictor.bet_optimizer import find_odds_for_bet
        bet = {"type": "umatan", "horses": [1, 2], "ordered": True}
        odds_data = {"umatan": [{"horses": [1, 2], "odds": 15.0, "payout": 1500}]}
        result = find_odds_for_bet(bet, odds_data)
        assert result is not None
        assert result["odds"] == 15.0

    def test_find_odds_for_bet_unordered_match(self):
        from backend.predictor.bet_optimizer import find_odds_for_bet
        bet = {"type": "umaren", "horses": [2, 1], "ordered": False}
        odds_data = {"umaren": [{"horses": [1, 2], "odds": 8.0, "payout": 800}]}
        result = find_odds_for_bet(bet, odds_data)
        assert result is not None
        assert result["odds"] == 8.0

    def test_find_odds_for_bet_preserves_odds_min_max(self):
        from backend.predictor.bet_optimizer import find_odds_for_bet
        bet = {"type": "fukusho", "horses": [1], "ordered": False}
        odds_data = {"fukusho": [{"horses": [1], "odds": 2.5, "payout": 250,
                                   "oddsMin": 2.0, "oddsMax": 3.0}]}
        result = find_odds_for_bet(bet, odds_data)
        assert result["oddsMin"] == 2.0
        assert result["oddsMax"] == 3.0
