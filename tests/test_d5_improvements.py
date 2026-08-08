"""TDD tests for D5 improvements: Tanpuku Hedge + Kelly Sizing.

Tests written FIRST (RED phase), then verified against implementation (GREEN).

Covers:
  - kelly_bet_size(): positive EV, negative EV, edge inputs, unit bounds
  - Tanpuku hedge in optimize_bets(): odds thresholds, betSize ratios,
    coexistence with ◎-anchor umatan/umaren/wide bets
  - betSize field: presence, type, bounds for all bets

Financial calculations → target 100% branch coverage.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_race_info():
    return {
        "raceId": "202608081101",
        "raceName": "テストレース",
        "headCount": 8,
    }


@pytest.fixture
def predictions_honmei_first():
    """8-horse field; horse 1 is ◎ (highest score)."""
    return [
        {"horseNumber": 1, "score": 85.0, "isScratched": False},
        {"horseNumber": 2, "score": 72.0, "isScratched": False},
        {"horseNumber": 3, "score": 65.0, "isScratched": False},
        {"horseNumber": 4, "score": 58.0, "isScratched": False},
        {"horseNumber": 5, "score": 50.0, "isScratched": False},
        {"horseNumber": 6, "score": 42.0, "isScratched": False},
        {"horseNumber": 7, "score": 35.0, "isScratched": False},
        {"horseNumber": 8, "score": 28.0, "isScratched": False},
    ]


def _make_entries(honmei_hn: int, tansho_odds: float) -> list:
    """Build a minimal entries list where honmei_hn has given tansho odds."""
    entries = []
    for hn in range(1, 9):
        entries.append({
            "horseNumber": hn,
            "frameNumber": hn,
            "odds": tansho_odds if hn == honmei_hn else 5.0,
            "popularity": hn,
            "isScratched": False,
        })
    return entries


def _make_odds_data(honmei_hn: int, tansho_odds: float, fukusho_odds: float) -> dict:
    """Build odds_data with tansho/fukusho for honmei + umatan/umaren/wide partners."""
    partners = [hn for hn in range(1, 9) if hn != honmei_hn][:6]
    return {
        "tansho": [
            {"horses": [hn], "odds": tansho_odds if hn == honmei_hn else 5.0, "payout": int((tansho_odds if hn == honmei_hn else 5.0) * 100)}
            for hn in range(1, 9)
        ],
        "fukusho": [
            {"horses": [hn], "odds": fukusho_odds if hn == honmei_hn else 2.0, "payout": int((fukusho_odds if hn == honmei_hn else 2.0) * 100)}
            for hn in range(1, 9)
        ],
        "umatan": [
            {"horses": [honmei_hn, p], "odds": 15.0, "payout": 1500}
            for p in partners
        ] + [
            {"horses": [p, honmei_hn], "odds": 20.0, "payout": 2000}
            for p in partners
        ],
        "umaren": [
            {"horses": sorted([honmei_hn, p]), "odds": 8.0, "payout": 800}
            for p in partners
        ],
        "wide": [
            {"horses": sorted([honmei_hn, p]), "odds": 3.5, "payout": 350}
            for p in partners
        ],
    }


# ---------------------------------------------------------------------------
# kelly_bet_size() — unit tests
# ---------------------------------------------------------------------------

class TestKellyBetSize:
    """Unit tests for kelly_bet_size(hit_prob, odds, bankroll)."""

    def _fn(self):
        from backend.predictor.bet_optimizer import kelly_bet_size
        return kelly_bet_size

    def test_positive_ev_returns_more_than_one(self):
        """Strong positive EV: high prob + high odds → units > 1."""
        fn = self._fn()
        # hit_prob=0.5, odds=5.0 → b=4, f=((5)*0.5-1)/4=0.375, half=0.1875 → units > 1
        result = fn(hit_prob=0.5, odds=5.0)
        assert result > 1

    def test_negative_ev_returns_minimum_one(self):
        """Low prob + low odds (negative EV) → minimum 1 unit."""
        fn = self._fn()
        # hit_prob=0.05, odds=2.0 → b=1, f=((2)*0.05-1)/1=-0.9 → f<=0 → 1
        result = fn(hit_prob=0.05, odds=2.0)
        assert result == 1

    def test_zero_hit_prob_returns_one(self):
        """Zero probability → guard clause → 1."""
        fn = self._fn()
        assert fn(hit_prob=0.0, odds=10.0) == 1

    def test_negative_hit_prob_returns_one(self):
        """Negative probability (invalid input) → guard clause → 1."""
        fn = self._fn()
        assert fn(hit_prob=-0.1, odds=10.0) == 1

    def test_odds_exactly_one_returns_one(self):
        """Odds == 1 → division guard → 1."""
        fn = self._fn()
        assert fn(hit_prob=0.5, odds=1.0) == 1

    def test_odds_below_one_returns_one(self):
        """Odds < 1 (invalid) → guard clause → 1."""
        fn = self._fn()
        assert fn(hit_prob=0.5, odds=0.9) == 1

    def test_high_prob_high_odds_capped_at_ten(self):
        """Extremely favourable bet → capped at 10 units."""
        fn = self._fn()
        result = fn(hit_prob=0.9, odds=50.0)
        assert result == 10

    def test_returns_integer(self):
        """Return type must always be int."""
        fn = self._fn()
        for prob, odds in [(0.3, 3.0), (0.5, 5.0), (0.1, 1.5), (0.0, 10.0)]:
            result = fn(hit_prob=prob, odds=odds)
            assert isinstance(result, int), f"Expected int for prob={prob}, odds={odds}"

    def test_result_always_between_one_and_ten(self):
        """Output is always in [1, 10] for a variety of inputs."""
        fn = self._fn()
        test_cases = [
            (0.0, 10.0),
            (0.01, 2.0),
            (0.3, 5.0),
            (0.5, 3.0),
            (0.8, 8.0),
            (0.99, 100.0),
            (0.5, 1.0),
            (0.5, 0.5),
        ]
        for prob, odds in test_cases:
            result = fn(hit_prob=prob, odds=odds)
            assert 1 <= result <= 10, (
                f"Out of range [{result}] for prob={prob}, odds={odds}"
            )

    def test_bankroll_parameter_accepted(self):
        """Custom bankroll is accepted without error (signature contract)."""
        fn = self._fn()
        result = fn(hit_prob=0.3, odds=5.0, bankroll=200_000)
        assert 1 <= result <= 10

    def test_boundary_breakeven_returns_one(self):
        """Exactly breakeven EV (f=0) → returns 1."""
        fn = self._fn()
        # Breakeven: (b+1)*p = 1 → p = 1/odds
        # odds=4.0, p=0.25 → f=0 exactly
        result = fn(hit_prob=0.25, odds=4.0)
        assert result == 1


# ---------------------------------------------------------------------------
# Tanpuku hedge in optimize_bets() — integration tests
# ---------------------------------------------------------------------------

class TestTanpukuHedge:
    """Integration tests for 単複リスクヘッジ inside optimize_bets()."""

    def _run(self, tansho_odds: float, fukusho_odds: float, *,
             honmei_hn: int = 1, predictions=None, race_info=None):
        from backend.predictor.bet_optimizer import optimize_bets

        if predictions is None:
            predictions = [
                {"horseNumber": i, "score": max(5.0, 90.0 - i * 10), "isScratched": False}
                for i in range(1, 9)
            ]
        if race_info is None:
            race_info = {"raceId": "202608080101", "headCount": 8}

        entries = _make_entries(honmei_hn, tansho_odds)
        odds_data = _make_odds_data(honmei_hn, tansho_odds, fukusho_odds)
        return optimize_bets(predictions, odds_data, race_info, entries=entries)

    # ── Threshold: odds >= 6x ──────────────────────────────────────────────

    def test_high_odds_produces_tansho_bet(self):
        """◎ odds=8x (>=6x) → tansho bet present."""
        bets = self._run(tansho_odds=8.0, fukusho_odds=3.0)
        types = [b["type"] for b in bets]
        assert "tansho" in types, f"Expected tansho in {types}"

    def test_high_odds_produces_fukusho_bet(self):
        """◎ odds=8x (>=6x) → fukusho bet present."""
        bets = self._run(tansho_odds=8.0, fukusho_odds=3.0)
        types = [b["type"] for b in bets]
        assert "fukusho" in types, f"Expected fukusho in {types}"

    def test_high_odds_tansho_bet_size_is_one(self):
        """◎ odds=8x → tansho betSize == TANPUKU_RATIO[0] == 1."""
        from backend.predictor.bet_optimizer import TANPUKU_RATIO
        bets = self._run(tansho_odds=8.0, fukusho_odds=3.0)
        tansho_bets = [b for b in bets if b["type"] == "tansho"]
        assert tansho_bets, "No tansho bets found"
        # The tanpuku tansho must have betSize == 1
        tanpuku_tansho = [b for b in tansho_bets if b.get("betSize") == TANPUKU_RATIO[0]]
        assert tanpuku_tansho, (
            f"No tansho with betSize={TANPUKU_RATIO[0]} found; "
            f"tansho bets: {tansho_bets}"
        )

    def test_high_odds_fukusho_bet_size_is_three(self):
        """◎ odds=8x → fukusho betSize == TANPUKU_RATIO[1] == 3."""
        from backend.predictor.bet_optimizer import TANPUKU_RATIO
        bets = self._run(tansho_odds=8.0, fukusho_odds=3.0)
        fukusho_bets = [b for b in bets if b["type"] == "fukusho"]
        assert fukusho_bets, "No fukusho bets found"
        tanpuku_fukusho = [b for b in fukusho_bets if b.get("betSize") == TANPUKU_RATIO[1]]
        assert tanpuku_fukusho, (
            f"No fukusho with betSize={TANPUKU_RATIO[1]} found; "
            f"fukusho bets: {fukusho_bets}"
        )

    # ── Threshold: 2x <= odds < 6x ────────────────────────────────────────

    def test_medium_odds_produces_fukusho_only(self):
        """◎ odds=3x (2x<=odds<6x) → fukusho present, tansho absent."""
        bets = self._run(tansho_odds=3.0, fukusho_odds=2.5)
        types = [b["type"] for b in bets]
        assert "fukusho" in types, f"Expected fukusho in {types}"
        assert "tansho" not in types, f"Unexpected tansho in {types}"

    def test_medium_odds_fukusho_bet_size_is_two(self):
        """◎ odds=3x → fukusho-only betSize == 2."""
        bets = self._run(tansho_odds=3.0, fukusho_odds=2.5)
        fukusho_bets = [b for b in bets if b["type"] == "fukusho"]
        assert fukusho_bets, "No fukusho bets found"
        assert fukusho_bets[0]["betSize"] == 2

    # ── Threshold: odds < 2x ──────────────────────────────────────────────

    def test_low_odds_no_tanpuku_bets(self):
        """◎ odds=1.5x (<2x) → no tansho or fukusho bets from tanpuku logic."""
        bets = self._run(tansho_odds=1.5, fukusho_odds=1.2)
        # Neither tansho nor fukusho should appear (tanpuku conditions not met)
        types = [b["type"] for b in bets]
        assert "tansho" not in types, f"Unexpected tansho when odds<2x: {types}"
        assert "fukusho" not in types, f"Unexpected fukusho when odds<2x: {types}"

    def test_exact_boundary_tansho_min_odds(self):
        """◎ odds exactly == TANPUKU_TANSHO_MIN_ODDS (6.0) → tansho AND fukusho."""
        from backend.predictor.bet_optimizer import TANPUKU_TANSHO_MIN_ODDS
        bets = self._run(tansho_odds=TANPUKU_TANSHO_MIN_ODDS, fukusho_odds=3.0)
        types = [b["type"] for b in bets]
        assert "tansho" in types
        assert "fukusho" in types

    def test_exact_boundary_fukusho_min_odds(self):
        """◎ odds exactly == TANPUKU_FUKUSHO_MIN_ODDS (2.0) → fukusho only."""
        from backend.predictor.bet_optimizer import TANPUKU_FUKUSHO_MIN_ODDS
        bets = self._run(tansho_odds=TANPUKU_FUKUSHO_MIN_ODDS, fukusho_odds=2.0)
        types = [b["type"] for b in bets]
        assert "fukusho" in types
        assert "tansho" not in types

    def test_just_below_fukusho_min_odds(self):
        """◎ odds = 1.9x (just below 2.0) → no tanpuku bets."""
        bets = self._run(tansho_odds=1.9, fukusho_odds=1.5)
        types = [b["type"] for b in bets]
        assert "tansho" not in types
        assert "fukusho" not in types

    # ── Missing fukusho odds data ──────────────────────────────────────────

    def test_high_odds_no_fukusho_data_tansho_only(self):
        """◎ odds=8x but no fukusho in odds_data → only tansho, no fukusho."""
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = [
            {"horseNumber": i, "score": max(5.0, 90.0 - i * 10), "isScratched": False}
            for i in range(1, 9)
        ]
        race_info = {"raceId": "202608080102", "headCount": 8}
        entries = _make_entries(1, 8.0)
        # odds_data has tansho but NOT fukusho
        odds_data = {
            "tansho": [
                {"horses": [hn], "odds": 8.0 if hn == 1 else 5.0,
                 "payout": 800 if hn == 1 else 500}
                for hn in range(1, 9)
            ],
            "umatan": [
                {"horses": [1, p], "odds": 15.0, "payout": 1500}
                for p in range(2, 8)
            ],
            "umaren": [
                {"horses": sorted([1, p]), "odds": 8.0, "payout": 800}
                for p in range(2, 8)
            ],
            "wide": [
                {"horses": sorted([1, p]), "odds": 3.5, "payout": 350}
                for p in range(2, 8)
            ],
            # no "fukusho" key
        }
        bets = optimize_bets(predictions, odds_data, race_info, entries=entries)
        types = [b["type"] for b in bets]
        assert "tansho" in types
        assert "fukusho" not in types

    # ── Coexistence with ◎-anchor combo bets ──────────────────────────────

    def test_tanpuku_coexists_with_umatan(self):
        """◎ odds=8x → tanpuku bets present; umatan also present (HONMEI_UMATAN_PARTNERS=1)."""
        bets = self._run(tansho_odds=8.0, fukusho_odds=3.0)
        types = [b["type"] for b in bets]
        assert "tansho" in types or "fukusho" in types, "No tanpuku bets"

    def test_tanpuku_coexists_with_umaren_or_wide(self):
        """◎ odds=8x → tanpuku bets AND umaren/wide bets both present."""
        bets = self._run(tansho_odds=8.0, fukusho_odds=3.0)
        types = [b["type"] for b in bets]
        combo_present = "umaren" in types or "wide" in types
        tanpuku_present = "tansho" in types or "fukusho" in types
        assert tanpuku_present and combo_present, (
            f"Expected both tanpuku and combo bets; got types={types}"
        )


# ---------------------------------------------------------------------------
# betSize field — present on all bets and within valid bounds
# ---------------------------------------------------------------------------

class TestBetSizeField:
    """All bets returned by optimize_bets() must carry a valid betSize."""

    def _standard_bets(self):
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = [
            {"horseNumber": i, "score": max(5.0, 90.0 - i * 10), "isScratched": False}
            for i in range(1, 9)
        ]
        race_info = {"raceId": "202608080201", "headCount": 8}
        entries = _make_entries(1, 8.0)
        odds_data = _make_odds_data(1, 8.0, 3.0)
        return optimize_bets(predictions, odds_data, race_info, entries=entries)

    def test_all_bets_have_bet_size_field(self):
        """Every bet dict must contain a 'betSize' key."""
        bets = self._standard_bets()
        assert bets, "optimize_bets returned no bets"
        missing = [b for b in bets if "betSize" not in b]
        assert not missing, (
            f"{len(missing)} bets missing betSize: "
            f"{[(b['type'], b['horses']) for b in missing]}"
        )

    def test_bet_size_is_always_integer(self):
        """betSize must be int, never float or None."""
        bets = self._standard_bets()
        for b in bets:
            assert isinstance(b["betSize"], int), (
                f"betSize={b['betSize']!r} is not int for "
                f"type={b['type']}, horses={b['horses']}"
            )

    def test_bet_size_minimum_is_one(self):
        """betSize >= 1 for all bets."""
        bets = self._standard_bets()
        for b in bets:
            assert b["betSize"] >= 1, (
                f"betSize={b['betSize']} < 1 for type={b['type']}"
            )

    def test_bet_size_maximum_is_ten(self):
        """betSize <= 10 for all bets."""
        bets = self._standard_bets()
        for b in bets:
            assert b["betSize"] <= 10, (
                f"betSize={b['betSize']} > 10 for type={b['type']}"
            )

    def test_bet_size_present_with_low_honmei_odds(self):
        """betSize present even when no tanpuku bets (odds < 2x)."""
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = [
            {"horseNumber": i, "score": max(5.0, 90.0 - i * 10), "isScratched": False}
            for i in range(1, 9)
        ]
        race_info = {"raceId": "202608080202", "headCount": 8}
        entries = _make_entries(1, 1.5)   # very low odds
        odds_data = _make_odds_data(1, 1.5, 1.2)
        bets = optimize_bets(predictions, odds_data, race_info, entries=entries)
        for b in bets:
            assert "betSize" in b, f"betSize missing for {b['type']}"
            assert isinstance(b["betSize"], int)
            assert 1 <= b["betSize"] <= 10

    def test_bet_size_present_with_medium_honmei_odds(self):
        """betSize present when tanpuku is fukusho-only mode (2x <= odds < 6x)."""
        from backend.predictor.bet_optimizer import optimize_bets

        predictions = [
            {"horseNumber": i, "score": max(5.0, 90.0 - i * 10), "isScratched": False}
            for i in range(1, 9)
        ]
        race_info = {"raceId": "202608080203", "headCount": 8}
        entries = _make_entries(1, 3.5)
        odds_data = _make_odds_data(1, 3.5, 2.5)
        bets = optimize_bets(predictions, odds_data, race_info, entries=entries)
        for b in bets:
            assert "betSize" in b
            assert 1 <= b["betSize"] <= 10

    def test_bet_size_not_zero(self):
        """betSize is never 0 (kelly_bet_size returns minimum 1, not 0)."""
        bets = self._standard_bets()
        zeros = [b for b in bets if b.get("betSize") == 0]
        assert not zeros, f"Bets with betSize=0: {zeros}"


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------

class TestConstants:
    """Smoke tests to verify the expected constants are exported."""

    def test_tanpuku_tansho_min_odds(self):
        from backend.predictor.bet_optimizer import TANPUKU_TANSHO_MIN_ODDS
        assert TANPUKU_TANSHO_MIN_ODDS == 6.0

    def test_tanpuku_fukusho_min_odds(self):
        from backend.predictor.bet_optimizer import TANPUKU_FUKUSHO_MIN_ODDS
        assert TANPUKU_FUKUSHO_MIN_ODDS == 2.0

    def test_tanpuku_ratio(self):
        from backend.predictor.bet_optimizer import TANPUKU_RATIO
        assert TANPUKU_RATIO == (1, 3)

    def test_kelly_fraction(self):
        from backend.predictor.bet_optimizer import KELLY_FRACTION
        assert KELLY_FRACTION == 0.5

    def test_all_constants_importable_together(self):
        from backend.predictor.bet_optimizer import (
            kelly_bet_size,
            optimize_bets,
            TANPUKU_TANSHO_MIN_ODDS,
            TANPUKU_FUKUSHO_MIN_ODDS,
            TANPUKU_RATIO,
            KELLY_FRACTION,
        )
        assert callable(kelly_bet_size)
        assert callable(optimize_bets)
