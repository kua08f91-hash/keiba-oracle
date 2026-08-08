"""TDD tests for frozen-cache immutability fixes.

Root cause addressed: multiple write paths could overwrite already-frozen
PredictionsCache data, causing predictions to change after race start.

Four fixes verified here:

  Fix 1 — _auto_freeze_and_cache() in backend/main.py
           Guards: if cache and cache.frozen: return
           Test: already-frozen cache → NO write to predictions_json
           Test: non-frozen cache → write occurs normally

  Fix 2 — generate_and_save_predictions() in backend/realtime_worker.py
           Guards: if cache and cache.frozen: return (before DB write)
           Test: already-frozen cache → function returns early, NO write
           Test: non-frozen cache → write occurs

  Fix 3 — _compute_live() double-check in backend/main.py
           Second call to _get_cached_predictions() at ~line 396 (after predictor.predict)
           If frozen cache found at that point → return cached data immediately
           Test: first call returns None, second call returns frozen → cached data returned
           Test: both calls return None → falls through to normal computation

  Fix 4 — analysis_text write guards (2 places in backend/main.py)
           _compute_live(): if cache and not cache.frozen → write analysis_text
           get_analysis():  if cache and cache.frozen: pass (never modify)
           Test: frozen cache → analysis_text NOT written
           Test: non-frozen cache → analysis_text written

  Fix 5 — FREEZE_THRESHOLD_MINS = 6 (changed from 7) in both files
           Test: constant equals 6 in backend.main
           Test: constant equals 6 in backend.realtime_worker

All external I/O is fully mocked.
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

RACE_ID = "202606030201"

# ---------------------------------------------------------------------------
# Helper factories — mirrors patterns from test_frozen_odds.py
# ---------------------------------------------------------------------------


def _make_cache_orm(frozen: bool = True, predictions_json: str | None = None) -> MagicMock:
    """Build a mock PredictionsCache ORM object."""
    cache = MagicMock()
    cache.frozen = frozen
    cache.predictions_json = predictions_json or json.dumps([{"horseNumber": 1, "score": 90}])
    cache.bets_json = json.dumps([])
    cache.longshot_json = None
    cache.pattern = "標準配置"
    cache.analysis_text = None
    cache.updated_at = MagicMock()
    cache.updated_at.isoformat.return_value = "2026-06-20T10:00:00"
    return cache


def _make_db_session(cache_orm: MagicMock | None = None) -> MagicMock:
    """Return a mock SQLAlchemy session.

    session.query(...).filter(...).first() returns cache_orm.
    session.query(...).filter(...).all()   returns [].
    """
    session = MagicMock()
    chain = MagicMock()
    chain.filter.return_value = chain
    chain.first.return_value = cache_orm
    chain.all.return_value = []
    session.query.return_value = chain
    return session


def _make_frozen_cache_dict(predictions: list | None = None) -> dict:
    """Return a frozen cache dict as returned by _get_cached_predictions()."""
    return {
        "predictions": predictions or [{"horseNumber": 1, "score": 90}],
        "bets": [],
        "longshot": None,
        "pattern": "標準配置",
        "frozen": True,
        "analysis": "AI分析テキスト",
        "updated_at": "2026-06-20T10:00:00",
    }


def _make_scraped_entry(horse_number: int, odds: float = 5.0) -> dict:
    return {
        "horseNumber": horse_number,
        "horseName": f"TestHorse{horse_number}",
        "odds": odds,
        "popularity": 1,
        "isScratched": False,
        "score": 0,
    }


def _make_fetch_race_card_data(entries: list | None = None) -> dict:
    if entries is None:
        entries = [_make_scraped_entry(1)]
    return {
        "race_info": {
            "raceId": RACE_ID,
            "headCount": len(entries),
            "date": "20260620",
        },
        "entries": entries,
    }


# ===========================================================================
# Fix 1 — _auto_freeze_and_cache() must not overwrite already-frozen cache
# ===========================================================================


class TestAutoFreezeAndCache:
    """_auto_freeze_and_cache() in backend/main.py must be idempotent for frozen caches."""

    def test_already_frozen_cache_returns_early_without_write(self):
        """When the existing DB cache has frozen=True, _auto_freeze_and_cache()
        must return immediately without writing predictions_json."""
        from backend.main import _auto_freeze_and_cache

        frozen_orm = _make_cache_orm(frozen=True, predictions_json='[{"horseNumber":1,"score":99}]')
        session = _make_db_session(cache_orm=frozen_orm)

        new_predictions = [{"horseNumber": 1, "score": 42}]

        with patch("backend.main.get_session", return_value=session):
            _auto_freeze_and_cache(RACE_ID, new_predictions, [], None, "")

        # predictions_json must NOT have been overwritten to the new value
        assert frozen_orm.predictions_json == '[{"horseNumber":1,"score":99}]', (
            "Already-frozen cache predictions_json must not be overwritten"
        )

    def test_already_frozen_cache_does_not_commit(self):
        """When frozen=True, no DB commit should occur."""
        from backend.main import _auto_freeze_and_cache

        frozen_orm = _make_cache_orm(frozen=True)
        session = _make_db_session(cache_orm=frozen_orm)

        with patch("backend.main.get_session", return_value=session):
            _auto_freeze_and_cache(RACE_ID, [], [], None, "")

        session.commit.assert_not_called()

    def test_non_frozen_cache_writes_predictions(self):
        """When cache exists but frozen=False, _auto_freeze_and_cache()
        must write the new predictions_json and commit."""
        from backend.main import _auto_freeze_and_cache

        non_frozen_orm = _make_cache_orm(frozen=False, predictions_json='[{"old":true}]')
        session = _make_db_session(cache_orm=non_frozen_orm)

        new_predictions = [{"horseNumber": 3, "score": 77}]

        with patch("backend.main.get_session", return_value=session), \
             patch("backend._tz.now_utc", return_value=MagicMock()):
            _auto_freeze_and_cache(RACE_ID, new_predictions, [], None, "")

        written = json.loads(non_frozen_orm.predictions_json)
        assert written == new_predictions, (
            "Non-frozen cache must have its predictions_json updated"
        )
        session.commit.assert_called_once()

    def test_no_existing_cache_creates_new_and_writes(self):
        """When no cache row exists yet (first freeze), a new row must be created."""
        from backend.main import _auto_freeze_and_cache

        session = _make_db_session(cache_orm=None)

        new_predictions = [{"horseNumber": 2, "score": 88}]

        with patch("backend.main.get_session", return_value=session), \
             patch("backend._tz.now_utc", return_value=MagicMock()):
            _auto_freeze_and_cache(RACE_ID, new_predictions, [], None, "")

        # db.add() must have been called to insert the new PredictionsCache row
        session.add.assert_called_once()
        session.commit.assert_called_once()

    def test_already_frozen_cache_closes_session(self):
        """db.close() must be called even when returning early for frozen cache."""
        from backend.main import _auto_freeze_and_cache

        frozen_orm = _make_cache_orm(frozen=True)
        session = _make_db_session(cache_orm=frozen_orm)

        with patch("backend.main.get_session", return_value=session):
            _auto_freeze_and_cache(RACE_ID, [], [], None, "")

        session.close.assert_called_once()

    def test_non_frozen_cache_closes_session(self):
        """db.close() must be called in the finally block on the normal write path."""
        from backend.main import _auto_freeze_and_cache

        non_frozen_orm = _make_cache_orm(frozen=False)
        session = _make_db_session(cache_orm=non_frozen_orm)

        with patch("backend.main.get_session", return_value=session), \
             patch("backend._tz.now_utc", return_value=MagicMock()):
            _auto_freeze_and_cache(RACE_ID, [], [], None, "")

        session.close.assert_called_once()


# ===========================================================================
# Fix 2 — generate_and_save_predictions() in realtime_worker must guard frozen
# ===========================================================================


class TestGenerateAndSavePredictions:
    """RealtimeWorker.generate_and_save_predictions() must skip DB write for frozen races."""

    def _make_worker(self):
        """Create a RealtimeWorker with mocked predictor so no model file is needed."""
        from backend.realtime_worker import RealtimeWorker

        with patch("backend.realtime_worker.init_db"), \
             patch("backend.realtime_worker.MLScoringModel") as MockModel:
            mock_predictor = MagicMock()
            mock_predictor.predict.return_value = [{"horseNumber": 1, "score": 80}]
            MockModel.return_value = mock_predictor
            worker = RealtimeWorker.__new__(RealtimeWorker)
            worker.predictor = mock_predictor
            worker.today = "20260620"
            worker._start_time_cache = {}
        return worker

    def test_frozen_cache_returns_early_no_write(self):
        """When PredictionsCache.frozen=True, generate_and_save_predictions()
        must return before writing anything to the DB."""
        worker = self._make_worker()

        frozen_orm = _make_cache_orm(frozen=True, predictions_json='[{"horseNumber":1,"score":99}]')
        # First two sessions: odds inject + combo odds read (frozen check uses 3rd session)
        # We need the final save session to return the frozen cache
        sessions_returned = [
            _make_db_session(cache_orm=None),   # session for HorseEntry odds inject
            _make_db_session(cache_orm=None),   # session for CombinationOdds read
            _make_db_session(cache_orm=frozen_orm),  # session for save — finds frozen
        ]
        session_iter = iter(sessions_returned)

        fetch_data = _make_fetch_race_card_data()
        mock_odds_data = {"umaren": []}

        with patch("backend.realtime_worker.fetch_race_card", return_value=fetch_data), \
             patch("backend.realtime_worker.get_session", side_effect=lambda: next(session_iter)), \
             patch("backend.realtime_worker.estimate_from_entries", return_value=mock_odds_data), \
             patch("backend.realtime_worker.optimize_bets", return_value=[]):
            worker.generate_and_save_predictions(RACE_ID)

        # The session that found a frozen cache must NOT have called commit
        save_session = sessions_returned[2]
        save_session.commit.assert_not_called()
        # predictions_json on the frozen ORM must be unchanged
        assert frozen_orm.predictions_json == '[{"horseNumber":1,"score":99}]'

    def test_frozen_cache_closes_session(self):
        """db.close() must be called even when returning early for frozen cache."""
        worker = self._make_worker()

        frozen_orm = _make_cache_orm(frozen=True)
        sessions = [
            _make_db_session(cache_orm=None),
            _make_db_session(cache_orm=None),
            _make_db_session(cache_orm=frozen_orm),
        ]
        session_iter = iter(sessions)

        fetch_data = _make_fetch_race_card_data()

        with patch("backend.realtime_worker.fetch_race_card", return_value=fetch_data), \
             patch("backend.realtime_worker.get_session", side_effect=lambda: next(session_iter)), \
             patch("backend.realtime_worker.estimate_from_entries", return_value={}), \
             patch("backend.realtime_worker.optimize_bets", return_value=[]):
            worker.generate_and_save_predictions(RACE_ID)

        # Every opened session must be closed
        for s in sessions:
            s.close.assert_called()

    def test_non_frozen_cache_writes_predictions(self):
        """When cache.frozen=False, predictions must be written to DB and committed."""
        worker = self._make_worker()

        non_frozen_orm = _make_cache_orm(frozen=False, predictions_json='[{"old":true}]')
        sessions = [
            _make_db_session(cache_orm=None),          # HorseEntry odds
            _make_db_session(cache_orm=None),          # CombinationOdds
            _make_db_session(cache_orm=non_frozen_orm), # save session
        ]
        session_iter = iter(sessions)

        fetch_data = _make_fetch_race_card_data()
        new_preds = [{"horseNumber": 1, "score": 75}]
        worker.predictor.predict.return_value = new_preds

        with patch("backend.realtime_worker.fetch_race_card", return_value=fetch_data), \
             patch("backend.realtime_worker.get_session", side_effect=lambda: next(session_iter)), \
             patch("backend.realtime_worker.estimate_from_entries", return_value={}), \
             patch("backend.realtime_worker.optimize_bets", return_value=[]), \
             patch("backend.realtime_worker.now_utc", return_value=MagicMock()):
            worker.generate_and_save_predictions(RACE_ID)

        save_session = sessions[2]
        save_session.commit.assert_called_once()
        written = json.loads(non_frozen_orm.predictions_json)
        assert written == new_preds

    def test_no_race_data_returns_early(self):
        """When fetch_race_card returns None, function returns without touching DB."""
        worker = self._make_worker()
        session = _make_db_session()

        with patch("backend.realtime_worker.fetch_race_card", return_value=None), \
             patch("backend.realtime_worker.get_session", return_value=session):
            worker.generate_and_save_predictions(RACE_ID)

        session.commit.assert_not_called()


# ===========================================================================
# Fix 3 — _compute_live() double-check: second _get_cached_predictions() call
# ===========================================================================


class TestComputeLiveDoubleCheck:
    """After predictor.predict(), _compute_live() re-checks the cache.
    If it is now frozen (race froze between the two checks), cached data
    is returned immediately without running optimize_bets or writing anything.
    """

    def _make_predictor_mock(self, preds: list | None = None) -> MagicMock:
        m = MagicMock()
        m.predict.return_value = preds or [{"horseNumber": 1, "score": 80}]
        return m

    def test_first_call_none_second_call_frozen_returns_cached(self):
        """Scenario: cache was None on entry to _compute_live(), but by the time
        the second _get_cached_predictions() is called (post-predict), the race
        has been frozen by the worker.

        Expected: _compute_live() returns the frozen cache data, not computed predictions.
        """
        from backend.main import _compute_live

        fetch_data = _make_fetch_race_card_data()
        frozen_dict = _make_frozen_cache_dict(predictions=[{"horseNumber": 1, "score": 99}])
        mock_predictor = self._make_predictor_mock(preds=[{"horseNumber": 1, "score": 55}])

        # First _get_cached_predictions call → None (not frozen yet)
        # Second call → frozen (frozen by worker after predict())
        side_effects = [None, frozen_dict]

        with patch("backend.main.fetch_race_card", return_value=fetch_data), \
             patch("backend.main._get_cached_predictions", side_effect=side_effects), \
             patch("backend.main._should_auto_freeze", return_value=False), \
             patch("backend.main.now_jst") as mock_now, \
             patch("backend.main._fetch_live_win_odds", return_value={}), \
             patch("backend.main.predictor", mock_predictor):
            mock_now.return_value.strftime.return_value = "20260620"
            result = _compute_live(RACE_ID, include_bets=False)

        assert result is not None
        assert result["frozen"] is True, (
            "Second-check frozen path must return frozen=True"
        )
        # Must return the frozen cache's predictions, NOT the predictor's output
        assert result["predictions"] == [{"horseNumber": 1, "score": 99}], (
            "When double-check finds frozen cache, must return cached predictions"
        )

    def test_first_call_none_second_call_frozen_does_not_call_optimize_bets(self):
        """When the second check finds a frozen cache, optimize_bets must NOT be called."""
        from backend.main import _compute_live

        fetch_data = _make_fetch_race_card_data()
        frozen_dict = _make_frozen_cache_dict()
        mock_predictor = self._make_predictor_mock()

        side_effects = [None, frozen_dict]

        with patch("backend.main.fetch_race_card", return_value=fetch_data), \
             patch("backend.main._get_cached_predictions", side_effect=side_effects), \
             patch("backend.main._should_auto_freeze", return_value=False), \
             patch("backend.main.now_jst") as mock_now, \
             patch("backend.main._fetch_live_win_odds", return_value={}), \
             patch("backend.main.predictor", mock_predictor), \
             patch("backend.main.optimize_bets") as mock_opt:
            mock_now.return_value.strftime.return_value = "20260620"
            _compute_live(RACE_ID, include_bets=True)

        mock_opt.assert_not_called()

    def test_both_calls_return_none_proceeds_to_compute(self):
        """When both _get_cached_predictions() calls return None, normal computation
        must proceed (no early return from the second check)."""
        from backend.main import _compute_live

        fetch_data = _make_fetch_race_card_data()
        mock_predictor = self._make_predictor_mock(preds=[{"horseNumber": 1, "score": 80}])

        side_effects = [None, None]

        with patch("backend.main.fetch_race_card", return_value=fetch_data), \
             patch("backend.main._get_cached_predictions", side_effect=side_effects), \
             patch("backend.main._should_auto_freeze", return_value=False), \
             patch("backend.main.now_jst") as mock_now, \
             patch("backend.main._fetch_live_win_odds", return_value={}), \
             patch("backend.main.predictor", mock_predictor):
            mock_now.return_value.strftime.return_value = "20260620"
            result = _compute_live(RACE_ID, include_bets=False)

        assert result is not None
        assert result["frozen"] is False, (
            "When both checks return None, result must not be marked frozen"
        )
        assert result["predictions"] == [{"horseNumber": 1, "score": 80}], (
            "Normal compute path must use predictor output"
        )

    def test_both_calls_none_non_frozen_result_uses_predictor_output(self):
        """Regression: normal (non-frozen) flow must still return predictor predictions."""
        from backend.main import _compute_live

        expected_preds = [{"horseNumber": 2, "score": 66}, {"horseNumber": 1, "score": 55}]
        fetch_data = _make_fetch_race_card_data(
            entries=[_make_scraped_entry(1), _make_scraped_entry(2)]
        )
        mock_predictor = self._make_predictor_mock(preds=expected_preds)

        with patch("backend.main.fetch_race_card", return_value=fetch_data), \
             patch("backend.main._get_cached_predictions", side_effect=[None, None]), \
             patch("backend.main._should_auto_freeze", return_value=False), \
             patch("backend.main.now_jst") as mock_now, \
             patch("backend.main._fetch_live_win_odds", return_value={}), \
             patch("backend.main.predictor", mock_predictor):
            mock_now.return_value.strftime.return_value = "20260620"
            result = _compute_live(RACE_ID, include_bets=False)

        assert result["predictions"] == expected_preds


# ===========================================================================
# Fix 4a — analysis_text write guard inside _compute_live() (include_bets path)
# ===========================================================================


class TestAnalysisTextGuardInComputeLive:
    """_compute_live() must not write analysis_text when cache is frozen."""

    def _run_compute_live_with_analysis(
        self,
        cache_orm: MagicMock | None,
        analysis_text_from_llm: str = "新しいAI分析",
    ) -> MagicMock:
        """Helper: run _compute_live(include_bets=True) with a mocked LLM and return
        the DB session that was used for the analysis_text write attempt."""
        from backend.main import _compute_live

        fetch_data = _make_fetch_race_card_data()
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = [{"horseNumber": 1, "score": 80}]

        analysis_write_session = _make_db_session(cache_orm=cache_orm)

        def get_session_factory():
            """Always return the same session for analysis write (simplest mock)."""
            return analysis_write_session

        with patch("backend.main.fetch_race_card", return_value=fetch_data), \
             patch("backend.main._get_cached_predictions", side_effect=[None, None]), \
             patch("backend.main._should_auto_freeze", return_value=False), \
             patch("backend.main.now_jst") as mock_now, \
             patch("backend.main._fetch_live_win_odds", return_value={}), \
             patch("backend.main.predictor", mock_predictor), \
             patch("backend.main.get_session", side_effect=get_session_factory), \
             patch("backend.main.optimize_bets", return_value=[]), \
             patch("backend.main.detect_race_pattern", return_value=""), \
             patch("backend.main.scores_to_probabilities", return_value=[]), \
             patch("backend.main.generate_candidates", return_value=[]), \
             patch("backend.main.monte_carlo_finish", return_value=[]), \
             patch("backend.main.estimate_hit_probabilities", return_value=[]), \
             patch("backend.main.pick_longshot", return_value=None), \
             patch("backend.main.evaluate_bet_confidence", return_value={}), \
             patch("backend.scraper.odds.estimate_from_entries", return_value={}):
            mock_now.return_value.strftime.return_value = "20260620"

            # Patch LLM path: no cached analysis_text, LLM returns new analysis
            if cache_orm is not None:
                cache_orm.analysis_text = None  # force LLM generation path

            with patch("backend.llm.analyzer.is_available", return_value=True), \
                 patch("backend.llm.analyzer.generate_race_analysis", return_value=analysis_text_from_llm):
                _compute_live(RACE_ID, include_bets=True)

        return analysis_write_session

    def test_frozen_cache_analysis_text_not_written(self):
        """When the DB cache is frozen, analysis_text must NOT be updated."""
        frozen_orm = _make_cache_orm(frozen=True)
        frozen_orm.analysis_text = None  # ensure LLM path runs

        session = self._run_compute_live_with_analysis(cache_orm=frozen_orm)

        # The frozen guard `if cache and not cache.frozen:` means no assignment
        assert frozen_orm.analysis_text is None, (
            "Frozen cache: analysis_text must not be written"
        )

    def test_non_frozen_cache_analysis_text_written(self):
        """When the DB cache is not frozen, analysis_text must be saved."""
        non_frozen_orm = _make_cache_orm(frozen=False)
        non_frozen_orm.analysis_text = None

        session = self._run_compute_live_with_analysis(cache_orm=non_frozen_orm)

        assert non_frozen_orm.analysis_text == "新しいAI分析", (
            "Non-frozen cache: analysis_text must be written"
        )


# ===========================================================================
# Fix 4b — analysis_text write guard inside get_analysis() endpoint
# ===========================================================================


class TestAnalysisTextGuardInGetAnalysis:
    """get_analysis() endpoint must not modify a frozen PredictionsCache row."""

    def test_frozen_cache_analysis_text_not_overwritten(self):
        """GET /api/analysis/{race_id}: when DB cache is frozen, analysis_text
        must not be set even when LLM returns a new value."""
        from backend.main import get_analysis

        frozen_orm = _make_cache_orm(frozen=True)
        frozen_orm.analysis_text = None  # no existing analysis → triggers LLM generation

        with patch("backend.main.get_session", side_effect=lambda: _make_db_session(cache_orm=frozen_orm)), \
             patch("backend.main.fetch_race_card", return_value=_make_fetch_race_card_data()), \
             patch("backend.main.predictor") as mp, \
             patch("backend.llm.analyzer.is_available", return_value=True), \
             patch("backend.llm.analyzer.generate_race_analysis", return_value="新しい分析"):
            mp.predict.return_value = []
            result = get_analysis(RACE_ID)

        # The endpoint must return the generated analysis text
        assert result["analysis"] == "新しい分析"
        # But the frozen ORM object must NOT have been updated
        assert frozen_orm.analysis_text is None, (
            "Frozen cache: get_analysis() must not write analysis_text to DB"
        )

    def test_non_frozen_cache_analysis_text_written(self):
        """GET /api/analysis/{race_id}: when DB cache is NOT frozen, analysis_text
        must be saved to DB."""
        from backend.main import get_analysis

        non_frozen_orm = _make_cache_orm(frozen=False)
        non_frozen_orm.analysis_text = None

        with patch("backend.main.fetch_race_card", return_value=_make_fetch_race_card_data()), \
             patch("backend.main.predictor") as mp, \
             patch("backend.llm.analyzer.is_available", return_value=True), \
             patch("backend.llm.analyzer.generate_race_analysis", return_value="分析テキスト"), \
             patch("backend.main.get_session", side_effect=lambda: _make_db_session(cache_orm=non_frozen_orm)):
            mp.predict.return_value = []
            result = get_analysis(RACE_ID)

        assert result["analysis"] == "分析テキスト"
        assert non_frozen_orm.analysis_text == "分析テキスト", (
            "Non-frozen cache: get_analysis() must write analysis_text to DB"
        )

    def test_frozen_cache_with_existing_analysis_returns_from_cache(self):
        """When a frozen cache already has analysis_text, it is returned immediately
        without calling the LLM at all."""
        from backend.main import get_analysis

        frozen_orm = _make_cache_orm(frozen=True)
        frozen_orm.analysis_text = "既存の分析テキスト"

        with patch("backend.main.get_session", return_value=_make_db_session(cache_orm=frozen_orm)), \
             patch("backend.llm.analyzer.generate_race_analysis") as mock_llm:
            result = get_analysis(RACE_ID)

        mock_llm.assert_not_called()
        assert result["analysis"] == "既存の分析テキスト"
        assert result["source"] == "cache"

    def test_no_analysis_generated_no_write_attempt(self):
        """When LLM returns empty string, no DB write must occur regardless of frozen state."""
        from backend.main import get_analysis

        non_frozen_orm = _make_cache_orm(frozen=False)
        non_frozen_orm.analysis_text = None

        with patch("backend.main.fetch_race_card", return_value=_make_fetch_race_card_data()), \
             patch("backend.main.predictor") as mp, \
             patch("backend.llm.analyzer.is_available", return_value=True), \
             patch("backend.llm.analyzer.generate_race_analysis", return_value=""), \
             patch("backend.main.get_session", side_effect=lambda: _make_db_session(cache_orm=non_frozen_orm)):
            mp.predict.return_value = []
            result = get_analysis(RACE_ID)

        # analysis should be empty; ORM should not be modified
        assert result["analysis"] == ""
        assert non_frozen_orm.analysis_text is None


# ===========================================================================
# Fix 5 — FREEZE_THRESHOLD_MINS constant must be 6 in both modules
# ===========================================================================


class TestFreezeThresholdConstant:
    """FREEZE_THRESHOLD_MINS must be exactly 6 (was previously 7)."""

    def test_main_py_freeze_threshold_is_6(self):
        """backend.main.FREEZE_THRESHOLD_MINS must equal 6."""
        import backend.main as main_module
        assert main_module.FREEZE_THRESHOLD_MINS == 6, (
            f"backend.main.FREEZE_THRESHOLD_MINS should be 6, got {main_module.FREEZE_THRESHOLD_MINS}"
        )

    def test_realtime_worker_freeze_threshold_is_6(self):
        """backend.realtime_worker.FREEZE_THRESHOLD_MINS must equal 6."""
        import backend.realtime_worker as worker_module
        assert worker_module.FREEZE_THRESHOLD_MINS == 6, (
            f"backend.realtime_worker.FREEZE_THRESHOLD_MINS should be 6, "
            f"got {worker_module.FREEZE_THRESHOLD_MINS}"
        )

    def test_both_constants_are_equal(self):
        """main.py and realtime_worker.py must agree on the threshold."""
        import backend.main as main_module
        import backend.realtime_worker as worker_module
        assert main_module.FREEZE_THRESHOLD_MINS == worker_module.FREEZE_THRESHOLD_MINS, (
            "FREEZE_THRESHOLD_MINS must be the same in main.py and realtime_worker.py"
        )

    def test_threshold_is_integer_not_float(self):
        """FREEZE_THRESHOLD_MINS must be an integer (used in comparison, not math)."""
        import backend.main as main_module
        assert isinstance(main_module.FREEZE_THRESHOLD_MINS, int), (
            "FREEZE_THRESHOLD_MINS must be int, not float"
        )


# ===========================================================================
# Integration — full immutability contract via HTTP endpoint
# ===========================================================================


class TestFrozenImmutabilityEndToEnd:
    """End-to-end checks: once frozen, racecard/analysis endpoints must never
    mutate the stored predictions or analysis_text."""

    def test_racecard_frozen_predictions_unchanged_across_calls(self):
        """Two consecutive calls to GET /api/racecard/{race_id} for a frozen race
        must return identical predictions both times."""
        from fastapi.testclient import TestClient
        from backend.main import app

        client = TestClient(app, raise_server_exceptions=True)

        frozen_preds = [{"horseNumber": 1, "score": 95, "label": "FROZEN"}]
        frozen_dict = _make_frozen_cache_dict(predictions=frozen_preds)
        frozen_orm = _make_cache_orm(frozen=True)

        def make_session():
            return _make_db_session(cache_orm=frozen_orm)

        fetch_data = _make_fetch_race_card_data()

        with patch("backend.main._get_cached_predictions", return_value=frozen_dict), \
             patch("backend.main.fetch_race_card", return_value=fetch_data), \
             patch("backend.main.get_session", side_effect=make_session):
            resp1 = client.get(f"/api/racecard/{RACE_ID}")
            resp2 = client.get(f"/api/racecard/{RACE_ID}")

        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.json()["predictions"] == frozen_preds
        assert resp2.json()["predictions"] == frozen_preds
        assert resp1.json()["predictions"] == resp2.json()["predictions"], (
            "Frozen predictions must be identical across successive requests"
        )

    def test_racecard_frozen_flag_is_true(self):
        """GET /api/racecard/{race_id} for a frozen race must return frozen=True."""
        from fastapi.testclient import TestClient
        from backend.main import app

        client = TestClient(app, raise_server_exceptions=True)

        frozen_dict = _make_frozen_cache_dict()
        frozen_orm = _make_cache_orm(frozen=True)
        fetch_data = _make_fetch_race_card_data()

        with patch("backend.main._get_cached_predictions", return_value=frozen_dict), \
             patch("backend.main.fetch_race_card", return_value=fetch_data), \
             patch("backend.main.get_session", return_value=_make_db_session(cache_orm=frozen_orm)):
            resp = client.get(f"/api/racecard/{RACE_ID}")

        assert resp.json()["frozen"] is True

    def test_auto_freeze_guard_prevents_double_freeze(self):
        """Calling _auto_freeze_and_cache() twice with different data only writes once.
        The second call must see frozen=True and return immediately."""
        from backend.main import _auto_freeze_and_cache

        # First call: no existing cache row
        session_first = _make_db_session(cache_orm=None)
        with patch("backend.main.get_session", return_value=session_first), \
             patch("backend._tz.now_utc", return_value=MagicMock()):
            _auto_freeze_and_cache(RACE_ID, [{"horseNumber": 1, "score": 90}], [], None, "")

        # Second call: simulates a frozen row now in DB
        frozen_orm = _make_cache_orm(frozen=True, predictions_json='[{"horseNumber":1,"score":90}]')
        session_second = _make_db_session(cache_orm=frozen_orm)

        with patch("backend.main.get_session", return_value=session_second):
            _auto_freeze_and_cache(RACE_ID, [{"horseNumber": 1, "score": 42}], [], None, "")

        # Second session must not have committed
        session_second.commit.assert_not_called()
        # The ORM object is unchanged
        assert frozen_orm.predictions_json == '[{"horseNumber":1,"score":90}]'

    def test_generate_and_save_predictions_frozen_idempotent(self):
        """RealtimeWorker: two prediction-generation cycles for a frozen race
        produce no DB writes after the first freeze."""
        from backend.realtime_worker import RealtimeWorker

        with patch("backend.realtime_worker.init_db"), \
             patch("backend.realtime_worker.MLScoringModel") as MockModel:
            mock_predictor = MagicMock()
            mock_predictor.predict.return_value = [{"horseNumber": 1, "score": 80}]
            MockModel.return_value = mock_predictor
            worker = RealtimeWorker.__new__(RealtimeWorker)
            worker.predictor = mock_predictor
            worker.today = "20260620"
            worker._start_time_cache = {}

        frozen_orm = _make_cache_orm(frozen=True, predictions_json='[{"horseNumber":1,"score":80}]')
        fetch_data = _make_fetch_race_card_data()

        call_count = [0]

        def counting_session():
            call_count[0] += 1
            # Odds sessions (1,2) return empty; save session returns frozen
            if call_count[0] <= 2:
                return _make_db_session(cache_orm=None)
            return _make_db_session(cache_orm=frozen_orm)

        with patch("backend.realtime_worker.fetch_race_card", return_value=fetch_data), \
             patch("backend.realtime_worker.get_session", side_effect=counting_session), \
             patch("backend.realtime_worker.estimate_from_entries", return_value={}), \
             patch("backend.realtime_worker.optimize_bets", return_value=[]):
            worker.generate_and_save_predictions(RACE_ID)

        # Reset counter for second cycle
        call_count[0] = 0

        with patch("backend.realtime_worker.fetch_race_card", return_value=fetch_data), \
             patch("backend.realtime_worker.get_session", side_effect=counting_session), \
             patch("backend.realtime_worker.estimate_from_entries", return_value={}), \
             patch("backend.realtime_worker.optimize_bets", return_value=[]):
            worker.generate_and_save_predictions(RACE_ID)

        # Predictions in ORM must remain the original frozen value
        assert frozen_orm.predictions_json == '[{"horseNumber":1,"score":80}]', (
            "Frozen predictions must survive a second generate_and_save_predictions() cycle"
        )
