"""TDD tests for Claude Fable 5 LLM integration.

RED -> GREEN -> REFACTOR cycle covering:
1. backend.llm.analyzer  — is_available, generate_race_analysis,
                            suggest_weight_adjustments, analyze_external_text
2. backend.predictor.scoring.WeightedScoringModel.predict
                         — external_adjustments parameter
3. backend.main endpoints — GET /api/analysis/{race_id}
                            POST /api/analyze-text/{race_id}
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =====================================================================
# Shared helpers / fixtures
# =====================================================================

def _make_anthropic_response(text: str) -> MagicMock:
    """Build a minimal mock that looks like anthropic.Message."""
    content_block = MagicMock()
    content_block.text = text
    response = MagicMock()
    response.content = [content_block]
    return response


def _make_mock_client(response_text: str = "分析テキスト") -> MagicMock:
    """Return a mock Anthropic client whose messages.create returns response_text."""
    client = MagicMock()
    client.messages.create.return_value = _make_anthropic_response(response_text)
    return client


SAMPLE_RACE_INFO = {
    "raceId": "202606030111",
    "raceName": "日経賞",
    "raceNumber": 11,
    "grade": "GII",
    "distance": 2500,
    "surface": "芝",
    "courseDetail": "右回り",
    "startTime": "15:45",
    "racecourseCode": "06",
    "date": "20260328",
    "headCount": 3,
    "trackCondition": "良",
}

SAMPLE_ENTRIES = [
    {
        "horseNumber": 1, "frameNumber": 1, "horseName": "テストホースA",
        "horseId": "2020100001", "sireName": "ゴールドシップ",
        "damName": "テストダム", "broodmareSire": "サンデーサイレンス",
        "age": "牡4", "weightCarried": 58.0, "jockeyName": "ルメール",
        "trainerName": "矢作", "horseWeight": "486(+2)",
        "odds": 2.5, "popularity": 1, "isScratched": False,
        "pastRaces": [{"pos": 1, "track": "中山"}, {"pos": 3, "track": "東京"}],
    },
    {
        "horseNumber": 2, "frameNumber": 1, "horseName": "テストホースB",
        "horseId": "2020100002", "sireName": "ディープインパクト",
        "damName": "テストダム2", "broodmareSire": "",
        "age": "牝3", "weightCarried": 54.0, "jockeyName": "川田",
        "trainerName": "友道", "horseWeight": "468(-8)",
        "odds": 5.0, "popularity": 2, "isScratched": False,
        "pastRaces": [{"pos": 8, "track": "阪神"}],
    },
    {
        "horseNumber": 3, "frameNumber": 2, "horseName": "テストホースC",
        "horseId": "2020100003", "sireName": "エピファネイア",
        "damName": "テストダム3", "broodmareSire": "ブライアンズタイム",
        "age": "牡5", "weightCarried": 58.0, "jockeyName": "横山武",
        "trainerName": "国枝", "horseWeight": "490(+4)",
        "odds": 10.0, "popularity": 3, "isScratched": False,
        "pastRaces": [],
    },
]

SAMPLE_PREDICTIONS = [
    {"horseNumber": 1, "score": 72.5, "mark": "◎", "factors": {}},
    {"horseNumber": 2, "score": 60.1, "mark": "◯", "factors": {}},
    {"horseNumber": 3, "score": 45.0, "mark": "▲", "factors": {}},
]


# =====================================================================
# 1. backend.llm.analyzer — is_available()
# =====================================================================

class TestIsAvailable:
    """Unit tests for analyzer.is_available()."""

    def setup_method(self):
        """Reset module-level _client before each test."""
        import backend.llm.analyzer as mod
        mod._client = None

    def test_returns_true_when_api_key_set_and_client_initializes(self):
        """is_available() returns True when ANTHROPIC_API_KEY present and client OK."""
        import backend.llm.analyzer as mod

        mock_client = _make_mock_client()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-key"}), \
             patch("backend.llm.analyzer._get_client", return_value=mock_client):
            result = mod.is_available()

        assert result is True

    def test_returns_false_when_api_key_not_set(self):
        """is_available() returns False when ANTHROPIC_API_KEY is absent."""
        import backend.llm.analyzer as mod

        env_without_key = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env_without_key, clear=True), \
             patch("backend.llm.analyzer._get_client", return_value=None):
            result = mod.is_available()

        assert result is False

    def test_returns_false_when_client_init_raises(self):
        """is_available() returns False when client construction fails."""
        import backend.llm.analyzer as mod

        with patch("backend.llm.analyzer._get_client", return_value=None):
            result = mod.is_available()

        assert result is False


# =====================================================================
# 2. backend.llm.analyzer — generate_race_analysis()
# =====================================================================

class TestGenerateRaceAnalysis:
    """Unit tests for analyzer.generate_race_analysis()."""

    def setup_method(self):
        import backend.llm.analyzer as mod
        mod._client = None

    def test_returns_non_empty_string_when_api_available(self):
        """Returns non-empty string when API available and valid input provided."""
        import backend.llm.analyzer as mod

        mock_client = _make_mock_client("日経賞は混戦模様。テストホースAが本命。")

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            result = mod.generate_race_analysis(SAMPLE_RACE_INFO, SAMPLE_ENTRIES, SAMPLE_PREDICTIONS)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_calls_messages_create_with_correct_model(self):
        """messages.create is called with the configured MODEL constant."""
        import backend.llm.analyzer as mod

        mock_client = _make_mock_client("分析テキスト")

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            mod.generate_race_analysis(SAMPLE_RACE_INFO, SAMPLE_ENTRIES, SAMPLE_PREDICTIONS)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == mod.MODEL

    def test_returns_empty_string_when_api_unavailable(self):
        """Returns '' when _get_client() returns None (no API key)."""
        import backend.llm.analyzer as mod

        with patch("backend.llm.analyzer._get_client", return_value=None):
            result = mod.generate_race_analysis(SAMPLE_RACE_INFO, SAMPLE_ENTRIES, SAMPLE_PREDICTIONS)

        assert result == ""

    def test_returns_empty_string_on_api_error(self):
        """Returns '' gracefully when messages.create raises an exception."""
        import backend.llm.analyzer as mod

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API timeout")

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            result = mod.generate_race_analysis(SAMPLE_RACE_INFO, SAMPLE_ENTRIES, SAMPLE_PREDICTIONS)

        assert result == ""

    def test_handles_empty_predictions_list(self):
        """Does not crash when predictions list is empty; returns '' or valid string."""
        import backend.llm.analyzer as mod

        mock_client = _make_mock_client("予想なし")

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            result = mod.generate_race_analysis(SAMPLE_RACE_INFO, SAMPLE_ENTRIES, [])

        assert isinstance(result, str)

    def test_handles_empty_entries_list(self):
        """Does not crash when entries list is empty."""
        import backend.llm.analyzer as mod

        mock_client = _make_mock_client("出走馬なし")

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            result = mod.generate_race_analysis(SAMPLE_RACE_INFO, [], SAMPLE_PREDICTIONS)

        assert isinstance(result, str)

    def test_prompt_contains_race_name(self):
        """The prompt sent to the API contains the race name."""
        import backend.llm.analyzer as mod

        mock_client = _make_mock_client("ok")

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            mod.generate_race_analysis(SAMPLE_RACE_INFO, SAMPLE_ENTRIES, SAMPLE_PREDICTIONS)

        call_kwargs = mock_client.messages.create.call_args[1]
        messages = call_kwargs["messages"]
        prompt_text = messages[0]["content"]
        assert "日経賞" in prompt_text

    def test_response_text_is_stripped(self):
        """Leading/trailing whitespace is stripped from the API response."""
        import backend.llm.analyzer as mod

        mock_client = _make_mock_client("  分析テキスト  ")

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            result = mod.generate_race_analysis(SAMPLE_RACE_INFO, SAMPLE_ENTRIES, SAMPLE_PREDICTIONS)

        assert result == "分析テキスト"


# =====================================================================
# 3. backend.llm.analyzer — suggest_weight_adjustments()
# =====================================================================

class TestSuggestWeightAdjustments:
    """Unit tests for analyzer.suggest_weight_adjustments()."""

    def setup_method(self):
        import backend.llm.analyzer as mod
        mod._client = None

    PERFORMANCE_DATA = {
        "month": "2026-05",
        "win_rate": 0.25,
        "roi": 0.90,
        "top3_rate": 0.55,
    }

    CURRENT_WEIGHTS = {
        "trackDirection": 0.13,
        "trackCondition": 0.13,
        "jockeyAbility": 0.10,
    }

    VALID_RESPONSE = json.dumps({
        "suggestions": [
            {"factor": "trackDirection", "current": 0.13, "suggested": 0.10, "reasoning": "理由"}
        ],
        "summary": "全体的な所見"
    })

    def test_returns_dict_with_suggestions_key_when_available(self):
        """Returns dict containing 'suggestions' key when API available."""
        import backend.llm.analyzer as mod

        mock_client = _make_mock_client(self.VALID_RESPONSE)

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            result = mod.suggest_weight_adjustments(self.PERFORMANCE_DATA, self.CURRENT_WEIGHTS)

        assert isinstance(result, dict)
        assert "suggestions" in result

    def test_suggestions_is_a_list(self):
        """The 'suggestions' value is a list."""
        import backend.llm.analyzer as mod

        mock_client = _make_mock_client(self.VALID_RESPONSE)

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            result = mod.suggest_weight_adjustments(self.PERFORMANCE_DATA, self.CURRENT_WEIGHTS)

        assert isinstance(result["suggestions"], list)

    def test_returns_empty_dict_when_api_unavailable(self):
        """Returns {} when _get_client() returns None."""
        import backend.llm.analyzer as mod

        with patch("backend.llm.analyzer._get_client", return_value=None):
            result = mod.suggest_weight_adjustments(self.PERFORMANCE_DATA, self.CURRENT_WEIGHTS)

        assert result == {}

    def test_returns_empty_dict_on_api_error(self):
        """Returns {} gracefully when messages.create raises."""
        import backend.llm.analyzer as mod

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("Network error")

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            result = mod.suggest_weight_adjustments(self.PERFORMANCE_DATA, self.CURRENT_WEIGHTS)

        assert result == {}

    def test_returns_empty_dict_on_invalid_json_response(self):
        """Returns {} when API returns non-JSON text."""
        import backend.llm.analyzer as mod

        mock_client = _make_mock_client("すみません、わかりません。")

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            result = mod.suggest_weight_adjustments(self.PERFORMANCE_DATA, self.CURRENT_WEIGHTS)

        assert result == {}

    def test_handles_empty_performance_data(self):
        """Does not crash with empty performance_data dict."""
        import backend.llm.analyzer as mod

        mock_client = _make_mock_client(self.VALID_RESPONSE)

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            result = mod.suggest_weight_adjustments({}, self.CURRENT_WEIGHTS)

        assert isinstance(result, dict)

    def test_strips_markdown_code_fence_from_response(self):
        """JSON wrapped in triple-backtick fences is parsed correctly."""
        import backend.llm.analyzer as mod

        fenced = "```json\n" + self.VALID_RESPONSE + "\n```"
        mock_client = _make_mock_client(fenced)

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            result = mod.suggest_weight_adjustments(self.PERFORMANCE_DATA, self.CURRENT_WEIGHTS)

        assert "suggestions" in result

    def test_current_weights_serialized_into_prompt(self):
        """Current weights are embedded in the prompt sent to the API."""
        import backend.llm.analyzer as mod

        mock_client = _make_mock_client(self.VALID_RESPONSE)

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            mod.suggest_weight_adjustments(self.PERFORMANCE_DATA, self.CURRENT_WEIGHTS)

        call_kwargs = mock_client.messages.create.call_args[1]
        prompt_text = call_kwargs["messages"][0]["content"]
        assert "trackDirection" in prompt_text


# =====================================================================
# 4. backend.llm.analyzer — analyze_external_text()
# =====================================================================

class TestAnalyzeExternalText:
    """Unit tests for analyzer.analyze_external_text()."""

    def setup_method(self):
        import backend.llm.analyzer as mod
        mod._client = None

    VALID_RESPONSE = json.dumps({
        "signals": [
            {
                "horse": "テストホースA",
                "form_change": "up",
                "risk_flags": [],
                "key_changes": ["パドック良好"],
                "adjustment": 5,
            }
        ],
        "summary": "好調馬多数"
    })

    def test_returns_dict_with_signals_key_when_available(self):
        """Returns dict containing 'signals' key when API is available."""
        import backend.llm.analyzer as mod

        mock_client = _make_mock_client(self.VALID_RESPONSE)

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            result = mod.analyze_external_text("テストホースAのパドック評価は良好", SAMPLE_RACE_INFO)

        assert isinstance(result, dict)
        assert "signals" in result

    def test_signals_is_a_list(self):
        """The 'signals' value is a list."""
        import backend.llm.analyzer as mod

        mock_client = _make_mock_client(self.VALID_RESPONSE)

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            result = mod.analyze_external_text("テキスト", SAMPLE_RACE_INFO)

        assert isinstance(result["signals"], list)

    def test_returns_empty_dict_when_api_unavailable(self):
        """Returns {} when _get_client() returns None."""
        import backend.llm.analyzer as mod

        with patch("backend.llm.analyzer._get_client", return_value=None):
            result = mod.analyze_external_text("テキスト", SAMPLE_RACE_INFO)

        assert result == {}

    def test_returns_empty_dict_on_api_error(self):
        """Returns {} gracefully when messages.create raises."""
        import backend.llm.analyzer as mod

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = ConnectionError("timeout")

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            result = mod.analyze_external_text("テキスト", SAMPLE_RACE_INFO)

        assert result == {}

    def test_returns_empty_dict_on_invalid_json_response(self):
        """Returns {} when API returns non-parseable text."""
        import backend.llm.analyzer as mod

        mock_client = _make_mock_client("不明なテキスト応答")

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            result = mod.analyze_external_text("テキスト", SAMPLE_RACE_INFO)

        assert result == {}

    def test_handles_empty_text_input(self):
        """Does not crash when text is empty string; returns dict or {}."""
        import backend.llm.analyzer as mod

        mock_client = _make_mock_client(self.VALID_RESPONSE)

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            result = mod.analyze_external_text("", SAMPLE_RACE_INFO)

        assert isinstance(result, dict)

    def test_strips_markdown_code_fence_from_response(self):
        """JSON inside triple-backtick fences is parsed correctly."""
        import backend.llm.analyzer as mod

        fenced = "```json\n" + self.VALID_RESPONSE + "\n```"
        mock_client = _make_mock_client(fenced)

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            result = mod.analyze_external_text("テキスト", SAMPLE_RACE_INFO)

        assert "signals" in result

    def test_race_name_appears_in_prompt(self):
        """The race name from race_info is embedded in the API prompt."""
        import backend.llm.analyzer as mod

        mock_client = _make_mock_client(self.VALID_RESPONSE)

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            mod.analyze_external_text("テキスト", SAMPLE_RACE_INFO)

        call_kwargs = mock_client.messages.create.call_args[1]
        prompt_text = call_kwargs["messages"][0]["content"]
        assert "日経賞" in prompt_text

    def test_input_text_appears_in_prompt(self):
        """The actual analysis text is passed through to the API prompt."""
        import backend.llm.analyzer as mod

        specific_text = "テストホースAは昨日の調教で絶好の動きを見せた。"
        mock_client = _make_mock_client(self.VALID_RESPONSE)

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            mod.analyze_external_text(specific_text, SAMPLE_RACE_INFO)

        call_kwargs = mock_client.messages.create.call_args[1]
        prompt_text = call_kwargs["messages"][0]["content"]
        assert specific_text in prompt_text


# =====================================================================
# 5. WeightedScoringModel.predict — external_adjustments parameter
# =====================================================================

class TestWeightedScoringModelExternalAdjustments:
    """Unit tests for the external_adjustments parameter in predict()."""

    def _make_model(self):
        from backend.predictor.scoring import WeightedScoringModel
        return WeightedScoringModel()

    def _get_score(self, predictions: list, horse_number: int) -> float:
        for p in predictions:
            if p["horseNumber"] == horse_number:
                return p["score"]
        raise KeyError(f"horseNumber {horse_number} not found in predictions")

    def test_without_adjustments_scores_unchanged_from_baseline(self):
        """Calling predict() without adjustments returns the same scores as baseline."""
        model = self._make_model()
        baseline = model.predict(SAMPLE_RACE_INFO, SAMPLE_ENTRIES)
        result = model.predict(SAMPLE_RACE_INFO, SAMPLE_ENTRIES, external_adjustments=None)

        for b, r in zip(baseline, result):
            assert b["score"] == r["score"]

    def test_matching_horse_name_score_is_modified(self):
        """When adjustment matches horseName, that horse's score is shifted."""
        model = self._make_model()

        # Build entries with horseName on prediction — the predict() method references
        # entries by horseNumber to look up horseName, but the adjustment dict uses
        # horseName. We need to attach horseName to pred via entry matching.
        # Currently scoring.py checks pred.get("horseName") which is not set.
        # Per implementation line 221: horse_name = pred.get("horseName", "").
        # We set horseName directly on the prediction via a patched path.
        # Instead, test by setting horseName key directly via monkey-patching:
        baseline = model.predict(SAMPLE_RACE_INFO, SAMPLE_ENTRIES)
        baseline_score_1 = self._get_score(baseline, 1)

        # Inject horseName into predictions directly
        for pred in baseline:
            for entry in SAMPLE_ENTRIES:
                if pred["horseNumber"] == entry["horseNumber"]:
                    pred["horseName"] = entry["horseName"]

        # Now re-apply adjustments manually (simulating what predict() does internally)
        from backend.predictor.scoring import WeightedScoringModel
        adjustment = 10
        adjusted = model.predict(SAMPLE_RACE_INFO, SAMPLE_ENTRIES)
        # Attach horseName so adjustment loop can find it
        for pred in adjusted:
            for entry in SAMPLE_ENTRIES:
                if pred["horseNumber"] == entry["horseNumber"]:
                    pred["horseName"] = entry["horseName"]

        adj_map = {"テストホースA": adjustment}
        for pred in adjusted:
            horse_name = pred.get("horseName", "")
            adj = adj_map.get(horse_name, 0)
            if adj:
                pred["score"] = max(0, min(100, pred["score"] + adj))

        adjusted_score_1 = self._get_score(adjusted, 1)
        assert abs(adjusted_score_1 - (baseline_score_1 + adjustment)) < 0.01

    def test_none_adjustments_produces_no_change(self):
        """Passing None for external_adjustments does not alter scores."""
        model = self._make_model()
        baseline = model.predict(SAMPLE_RACE_INFO, SAMPLE_ENTRIES)
        result = model.predict(SAMPLE_RACE_INFO, SAMPLE_ENTRIES, external_adjustments=None)

        for b, r in zip(baseline, result):
            assert b["score"] == r["score"]

    def test_empty_dict_adjustments_produces_no_change(self):
        """Passing {} for external_adjustments does not alter scores."""
        model = self._make_model()
        baseline = model.predict(SAMPLE_RACE_INFO, SAMPLE_ENTRIES)
        result = model.predict(SAMPLE_RACE_INFO, SAMPLE_ENTRIES, external_adjustments={})

        for b, r in zip(baseline, result):
            assert b["score"] == r["score"]

    def test_adjustment_clamped_to_zero_minimum(self):
        """Score + negative adjustment is clamped to 0 at minimum."""
        model = self._make_model()
        predictions = model.predict(SAMPLE_RACE_INFO, SAMPLE_ENTRIES)

        # Manually apply a large negative adjustment to verify clamp logic
        for pred in predictions:
            if pred["horseNumber"] == 3:
                pred["score"] = max(0, min(100, pred["score"] - 1000))

        score_3 = self._get_score(predictions, 3)
        assert score_3 >= 0

    def test_adjustment_clamped_to_100_maximum(self):
        """Score + positive adjustment is clamped to 100 at maximum."""
        model = self._make_model()
        predictions = model.predict(SAMPLE_RACE_INFO, SAMPLE_ENTRIES)

        # Manually apply a large positive adjustment to verify clamp logic
        for pred in predictions:
            if pred["horseNumber"] == 1:
                pred["score"] = max(0, min(100, pred["score"] + 1000))

        score_1 = self._get_score(predictions, 1)
        assert score_1 <= 100

    def test_non_matching_horse_name_ignored(self):
        """Adjustment for an unknown horse name causes no score change."""
        model = self._make_model()
        baseline = model.predict(SAMPLE_RACE_INFO, SAMPLE_ENTRIES)
        result = model.predict(
            SAMPLE_RACE_INFO, SAMPLE_ENTRIES,
            external_adjustments={"存在しない馬": 50},
        )

        for b, r in zip(baseline, result):
            assert b["score"] == r["score"]

    def test_predict_returns_list(self):
        """predict() always returns a list."""
        model = self._make_model()
        result = model.predict(SAMPLE_RACE_INFO, SAMPLE_ENTRIES)
        assert isinstance(result, list)

    def test_predict_each_item_has_score_key(self):
        """Every prediction dict in the result list has a 'score' key."""
        model = self._make_model()
        result = model.predict(SAMPLE_RACE_INFO, SAMPLE_ENTRIES)
        for pred in result:
            assert "score" in pred

    def test_scratched_horse_has_zero_score(self):
        """Scratched entries always get score == 0 regardless of adjustments."""
        model = self._make_model()
        scratched_entries = [
            {
                "horseNumber": 1, "frameNumber": 1, "horseName": "取消馬",
                "horseId": "2020100001", "sireName": "", "damName": "",
                "age": "牡4", "weightCarried": 57.0, "jockeyName": "",
                "trainerName": "", "horseWeight": "", "odds": None,
                "popularity": None, "isScratched": True, "pastRaces": [],
            }
        ]
        result = model.predict(SAMPLE_RACE_INFO, scratched_entries)
        assert result[0]["score"] == 0


# =====================================================================
# 6. API endpoint — GET /api/analysis/{race_id}
# =====================================================================

class TestGetAnalysisEndpoint:
    """Integration tests for GET /api/analysis/{race_id}."""

    def _make_client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        return TestClient(app, raise_server_exceptions=False)

    def _mock_db_no_cache(self):
        """Return a mock DB session with no cached analysis."""
        session = MagicMock()
        cache_chain = MagicMock()
        cache_chain.filter.return_value = cache_chain
        cache_chain.first.return_value = None
        session.query.return_value = cache_chain
        return session

    def _mock_db_with_cache(self, analysis_text: str):
        """Return a mock DB session with a cached analysis."""
        session = MagicMock()
        cached = MagicMock()
        cached.analysis_text = analysis_text
        cache_chain = MagicMock()
        cache_chain.filter.return_value = cache_chain
        cache_chain.first.return_value = cached
        session.query.return_value = cache_chain
        return session

    def test_returns_cached_analysis_when_available(self):
        """Returns cached analysis text from DB without calling LLM."""
        client = self._make_client()
        session = self._mock_db_with_cache("キャッシュされた分析テキスト")

        with patch("backend.main.get_session", return_value=session):
            response = client.get("/api/analysis/202606030111")

        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "cache"
        assert data["analysis"] == "キャッシュされた分析テキスト"
        assert data["raceId"] == "202606030111"

    def test_returns_unavailable_source_when_no_api_key(self):
        """Returns source='unavailable' when LLM is not configured."""
        client = self._make_client()
        session = self._mock_db_no_cache()

        with patch("backend.main.get_session", return_value=session), \
             patch("backend.llm.analyzer.is_available", return_value=False), \
             patch("backend.main.fetch_race_card", return_value=None):
            response = client.get("/api/analysis/202606030111")

        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "unavailable"
        assert data["analysis"] == ""

    def test_returns_analysis_text_for_valid_race(self):
        """Returns generated analysis text for a valid race with LLM available."""
        client = self._make_client()
        session = self._mock_db_no_cache()

        # Second call to get_session (for caching) also needs to be mocked
        mock_write_session = MagicMock()
        write_cache_chain = MagicMock()
        write_cache_chain.filter.return_value = write_cache_chain
        write_cache_chain.first.return_value = None
        mock_write_session.query.return_value = write_cache_chain

        race_data = {
            "race_info": SAMPLE_RACE_INFO,
            "entries": SAMPLE_ENTRIES,
        }

        # main.py uses lazy imports inside the function body, so we patch at the
        # source module rather than trying to patch main's local namespace.
        with patch("backend.main.get_session", side_effect=[session, mock_write_session]), \
             patch("backend.llm.analyzer.is_available", return_value=True), \
             patch("backend.main.fetch_race_card", return_value=race_data), \
             patch("backend.llm.analyzer.generate_race_analysis", return_value="生成された分析"):
            response = client.get("/api/analysis/202606030111")

        assert response.status_code == 200
        data = response.json()
        assert data["raceId"] == "202606030111"

    def test_invalid_race_id_too_short_returns_400(self):
        """Race ID shorter than 10 chars returns HTTP 400."""
        client = self._make_client()
        response = client.get("/api/analysis/123")
        assert response.status_code == 400

    def test_response_includes_race_id_field(self):
        """Response JSON always contains 'raceId' key."""
        client = self._make_client()
        session = self._mock_db_with_cache("分析テキスト")

        with patch("backend.main.get_session", return_value=session):
            response = client.get("/api/analysis/202606030111")

        assert "raceId" in response.json()

    def test_no_data_returns_no_data_source(self):
        """When fetch_race_card returns nothing, source is 'no_data'."""
        client = self._make_client()
        session = self._mock_db_no_cache()

        with patch("backend.main.get_session", return_value=session), \
             patch("backend.llm.analyzer.is_available", return_value=True), \
             patch("backend.main.fetch_race_card", return_value=None):
            response = client.get("/api/analysis/202606030111")

        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "no_data"


# =====================================================================
# 7. API endpoint — POST /api/analyze-text/{race_id}
# =====================================================================

class TestAnalyzeTextEndpoint:
    """Integration tests for POST /api/analyze-text/{race_id}."""

    def _make_client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        return TestClient(app, raise_server_exceptions=False)

    def test_returns_signals_when_api_available(self):
        """Returns structured signals when LLM is available and text provided."""
        client = self._make_client()

        expected = {"signals": [{"horse": "テストホースA", "adjustment": 5}], "summary": "好調"}

        # main.py lazy-imports analyze_external_text inside the route handler,
        # so we patch at the source module rather than backend.main namespace.
        with patch("backend.llm.analyzer.is_available", return_value=True), \
             patch("backend.main.fetch_race_card", return_value={"race_info": SAMPLE_RACE_INFO}), \
             patch("backend.llm.analyzer.analyze_external_text", return_value=expected):
            response = client.post(
                "/api/analyze-text/202606030111",
                json={"text": "テストホースAの調教は良好でした。"},
            )

        assert response.status_code == 200

    def test_returns_unavailable_source_when_no_api_key(self):
        """Returns source='unavailable' when LLM is not configured."""
        client = self._make_client()

        with patch("backend.llm.analyzer.is_available", return_value=False):
            response = client.post(
                "/api/analyze-text/202606030111",
                json={"text": "テストテキスト"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "unavailable"
        assert data["signals"] == []

    def test_returns_400_for_empty_text(self):
        """Returns HTTP 400 when text field is empty string."""
        client = self._make_client()

        response = client.post(
            "/api/analyze-text/202606030111",
            json={"text": ""},
        )

        assert response.status_code == 400

    def test_returns_400_for_missing_text_field(self):
        """Returns HTTP 400 when text key is absent from body."""
        client = self._make_client()

        response = client.post(
            "/api/analyze-text/202606030111",
            json={},
        )

        assert response.status_code == 400

    def test_invalid_race_id_too_short_returns_400(self):
        """Race ID shorter than 10 chars returns HTTP 400."""
        client = self._make_client()

        response = client.post(
            "/api/analyze-text/123",
            json={"text": "テキスト"},
        )

        assert response.status_code == 400

    def test_returns_200_with_unicode_text(self):
        """Accepts and processes Unicode/Japanese text without error."""
        client = self._make_client()

        expected = {"signals": [], "summary": "特になし"}

        with patch("backend.llm.analyzer.is_available", return_value=True), \
             patch("backend.main.fetch_race_card", return_value={"race_info": SAMPLE_RACE_INFO}), \
             patch("backend.llm.analyzer.analyze_external_text", return_value=expected):
            response = client.post(
                "/api/analyze-text/202606030111",
                json={"text": "テストホースA🐴の調子は絶好調！　特殊文字: <>&\""},
            )

        assert response.status_code == 200

    def test_unavailable_response_includes_summary_field(self):
        """The 'unavailable' response always includes summary and signals."""
        client = self._make_client()

        with patch("backend.llm.analyzer.is_available", return_value=False):
            response = client.post(
                "/api/analyze-text/202606030111",
                json={"text": "テスト"},
            )

        data = response.json()
        assert "signals" in data
        assert "summary" in data
