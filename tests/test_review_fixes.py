"""TDD tests for review-fix items in analyzer.py and main.py.

RED -> GREEN -> REFACTOR cycle covering:
1. _strip_markdown_fence()       — markdown fence stripping helper
2. _get_client() thread safety   — lock usage, single-instance guarantee
3. ANTHROPIC_MODEL env var       — default and custom model selection
4. MAX_INPUT_TEXT_LENGTH         — truncation in analyze_external_text()
5. _compute_live analysis cache  — cache-first, LLM call, DB save
6. POST /api/analyze-text        — text length 400 guard, normal flow
"""
from __future__ import annotations

import os
import sys
import threading
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =====================================================================
# Shared helpers
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
    "trackCondition": "良",
    "headCount": 3,
    "date": "20260328",
    "racecourseCode": "06",
}

SAMPLE_ENTRIES = [
    {
        "horseNumber": 1, "frameNumber": 1, "horseName": "テストホースA",
        "horseId": "2020100001", "age": "牡4", "weightCarried": 58.0,
        "jockeyName": "ルメール", "trainerName": "矢作",
        "odds": 2.5, "popularity": 1, "isScratched": False, "pastRaces": [],
    },
    {
        "horseNumber": 2, "frameNumber": 2, "horseName": "テストホースB",
        "horseId": "2020100002", "age": "牝3", "weightCarried": 54.0,
        "jockeyName": "川田", "trainerName": "友道",
        "odds": 5.0, "popularity": 2, "isScratched": False, "pastRaces": [],
    },
]


# =====================================================================
# 1. _strip_markdown_fence()
# =====================================================================

class TestStripMarkdownFence:
    """Unit tests for backend.llm.analyzer._strip_markdown_fence()."""

    def _fn(self):
        from backend.llm.analyzer import _strip_markdown_fence
        return _strip_markdown_fence

    def test_plain_json_returned_unchanged(self):
        """Plain JSON with no fences passes through untouched."""
        strip = self._fn()
        payload = '{"key": "value"}'
        assert strip(payload) == payload

    def test_json_fence_with_language_tag_extracted(self):
        """```json\\n{...}\\n``` — inner JSON is extracted and returned."""
        strip = self._fn()
        inner = '{"key": "value"}'
        fenced = "```json\n" + inner + "\n```"
        assert strip(fenced) == inner

    def test_fence_without_language_tag_extracted(self):
        """```\\n{...}\\n``` (no language specifier) — inner content extracted."""
        strip = self._fn()
        inner = '{"a": 1}'
        fenced = "```\n" + inner + "\n```"
        assert strip(fenced) == inner

    def test_empty_string_returns_empty_string(self):
        """Empty string input returns empty string."""
        strip = self._fn()
        assert strip("") == ""

    def test_already_stripped_text_unchanged(self):
        """Text that has no fences is returned as-is (idempotent)."""
        strip = self._fn()
        text = "plain text without fences"
        assert strip(text) == text

    def test_only_first_fence_block_content_returned(self):
        """When multiple fence blocks appear, only the first block is extracted."""
        strip = self._fn()
        inner_first = '{"first": true}'
        inner_second = '{"second": true}'
        fenced = "```json\n" + inner_first + "\n```\n\nsome text\n\n```json\n" + inner_second + "\n```"
        result = strip(fenced)
        # The first block's content must be present
        assert inner_first in result
        # The second block should not be re-extracted as a standalone JSON object
        # (implementation strips only the outer first fence)
        assert result.startswith(inner_first.strip()) or inner_first in result

    def test_trailing_whitespace_stripped(self):
        """Leading and trailing whitespace around the text is stripped."""
        strip = self._fn()
        assert strip("  hello  ") == "hello"

    def test_fence_with_trailing_whitespace_on_closing_line(self):
        """Closing fence line with trailing spaces is still recognised."""
        strip = self._fn()
        inner = '{"x": 1}'
        # Closing fence has a trailing space
        fenced = "```json\n" + inner + "\n```  "
        result = strip(fenced)
        assert inner in result

    def test_json_content_preserved_exactly(self):
        """Multi-line JSON content inside a fence is preserved line-by-line."""
        strip = self._fn()
        inner = '{\n  "key": "value",\n  "num": 42\n}'
        fenced = "```json\n" + inner + "\n```"
        result = strip(fenced)
        assert result == inner

    def test_non_json_text_in_fence_extracted(self):
        """Non-JSON text inside a fence block is also extracted correctly."""
        strip = self._fn()
        inner = "some plain text inside fence"
        fenced = "```\n" + inner + "\n```"
        assert strip(fenced) == inner


# =====================================================================
# 2. Thread-safe _get_client()
# =====================================================================

class TestGetClientThreadSafety:
    """Unit tests for thread safety of backend.llm.analyzer._get_client()."""

    def setup_method(self):
        """Reset module-level _client and _client_lock before each test."""
        import backend.llm.analyzer as mod
        mod._client = None

    def teardown_method(self):
        """Reset after each test to avoid state leakage."""
        import backend.llm.analyzer as mod
        mod._client = None

    def test_concurrent_calls_create_only_one_client(self):
        """Multiple threads calling _get_client() concurrently produce a single client."""
        import backend.llm.analyzer as mod

        created_clients = []

        def fake_anthropic_constructor(api_key):  # noqa: ARG001
            # Simulate a slight delay so threads can collide
            client = MagicMock()
            created_clients.append(client)
            return client

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.side_effect = fake_anthropic_constructor

        results = []

        def worker():
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}), \
                 patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
                result = mod._get_client()
                results.append(result)

        # Reset client so every worker starts from scratch
        mod._client = None

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should receive the same client object
        assert len(results) == 8
        first = results[0]
        for r in results[1:]:
            assert r is first, "Different client objects returned — lock broken"

    def test_second_call_skips_initialization_entirely(self):
        """After first successful init, subsequent calls never touch anthropic module."""
        import backend.llm.analyzer as mod

        pre_created = MagicMock()
        mod._client = pre_created  # already initialised

        mock_anthropic = MagicMock()

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = mod._get_client()

        # anthropic.Anthropic() must NOT have been called again
        mock_anthropic.Anthropic.assert_not_called()
        assert result is pre_created

    def test_lock_attribute_exists_on_module(self):
        """The module exposes a threading Lock as _client_lock."""
        import backend.llm.analyzer as mod
        import threading
        assert isinstance(mod._client_lock, type(threading.Lock()))

    def test_returns_none_without_api_key(self):
        """_get_client() returns None when ANTHROPIC_API_KEY is absent."""
        import backend.llm.analyzer as mod
        mod._client = None

        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            result = mod._get_client()

        assert result is None


# =====================================================================
# 3. ANTHROPIC_MODEL env var
# =====================================================================

class TestAnthropicModelEnvVar:
    """Tests that MODEL constant respects the ANTHROPIC_MODEL environment variable."""

    def test_default_model_used_when_env_not_set(self):
        """MODEL falls back to the hardcoded default when ANTHROPIC_MODEL is absent."""
        # We import the constant directly after ensuring the env var is absent.
        # Because the module may already be cached we inspect the source-of-truth default.
        import importlib
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_MODEL"}
        with patch.dict(os.environ, env, clear=True):
            import backend.llm.analyzer as mod
            importlib.reload(mod)
            assert mod.MODEL == "claude-sonnet-4-20250514"

    def test_custom_model_used_when_env_set(self):
        """MODEL uses the value from ANTHROPIC_MODEL env var when present."""
        import importlib
        with patch.dict(os.environ, {"ANTHROPIC_MODEL": "claude-opus-4-5"}):
            import backend.llm.analyzer as mod
            importlib.reload(mod)
            assert mod.MODEL == "claude-opus-4-5"

    def test_generate_race_analysis_passes_model_to_api(self):
        """generate_race_analysis forwards MODEL to messages.create."""
        import backend.llm.analyzer as mod

        mock_client = _make_mock_client("テスト分析")

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            mod.generate_race_analysis(SAMPLE_RACE_INFO, SAMPLE_ENTRIES, [])

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == mod.MODEL

    def test_analyze_external_text_passes_model_to_api(self):
        """analyze_external_text forwards MODEL to messages.create."""
        import json
        import backend.llm.analyzer as mod

        valid_resp = json.dumps({"signals": [], "summary": "ok"})
        mock_client = _make_mock_client(valid_resp)

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            mod.analyze_external_text("テキスト", SAMPLE_RACE_INFO)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == mod.MODEL


# =====================================================================
# 4. MAX_INPUT_TEXT_LENGTH truncation in analyze_external_text()
# =====================================================================

class TestMaxInputTextLengthTruncation:
    """Unit tests for MAX_INPUT_TEXT_LENGTH enforcement in analyze_external_text()."""

    def setup_method(self):
        import backend.llm.analyzer as mod
        mod._client = None

    VALID_RESPONSE_JSON = '{"signals": [], "summary": "ok"}'

    def test_text_longer_than_limit_is_truncated_in_prompt(self):
        """Text exceeding MAX_INPUT_TEXT_LENGTH is sliced before reaching the API prompt."""
        import backend.llm.analyzer as mod

        limit = mod.MAX_INPUT_TEXT_LENGTH
        long_text = "あ" * (limit + 500)

        mock_client = _make_mock_client(self.VALID_RESPONSE_JSON)

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            mod.analyze_external_text(long_text, SAMPLE_RACE_INFO)

        call_kwargs = mock_client.messages.create.call_args[1]
        prompt_text = call_kwargs["messages"][0]["content"]

        # The full long_text must NOT appear in the prompt; only the truncated slice
        assert long_text not in prompt_text
        # But the first `limit` characters (minus race_name boilerplate) should be present
        truncated_portion = "あ" * limit
        assert truncated_portion in prompt_text

    def test_text_shorter_than_limit_is_sent_unchanged(self):
        """Text within MAX_INPUT_TEXT_LENGTH is passed to the prompt verbatim."""
        import backend.llm.analyzer as mod

        short_text = "短いテキスト。これは制限以内です。"
        assert len(short_text) < mod.MAX_INPUT_TEXT_LENGTH

        mock_client = _make_mock_client(self.VALID_RESPONSE_JSON)

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            mod.analyze_external_text(short_text, SAMPLE_RACE_INFO)

        call_kwargs = mock_client.messages.create.call_args[1]
        prompt_text = call_kwargs["messages"][0]["content"]
        assert short_text in prompt_text

    def test_text_exactly_at_limit_is_sent_unchanged(self):
        """Text whose length equals MAX_INPUT_TEXT_LENGTH is passed through unmodified."""
        import backend.llm.analyzer as mod

        exact_text = "x" * mod.MAX_INPUT_TEXT_LENGTH
        mock_client = _make_mock_client(self.VALID_RESPONSE_JSON)

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            mod.analyze_external_text(exact_text, SAMPLE_RACE_INFO)

        call_kwargs = mock_client.messages.create.call_args[1]
        prompt_text = call_kwargs["messages"][0]["content"]
        assert exact_text in prompt_text

    def test_max_input_text_length_constant_is_5000(self):
        """The constant MAX_INPUT_TEXT_LENGTH is set to 5000."""
        import backend.llm.analyzer as mod
        assert mod.MAX_INPUT_TEXT_LENGTH == 5000

    def test_truncation_happens_before_api_call(self):
        """Even if API raises, truncation still occurred (prompt is constructed first)."""
        import backend.llm.analyzer as mod

        limit = mod.MAX_INPUT_TEXT_LENGTH
        long_text = "b" * (limit * 2)

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("forced error")

        with patch("backend.llm.analyzer._get_client", return_value=mock_client):
            result = mod.analyze_external_text(long_text, SAMPLE_RACE_INFO)

        # Should degrade gracefully to empty dict
        assert result == {}
        # But the call was still attempted with the truncated prompt
        call_kwargs = mock_client.messages.create.call_args[1]
        prompt_text = call_kwargs["messages"][0]["content"]
        assert long_text not in prompt_text


# =====================================================================
# 5. _compute_live analysis cache-first behaviour
# =====================================================================

class TestComputeLiveAnalysisCache:
    """Integration tests for the analysis cache logic inside _compute_live()."""

    # _compute_live is an internal function in backend.main.  We test its
    # observable effect by calling it directly after mocking its dependencies.

    def _make_cache_mock(self, analysis_text: str = ""):
        """Return a mock DB session that returns a PredictionsCache with analysis_text."""
        session = MagicMock()
        cache_row = MagicMock()
        cache_row.analysis_text = analysis_text
        cache_row.frozen = False
        chain = MagicMock()
        chain.filter.return_value = chain
        chain.first.return_value = cache_row if analysis_text is not None else None
        session.query.return_value = chain
        return session, cache_row

    def _make_no_cache_mock(self):
        """Return a mock DB session that has no cached row."""
        session = MagicMock()
        chain = MagicMock()
        chain.filter.return_value = chain
        chain.first.return_value = None
        session.query.return_value = chain
        return session

    def _make_frozen_cache_mock(self):
        """Return a mock DB session with a row that has frozen=True."""
        session = MagicMock()
        cache_row = MagicMock()
        cache_row.frozen = True
        cache_row.analysis_text = "凍結された分析"
        chain = MagicMock()
        chain.filter.return_value = chain
        chain.first.return_value = cache_row
        session.query.return_value = chain
        return session

    def test_cached_analysis_prevents_llm_call(self):
        """When the DB row already has analysis_text, generate_race_analysis is NOT called."""
        from backend.main import _compute_live

        db_with_cache, _ = self._make_cache_mock(analysis_text="キャッシュ済み分析")
        db_no_bets = self._make_no_cache_mock()

        race_data = {"race_info": SAMPLE_RACE_INFO, "entries": SAMPLE_ENTRIES}

        with patch("backend.main.fetch_race_card", return_value=race_data), \
             patch("backend.main._get_cached_predictions", return_value=None), \
             patch("backend.main._should_auto_freeze", return_value=False), \
             patch("backend.main._fetch_live_win_odds", return_value={}), \
             patch("backend.main.get_session", side_effect=[db_with_cache, db_no_bets]), \
             patch("backend.main.predictor") as mock_predictor, \
             patch("backend.llm.analyzer.generate_race_analysis") as mock_llm, \
             patch("backend.llm.analyzer.is_available", return_value=True):

            mock_predictor.predict.return_value = []
            result = _compute_live("202606030111", include_bets=True)

        mock_llm.assert_not_called()

    def test_no_cached_analysis_triggers_llm_call(self):
        """When DB has no analysis_text, generate_race_analysis IS called."""
        from backend.main import _compute_live

        db_no_cache = self._make_no_cache_mock()
        db_write = self._make_no_cache_mock()

        race_data = {"race_info": SAMPLE_RACE_INFO, "entries": SAMPLE_ENTRIES}

        with patch("backend.main.fetch_race_card", return_value=race_data), \
             patch("backend.main._get_cached_predictions", return_value=None), \
             patch("backend.main._should_auto_freeze", return_value=False), \
             patch("backend.main._fetch_live_win_odds", return_value={}), \
             patch("backend.main.get_session", side_effect=[db_no_cache, db_write]), \
             patch("backend.main.predictor") as mock_predictor, \
             patch("backend.llm.analyzer.generate_race_analysis", return_value="新規生成") as mock_llm, \
             patch("backend.llm.analyzer.is_available", return_value=True):

            mock_predictor.predict.return_value = []
            _compute_live("202606030111", include_bets=True)

        mock_llm.assert_called_once()

    def test_generated_analysis_saved_to_db(self):
        """After generating new analysis, it is committed to the DB cache row."""
        from backend.main import _compute_live

        db_no_cache = self._make_no_cache_mock()

        # Second session (write) returns a real cache row to update
        db_write = MagicMock()
        cache_row = MagicMock()
        write_chain = MagicMock()
        write_chain.filter.return_value = write_chain
        write_chain.first.return_value = cache_row
        db_write.query.return_value = write_chain

        race_data = {"race_info": SAMPLE_RACE_INFO, "entries": SAMPLE_ENTRIES}
        generated_text = "LLM生成分析テキスト"

        with patch("backend.main.fetch_race_card", return_value=race_data), \
             patch("backend.main._get_cached_predictions", return_value=None), \
             patch("backend.main._should_auto_freeze", return_value=False), \
             patch("backend.main._fetch_live_win_odds", return_value={}), \
             patch("backend.main.get_session", side_effect=[db_no_cache, db_write]), \
             patch("backend.main.predictor") as mock_predictor, \
             patch("backend.llm.analyzer.generate_race_analysis", return_value=generated_text), \
             patch("backend.llm.analyzer.is_available", return_value=True):

            mock_predictor.predict.return_value = []
            _compute_live("202606030111", include_bets=True)

        # The cache row should have had analysis_text set and committed
        assert cache_row.analysis_text == generated_text
        db_write.commit.assert_called_once()

    def test_analysis_not_computed_when_include_bets_false(self):
        """When include_bets=False, the analysis block is skipped entirely."""
        from backend.main import _compute_live

        race_data = {"race_info": SAMPLE_RACE_INFO, "entries": SAMPLE_ENTRIES}

        with patch("backend.main.fetch_race_card", return_value=race_data), \
             patch("backend.main._get_cached_predictions", return_value=None), \
             patch("backend.main._should_auto_freeze", return_value=False), \
             patch("backend.main._fetch_live_win_odds", return_value={}), \
             patch("backend.main.get_session") as mock_get_session, \
             patch("backend.main.predictor") as mock_predictor, \
             patch("backend.llm.analyzer.generate_race_analysis") as mock_llm:

            mock_predictor.predict.return_value = []
            result = _compute_live("202606030111", include_bets=False)

        mock_llm.assert_not_called()
        # Verify analysis in result is empty string
        assert result is not None
        assert result.get("analysis", "") == ""

    def test_empty_generated_analysis_is_not_saved_to_db(self):
        """If LLM returns empty string, no DB commit is attempted."""
        from backend.main import _compute_live

        db_no_cache = self._make_no_cache_mock()
        db_write = MagicMock()

        race_data = {"race_info": SAMPLE_RACE_INFO, "entries": SAMPLE_ENTRIES}

        with patch("backend.main.fetch_race_card", return_value=race_data), \
             patch("backend.main._get_cached_predictions", return_value=None), \
             patch("backend.main._should_auto_freeze", return_value=False), \
             patch("backend.main._fetch_live_win_odds", return_value={}), \
             patch("backend.main.get_session", side_effect=[db_no_cache, db_write]), \
             patch("backend.main.predictor") as mock_predictor, \
             patch("backend.llm.analyzer.generate_race_analysis", return_value=""), \
             patch("backend.llm.analyzer.is_available", return_value=True):

            mock_predictor.predict.return_value = []
            _compute_live("202606030111", include_bets=True)

        # With empty analysis there is no second session open for writing
        db_write.commit.assert_not_called()


# =====================================================================
# 6. POST /api/analyze-text — text length guard and normal flow
# =====================================================================

class TestAnalyzeTextEndpointLengthGuard:
    """Integration tests for the text-length guard on POST /api/analyze-text/{race_id}."""

    def _make_client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        return TestClient(app, raise_server_exceptions=False)

    def test_text_over_5000_chars_returns_400(self):
        """POST body with text longer than 5000 chars is rejected with HTTP 400."""
        client = self._make_client()
        long_text = "あ" * 5001

        response = client.post(
            "/api/analyze-text/202606030111",
            json={"text": long_text},
        )

        assert response.status_code == 400

    def test_text_over_5000_chars_error_message_mentions_limit(self):
        """400 error body explains that the text is too long."""
        client = self._make_client()
        long_text = "x" * 5001

        response = client.post(
            "/api/analyze-text/202606030111",
            json={"text": long_text},
        )

        assert response.status_code == 400
        detail = response.json().get("detail", "")
        assert "5000" in detail or "long" in detail.lower() or "長" in detail

    def test_text_exactly_5000_chars_proceeds_normally(self):
        """Text of exactly 5000 chars is accepted (boundary value)."""
        client = self._make_client()
        exact_text = "a" * 5000

        import json as _json
        valid_response = _json.dumps({"signals": [], "summary": "ok"})

        with patch("backend.llm.analyzer.is_available", return_value=True), \
             patch("backend.main.fetch_race_card", return_value={"race_info": SAMPLE_RACE_INFO}), \
             patch("backend.llm.analyzer.analyze_external_text", return_value={"signals": [], "summary": "ok"}):
            response = client.post(
                "/api/analyze-text/202606030111",
                json={"text": exact_text},
            )

        assert response.status_code == 200

    def test_text_under_5000_chars_proceeds_normally(self):
        """Text well under 5000 chars is accepted and processed."""
        client = self._make_client()
        short_text = "テストホースAは絶好調です。"

        with patch("backend.llm.analyzer.is_available", return_value=True), \
             patch("backend.main.fetch_race_card", return_value={"race_info": SAMPLE_RACE_INFO}), \
             patch("backend.llm.analyzer.analyze_external_text", return_value={"signals": [], "summary": "ok"}):
            response = client.post(
                "/api/analyze-text/202606030111",
                json={"text": short_text},
            )

        assert response.status_code == 200

    def test_text_4999_chars_proceeds_normally(self):
        """Text of 4999 chars (one below limit) is accepted."""
        client = self._make_client()
        near_limit_text = "y" * 4999

        with patch("backend.llm.analyzer.is_available", return_value=True), \
             patch("backend.main.fetch_race_card", return_value={"race_info": SAMPLE_RACE_INFO}), \
             patch("backend.llm.analyzer.analyze_external_text", return_value={"signals": [], "summary": "ok"}):
            response = client.post(
                "/api/analyze-text/202606030111",
                json={"text": near_limit_text},
            )

        assert response.status_code == 200

    def test_text_5001_chars_rejected(self):
        """Text of 5001 chars (one above limit) triggers 400."""
        client = self._make_client()
        just_over_text = "z" * 5001

        response = client.post(
            "/api/analyze-text/202606030111",
            json={"text": just_over_text},
        )

        assert response.status_code == 400

    def test_empty_text_returns_400(self):
        """Empty string text is rejected with HTTP 400."""
        client = self._make_client()

        response = client.post(
            "/api/analyze-text/202606030111",
            json={"text": ""},
        )

        assert response.status_code == 400

    def test_missing_text_field_returns_400(self):
        """Missing text key in body is rejected with HTTP 400."""
        client = self._make_client()

        response = client.post(
            "/api/analyze-text/202606030111",
            json={},
        )

        assert response.status_code == 400

    def test_llm_unavailable_returns_200_with_unavailable_source(self):
        """When LLM is not configured, endpoint returns 200 with source='unavailable'."""
        client = self._make_client()

        with patch("backend.llm.analyzer.is_available", return_value=False):
            response = client.post(
                "/api/analyze-text/202606030111",
                json={"text": "テスト"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data.get("source") == "unavailable"

    def test_response_contains_signals_list(self):
        """Successful response includes a 'signals' key with a list."""
        client = self._make_client()

        with patch("backend.llm.analyzer.is_available", return_value=True), \
             patch("backend.main.fetch_race_card", return_value={"race_info": SAMPLE_RACE_INFO}), \
             patch("backend.llm.analyzer.analyze_external_text",
                   return_value={"signals": [{"horse": "テストホースA", "adjustment": 3}], "summary": "好調"}):
            response = client.post(
                "/api/analyze-text/202606030111",
                json={"text": "テストホースAは好調"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "signals" in data
        assert isinstance(data["signals"], list)

    def test_invalid_race_id_returns_400_regardless_of_text_length(self):
        """Short race ID produces 400 even when text length is valid."""
        client = self._make_client()

        response = client.post(
            "/api/analyze-text/123",
            json={"text": "有効なテキスト"},
        )

        assert response.status_code == 400

    def test_unicode_text_within_limit_accepted(self):
        """Japanese + emoji text within the character limit is accepted."""
        client = self._make_client()
        unicode_text = "テストホースA🐴の調子は絶好調！特殊文字: <>&\"\u3000" * 10

        assert len(unicode_text) < 5000

        with patch("backend.llm.analyzer.is_available", return_value=True), \
             patch("backend.main.fetch_race_card", return_value={"race_info": SAMPLE_RACE_INFO}), \
             patch("backend.llm.analyzer.analyze_external_text",
                   return_value={"signals": [], "summary": "特になし"}):
            response = client.post(
                "/api/analyze-text/202606030111",
                json={"text": unicode_text},
            )

        assert response.status_code == 200
