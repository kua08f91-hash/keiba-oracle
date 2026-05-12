"""TDD tests for fetch_live_odds retry logic (export_predictions.py).

Change: fetch_live_odds(rid, max_retries=3) now retries up to 3 times with
exponential backoff on empty result or exception.  Returns {} only after all
retries are exhausted.

RED → GREEN → REFACTOR cycle:
  - Tests written first against the expected contract.
  - All HTTP calls mocked via unittest.mock to avoid real network traffic.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_odds_response(horse_data: dict) -> MagicMock:
    """Build a mock requests.Response whose .text is a valid netkeiba API
    payload containing the supplied horse_data dict under data.odds."1"."""
    payload = {
        "data": {
            "odds": {
                "1": horse_data,
            }
        }
    }
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(payload)
    return mock_resp


def _make_empty_response() -> MagicMock:
    """Build a mock response that yields an empty odds dict — the condition
    that triggers a retry in fetch_live_odds."""
    payload = {"data": {"odds": {"1": {}}}}
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(payload)
    return mock_resp


# Valid horse data that parse successfully into {horse_number: {odds, popularity}}
_VALID_HORSE_DATA = {
    "1": ["2.5", "0", "1"],
    "2": ["5.0", "0", "2"],
    "3": ["10.0", "0", "3"],
}


# =====================================================================
# TestFetchLiveOddsRetry
# =====================================================================
class TestFetchLiveOddsRetry:
    """Unit tests for fetch_live_odds retry and backoff behaviour.

    All requests.get calls are patched so no real HTTP traffic occurs.
    time.sleep is also patched to keep tests fast.
    """

    # ------------------------------------------------------------------
    # Test 1: retries on empty result, succeeds on second attempt
    # ------------------------------------------------------------------
    def test_retries_on_empty_result_succeeds_on_second_call(self):
        """When the first response has an empty odds dict, fetch_live_odds
        must retry and return the data from the second (successful) call."""
        from backend.export_predictions import fetch_live_odds

        empty_resp = _make_empty_response()
        good_resp = _make_odds_response(_VALID_HORSE_DATA)

        with patch("backend.export_predictions.requests.get",
                   side_effect=[empty_resp, good_resp]) as mock_get, \
             patch("backend.export_predictions.time.sleep") as mock_sleep:

            result = fetch_live_odds("202606030201", max_retries=3)

        assert isinstance(result, dict), "Return value must be a dict"
        assert len(result) > 0, (
            "fetch_live_odds should return non-empty data after retrying "
            "the empty first response."
        )
        # Verify two HTTP calls were made (first empty → second hit)
        assert mock_get.call_count == 2, (
            f"Expected 2 HTTP calls (1 empty + 1 success), got {mock_get.call_count}"
        )
        # Verify sleep was called once between the two attempts
        mock_sleep.assert_called_once()

    # ------------------------------------------------------------------
    # Test 2: returns {} after all retries exhausted (always empty)
    # ------------------------------------------------------------------
    def test_returns_empty_dict_after_all_retries_exhausted(self):
        """When every attempt returns an empty odds dict, fetch_live_odds
        must return {} after max_retries attempts without raising."""
        from backend.export_predictions import fetch_live_odds

        max_retries = 3
        empty_resp = _make_empty_response()

        with patch("backend.export_predictions.requests.get",
                   return_value=empty_resp) as mock_get, \
             patch("backend.export_predictions.time.sleep"):

            result = fetch_live_odds("202606030201", max_retries=max_retries)

        assert result == {}, (
            f"After {max_retries} failed attempts the function must return {{}}; "
            f"got {result!r}"
        )
        assert mock_get.call_count == max_retries, (
            f"Expected exactly {max_retries} HTTP calls, got {mock_get.call_count}"
        )

    # ------------------------------------------------------------------
    # Test 3: returns data immediately on first success (no retry needed)
    # ------------------------------------------------------------------
    def test_returns_data_immediately_on_first_success(self):
        """When the very first call returns valid odds data, the function
        must return it immediately without any retries or sleep calls."""
        from backend.export_predictions import fetch_live_odds

        good_resp = _make_odds_response(_VALID_HORSE_DATA)

        with patch("backend.export_predictions.requests.get",
                   return_value=good_resp) as mock_get, \
             patch("backend.export_predictions.time.sleep") as mock_sleep:

            result = fetch_live_odds("202606030201", max_retries=3)

        assert len(result) > 0, "First-call success should return non-empty data"
        assert mock_get.call_count == 1, (
            f"Only 1 HTTP call should be made on first success; "
            f"got {mock_get.call_count}"
        )
        mock_sleep.assert_not_called(), (
            "sleep must not be called when data is returned on the first attempt"
        )

    # ------------------------------------------------------------------
    # Test 4: handles timeout exception and retries
    # ------------------------------------------------------------------
    def test_retries_on_timeout_exception(self):
        """When requests.get raises an exception (e.g. Timeout), the function
        must catch it, sleep, and retry — returning data on a later attempt."""
        from backend.export_predictions import fetch_live_odds
        import requests as _requests

        good_resp = _make_odds_response(_VALID_HORSE_DATA)

        with patch("backend.export_predictions.requests.get",
                   side_effect=[
                       _requests.exceptions.Timeout("timed out"),
                       good_resp,
                   ]) as mock_get, \
             patch("backend.export_predictions.time.sleep") as mock_sleep:

            result = fetch_live_odds("202606030201", max_retries=3)

        assert len(result) > 0, (
            "Function must recover from a Timeout on attempt 1 and return "
            "data from attempt 2."
        )
        assert mock_get.call_count == 2
        # Sleep must have been called to implement backoff
        mock_sleep.assert_called_once()

    # ------------------------------------------------------------------
    # Test 5: respects max_retries parameter
    # ------------------------------------------------------------------
    def test_respects_max_retries_parameter(self):
        """The number of HTTP attempts must equal max_retries when every
        attempt fails (empty result), confirming the parameter is honoured."""
        from backend.export_predictions import fetch_live_odds

        empty_resp = _make_empty_response()

        for max_r in (1, 2, 5):
            with patch("backend.export_predictions.requests.get",
                       return_value=empty_resp) as mock_get, \
                 patch("backend.export_predictions.time.sleep"):

                result = fetch_live_odds("202606030201", max_retries=max_r)

            assert result == {}, (
                f"max_retries={max_r}: expected {{}} after exhaustion"
            )
            assert mock_get.call_count == max_r, (
                f"max_retries={max_r}: expected {max_r} HTTP calls, "
                f"got {mock_get.call_count}"
            )

    # ------------------------------------------------------------------
    # Test 6: backoff sleep duration increases with each attempt
    # ------------------------------------------------------------------
    def test_backoff_sleep_duration_increases(self):
        """sleep() is called with a duration proportional to the attempt
        number (3 * (attempt + 1)): 3 s after attempt 0, 6 s after attempt 1.
        This verifies the backoff formula in the implementation."""
        from backend.export_predictions import fetch_live_odds

        empty_resp = _make_empty_response()

        with patch("backend.export_predictions.requests.get",
                   return_value=empty_resp), \
             patch("backend.export_predictions.time.sleep") as mock_sleep:

            fetch_live_odds("202606030201", max_retries=3)

        # Attempts 0 and 1 trigger sleep (attempt 2 is the last → no sleep after)
        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert len(sleep_calls) == 2, (
            f"Expected 2 sleep calls for max_retries=3; got {len(sleep_calls)}"
        )
        # First sleep: 3 * (0 + 1) = 3, second sleep: 3 * (1 + 1) = 6
        assert sleep_calls[0] == 3, (
            f"First backoff should be 3 s; got {sleep_calls[0]}"
        )
        assert sleep_calls[1] == 6, (
            f"Second backoff should be 6 s; got {sleep_calls[1]}"
        )

    # ------------------------------------------------------------------
    # Test 7: returns {} after all retries exhausted with exception
    # ------------------------------------------------------------------
    def test_returns_empty_dict_when_all_attempts_raise_exception(self):
        """When every attempt raises an exception (network unreachable, etc.),
        the function must still return {} rather than propagating the error."""
        from backend.export_predictions import fetch_live_odds
        import requests as _requests

        with patch("backend.export_predictions.requests.get",
                   side_effect=_requests.exceptions.ConnectionError("offline")), \
             patch("backend.export_predictions.time.sleep"):

            result = fetch_live_odds("202606030201", max_retries=3)

        assert result == {}, (
            "All attempts raising exceptions must produce {} final return, "
            f"not raise or return {result!r}"
        )
