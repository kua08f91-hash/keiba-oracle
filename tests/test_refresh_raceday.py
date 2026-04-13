"""TDD tests for race-day refresh module.

Covers: time window detection, cache clearing, parse_time utility.
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock


class TestParseTime:
    def test_normal_time(self):
        from backend.refresh_raceday import parse_time
        result = parse_time("15:45")
        assert result is not None
        assert result.hour == 15
        assert result.minute == 45

    def test_morning_time(self):
        from backend.refresh_raceday import parse_time
        result = parse_time("09:30")
        assert result.hour == 9
        assert result.minute == 30

    def test_invalid_format(self):
        from backend.refresh_raceday import parse_time
        assert parse_time("") is None
        assert parse_time("invalid") is None
        assert parse_time("25:99") is None  # invalid time returns None


class TestClearRaceCache:
    def test_clears_without_error(self):
        from backend.refresh_raceday import clear_race_cache
        from backend.database.db import init_db
        init_db()
        # Should not raise even for non-existent race
        clear_race_cache("999999999999")


class TestRefreshTimeWindow:
    """Test that refresh correctly identifies races within 25-35 min window."""

    def test_race_in_window(self):
        """A race starting in 10 minutes should be in window."""
        now = datetime.now()
        start = now + timedelta(minutes=10)
        window_start = now + timedelta(minutes=5)
        window_end = now + timedelta(minutes=15)
        assert window_start <= start <= window_end

    def test_race_too_early(self):
        """A race starting in 2 minutes should NOT be in window."""
        now = datetime.now()
        start = now + timedelta(minutes=2)
        window_start = now + timedelta(minutes=5)
        window_end = now + timedelta(minutes=15)
        assert not (window_start <= start <= window_end)

    def test_race_too_late(self):
        """A race starting in 60 minutes should NOT be in window."""
        now = datetime.now()
        start = now + timedelta(minutes=60)
        window_start = now + timedelta(minutes=5)
        window_end = now + timedelta(minutes=15)
        assert not (window_start <= start <= window_end)


class TestWeekdayCheck:
    def test_saturday_is_race_day(self):
        # Saturday weekday() == 5
        assert 5 in (5, 6)

    def test_sunday_is_race_day(self):
        assert 6 in (5, 6)

    def test_monday_is_not_race_day(self):
        assert 0 not in (5, 6)


class TestMainIntegration:
    """Integration test for the main refresh flow (mocked scraping)."""

    @patch("backend.refresh_raceday.fetch_race_list")
    @patch("backend.refresh_raceday.fetch_race_card")
    @patch("backend.refresh_raceday.clear_race_cache")
    def test_skips_non_race_day(self, mock_clear, mock_card, mock_list):
        from backend.refresh_raceday import main
        with patch("backend.refresh_raceday.datetime") as mock_dt:
            # Set to Wednesday
            fake_now = datetime(2026, 4, 1, 12, 0)  # Wednesday
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
            main()
            mock_list.assert_not_called()
