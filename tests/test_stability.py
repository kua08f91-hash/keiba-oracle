"""TDD tests for race-day stability features.

RED → GREEN cycle covering:
1. _create_engine_safe  — SQLite passthrough, PostgreSQL success, PostgreSQL fallback
2. _should_auto_freeze  — RaceStatus lookup, Race table fallback, boundary math
3. _acquire_lock / _release_lock  — PID file creation, stale detection, release guard
"""
from __future__ import annotations

import os
import sys
import pytest
from unittest.mock import patch, MagicMock, mock_open, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =====================================================================
# 1. _create_engine_safe
# =====================================================================

class TestCreateEngineSafe:
    """Unit tests for backend.database.db._create_engine_safe.

    The function lives in db.py but accesses the module-level DATABASE_URL.
    We patch create_engine and the module-level DATABASE_URL to isolate behavior.
    """

    def test_sqlite_url_returns_sqlite_engine_directly(self):
        """When DATABASE_URL starts with 'sqlite', return engine without connecting."""
        from backend.database import db as db_module

        mock_engine = MagicMock()
        with patch.object(db_module, "DATABASE_URL", "sqlite:///test.db"), \
             patch("backend.database.db.create_engine", return_value=mock_engine) as mock_ce:
            result = db_module._create_engine_safe()

        assert result is mock_engine
        # create_engine called exactly once with the SQLite URL
        mock_ce.assert_called_once()
        call_args = mock_ce.call_args
        assert call_args[0][0] == "sqlite:///test.db"

    def test_sqlite_url_does_not_attempt_connection(self):
        """SQLite path must not call engine.connect() at all."""
        from backend.database import db as db_module

        mock_engine = MagicMock()
        with patch.object(db_module, "DATABASE_URL", "sqlite:///test.db"), \
             patch("backend.database.db.create_engine", return_value=mock_engine):
            db_module._create_engine_safe()

        mock_engine.connect.assert_not_called()

    def test_postgres_success_returns_postgres_engine(self):
        """When PostgreSQL connection succeeds, return the PostgreSQL engine."""
        from backend.database import db as db_module

        pg_engine = MagicMock()
        mock_conn = MagicMock()
        # context manager protocol: engine.connect() returns a context manager
        pg_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        pg_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(db_module, "DATABASE_URL", "postgresql://user:pass@host/db"), \
             patch("backend.database.db.create_engine", return_value=pg_engine):
            result = db_module._create_engine_safe()

        assert result is pg_engine
        pg_engine.connect.assert_called_once()

    def test_postgres_connection_failure_falls_back_to_sqlite(self):
        """When PostgreSQL connect() raises, return a SQLite engine instead."""
        from backend.database import db as db_module

        pg_engine = MagicMock()
        # Simulate a timeout / connection refused
        pg_engine.connect.side_effect = Exception("connection refused")

        sqlite_engine = MagicMock()

        def fake_create_engine(url, **kwargs):
            if url.startswith("postgresql"):
                return pg_engine
            return sqlite_engine

        with patch.object(db_module, "DATABASE_URL", "postgresql://user:pass@host/db"), \
             patch("backend.database.db.create_engine", side_effect=fake_create_engine), \
             patch("os.makedirs"):
            result = db_module._create_engine_safe()

        assert result is sqlite_engine

    def test_postgres_failure_fallback_uses_sqlite_url(self):
        """Fallback engine must be created with the module-level SQLITE_URL."""
        from backend.database import db as db_module

        pg_engine = MagicMock()
        pg_engine.connect.side_effect = Exception("timeout")

        captured_urls = []

        def fake_create_engine(url, **kwargs):
            captured_urls.append(url)
            if url.startswith("postgresql"):
                return pg_engine
            return MagicMock()

        with patch.object(db_module, "DATABASE_URL", "postgresql://user:pass@host/db"), \
             patch("backend.database.db.create_engine", side_effect=fake_create_engine), \
             patch("os.makedirs"):
            db_module._create_engine_safe()

        assert len(captured_urls) == 2
        assert captured_urls[0].startswith("postgresql")
        assert captured_urls[1] == db_module.SQLITE_URL

    def test_postgres_failure_creates_data_directory(self):
        """On PostgreSQL fallback, DATA_DIR must be created if it does not exist."""
        from backend.database import db as db_module

        pg_engine = MagicMock()
        pg_engine.connect.side_effect = Exception("no route to host")

        with patch.object(db_module, "DATABASE_URL", "postgresql://user:pass@host/db"), \
             patch("backend.database.db.create_engine", return_value=pg_engine), \
             patch("os.makedirs") as mock_makedirs:
            db_module._create_engine_safe()

        mock_makedirs.assert_called_once_with(db_module.DATA_DIR, exist_ok=True)


# =====================================================================
# 2. _should_auto_freeze
# =====================================================================

class TestShouldAutoFreeze:
    """Unit tests for backend.main._should_auto_freeze.

    The function queries RaceStatus first, falls back to Race, then computes
    minutes-to-post relative to JST now.  We mock the DB session and now_jst.
    """

    # ------ helpers ------

    def _make_session(self, race_status=None, race=None):
        """Build a mock session that returns race_status then race on successive queries."""
        session = MagicMock()

        # Each call to session.query(...).filter(...).first() is chained differently
        # depending on the model class passed.  We track calls via side_effect.
        status_chain = MagicMock()
        status_chain.filter.return_value = status_chain
        status_chain.first.return_value = race_status

        race_chain = MagicMock()
        race_chain.filter.return_value = race_chain
        race_chain.first.return_value = race

        # Return appropriate chain based on which model class is queried
        from backend.database.models import RaceStatus, Race as RaceModel

        def side_effect(model):
            if model is RaceStatus:
                return status_chain
            if model is RaceModel:
                return race_chain
            return MagicMock()

        session.query.side_effect = side_effect
        return session

    def _mock_now_jst(self, hour: int, minute: int):
        """Return a fixed JST datetime at the given hour:minute."""
        from datetime import datetime, timezone, timedelta
        JST = timezone(timedelta(hours=9), name="JST")
        return datetime(2026, 5, 23, hour, minute, 0, tzinfo=JST)

    # ------ no start_time in either table ------

    def test_no_race_status_no_race_returns_false(self):
        """No RaceStatus AND no Race record → False."""
        from backend.main import _should_auto_freeze

        session = self._make_session(race_status=None, race=None)
        with patch("backend.main.get_session", return_value=session):
            result = _should_auto_freeze("202605230601")

        assert result is False

    def test_race_status_exists_but_no_start_time_falls_back_to_race(self):
        """RaceStatus row with empty start_time → falls back to Race, no Race → False."""
        from backend.main import _should_auto_freeze

        status = MagicMock()
        status.start_time = ""  # falsy

        session = self._make_session(race_status=status, race=None)
        with patch("backend.main.get_session", return_value=session):
            result = _should_auto_freeze("202605230601")

        assert result is False

    def test_race_status_none_start_time_falls_back_to_race(self):
        """RaceStatus.start_time is None → falls back to Race, no Race → False."""
        from backend.main import _should_auto_freeze

        status = MagicMock()
        status.start_time = None

        session = self._make_session(race_status=status, race=None)
        with patch("backend.main.get_session", return_value=session):
            result = _should_auto_freeze("202605230601")

        assert result is False

    # ------ boundary: exactly at threshold ------

    def test_exactly_at_threshold_returns_true(self):
        """mins_to_post == FREEZE_THRESHOLD_MINS (6) → True (<=)."""
        from backend.main import _should_auto_freeze

        status = MagicMock()
        status.start_time = "15:30"

        session = self._make_session(race_status=status)
        # now is 15:24 → exactly 6 minutes to post
        now = self._mock_now_jst(15, 24)

        with patch("backend.main.get_session", return_value=session), \
             patch("backend.main.now_jst", return_value=now):
            result = _should_auto_freeze("202605230601")

        assert result is True

    def test_one_minute_inside_threshold_returns_true(self):
        """6 minutes to post (< 7) → True."""
        from backend.main import _should_auto_freeze

        status = MagicMock()
        status.start_time = "15:30"

        session = self._make_session(race_status=status)
        now = self._mock_now_jst(15, 24)  # 6 min to post

        with patch("backend.main.get_session", return_value=session), \
             patch("backend.main.now_jst", return_value=now):
            result = _should_auto_freeze("202605230601")

        assert result is True

    def test_one_minute_outside_threshold_returns_false(self):
        """8 minutes to post (> 7) → False."""
        from backend.main import _should_auto_freeze

        status = MagicMock()
        status.start_time = "15:30"

        session = self._make_session(race_status=status)
        now = self._mock_now_jst(15, 22)  # 8 min to post

        with patch("backend.main.get_session", return_value=session), \
             patch("backend.main.now_jst", return_value=now):
            result = _should_auto_freeze("202605230601")

        assert result is False

    def test_post_time_already_passed_returns_true(self):
        """Negative mins_to_post (race already started) → True (<= 7)."""
        from backend.main import _should_auto_freeze

        status = MagicMock()
        status.start_time = "15:30"

        session = self._make_session(race_status=status)
        now = self._mock_now_jst(15, 45)  # 15 min AFTER post

        with patch("backend.main.get_session", return_value=session), \
             patch("backend.main.now_jst", return_value=now):
            result = _should_auto_freeze("202605230601")

        assert result is True

    def test_far_future_race_returns_false(self):
        """Race is 60 minutes away → False."""
        from backend.main import _should_auto_freeze

        status = MagicMock()
        status.start_time = "16:30"

        session = self._make_session(race_status=status)
        now = self._mock_now_jst(15, 30)  # 60 min before

        with patch("backend.main.get_session", return_value=session), \
             patch("backend.main.now_jst", return_value=now):
            result = _should_auto_freeze("202605230601")

        assert result is False

    # ------ fallback path: RaceStatus missing, Race table used ------

    def test_no_race_status_uses_race_table_start_time(self):
        """When RaceStatus is absent, start_time from Race table is used."""
        from backend.main import _should_auto_freeze

        race = MagicMock()
        race.start_time = "10:05"

        session = self._make_session(race_status=None, race=race)
        now = self._mock_now_jst(10, 0)  # 5 min to post → within threshold

        with patch("backend.main.get_session", return_value=session), \
             patch("backend.main.now_jst", return_value=now):
            result = _should_auto_freeze("202605230601")

        assert result is True

    def test_race_table_used_when_status_has_no_start_time(self):
        """RaceStatus row exists but start_time is blank — Race table used."""
        from backend.main import _should_auto_freeze

        status = MagicMock()
        status.start_time = ""

        race = MagicMock()
        race.start_time = "11:00"

        session = self._make_session(race_status=status, race=race)
        now = self._mock_now_jst(11, 0)  # at post time → True

        with patch("backend.main.get_session", return_value=session), \
             patch("backend.main.now_jst", return_value=now):
            result = _should_auto_freeze("202605230601")

        assert result is True

    def test_malformed_start_time_returns_false(self):
        """Unparseable start_time string (e.g. 'N/A') → False (exception swallowed)."""
        from backend.main import _should_auto_freeze

        status = MagicMock()
        status.start_time = "N/A"

        session = self._make_session(race_status=status)
        now = self._mock_now_jst(12, 0)

        with patch("backend.main.get_session", return_value=session), \
             patch("backend.main.now_jst", return_value=now):
            result = _should_auto_freeze("202605230601")

        assert result is False

    def test_db_session_always_closed(self):
        """DB session must be closed even when an exception is raised."""
        from backend.main import _should_auto_freeze

        session = MagicMock()
        session.query.side_effect = RuntimeError("DB exploded")

        with patch("backend.main.get_session", return_value=session):
            result = _should_auto_freeze("202605230601")

        session.close.assert_called_once()
        # Should not propagate the error — safe fallback
        assert result is False


# =====================================================================
# 3. _acquire_lock / _release_lock
# =====================================================================

class TestPidLock:
    """Unit tests for _acquire_lock and _release_lock in realtime_worker.

    All OS operations (path existence, kill, file I/O, remove) are patched so
    tests never touch the real filesystem or signal real processes.
    """

    LOCK_PATH = "/tmp/keiba-realtime-worker.lock"

    # ------ _acquire_lock ------

    def test_acquire_when_no_lock_file_returns_true_and_writes_pid(self):
        """No existing lock file → acquire succeeds, PID is written."""
        from backend.realtime_worker import _acquire_lock

        written = {}

        def fake_open(path, mode="r"):
            m = mock_open()()
            if mode == "w":
                # Capture what was written
                def fake_write(data):
                    written["pid"] = data
                m.write = fake_write
                m.__enter__ = MagicMock(return_value=m)
                m.__exit__ = MagicMock(return_value=False)
            return m

        with patch("os.path.exists", return_value=False), \
             patch("builtins.open", mock_open()) as m, \
             patch("os.getpid", return_value=12345):
            result = _acquire_lock()

        assert result is True
        # Verify the write call contained the PID
        m.return_value.__enter__.return_value.write.assert_called_once_with("12345")

    def test_acquire_when_live_process_holds_lock_returns_false(self):
        """Lock file exists and process is alive → return False."""
        from backend.realtime_worker import _acquire_lock

        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", mock_open(read_data="99999")), \
             patch("os.kill", return_value=None):  # kill(pid, 0) succeeds → alive
            result = _acquire_lock()

        assert result is False

    def test_acquire_when_stale_lock_dead_pid_returns_true(self):
        """Lock file exists but process is gone (ProcessLookupError) → acquire succeeds."""
        from backend.realtime_worker import _acquire_lock

        removed = []

        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", mock_open(read_data="55555")), \
             patch("os.kill", side_effect=ProcessLookupError("no such process")), \
             patch("os.remove", side_effect=lambda p: removed.append(p)), \
             patch("os.getpid", return_value=12345):
            result = _acquire_lock()

        assert result is True
        assert self.LOCK_PATH in removed

    def test_acquire_stale_lock_with_permission_error_returns_true(self):
        """PermissionError from os.kill (process owned by another user) → treated as stale."""
        from backend.realtime_worker import _acquire_lock

        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", mock_open(read_data="1")), \
             patch("os.kill", side_effect=PermissionError("not permitted")), \
             patch("os.remove"), \
             patch("os.getpid", return_value=12345):
            result = _acquire_lock()

        assert result is True

    def test_acquire_stale_lock_removes_lock_file_before_writing(self):
        """After removing stale lock, a new lock file is written with current PID."""
        from backend.realtime_worker import _acquire_lock

        calls = []

        def fake_open(path, mode="r"):
            calls.append(("open", mode))
            if mode == "r":
                return mock_open(read_data="55555")()
            m = MagicMock()
            m.__enter__ = MagicMock(return_value=m)
            m.__exit__ = MagicMock(return_value=False)
            return m

        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", side_effect=fake_open), \
             patch("os.kill", side_effect=ProcessLookupError()), \
             patch("os.remove"), \
             patch("os.getpid", return_value=77777):
            result = _acquire_lock()

        assert result is True
        modes = [c[1] for c in calls]
        assert "r" in modes
        assert "w" in modes

    # ------ _release_lock ------

    def test_release_removes_lock_when_pid_matches(self):
        """release with matching PID removes the lock file."""
        from backend.realtime_worker import _release_lock

        removed = []

        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", mock_open(read_data="42")), \
             patch("os.getpid", return_value=42), \
             patch("os.remove", side_effect=lambda p: removed.append(p)):
            _release_lock()

        assert self.LOCK_PATH in removed

    def test_release_does_nothing_when_pid_does_not_match(self):
        """release when PID differs → lock file not removed."""
        from backend.realtime_worker import _release_lock

        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", mock_open(read_data="9999")), \
             patch("os.getpid", return_value=42), \
             patch("os.remove") as mock_remove:
            _release_lock()

        mock_remove.assert_not_called()

    def test_release_does_nothing_when_no_lock_file(self):
        """release when no lock file exists → no exception, no remove."""
        from backend.realtime_worker import _release_lock

        with patch("os.path.exists", return_value=False), \
             patch("os.remove") as mock_remove:
            _release_lock()  # must not raise

        mock_remove.assert_not_called()

    def test_release_is_silent_on_io_error(self):
        """Exceptions during release are silently swallowed."""
        from backend.realtime_worker import _release_lock

        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", side_effect=IOError("disk full")):
            _release_lock()  # must not propagate

    # ------ integration: acquire then release cycle ------

    def test_acquire_returns_false_on_unexpected_os_error(self):
        """If open(LOCK_FILE, 'w') raises an unexpected exception, return False."""
        from backend.realtime_worker import _acquire_lock

        def fake_open(path, mode="r"):
            if mode == "w":
                raise OSError("read-only filesystem")
            return mock_open(read_data="")()

        with patch("os.path.exists", return_value=False), \
             patch("builtins.open", side_effect=fake_open):
            result = _acquire_lock()

        assert result is False

    def test_acquire_then_release_full_cycle(self):
        """Acquire writes PID, release with same PID removes file."""
        from backend.realtime_worker import _acquire_lock, _release_lock

        removed = []
        written_pid = []

        write_mock = MagicMock()
        write_mock.__enter__ = MagicMock(return_value=write_mock)
        write_mock.__exit__ = MagicMock(return_value=False)
        write_mock.write = lambda data: written_pid.append(data)

        def fake_open(path, mode="r"):
            if mode == "w":
                return write_mock
            # For release read: return what was "written"
            pid_str = written_pid[0] if written_pid else "0"
            m = mock_open(read_data=pid_str)()
            m.__enter__ = MagicMock(return_value=m)
            m.__exit__ = MagicMock(return_value=False)
            return m

        with patch("os.path.exists", return_value=False), \
             patch("builtins.open", side_effect=fake_open), \
             patch("os.getpid", return_value=5050), \
             patch("os.remove", side_effect=lambda p: removed.append(p)):
            acquired = _acquire_lock()
            # Now simualte file exists for release
            with patch("os.path.exists", return_value=True):
                _release_lock()

        assert acquired is True
        assert written_pid == ["5050"]
        assert self.LOCK_PATH in removed
