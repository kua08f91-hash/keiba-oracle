"""TDD tests for _auto_migrate_sqlite() in backend/database/db.py.

RED → GREEN cycle covering:
1. Missing column is added via ALTER TABLE
2. Existing columns are not modified (no errors, no duplication)
3. Multiple missing columns are all added
4. Table not yet created is silently skipped (create_all handles it)
5. Non-SQLite engine causes _auto_migrate_sqlite to not be called
6. Integration: init_db calls _auto_migrate_sqlite for SQLite engines
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest
from unittest.mock import patch, MagicMock, call
from sqlalchemy import create_engine, text, inspect, Column, String, Integer, Float
from sqlalchemy.orm import declarative_base

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_temp_sqlite_engine():
    """Return a fresh in-process SQLite engine backed by a temp file.

    Using a file (not :memory:) so that a second connection via inspect sees
    the same schema that was created by the first connection.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    url = f"sqlite:///{tmp.name}"
    engine = create_engine(url, echo=False)
    return engine, tmp.name


def _column_names(engine, table_name: str) -> set[str]:
    """Return the set of column names for *table_name* as seen by SQLAlchemy inspect."""
    insp = inspect(engine)
    return {col["name"] for col in insp.get_columns(table_name)}


# ---------------------------------------------------------------------------
# 1. Missing column is added
# ---------------------------------------------------------------------------

class TestMissingColumnIsAdded:
    """_auto_migrate_sqlite adds a column that is absent from an existing table."""

    def test_single_missing_column_is_added(self):
        """Column present in SQLAlchemy model but absent in DB is added via ALTER TABLE."""
        from backend.database import db as db_module

        engine, db_path = _make_temp_sqlite_engine()
        try:
            # Create the 'races' table WITHOUT the 'grade' column
            with engine.connect() as conn:
                conn.execute(text(
                    "CREATE TABLE races ("
                    "  race_id TEXT PRIMARY KEY,"
                    "  race_name TEXT NOT NULL,"
                    "  race_number INTEGER NOT NULL,"
                    "  distance INTEGER NOT NULL,"
                    "  surface TEXT NOT NULL"
                    ")"
                ))
                conn.commit()

            before = _column_names(engine, "races")
            assert "grade" not in before, "pre-condition: grade must be absent"

            # Patch the module-level engine so _auto_migrate_sqlite uses our DB
            with patch.object(db_module, "engine", engine):
                db_module._auto_migrate_sqlite()

            after = _column_names(engine, "races")
            assert "grade" in after, "grade column must be present after migration"
        finally:
            engine.dispose()
            os.unlink(db_path)

    def test_added_column_has_correct_type(self):
        """The migrated column stores values of the expected SQLAlchemy type."""
        from backend.database import db as db_module

        engine, db_path = _make_temp_sqlite_engine()
        try:
            with engine.connect() as conn:
                conn.execute(text(
                    "CREATE TABLE races ("
                    "  race_id TEXT PRIMARY KEY,"
                    "  race_name TEXT NOT NULL,"
                    "  race_number INTEGER NOT NULL,"
                    "  distance INTEGER NOT NULL,"
                    "  surface TEXT NOT NULL"
                    ")"
                ))
                conn.commit()

            with patch.object(db_module, "engine", engine):
                db_module._auto_migrate_sqlite()

            # Insert a value into the newly added 'grade' column (VARCHAR/TEXT)
            with engine.connect() as conn:
                conn.execute(text(
                    "INSERT INTO races (race_id, race_name, race_number, distance, surface, grade)"
                    " VALUES ('r1', 'テスト', 1, 1800, '芝', 'G1')"
                ))
                conn.commit()
                row = conn.execute(text("SELECT grade FROM races WHERE race_id='r1'")).fetchone()

            assert row[0] == "G1"
        finally:
            engine.dispose()
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# 2. Existing columns are not modified
# ---------------------------------------------------------------------------

class TestExistingColumnsNotModified:
    """When all model columns already exist _auto_migrate_sqlite runs without error."""

    def test_no_error_when_all_columns_present(self):
        """Full schema table causes no exception and no ALTER TABLE."""
        from backend.database import db as db_module
        from backend.database.models import Base

        engine, db_path = _make_temp_sqlite_engine()
        try:
            # Create all tables exactly as the models define them
            Base.metadata.create_all(engine)

            columns_before = _column_names(engine, "races")

            with patch.object(db_module, "engine", engine):
                # Must not raise
                db_module._auto_migrate_sqlite()

            columns_after = _column_names(engine, "races")
            assert columns_before == columns_after, "column set must be unchanged"
        finally:
            engine.dispose()
            os.unlink(db_path)

    def test_alter_table_not_called_when_schema_is_current(self):
        """No ALTER TABLE statement is executed when the schema is already up-to-date."""
        from backend.database import db as db_module
        from backend.database.models import Base
        from sqlalchemy.engine.base import Connection as SAConnection

        engine, db_path = _make_temp_sqlite_engine()
        try:
            Base.metadata.create_all(engine)

            executed_statements: list[str] = []

            # Patch the real SQLAlchemy Connection.execute to record calls.
            # This lets inspect() keep using real Connection objects (required
            # by dialect.has_table) while still capturing SQL statements.
            original_execute = SAConnection.execute

            def recording_execute(self, stmt, *args, **kwargs):
                executed_statements.append(str(stmt))
                return original_execute(self, stmt, *args, **kwargs)

            with patch.object(db_module, "engine", engine), \
                 patch.object(SAConnection, "execute", recording_execute):
                db_module._auto_migrate_sqlite()

            alter_calls = [s for s in executed_statements if "ALTER" in s.upper()]
            assert alter_calls == [], f"unexpected ALTER TABLE calls: {alter_calls}"
        finally:
            engine.dispose()
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# 3. Multiple missing columns added
# ---------------------------------------------------------------------------

class TestMultipleMissingColumnsAdded:
    """All missing columns are added when several are absent."""

    def test_two_missing_columns_both_added(self):
        """horse_entries table missing two columns gets both added."""
        from backend.database import db as db_module

        engine, db_path = _make_temp_sqlite_engine()
        try:
            # Build horse_entries without 'brood_mare_sire' and 'past_races_json'
            with engine.connect() as conn:
                conn.execute(text(
                    "CREATE TABLE horse_entries ("
                    "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "  race_id TEXT NOT NULL,"
                    "  frame_number INTEGER NOT NULL,"
                    "  horse_number INTEGER NOT NULL,"
                    "  horse_name TEXT NOT NULL"
                    ")"
                ))
                conn.commit()

            before = _column_names(engine, "horse_entries")
            assert "brood_mare_sire" not in before
            assert "past_races_json" not in before

            with patch.object(db_module, "engine", engine):
                db_module._auto_migrate_sqlite()

            after = _column_names(engine, "horse_entries")
            assert "brood_mare_sire" in after, "brood_mare_sire must be present after migration"
            assert "past_races_json" in after, "past_races_json must be present after migration"
        finally:
            engine.dispose()
            os.unlink(db_path)

    def test_all_missing_columns_added_across_multiple_tables(self):
        """Missing columns in two different tables are both migrated in one call."""
        from backend.database import db as db_module

        engine, db_path = _make_temp_sqlite_engine()
        try:
            with engine.connect() as conn:
                # races — missing 'grade' and 'start_time'
                conn.execute(text(
                    "CREATE TABLE races ("
                    "  race_id TEXT PRIMARY KEY,"
                    "  race_name TEXT NOT NULL,"
                    "  race_number INTEGER NOT NULL,"
                    "  distance INTEGER NOT NULL,"
                    "  surface TEXT NOT NULL"
                    ")"
                ))
                # predictions_cache — missing 'frozen' and 'analysis_text'
                conn.execute(text(
                    "CREATE TABLE predictions_cache ("
                    "  race_id TEXT PRIMARY KEY,"
                    "  predictions_json TEXT,"
                    "  bets_json TEXT"
                    ")"
                ))
                conn.commit()

            with patch.object(db_module, "engine", engine):
                db_module._auto_migrate_sqlite()

            races_cols = _column_names(engine, "races")
            pc_cols = _column_names(engine, "predictions_cache")

            assert "grade" in races_cols
            assert "start_time" in races_cols
            assert "frozen" in pc_cols
            assert "analysis_text" in pc_cols
        finally:
            engine.dispose()
            os.unlink(db_path)

    def test_correct_count_of_columns_after_migration(self):
        """After migration the total column count matches the model definition."""
        from backend.database import db as db_module
        from backend.database.models import Base

        engine, db_path = _make_temp_sqlite_engine()
        try:
            # Create races with minimal columns only
            with engine.connect() as conn:
                conn.execute(text(
                    "CREATE TABLE races ("
                    "  race_id TEXT PRIMARY KEY,"
                    "  race_name TEXT NOT NULL,"
                    "  race_number INTEGER NOT NULL,"
                    "  distance INTEGER NOT NULL,"
                    "  surface TEXT NOT NULL"
                    ")"
                ))
                conn.commit()

            with patch.object(db_module, "engine", engine):
                db_module._auto_migrate_sqlite()

            expected = {col.name for col in Base.metadata.tables["races"].columns}
            actual = _column_names(engine, "races")
            assert expected == actual, (
                f"column mismatch — missing: {expected - actual}, extra: {actual - expected}"
            )
        finally:
            engine.dispose()
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# 4. Table that doesn't exist yet is silently skipped
# ---------------------------------------------------------------------------

class TestMissingTableIsSkipped:
    """_auto_migrate_sqlite skips tables that have not been created yet."""

    def test_no_error_when_table_does_not_exist(self):
        """Calling _auto_migrate_sqlite on an empty DB raises no exception."""
        from backend.database import db as db_module

        engine, db_path = _make_temp_sqlite_engine()
        try:
            # Absolutely empty database — no tables at all
            with patch.object(db_module, "engine", engine):
                db_module._auto_migrate_sqlite()  # must not raise
        finally:
            engine.dispose()
            os.unlink(db_path)

    def test_nonexistent_table_does_not_produce_alter_table(self):
        """A table absent from the DB is not ALTERed (create_all is responsible)."""
        from backend.database import db as db_module
        from sqlalchemy.engine.base import Connection as SAConnection

        engine, db_path = _make_temp_sqlite_engine()
        try:
            executed_sql: list[str] = []

            # Patch the real Connection.execute to capture statements.
            # This avoids bypassing dialect.has_table()'s type-check on the
            # connection object that would occur with a proxy wrapper approach.
            original_execute = SAConnection.execute

            def recording_execute(self, stmt, *a, **kw):
                executed_sql.append(str(stmt))
                return original_execute(self, stmt, *a, **kw)

            with patch.object(db_module, "engine", engine), \
                 patch.object(SAConnection, "execute", recording_execute):
                db_module._auto_migrate_sqlite()

            alter_sql = [s for s in executed_sql if "ALTER" in s.upper()]
            assert alter_sql == [], "no SQL should be issued against a completely empty DB"
        finally:
            engine.dispose()
            os.unlink(db_path)

    def test_existing_tables_migrated_even_when_some_tables_absent(self):
        """Only the tables that exist are migrated; absent ones are safely skipped."""
        from backend.database import db as db_module

        engine, db_path = _make_temp_sqlite_engine()
        try:
            # Only create 'races' — all other model tables are absent
            with engine.connect() as conn:
                conn.execute(text(
                    "CREATE TABLE races ("
                    "  race_id TEXT PRIMARY KEY,"
                    "  race_name TEXT NOT NULL,"
                    "  race_number INTEGER NOT NULL,"
                    "  distance INTEGER NOT NULL,"
                    "  surface TEXT NOT NULL"
                    ")"
                ))
                conn.commit()

            with patch.object(db_module, "engine", engine):
                db_module._auto_migrate_sqlite()  # must not raise

            # 'races' should have been migrated
            assert "grade" in _column_names(engine, "races")
        finally:
            engine.dispose()
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# 5. Non-SQLite engine: _auto_migrate_sqlite is not called
# ---------------------------------------------------------------------------

class TestNonSqliteEngineSkipped:
    """init_db must NOT call _auto_migrate_sqlite when the engine is PostgreSQL."""

    def test_init_db_skips_migration_for_postgresql_engine(self):
        """PostgreSQL engine URL causes _auto_migrate_sqlite to be bypassed."""
        from backend.database import db as db_module

        mock_engine = MagicMock()
        mock_engine.url = MagicMock()
        mock_engine.url.__str__ = lambda self: "postgresql://user:pass@host/db"

        with patch.object(db_module, "engine", mock_engine), \
             patch.object(db_module.Base.metadata, "create_all"), \
             patch.object(db_module, "_auto_migrate_sqlite") as mock_migrate, \
             patch("os.makedirs"):
            db_module.init_db()

        mock_migrate.assert_not_called()

    def test_init_db_skips_migration_for_postgres_scheme(self):
        """Both 'postgresql://' and 'postgres://' (before normalisation) are non-SQLite."""
        from backend.database import db as db_module

        for scheme in ("postgresql://user:pass@host/db", "postgres://user:pass@host/db"):
            mock_engine = MagicMock()
            mock_engine.url = MagicMock()
            mock_engine.url.__str__ = lambda self, s=scheme: s

            with patch.object(db_module, "engine", mock_engine), \
                 patch.object(db_module.Base.metadata, "create_all"), \
                 patch.object(db_module, "_auto_migrate_sqlite") as mock_migrate, \
                 patch("os.makedirs"):
                db_module.init_db()

            mock_migrate.assert_not_called(), f"should not migrate for scheme: {scheme}"

    def test_init_db_still_calls_create_all_for_postgresql(self):
        """Even when skipping migration, create_all is still invoked for PostgreSQL."""
        from backend.database import db as db_module

        mock_engine = MagicMock()
        mock_engine.url = MagicMock()
        mock_engine.url.__str__ = lambda self: "postgresql://user:pass@host/db"

        with patch.object(db_module, "engine", mock_engine), \
             patch.object(db_module.Base.metadata, "create_all") as mock_create_all, \
             patch.object(db_module, "_auto_migrate_sqlite"), \
             patch("os.makedirs"):
            db_module.init_db()

        mock_create_all.assert_called_once_with(mock_engine)


# ---------------------------------------------------------------------------
# 6. Integration: init_db calls _auto_migrate_sqlite for SQLite
# ---------------------------------------------------------------------------

class TestInitDbIntegration:
    """init_db triggers _auto_migrate_sqlite when the active engine is SQLite."""

    def test_init_db_calls_migrate_for_sqlite_engine(self):
        """init_db invokes _auto_migrate_sqlite when engine URL starts with 'sqlite'."""
        from backend.database import db as db_module

        mock_engine = MagicMock()
        mock_engine.url = MagicMock()
        mock_engine.url.__str__ = lambda self: "sqlite:////tmp/test.db"

        with patch.object(db_module, "engine", mock_engine), \
             patch.object(db_module.Base.metadata, "create_all"), \
             patch.object(db_module, "_auto_migrate_sqlite") as mock_migrate, \
             patch("os.makedirs"):
            db_module.init_db()

        mock_migrate.assert_called_once()

    def test_init_db_calls_create_all_before_migrate(self):
        """create_all must be called before _auto_migrate_sqlite so tables exist first."""
        from backend.database import db as db_module

        call_order: list[str] = []

        mock_engine = MagicMock()
        mock_engine.url = MagicMock()
        mock_engine.url.__str__ = lambda self: "sqlite:////tmp/test.db"

        def record_create_all(*args, **kwargs):
            call_order.append("create_all")

        def record_migrate():
            call_order.append("_auto_migrate_sqlite")

        with patch.object(db_module, "engine", mock_engine), \
             patch.object(db_module.Base.metadata, "create_all", side_effect=record_create_all), \
             patch.object(db_module, "_auto_migrate_sqlite", side_effect=record_migrate), \
             patch("os.makedirs"):
            db_module.init_db()

        assert call_order == ["create_all", "_auto_migrate_sqlite"], (
            f"expected create_all before migration, got: {call_order}"
        )

    def test_init_db_end_to_end_on_real_sqlite(self):
        """Full init_db run on a real temporary SQLite database succeeds end-to-end."""
        from backend.database import db as db_module
        from backend.database.models import Base

        engine, db_path = _make_temp_sqlite_engine()
        try:
            # Create races table missing a column to prove migration runs
            with engine.connect() as conn:
                conn.execute(text(
                    "CREATE TABLE races ("
                    "  race_id TEXT PRIMARY KEY,"
                    "  race_name TEXT NOT NULL,"
                    "  race_number INTEGER NOT NULL,"
                    "  distance INTEGER NOT NULL,"
                    "  surface TEXT NOT NULL"
                    ")"
                ))
                conn.commit()

            with patch.object(db_module, "engine", engine), \
                 patch("os.makedirs"):
                db_module.init_db()

            # All model columns must now be present
            expected = {col.name for col in Base.metadata.tables["races"].columns}
            actual = _column_names(engine, "races")
            assert expected == actual
        finally:
            engine.dispose()
            os.unlink(db_path)

    def test_init_db_sqlite_creates_data_dir(self):
        """init_db creates DATA_DIR when running on SQLite."""
        from backend.database import db as db_module

        mock_engine = MagicMock()
        mock_engine.url = MagicMock()
        mock_engine.url.__str__ = lambda self: "sqlite:////tmp/test.db"

        with patch.object(db_module, "engine", mock_engine), \
             patch.object(db_module.Base.metadata, "create_all"), \
             patch.object(db_module, "_auto_migrate_sqlite"), \
             patch("os.makedirs") as mock_makedirs:
            db_module.init_db()

        mock_makedirs.assert_called_once_with(db_module.DATA_DIR, exist_ok=True)
