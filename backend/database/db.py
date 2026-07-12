"""Database connection — supports SQLite (local) and PostgreSQL (Railway).

Uses DATABASE_URL env var when set (Railway/cloud), falls back to local SQLite.
If PostgreSQL connection fails, automatically falls back to SQLite.
"""
import logging
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from .models import Base

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
DB_PATH = os.path.join(DATA_DIR, "jra_races.db")
SQLITE_URL = f"sqlite:///{DB_PATH}"

# DATABASE_URL from environment (Railway sets this automatically for Postgres)
DATABASE_URL = os.environ.get("DATABASE_URL", SQLITE_URL)

# Railway/Heroku use postgres:// but SQLAlchemy requires postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)


def _create_engine_safe():
    """Create DB engine with fallback to SQLite if PostgreSQL fails."""
    if DATABASE_URL.startswith("sqlite"):
        return create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)

    # Try PostgreSQL first
    try:
        pg_engine = create_engine(
            DATABASE_URL, echo=False, pool_pre_ping=True,
            connect_args={"connect_timeout": 5},
        )
        with pg_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("PostgreSQL connection OK")
        return pg_engine
    except Exception as e:
        logger.warning("PostgreSQL connection failed: %s — falling back to SQLite", e)
        os.makedirs(DATA_DIR, exist_ok=True)
        return create_engine(SQLITE_URL, echo=False, pool_pre_ping=True)


engine = _create_engine_safe()


def init_db():
    if str(engine.url).startswith("sqlite"):
        os.makedirs(DATA_DIR, exist_ok=True)
    Base.metadata.create_all(engine)
    # Auto-migrate: add missing columns to existing tables (SQLite only)
    if str(engine.url).startswith("sqlite"):
        _auto_migrate_sqlite()


def _auto_migrate_sqlite():
    """Add missing columns to existing SQLite tables."""
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    for table_name, table in Base.metadata.tables.items():
        if not insp.has_table(table_name):
            continue
        existing_cols = {col['name'] for col in insp.get_columns(table_name)}
        for col in table.columns:
            if col.name not in existing_cols:
                col_type = col.type.compile(engine.dialect)
                with engine.connect() as conn:
                    conn.execute(text(
                        "ALTER TABLE %s ADD COLUMN %s %s" % (table_name, col.name, col_type)
                    ))
                    conn.commit()
                logger.info("Auto-migrated: %s.%s (%s)", table_name, col.name, col_type)


def get_session() -> Session:
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()
