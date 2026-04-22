"""Database connection — supports SQLite (local) and PostgreSQL (Railway).

Uses DATABASE_URL env var when set (Railway/cloud), falls back to local SQLite.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .models import Base

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
DB_PATH = os.path.join(DATA_DIR, "jra_races.db")

# DATABASE_URL from environment (Railway sets this automatically for Postgres)
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DB_PATH}")

# Railway/Heroku use postgres:// but SQLAlchemy requires postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)


def init_db():
    if DATABASE_URL.startswith("sqlite"):
        os.makedirs(DATA_DIR, exist_ok=True)
    Base.metadata.create_all(engine)


def get_session() -> Session:
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()
