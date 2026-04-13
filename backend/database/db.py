import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .models import Base

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
DB_PATH = os.path.join(DATA_DIR, "jra_races.db")

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)


def init_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    Base.metadata.create_all(engine)


def get_session() -> Session:
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()
