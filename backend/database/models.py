from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()


class Race(Base):
    __tablename__ = "races"

    race_id = Column(String, primary_key=True)
    race_name = Column(String, nullable=False)
    race_number = Column(Integer, nullable=False)
    grade = Column(String, nullable=True)
    distance = Column(Integer, nullable=False)
    surface = Column(String, nullable=False)
    course_detail = Column(String, default="")
    start_time = Column(String, default="")
    racecourse_code = Column(String, nullable=False)
    date = Column(String, nullable=False)
    head_count = Column(Integer, default=0)
    scraped_at = Column(DateTime, default=datetime.utcnow)


class HorseEntry(Base):
    __tablename__ = "horse_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(String, nullable=False, index=True)
    frame_number = Column(Integer, nullable=False)
    horse_number = Column(Integer, nullable=False)
    horse_name = Column(String, nullable=False)
    horse_id = Column(String, default="")
    sire_name = Column(String, default="")
    dam_name = Column(String, default="")
    coat_color = Column(String, default="")
    weight_carried = Column(Float, default=0.0)
    age = Column(String, default="")
    jockey_name = Column(String, default="")
    jockey_id = Column(String, default="")
    trainer_name = Column(String, default="")
    trainer_id = Column(String, default="")
    horse_weight = Column(String, default="")
    odds = Column(Float, nullable=True)
    popularity = Column(Integer, nullable=True)
    is_scratched = Column(Boolean, default=False)
    brood_mare_sire = Column(String, default="")
    past_races_json = Column(Text, default="[]")


class OddsSnapshot(Base):
    """Time-series odds history for each horse."""
    __tablename__ = "odds_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(String, nullable=False, index=True)
    horse_number = Column(Integer, nullable=False)
    odds_type = Column(String, default="tansho")  # tansho, fukusho
    odds = Column(Float, nullable=False)
    popularity = Column(Integer, nullable=True)
    captured_at = Column(DateTime, default=datetime.utcnow)


class CombinationOdds(Base):
    """Real-time combination odds (馬連/ワイド/3連複/3連単)."""
    __tablename__ = "combination_odds"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(String, nullable=False, index=True)
    bet_type = Column(String, nullable=False)  # umaren, wide, sanrenpuku, sanrentan
    horses_key = Column(String, nullable=False)  # "04-07", "05-08-12"
    odds = Column(Float, nullable=False)
    captured_at = Column(DateTime, default=datetime.utcnow)


class PredictionsCache(Base):
    """Cached predictions, bets, and longshot for each race."""
    __tablename__ = "predictions_cache"

    race_id = Column(String, primary_key=True)
    predictions_json = Column(Text, default="[]")
    bets_json = Column(Text, default="[]")
    longshot_json = Column(Text, nullable=True)
    pattern = Column(String, default="")
    frozen = Column(Boolean, default=False)
    updated_at = Column(DateTime, default=datetime.utcnow)


class RaceStatus(Base):
    """Race lifecycle state management."""
    __tablename__ = "race_status"

    race_id = Column(String, primary_key=True)
    status = Column(String, default="upcoming")  # upcoming, active, frozen, finished
    start_time = Column(String, default="")
    track_condition = Column(String, default="")
    last_odds_update = Column(DateTime, nullable=True)
    last_prediction_update = Column(DateTime, nullable=True)


class RaceResult(Base):
    __tablename__ = "race_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(String, nullable=False, index=True)
    horse_number = Column(Integer, nullable=False)
    finish_position = Column(Integer, nullable=True)
    finish_time = Column(String, default="")
    margin = Column(String, default="")
