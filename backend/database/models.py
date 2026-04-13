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


class RaceResult(Base):
    __tablename__ = "race_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(String, nullable=False, index=True)
    horse_number = Column(Integer, nullable=False)
    finish_position = Column(Integer, nullable=True)
    finish_time = Column(String, default="")
    margin = Column(String, default="")
