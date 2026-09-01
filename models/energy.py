"""Energy data models for EcoHome Energy Advisor."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class EnergyUsage(Base):
    __tablename__ = "energy_usage"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    consumption_kwh = Column(Float, nullable=False)
    device_type = Column(String(50), nullable=True)
    device_name = Column(String(100), nullable=True)
    cost_usd = Column(Float, nullable=True)

    def __repr__(self):
        return (
            f"<EnergyUsage(timestamp={self.timestamp}, "
            f"consumption={self.consumption_kwh}kWh, device={self.device_name})>"
        )


class SolarGeneration(Base):
    __tablename__ = "solar_generation"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    generation_kwh = Column(Float, nullable=False)
    weather_condition = Column(String(50), nullable=True)
    temperature_c = Column(Float, nullable=True)
    solar_irradiance = Column(Float, nullable=True)

    def __repr__(self):
        return (
            f"<SolarGeneration(timestamp={self.timestamp}, "
            f"generation={self.generation_kwh}kWh, weather={self.weather_condition})>"
        )


def _pick_db_path(db_path: str) -> str:
    import sqlite3
    requested = Path(db_path)
    candidates = [Path("/tmp") / Path(db_path).name, requested]
    for path in candidates:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            con = sqlite3.connect(str(path))
            con.execute("create table if not exists _probe(x int)")
            con.execute("insert into _probe values (1)")
            con.commit()
            con.close()
            return str(path)
        except OSError:
            continue
        except Exception:
            continue
    return str(candidates[0])


class DatabaseManager:
    def __init__(self, db_path: str = "data/energy_data.db"):
        self.requested_path = db_path
        self.db_path = _pick_db_path(db_path)
        self.engine = create_engine(f"sqlite:///{self.db_path}", future=True)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def create_tables(self):
        Base.metadata.create_all(bind=self.engine)
        print(f"Database tables created at {self.db_path}")

    def get_session(self):
        return self.SessionLocal()

    def add_usage_record(
        self,
        timestamp: datetime,
        consumption_kwh: float,
        device_type: str = None,
        device_name: str = None,
        cost_usd: float = None,
    ):
        session = self.get_session()
        try:
            record = EnergyUsage(
                timestamp=timestamp,
                consumption_kwh=consumption_kwh,
                device_type=device_type,
                device_name=device_name,
                cost_usd=cost_usd,
            )
            session.add(record)
            session.commit()
            return record
        finally:
            session.close()

    def add_generation_record(
        self,
        timestamp: datetime,
        generation_kwh: float,
        weather_condition: str = None,
        temperature_c: float = None,
        solar_irradiance: float = None,
    ):
        session = self.get_session()
        try:
            record = SolarGeneration(
                timestamp=timestamp,
                generation_kwh=generation_kwh,
                weather_condition=weather_condition,
                temperature_c=temperature_c,
                solar_irradiance=solar_irradiance,
            )
            session.add(record)
            session.commit()
            return record
        finally:
            session.close()

    def add_usage_records(self, rows: list):
        session = self.get_session()
        try:
            session.bulk_insert_mappings(EnergyUsage, rows)
            session.commit()
        finally:
            session.close()

    def add_generation_records(self, rows: list):
        session = self.get_session()
        try:
            session.bulk_insert_mappings(SolarGeneration, rows)
            session.commit()
        finally:
            session.close()

    def get_usage_by_date_range(self, start_date: datetime, end_date: datetime):
        session = self.get_session()
        try:
            return (
                session.query(EnergyUsage)
                .filter(
                    EnergyUsage.timestamp >= start_date,
                    EnergyUsage.timestamp <= end_date,
                )
                .order_by(EnergyUsage.timestamp)
                .all()
            )
        finally:
            session.close()

    def get_generation_by_date_range(self, start_date: datetime, end_date: datetime):
        session = self.get_session()
        try:
            return (
                session.query(SolarGeneration)
                .filter(
                    SolarGeneration.timestamp >= start_date,
                    SolarGeneration.timestamp <= end_date,
                )
                .order_by(SolarGeneration.timestamp)
                .all()
            )
        finally:
            session.close()

    def get_recent_usage(self, hours: int = 24):
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        return self.get_usage_by_date_range(start_time, end_time)

    def get_recent_generation(self, hours: int = 24):
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        return self.get_generation_by_date_range(start_time, end_time)
