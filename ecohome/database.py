"""SQLite persistence helpers for EcoHome energy data."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from ecohome.config import DB_PATH
from ecohome.models import Base, EnergyUsage, SolarGeneration


class DatabaseManager:
    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,
            future=True,
        )
        self.SessionLocal = sessionmaker(
            bind=self.engine, autoflush=False, autocommit=False, future=True
        )

    def create_tables(self) -> None:
        Base.metadata.create_all(self.engine)

    def drop_tables(self) -> None:
        Base.metadata.drop_all(self.engine)

    def session(self) -> Session:
        return self.SessionLocal()

    def usage_count(self) -> int:
        self.create_tables()
        with self.session() as s:
            return s.scalar(select(func.count(EnergyUsage.id))) or 0

    def generation_count(self) -> int:
        with self.session() as s:
            return s.scalar(select(func.count(SolarGeneration.id))) or 0

    def get_usage_by_date_range(
        self,
        start: datetime,
        end: datetime,
        device_type: str | None = None,
    ) -> list[EnergyUsage]:
        with self.session() as s:
            stmt = select(EnergyUsage).where(
                EnergyUsage.timestamp >= start,
                EnergyUsage.timestamp <= end,
            )
            if device_type:
                stmt = stmt.where(EnergyUsage.device_type == device_type)
            stmt = stmt.order_by(EnergyUsage.timestamp)
            return list(s.scalars(stmt).all())

    def get_generation_by_date_range(
        self, start: datetime, end: datetime
    ) -> list[SolarGeneration]:
        with self.session() as s:
            stmt = (
                select(SolarGeneration)
                .where(
                    SolarGeneration.timestamp >= start,
                    SolarGeneration.timestamp <= end,
                )
                .order_by(SolarGeneration.timestamp)
            )
            return list(s.scalars(stmt).all())

    def get_recent_usage(self, hours: int = 24) -> list[EnergyUsage]:
        end = datetime.now()
        return self.get_usage_by_date_range(end - timedelta(hours=hours), end)

    def get_recent_generation(self, hours: int = 24) -> list[SolarGeneration]:
        end = datetime.now()
        return self.get_generation_by_date_range(end - timedelta(hours=hours), end)

    def device_totals(
        self, start: datetime, end: datetime
    ) -> dict[str, dict[str, float]]:
        rows = self.get_usage_by_date_range(start, end)
        out: dict[str, dict[str, float]] = {}
        for r in rows:
            key = r.device_type or "unknown"
            bucket = out.setdefault(
                key, {"consumption_kwh": 0.0, "cost": 0.0, "records": 0}
            )
            bucket["consumption_kwh"] += r.consumption_kwh
            bucket["cost"] += r.cost or 0.0
            bucket["records"] += 1
        for b in out.values():
            b["consumption_kwh"] = round(b["consumption_kwh"], 2)
            b["cost"] = round(b["cost"], 2)
        return out
