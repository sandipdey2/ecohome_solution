"""SQLAlchemy models for household energy usage and rooftop solar generation."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class EnergyUsage(Base):
    """One interval of consumption for a named device."""

    __tablename__ = "energy_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    consumption_kwh: Mapped[float] = mapped_column(Float, nullable=False)
    device_type: Mapped[str | None] = mapped_column(String(50), index=True)
    device_name: Mapped[str | None] = mapped_column(String(100))
    cost: Mapped[float | None] = mapped_column(Float)  # in home currency
    period: Mapped[str | None] = mapped_column(String(20))

    def __repr__(self) -> str:
        return (
            f"<EnergyUsage {self.timestamp} {self.device_type} "
            f"{self.consumption_kwh} kWh>"
        )


class SolarGeneration(Base):
    """One interval of rooftop PV production."""

    __tablename__ = "solar_generation"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    generation_kwh: Mapped[float] = mapped_column(Float, nullable=False)
    weather_condition: Mapped[str | None] = mapped_column(String(50))
    temperature_c: Mapped[float | None] = mapped_column(Float)
    solar_irradiance: Mapped[float | None] = mapped_column(Float)  # W/m²
    self_consumed_kwh: Mapped[float | None] = mapped_column(Float)
    exported_kwh: Mapped[float | None] = mapped_column(Float)

    def __repr__(self) -> str:
        return f"<SolarGeneration {self.timestamp} {self.generation_kwh} kWh>"
