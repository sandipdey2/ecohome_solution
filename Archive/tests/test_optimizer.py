"""Sanity checks for the net-cost scheduler."""

from ecohome.config import HOME
from ecohome.optimizer import Slot, schedule_flexible_load


def _slot(hour, price, solar, day="2026-09-02"):
    return Slot(
        timestamp=f"{day}T{hour:02d}:00:00",
        hour=hour,
        date=day,
        price=price,
        period="on_peak" if 18 <= hour < 23 else "off_peak" if hour >= 23 or hour < 6 else "mid_peak",
        solar_kwh=solar,
        irradiance=solar * 200,
        temperature_c=30,
        condition="sunny" if solar > 1 else "night",
    )


def test_prefers_solar_over_evening_peak():
    slots = [_slot(h, 12.4 if 18 <= h < 23 else 8.2 if 10 <= h < 16 else 5.5, 3.0 if 11 <= h <= 14 else 0.0) for h in range(24)]
    plan = schedule_flexible_load(kwh=6.6, power_kw=3.3, slots=slots, prefer_contiguous=True)
    assert plan["savings_vs_evening"] > 0
    assert not any(18 <= h <= 21 for h in plan["hours"])
    assert plan["solar_kwh"] > 0
