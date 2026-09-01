"""Generate a realistic 60-day history for the demo household."""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

from ecohome.config import HOME, tou_period, tou_rate
from ecohome.database import DatabaseManager
from ecohome.models import EnergyUsage, SolarGeneration
from ecohome.weather import _solar_elevation_proxy


def seed_database(days: int = 60, seed: int = 42, reset: bool = True) -> dict:
    rng = random.Random(seed)
    db = DatabaseManager()
    if reset:
        db.drop_tables()
    db.create_tables()

    end = datetime.now().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)

    usage_rows: list[EnergyUsage] = []
    solar_rows: list[SolarGeneration] = []

    cursor = start
    while cursor <= end:
        hour = cursor.hour
        is_weekend = cursor.weekday() >= 5
        # Mild monsoon-ish variability: wetter every 6th day
        monsoon = 0.5 + 0.5 * rng.random() if cursor.timetuple().tm_yday % 6 == 0 else 0.85 + 0.15 * rng.random()
        elev = _solar_elevation_proxy(hour, HOME.latitude)
        irr = max(0.0, 980 * elev * monsoon + rng.uniform(-40, 40) * elev)
        gen = HOME.solar_capacity_kw * (irr / 1000.0) * HOME.solar_performance_ratio
        gen = max(0.0, gen + rng.uniform(-0.05, 0.05) * HOME.solar_capacity_kw * elev)
        if irr < 20:
            cond = "night"
        elif monsoon < 0.65:
            cond = "overcast"
        elif monsoon < 0.85:
            cond = "partly_cloudy"
        else:
            cond = "sunny"
        temp = 23 + 9 * max(0, math.sin((hour - 8) / 16 * math.pi)) + rng.uniform(-1.2, 1.2)

        solar_rows.append(
            SolarGeneration(
                timestamp=cursor,
                generation_kwh=round(gen, 3),
                weather_condition=cond,
                temperature_c=round(temp, 1),
                solar_irradiance=round(max(irr, 0), 1),
                self_consumed_kwh=None,
                exported_kwh=None,
            )
        )

        rate = tou_rate(hour, is_weekend)
        period = tou_period(hour)

        # Always-on + lighting / electronics
        base = 0.18 + (0.12 if 18 <= hour <= 23 else 0.0) + rng.uniform(0, 0.05)
        usage_rows.append(_usage(cursor, base, "other", "Always-on + lights", rate, period))

        # HVAC: heavy 13:00-23:00, lighter overnight
        if 13 <= hour <= 23:
            hvac = HOME.hvac_typical_kw * rng.uniform(0.55, 0.95)
        elif 0 <= hour <= 6:
            hvac = HOME.hvac_typical_kw * rng.uniform(0.15, 0.35)
        else:
            hvac = HOME.hvac_typical_kw * rng.uniform(0.25, 0.55)
        usage_rows.append(_usage(cursor, hvac, "HVAC", "Split AC living+bed", rate, period))

        # EV: naïve household currently charges 19:00-23:00 on weeknights
        # plus a longer Sunday top-up. This is *intentionally suboptimal*.
        if (not is_weekend and 19 <= hour <= 22) or (cursor.weekday() == 6 and 10 <= hour <= 14):
            ev = HOME.ev_charge_rate_kw * rng.uniform(0.85, 1.0)
            usage_rows.append(_usage(cursor, ev, "EV", "EV charger", rate, period))

        # Dishwasher: evenings ~5 nights a week (bad habit)
        if hour == 20 and rng.random() < 0.72:
            usage_rows.append(_usage(cursor, 1.35, "appliance", "Dishwasher", rate, period))

        # Washer: 3 mornings + sometimes evening
        if hour == 21 and cursor.weekday() in (2, 5) and rng.random() < 0.8:
            usage_rows.append(_usage(cursor, 0.85, "appliance", "Washing machine", rate, period))
        if hour == 8 and cursor.weekday() == 0 and rng.random() < 0.5:
            usage_rows.append(_usage(cursor, 0.85, "appliance", "Washing machine", rate, period))

        # Dryer after evening wash
        if hour == 22 and cursor.weekday() in (2, 5) and rng.random() < 0.6:
            usage_rows.append(_usage(cursor, 2.1, "appliance", "Clothes dryer", rate, period))

        # Water heater morning + evening
        if hour in (6, 19) and rng.random() < 0.9:
            usage_rows.append(
                _usage(cursor, 2.2, "water_heater", "Storage geyser", rate, period)
            )

        # Pool pump: builder default 8 hours including peak — bad
        if 14 <= hour <= 21:
            usage_rows.append(_usage(cursor, 1.15, "pool", "Pool pump", rate, period))

        cursor += timedelta(hours=1)

    with db.session() as s:
        batch = 400
        for i in range(0, len(solar_rows), batch):
            s.add_all(solar_rows[i : i + batch])
            s.commit()
        for i in range(0, len(usage_rows), batch):
            s.add_all(usage_rows[i : i + batch])
            s.commit()

    return {
        "days": days,
        "usage_rows": len(usage_rows),
        "solar_rows": len(solar_rows),
        "db_path": str(db.db_path),
        "start": start.isoformat(timespec="hours"),
        "end": end.isoformat(timespec="hours"),
    }


def _usage(ts, kwh, dtype, name, rate, period) -> EnergyUsage:
    return EnergyUsage(
        timestamp=ts,
        consumption_kwh=round(kwh, 3),
        device_type=dtype,
        device_name=name,
        cost=round(kwh * rate, 3),
        period=period,
    )


if __name__ == "__main__":
    info = seed_database()
    print(info)
