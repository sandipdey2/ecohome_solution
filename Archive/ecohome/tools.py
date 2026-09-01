"""LangChain tools the Energy Advisor is allowed to call."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Optional

from langchain_core.tools import tool

from ecohome.config import DEVICE_CATALOGUE, HOME, tou_period, tou_rate
from ecohome.database import DatabaseManager
from ecohome.optimizer import (
    build_horizon,
    savings_report,
    schedule_flexible_load,
    thermostat_plan,
)
from ecohome.rag import KB
from ecohome.weather import forecast as weather_forecast

_db = DatabaseManager()


def _parse_date(value: Optional[str]) -> datetime:
    if not value:
        return datetime.now()
    return datetime.strptime(value, "%Y-%m-%d")


def _trim_records(records: list[dict], limit: int = 48) -> list[dict]:
    if len(records) <= limit:
        return records
    return records[:24] + records[-24:]


@tool
def get_weather_forecast(location: str = "", days: int = 3) -> str:
    """Get hourly weather and solar-irradiance forecast for a location.

    Args:
        location: City name, e.g. "Bengaluru, IN". Defaults to the home profile.
        days: Forecast horizon from 1 to 7 days.
    """
    loc = location.strip() or HOME.location
    data = weather_forecast(loc, days)
    # Keep the payload compact for the LLM / engine.
    hourly = data.get("hourly", [])
    compact = []
    for row in hourly:
        compact.append(
            {
                "timestamp": row.get("timestamp"),
                "hour": row.get("hour"),
                "temp_c": row.get("temperature_c"),
                "condition": row.get("condition"),
                "irradiance": row.get("solar_irradiance"),
                "cloud": row.get("cloud_cover"),
            }
        )
    out = {
        "location": data.get("location", loc),
        "source": data.get("source"),
        "warning": data.get("warning"),
        "forecast_days": data.get("forecast_days"),
        "current": data.get("current"),
        "solar_friendly_hours": data.get("solar_friendly_hours"),
        "hourly": compact,
    }
    return json.dumps(out)


@tool
def get_electricity_prices(date: str = "") -> str:
    """Get hourly time-of-use electricity prices for a date (YYYY-MM-DD).

    Peak windows and dollars change with season and weekday. This is not
    a static 16:00-21:00 table.

    Args:
        date: Date in YYYY-MM-DD. Defaults to today.
    """
    import random

    dt = _parse_date(date or None)
    # Reuse the same dynamic windows as tools.py.
    from tools import _tou_windows

    cfg = _tou_windows(dt)
    rng = random.Random(f"tou-{dt.strftime('%Y-%m-%d')}")
    hourly = []
    for hour in range(24):
        if hour in cfg["peak_hours"]:
            period, shape = "on_peak", cfg["peak_mult"]
        elif hour in cfg["off_hours"]:
            period, shape = "off_peak", cfg["off_mult"]
        else:
            period, shape = "mid_peak", 1.0
        jitter = rng.uniform(-0.04, 0.04)
        rate = max(0.04, cfg["base"] * shape * cfg["dow_mult"] * (1.0 + jitter))
        hourly.append({"hour": hour, "rate": round(rate, 4), "period": period})
    off = [h["hour"] for h in hourly if h["period"] == "off_peak"]
    peak = [h["hour"] for h in hourly if h["period"] == "on_peak"]
    out = {
        "date": dt.strftime("%Y-%m-%d"),
        "weekday": dt.strftime("%A"),
        "season": cfg["season"],
        "pricing_type": "time_of_use",
        "currency": HOME.currency,
        "unit": "per_kWh",
        "is_weekend": cfg["is_weekend"],
        "day_of_week_multiplier": cfg["dow_mult"],
        "location": HOME.location,
        "off_peak_hours": off,
        "on_peak_hours": peak,
        "base_rate_usd_per_kwh": cfg["base"],
        "min_rate": min(h["rate"] for h in hourly),
        "max_rate": max(h["rate"] for h in hourly),
        "profile_note": f"{cfg['season']} {dt.strftime('%A')}: on-peak={peak or 'none'}",
        "hourly_rates": hourly,
    }
    return json.dumps(out)


@tool
def query_energy_usage(
    start_date: str, end_date: str, device_type: str = ""
) -> str:
    """Query historical household energy usage between two dates.

    Args:
        start_date: Inclusive start date YYYY-MM-DD.
        end_date: Inclusive end date YYYY-MM-DD.
        device_type: Optional filter: EV, HVAC, appliance, water_heater, pool, other.
    """
    start = _parse_date(start_date)
    end = _parse_date(end_date) + timedelta(days=1)
    dtype = device_type.strip() or None
    rows = _db.get_usage_by_date_range(start, end, dtype)
    records = [
        {
            "timestamp": r.timestamp.isoformat(timespec="minutes"),
            "kwh": r.consumption_kwh,
            "device_type": r.device_type,
            "device_name": r.device_name,
            "cost": r.cost,
            "period": r.period,
        }
        for r in rows
    ]
    by_device: dict[str, dict[str, float]] = {}
    by_period: dict[str, float] = {}
    for r in rows:
        key = r.device_type or "unknown"
        bucket = by_device.setdefault(key, {"kwh": 0.0, "cost": 0.0})
        bucket["kwh"] += r.consumption_kwh
        bucket["cost"] += r.cost or 0.0
        p = r.period or "unknown"
        by_period[p] = by_period.get(p, 0.0) + r.consumption_kwh
    for b in by_device.values():
        b["kwh"] = round(b["kwh"], 2)
        b["cost"] = round(b["cost"], 2)
    out = {
        "start_date": start_date,
        "end_date": end_date,
        "device_type": dtype,
        "total_records": len(rows),
        "total_kwh": round(sum(r.consumption_kwh for r in rows), 2),
        "total_cost": round(sum(r.cost or 0 for r in rows), 2),
        "currency": HOME.currency,
        "by_device": by_device,
        "kwh_by_period": {k: round(v, 2) for k, v in by_period.items()},
        "sample_records": _trim_records(records),
    }
    return json.dumps(out)


@tool
def query_solar_generation(start_date: str, end_date: str) -> str:
    """Query historical rooftop solar generation between two dates.

    Args:
        start_date: Inclusive start date YYYY-MM-DD.
        end_date: Inclusive end date YYYY-MM-DD.
    """
    start = _parse_date(start_date)
    end = _parse_date(end_date) + timedelta(days=1)
    rows = _db.get_generation_by_date_range(start, end)
    n_days = max((end - start).days, 1)
    total = sum(r.generation_kwh for r in rows)
    records = [
        {
            "timestamp": r.timestamp.isoformat(timespec="minutes"),
            "kwh": r.generation_kwh,
            "weather": r.weather_condition,
            "temp_c": r.temperature_c,
            "irradiance": r.solar_irradiance,
        }
        for r in rows
    ]
    # Hour-of-day average yield — useful for scheduling
    by_hour = {h: [] for h in range(24)}
    for r in rows:
        by_hour[r.timestamp.hour].append(r.generation_kwh)
    hourly_avg = {
        h: round(sum(v) / len(v), 3) if v else 0.0 for h, v in by_hour.items()
    }
    best_hours = sorted(hourly_avg, key=lambda h: hourly_avg[h], reverse=True)[:5]
    out = {
        "start_date": start_date,
        "end_date": end_date,
        "total_records": len(rows),
        "total_generation_kwh": round(total, 2),
        "average_daily_kwh": round(total / n_days, 2),
        "capacity_kw": HOME.solar_capacity_kw,
        "best_historical_hours": best_hours,
        "average_kwh_by_hour": hourly_avg,
        "sample_records": _trim_records(records),
    }
    return json.dumps(out)


@tool
def get_recent_energy_summary(hours: int = 24) -> str:
    """Summarise recent consumption and solar generation.

    Args:
        hours: Lookback window in hours (default 24, try 168 for a week).
    """
    hours = max(1, min(int(hours), 24 * 90))
    usage = _db.get_recent_usage(hours)
    gen = _db.get_recent_generation(hours)
    breakdown: dict[str, dict[str, float]] = {}
    peak_kwh = 0.0
    for r in usage:
        key = r.device_type or "unknown"
        b = breakdown.setdefault(key, {"kwh": 0.0, "cost": 0.0})
        b["kwh"] += r.consumption_kwh
        b["cost"] += r.cost or 0.0
        if r.period == "on_peak":
            peak_kwh += r.consumption_kwh
    for b in breakdown.values():
        b["kwh"] = round(b["kwh"], 2)
        b["cost"] = round(b["cost"], 2)
    total_use = sum(r.consumption_kwh for r in usage)
    total_gen = sum(r.generation_kwh for r in gen)
    out = {
        "hours": hours,
        "location": HOME.location,
        "currency": HOME.currency,
        "usage_kwh": round(total_use, 2),
        "usage_cost": round(sum(r.cost or 0 for r in usage), 2),
        "solar_kwh": round(total_gen, 2),
        "self_sufficiency_pct": round(100 * min(total_gen / total_use, 1.0), 1)
        if total_use
        else 0,
        "on_peak_usage_kwh": round(peak_kwh, 2),
        "by_device": breakdown,
        "biggest_load": max(breakdown, key=lambda k: breakdown[k]["kwh"])
        if breakdown
        else None,
    }
    return json.dumps(out)


@tool
def search_energy_tips(query: str, max_results: int = 4) -> str:
    """Retrieve energy-saving tips from the EcoHome knowledge base (RAG).

    Args:
        query: Natural-language topic, e.g. "EV charging with solar".
        max_results: How many passages to return (1-8).
    """
    k = max(1, min(int(max_results), 8))
    tips = KB.search(query, k=k)
    return json.dumps(
        {
            "query": query,
            "total_results": len(tips),
            "tips": tips,
            "citation_note": "Passages come from the EcoHome energy-saving knowledge base.",
        }
    )


@tool
def calculate_energy_savings(
    device_type: str,
    current_usage_kwh: float,
    optimized_usage_kwh: float,
    price_per_kwh: float,
    cycles_per_week: float = 7.0,
) -> str:
    """Calculate kWh, money and CO2 savings from an optimization.

    Args:
        device_type: EV, HVAC, appliance, water_heater, pool, other.
        current_usage_kwh: kWh (or effective billed kWh) today.
        optimized_usage_kwh: kWh after the change. For a pure tariff shift
            pass current_kwh * peak_rate as current and current_kwh * new_rate
            as optimized only if you first convert to "cost-equivalent kWh";
            prefer passing the actual kWh and the *price being avoided*.
        price_per_kwh: Tariff applied to the saved kWh, in home currency.
        cycles_per_week: How often the event happens; used for annualisation.
    """
    report = savings_report(
        device_type=device_type,
        current_kwh=float(current_usage_kwh),
        optimized_kwh=float(optimized_usage_kwh),
        price_per_kwh=float(price_per_kwh),
        cycles_per_week=float(cycles_per_week),
    )
    return json.dumps(report)


@tool
def optimize_device_schedule(
    device_type: str,
    kwh: float = 0.0,
    date: str = "",
    days: int = 2,
) -> str:
    """Compute the lowest-net-cost schedule for a flexible device.

    Combines TOU prices with the solar forecast and returns a concrete
    hour plan plus savings versus charging/running in the evening peak.

    Args:
        device_type: EV, appliance, water_heater, or pool.
        kwh: Energy the device needs. 0 = use the catalogue default.
        date: Unused placeholder kept for callers that pass a date.
        days: Weather horizon to plan over (1-3 is typical).
    """
    spec = DEVICE_CATALOGUE.get(device_type, DEVICE_CATALOGUE["appliance"])
    need = float(kwh) if kwh and kwh > 0 else float(spec["typical_kwh"])
    power = {
        "EV": HOME.ev_charge_rate_kw,
        "appliance": 1.2,
        "water_heater": 2.0,
        "pool": 1.15,
        "HVAC": HOME.hvac_typical_kw,
    }.get(device_type, 1.0)

    prices = json.loads(get_electricity_prices.invoke({"date": date or ""}))
    weather = weather_forecast(HOME.location, days)
    slots = build_horizon(prices, weather)
    contiguous = device_type in {"EV", "appliance", "water_heater"}
    plan = schedule_flexible_load(
        kwh=need,
        power_kw=power,
        slots=slots,
        prefer_contiguous=contiguous,
    )
    plan["device_type"] = device_type
    plan["device_name"] = spec["name"]
    plan["preferred_strategy"] = spec["preferred"]
    if device_type == "HVAC":
        plan["thermostat"] = thermostat_plan(slots)
    return json.dumps(plan)


@tool
def get_home_profile() -> str:
    """Return the configured household profile (location, solar size, EV, tariff)."""
    return json.dumps(
        {
            "location": HOME.location,
            "latitude": HOME.latitude,
            "longitude": HOME.longitude,
            "currency": HOME.currency,
            "solar_capacity_kw": HOME.solar_capacity_kw,
            "ev_battery_kwh": HOME.ev_battery_kwh,
            "ev_charge_rate_kw": HOME.ev_charge_rate_kw,
            "tariff": {
                "off_peak": HOME.off_peak_rate,
                "mid_peak": HOME.mid_peak_rate,
                "on_peak": HOME.on_peak_rate,
                "weekend_discount": HOME.weekend_discount,
                "off_peak_hours": "23:00-06:00",
                "on_peak_hours": "18:00-23:00",
            },
            "devices": {k: v["name"] for k, v in DEVICE_CATALOGUE.items()},
        }
    )


TOOL_KIT = [
    get_weather_forecast,
    get_electricity_prices,
    query_energy_usage,
    query_solar_generation,
    get_recent_energy_summary,
    search_energy_tips,
    calculate_energy_savings,
    optimize_device_schedule,
    get_home_profile,
]

TOOL_BY_NAME = {t.name: t for t in TOOL_KIT}
