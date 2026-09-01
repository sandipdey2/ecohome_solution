"""Deterministic energy optimizer.

This is the part that actually *decides*, rather than just narrating.
Given a flexible load, a TOU tariff and a solar forecast, it searches
hour combinations that minimize net grid cost (import after self-consumption).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ecohome.config import HOME, tou_period, tou_rate
from ecohome.weather import expected_solar_kwh


@dataclass
class Slot:
    timestamp: str
    hour: int
    date: str
    price: float
    period: str
    solar_kwh: float
    irradiance: float
    temperature_c: float
    condition: str


def build_horizon(
    prices: dict[str, Any],
    weather: dict[str, Any],
) -> list[Slot]:
    hourly_rates = {int(r["hour"]): r for r in prices.get("hourly_rates", [])}
    slots: list[Slot] = []
    for row in weather.get("hourly", []):
        hour = int(row.get("hour", 0))
        rate_row = hourly_rates.get(hour, {})
        price = float(rate_row.get("rate", tou_rate(hour)))
        period = rate_row.get("period", tou_period(hour))
        irr = float(row.get("solar_irradiance") or 0)
        ts = str(row.get("timestamp", ""))
        slots.append(
            Slot(
                timestamp=ts,
                hour=hour,
                date=ts[:10],
                price=price,
                period=period,
                solar_kwh=expected_solar_kwh(irr),
                irradiance=irr,
                temperature_c=float(row.get("temperature_c") or 0),
                condition=str(row.get("condition") or ""),
            )
        )
    return slots


def schedule_flexible_load(
    kwh: float,
    power_kw: float,
    slots: list[Slot],
    prefer_contiguous: bool = True,
    hours_ahead: int = 36,
) -> dict[str, Any]:
    """Pick hours that cover `kwh` at `power_kw` with minimum net grid cost."""
    if kwh <= 0 or power_kw <= 0 or not slots:
        return {"error": "Invalid load or empty forecast horizon."}

    horizon = slots[: max(hours_ahead, 8)]
    hours_needed = max(1, math_ceil(kwh / power_kw))
    hours_needed = min(hours_needed, len(horizon))

    def net_cost(slot: Slot, draw_kwh: float) -> tuple[float, float, float]:
        solar_used = min(slot.solar_kwh, draw_kwh)
        grid = max(draw_kwh - solar_used, 0.0)
        return grid * slot.price, grid, solar_used

    candidates: list[dict[str, Any]] = []

    if prefer_contiguous:
        for i in range(0, len(horizon) - hours_needed + 1):
            window = horizon[i : i + hours_needed]
            remaining = kwh
            cost = 0.0
            grid = 0.0
            solar = 0.0
            plan = []
            for slot in window:
                draw = min(power_kw, remaining)
                c, g, s = net_cost(slot, draw)
                cost += c
                grid += g
                solar += s
                remaining -= draw
                plan.append(_plan_row(slot, draw, g, s, c))
            candidates.append(
                {
                    "start": window[0].timestamp,
                    "end": window[-1].timestamp,
                    "hours": [s.hour for s in window],
                    "cost": round(cost, 2),
                    "grid_kwh": round(grid, 2),
                    "solar_kwh": round(solar, 2),
                    "plan": plan,
                    "style": "contiguous",
                }
            )

    # Also consider the cheapest individual hours (non-contiguous)
    rated = []
    for slot in horizon:
        draw = min(power_kw, kwh)
        c, g, s = net_cost(slot, draw)
        effective = c / max(draw, 1e-6)
        rated.append((effective, -s, slot.timestamp, slot))
    rated.sort(key=lambda item: item[:3])
    remaining = kwh
    cost = grid = solar = 0.0
    picked: list[Slot] = []
    plan = []
    for _, __, ___, slot in rated:
        if remaining <= 0:
            break
        draw = min(power_kw, remaining)
        c, g, s = net_cost(slot, draw)
        cost += c
        grid += g
        solar += s
        remaining -= draw
        picked.append(slot)
        plan.append(_plan_row(slot, draw, g, s, c))
    picked_sorted = sorted(picked, key=lambda s: s.timestamp)
    candidates.append(
        {
            "start": picked_sorted[0].timestamp if picked_sorted else None,
            "end": picked_sorted[-1].timestamp if picked_sorted else None,
            "hours": [s.hour for s in picked_sorted],
            "cost": round(cost, 2),
            "grid_kwh": round(grid, 2),
            "solar_kwh": round(solar, 2),
            "plan": sorted(plan, key=lambda r: r["timestamp"]),
            "style": "cheapest_hours",
        }
    )

    naive = _naive_evening_cost(kwh, power_kw, horizon)

    best = min(candidates, key=lambda c: (c["cost"], -c["solar_kwh"]))
    best["naive_evening_cost"] = naive["cost"]
    best["savings_vs_evening"] = round(naive["cost"] - best["cost"], 2)
    best["naive_plan_hours"] = naive["hours"]
    best["currency"] = HOME.currency
    best["kwh_requested"] = kwh
    best["power_kw"] = power_kw
    return best


def math_ceil(x: float) -> int:
    n = int(x)
    return n if n == x else n + 1


def _plan_row(slot: Slot, draw: float, grid: float, solar: float, cost: float) -> dict:
    return {
        "timestamp": slot.timestamp,
        "hour": slot.hour,
        "period": slot.period,
        "price": slot.price,
        "draw_kwh": round(draw, 3),
        "grid_kwh": round(grid, 3),
        "solar_kwh": round(solar, 3),
        "cost": round(cost, 2),
        "irradiance": slot.irradiance,
        "condition": slot.condition,
    }


def _naive_evening_cost(kwh: float, power_kw: float, horizon: list[Slot]) -> dict:
    """Cost of starting at 19:00 the first evening in the horizon."""
    evenings = [s for s in horizon if s.hour >= 18]
    if not evenings:
        evenings = horizon[: math_ceil(kwh / power_kw)]
    remaining = kwh
    cost = 0.0
    hours = []
    for slot in evenings:
        if remaining <= 0:
            break
        draw = min(power_kw, remaining)
        solar_used = min(slot.solar_kwh, draw)
        grid = max(draw - solar_used, 0.0)
        cost += grid * slot.price
        remaining -= draw
        hours.append(slot.hour)
    return {"cost": round(cost, 2), "hours": hours}


def thermostat_plan(slots: list[Slot], peak_setpoint_c: float = 26.5) -> dict[str, Any]:
    """Recommend pre-cool vs peak-coast using forecast temps + tariff."""
    advice = []
    by_day: dict[str, list[Slot]] = {}
    for s in slots:
        by_day.setdefault(s.date, []).append(s)
    for day, day_slots in list(by_day.items())[:3]:
        peak = [s for s in day_slots if s.period == "on_peak"]
        pre = [s for s in day_slots if 14 <= s.hour <= 17]
        max_temp = max((s.temperature_c for s in day_slots), default=32)
        peak_rate = max((s.price for s in peak), default=HOME.on_peak_rate)
        advice.append(
            {
                "date": day,
                "max_temperature_c": max_temp,
                "precool_window": [s.hour for s in pre],
                "coast_window": [s.hour for s in peak],
                "recommended_setpoint_c": peak_setpoint_c if max_temp >= 30 else 26.0,
                "precool_setpoint_c": peak_setpoint_c - 1.0,
                "rationale": (
                    f"Peak tariff {HOME.currency_symbol}{peak_rate}/kWh coincides with "
                    f"late-day heat (max {max_temp:.0f}°C). Pre-cool 1°C in the "
                    f"14:00–17:00 shoulder, then hold {peak_setpoint_c}°C with fans "
                    f"through 18:00–23:00."
                ),
            }
        )
    return {"days": advice, "currency": HOME.currency}


def savings_report(
    device_type: str,
    current_kwh: float,
    optimized_kwh: float,
    price_per_kwh: float,
    cycles_per_week: float = 7.0,
) -> dict[str, Any]:
    kwh_saved = max(current_kwh - optimized_kwh, 0.0)
    money = kwh_saved * price_per_kwh
    weekly = money * cycles_per_week
    annual = weekly * 52
    kg = kwh_saved * HOME.grid_emission_factor_kg_per_kwh
    return {
        "device_type": device_type,
        "current_usage_kwh": current_kwh,
        "optimized_usage_kwh": optimized_kwh,
        "kwh_saved_per_event": round(kwh_saved, 3),
        "price_per_kwh": price_per_kwh,
        "currency": HOME.currency,
        "savings_per_event": round(money, 2),
        "savings_per_week": round(weekly, 2),
        "savings_per_year": round(annual, 2),
        "co2_kg_saved_per_event": round(kg, 3),
        "co2_kg_saved_per_year": round(kg * cycles_per_week * 52, 1),
        "note": (
            "kWh saved captures efficiency cuts. Tariff-shift savings "
            "(same kWh, cheaper hour) should be passed as current_kwh * "
            "peak_price vs optimized_kwh * offpeak_price with the prices "
            "already baked into those two kWh figures — or compared via "
            "schedule_flexible_load."
        ),
    }
