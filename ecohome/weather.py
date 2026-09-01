"""Weather + irradiance forecasts.

Tries the free Open-Meteo API (no key). Falls back to a deterministic
synthetic forecast anchored on the home's lat/lon so the advisor never
hard-fails when the network is unhappy.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

import requests

from ecohome.config import HOME

_CACHE: dict[tuple, dict[str, Any]] = {}


def forecast(location: str | None = None, days: int = 3) -> dict[str, Any]:
    days = max(1, min(int(days), 7))
    loc = location or HOME.location
    lat, lon = HOME.latitude, HOME.longitude
    key = (round(lat, 3), round(lon, 3), days, datetime.now().strftime("%Y-%m-%d-%H"))
    if key in _CACHE:
        return _CACHE[key]
    try:
        data = _open_meteo(lat, lon, days)
        data["location"] = loc
        data["source"] = "open-meteo"
        _CACHE[key] = data
        return data
    except Exception as exc:
        synthetic = _synthetic(lat, lon, days)
        synthetic["location"] = loc
        synthetic["source"] = "synthetic_fallback"
        reason = type(exc).__name__
        if hasattr(exc, "response") and getattr(exc, "response") is not None:
            reason = f"HTTP {exc.response.status_code}"
        synthetic["warning"] = (
            f"Live weather unavailable ({reason}). Using a solar-typical profile for {loc}."
        )
        _CACHE[key] = synthetic
        return synthetic


def _open_meteo(lat: float, lon: float, days: int) -> dict[str, Any]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,shortwave_radiation,wind_speed_10m,weather_code",
        "forecast_days": days,
        "timezone": HOME.timezone,
    }
    resp = requests.get(url, params=params, timeout=12)
    resp.raise_for_status()
    payload = resp.json()
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    rows = []
    for i, ts in enumerate(times):
        irr = hourly.get("shortwave_radiation", [0] * len(times))[i] or 0
        cloud = hourly.get("cloud_cover", [50] * len(times))[i] or 0
        temp = hourly.get("temperature_2m", [28] * len(times))[i]
        rows.append(
            {
                "timestamp": ts,
                "hour": int(ts[11:13]) if len(ts) >= 13 else i % 24,
                "temperature_c": temp,
                "humidity": hourly.get("relative_humidity_2m", [60] * len(times))[i],
                "wind_speed": hourly.get("wind_speed_10m", [8] * len(times))[i],
                "cloud_cover": cloud,
                "solar_irradiance": round(float(irr), 1),
                "condition": _condition_from_code(
                    (hourly.get("weather_code") or [0] * len(times))[i], cloud
                ),
            }
        )
    current = rows[0] if rows else {}
    return {
        "latitude": lat,
        "longitude": lon,
        "forecast_days": days,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "current": current,
        "hourly": rows,
        "solar_friendly_hours": _solar_friendly(rows),
    }


def _synthetic(lat: float, lon: float, days: int) -> dict[str, Any]:
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    rows = []
    for h in range(days * 24):
        ts = now + timedelta(hours=h)
        hour = ts.hour
        # Tropical clear-ish profile with a mild monsoon dip on day 2
        monsoon = 0.55 if (h // 24) == 1 else 1.0
        elev = _solar_elevation_proxy(hour, lat)
        irr = max(0.0, 980 * elev * monsoon)
        cloud = 25 if monsoon > 0.8 else 70
        if elev <= 0:
            cond = "clear_night" if cloud < 40 else "cloudy_night"
        elif cloud > 60:
            cond = "overcast"
        elif cloud > 35:
            cond = "partly_cloudy"
        else:
            cond = "sunny"
        temp = 24 + 8 * max(0, math.sin((hour - 8) / 16 * math.pi))
        rows.append(
            {
                "timestamp": ts.isoformat(timespec="seconds"),
                "hour": hour,
                "temperature_c": round(temp, 1),
                "humidity": 55 if cond == "sunny" else 75,
                "wind_speed": 9.0,
                "cloud_cover": cloud,
                "solar_irradiance": round(irr, 1),
                "condition": cond,
            }
        )
    return {
        "latitude": lat,
        "longitude": lon,
        "forecast_days": days,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "current": rows[0],
        "hourly": rows,
        "solar_friendly_hours": _solar_friendly(rows),
    }


def _solar_elevation_proxy(hour: int, lat: float) -> float:
    sunrise, sunset = 6.1, 18.4
    if hour < sunrise or hour > sunset:
        return 0.0
    mid = 12.2
    span = (sunset - sunrise) / 2
    x = 1 - abs(hour + 0.5 - mid) / span
    return max(0.0, x) ** 1.15


def _condition_from_code(code: int, cloud: float) -> str:
    if code in (0,):
        return "sunny" if cloud < 30 else "partly_cloudy"
    if code in (1, 2):
        return "partly_cloudy"
    if code in (3,):
        return "overcast"
    if code in (45, 48):
        return "fog"
    if 51 <= code <= 67 or 80 <= code <= 82:
        return "rain"
    if 71 <= code <= 77 or 85 <= code <= 86:
        return "snow"
    if 95 <= code <= 99:
        return "thunderstorm"
    return "cloudy"


def _solar_friendly(rows: list[dict]) -> list[dict]:
    friendly = []
    by_day: dict[str, list[dict]] = {}
    for r in rows:
        day = str(r["timestamp"])[:10]
        by_day.setdefault(day, []).append(r)
    for day, hrs in by_day.items():
        ranked = sorted(hrs, key=lambda r: r["solar_irradiance"], reverse=True)
        top = [r for r in ranked if r["solar_irradiance"] >= 350][:6]
        if not top:
            continue
        hours = sorted(r["hour"] for r in top)
        friendly.append(
            {
                "date": day,
                "best_hours": hours,
                "peak_irradiance": max(r["solar_irradiance"] for r in top),
                "mean_irradiance": round(
                    sum(r["solar_irradiance"] for r in top) / len(top), 1
                ),
            }
        )
    return friendly


def expected_solar_kwh(irradiance_wm2: float) -> float:
    """Rough hourly AC yield from the home's array given GHI."""
    # 1 kW array ≈ 1 kWh at 1000 W/m² * PR
    return round(
        HOME.solar_capacity_kw
        * (max(irradiance_wm2, 0.0) / 1000.0)
        * HOME.solar_performance_ratio,
        3,
    )
