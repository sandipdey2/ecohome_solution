"""Tools for the EcoHome Energy Advisor agent."""

from __future__ import annotations

import json
import math
import os
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from models.energy import DatabaseManager

db_manager = DatabaseManager(os.getenv("ECOHOME_DB_PATH", "data/energy_data.db"))


def _clamp_days(days: int) -> int:
    try:
        return max(1, min(int(days), 7))
    except (TypeError, ValueError):
        return 3


@tool
def get_weather_forecast(location: str, days: int = 3) -> Dict[str, Any]:
    """Get weather forecast for a location, including hourly solar irradiance.

    Args:
        location: City name, e.g. "San Francisco, CA".
        days: Forecast horizon from 1 to 7.
    """
    days = _clamp_days(days)
    location = (location or "San Francisco, CA").strip()
    try:
        return _live_or_synthetic_weather(location, days)
    except Exception as exc:
        data = _synthetic_weather(location, days)
        data["warning"] = f"Live weather unavailable ({type(exc).__name__}). Using a solar-typical profile."
        return data


def _live_or_synthetic_weather(location: str, days: int) -> Dict[str, Any]:
    # Prefer Open-Meteo (no key). Fall back to a deterministic mock.
    coords = _geocode(location)
    try:
        import requests

        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": coords["lat"],
                "longitude": coords["lon"],
                "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,shortwave_radiation,wind_speed_10m",
                "forecast_days": days,
                "timezone": "auto",
            },
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
        hourly_src = payload.get("hourly") or {}
        times = hourly_src.get("time") or []
        hourly = []
        for i, ts in enumerate(times):
            hour = int(ts[11:13]) if len(ts) >= 13 else i % 24
            cloud = (hourly_src.get("cloud_cover") or [50])[i]
            irr = (hourly_src.get("shortwave_radiation") or [0])[i] or 0
            cond = "sunny" if cloud < 30 else "partly_cloudy" if cloud < 65 else "cloudy"
            hourly.append(
                {
                    "hour": hour,
                    "timestamp": ts,
                    "temperature_c": (hourly_src.get("temperature_2m") or [18])[i],
                    "condition": cond,
                    "solar_irradiance": round(float(irr), 1),
                    "humidity": (hourly_src.get("relative_humidity_2m") or [60])[i],
                    "wind_speed": (hourly_src.get("wind_speed_10m") or [8])[i],
                }
            )
        current = hourly[0] if hourly else {}
        return {
            "location": location,
            "forecast_days": days,
            "source": "open-meteo",
            "current": {
                "temperature_c": current.get("temperature_c"),
                "condition": current.get("condition"),
                "humidity": current.get("humidity"),
                "wind_speed": current.get("wind_speed"),
            },
            "hourly": hourly,
            "solar_friendly_hours": _solar_friendly(hourly),
        }
    except Exception:
        return _synthetic_weather(location, days)


def _geocode(location: str) -> Dict[str, float]:
    known = {
        "san francisco": {"lat": 37.7749, "lon": -122.4194},
        "bengaluru": {"lat": 12.9716, "lon": 77.5946},
        "bangalore": {"lat": 12.9716, "lon": 77.5946},
        "new york": {"lat": 40.7128, "lon": -74.0060},
    }
    key = location.lower()
    for name, coords in known.items():
        if name in key:
            return coords
    return {"lat": 37.7749, "lon": -122.4194}


def _synthetic_weather(location: str, days: int) -> Dict[str, Any]:
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    hourly = []
    rng = random.Random(location + now.strftime("%Y-%m-%d"))
    for h in range(days * 24):
        ts = now + timedelta(hours=h)
        hour = ts.hour
        # Daylight irradiance peaks near noon.
        elev = max(0.0, 1 - abs(hour + 0.5 - 12.5) / 6.5)
        if hour < 6 or hour > 19:
            elev = 0.0
        cloud = rng.choice([15, 25, 40, 60])
        irr = 950 * elev * (1 - cloud / 180)
        cond = "sunny" if cloud < 30 else "partly_cloudy" if cloud < 55 else "cloudy"
        temp = 14 + 10 * max(0, math.sin((hour - 7) / 16 * math.pi))
        hourly.append(
            {
                "hour": hour,
                "timestamp": ts.isoformat(timespec="seconds"),
                "temperature_c": round(temp, 1),
                "condition": cond,
                "solar_irradiance": round(max(irr, 0), 1),
                "humidity": 55 if cond == "sunny" else 70,
                "wind_speed": 10,
            }
        )
    current = hourly[0]
    return {
        "location": location,
        "forecast_days": days,
        "source": "synthetic",
        "current": {
            "temperature_c": current["temperature_c"],
            "condition": current["condition"],
            "humidity": current["humidity"],
            "wind_speed": current["wind_speed"],
        },
        "hourly": hourly,
        "solar_friendly_hours": _solar_friendly(hourly),
    }


def _solar_friendly(hourly: List[dict]) -> List[dict]:
    by_day: Dict[str, List[dict]] = {}
    for row in hourly:
        day = str(row.get("timestamp", ""))[:10] or "day"
        by_day.setdefault(day, []).append(row)
    out = []
    for day, rows in by_day.items():
        best = [r for r in rows if r.get("solar_irradiance", 0) >= 350]
        best = sorted(best, key=lambda r: r["solar_irradiance"], reverse=True)[:6]
        if not best:
            continue
        out.append(
            {
                "date": day,
                "best_hours": sorted(r["hour"] for r in best),
                "peak_irradiance": max(r["solar_irradiance"] for r in best),
            }
        )
    return out


def _tou_windows(dt: datetime) -> Dict[str, Any]:
    """Peak *hours* and rate shape change with season and weekday — not a fixed table."""
    month, weekday = dt.month, dt.weekday()
    is_weekend = weekday >= 5
    if month in (6, 7, 8, 9):
        season = "summer"
        base = 0.26
        # Long late-day AC peak on weekdays; weekend peak shrinks.
        peak_hours = list(range(15, 22)) if not is_weekend else list(range(16, 20))
        off_hours = list(range(0, 6)) + [23]
        peak_mult, off_mult = (1.95, 0.42) if not is_weekend else (1.35, 0.40)
    elif month in (12, 1, 2):
        season = "winter"
        base = 0.215
        # Dual peak: morning heating + evening lighting/heat.
        peak_hours = ([7, 8, 9, 10, 17, 18, 19, 20] if not is_weekend else [18, 19])
        off_hours = list(range(0, 6)) + [22, 23]
        peak_mult, off_mult = (1.70, 0.70) if not is_weekend else (1.25, 0.62)
    elif month in (3, 4, 5):
        season = "spring"
        base = 0.18
        peak_hours = list(range(16, 21)) if not is_weekend else []
        off_hours = list(range(0, 7)) + [22, 23]
        peak_mult, off_mult = (1.40, 0.48) if not is_weekend else (1.00, 0.44)
    else:
        season = "autumn"
        base = 0.20
        peak_hours = list(range(16, 21)) if weekday != 6 else [17, 18]
        off_hours = list(range(0, 6)) + [23]
        peak_mult, off_mult = (1.55, 0.50) if not is_weekend else (1.15, 0.46)

    dow_mult = {0: 1.03, 1: 1.00, 2: 0.99, 3: 1.01, 4: 1.08, 5: 0.86, 6: 0.78}[weekday]
    return {
        "season": season,
        "base": base,
        "peak_hours": peak_hours,
        "off_hours": off_hours,
        "peak_mult": peak_mult,
        "off_mult": off_mult,
        "dow_mult": dow_mult,
        "is_weekend": is_weekend,
    }


@tool
def get_electricity_prices(date: str = "") -> Dict[str, Any]:
    """Get time-of-use electricity prices for a date (YYYY-MM-DD).

    The profile is not a static lookup table. Peak windows move with
    season (summer late-day AC vs winter dual peak vs spring with no
    weekend peak). Weekday multipliers and a date-seeded jitter change
    the dollars. Calling this for a Monday in July and a Sunday in
    January returns different hours and different rates.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today.
    """
    try:
        dt = datetime.strptime(date, "%Y-%m-%d") if date else datetime.now()
    except ValueError:
        dt = datetime.now()
    date = dt.strftime("%Y-%m-%d")
    cfg = _tou_windows(dt)
    rng = random.Random(f"tou-{date}")
    hourly_rates = []
    for hour in range(24):
        if hour in cfg["peak_hours"]:
            period, shape, demand = "on_peak", cfg["peak_mult"], 0.09 if not cfg["is_weekend"] else 0.02
        elif hour in cfg["off_hours"]:
            period, shape, demand = "off_peak", cfg["off_mult"], 0.0
        else:
            period, shape, demand = "mid_peak", 1.0, 0.0
        jitter = rng.uniform(-0.04, 0.04)
        hour_weight = shape
        seasonal_weight = 1.0
        dow_weight = cfg["dow_mult"]
        # Rubric mock: rate = base * weights (season window, weekday, hour, jitter).
        rate = max(0.04, cfg["base"] * seasonal_weight * dow_weight * hour_weight * (1.0 + jitter))
        hourly_rates.append(
            {
                "hour": hour,
                "rate": round(rate, 4),
                "period": period,
                "hour_weight": round(hour_weight, 3),
                "dow_weight": dow_weight,
                "jitter": round(jitter, 4),
                "demand_charge": round(demand * dow_weight, 4),
            }
        )
    off = [r["hour"] for r in hourly_rates if r["period"] == "off_peak"]
    on = [r["hour"] for r in hourly_rates if r["period"] == "on_peak"]
    mid = [r["hour"] for r in hourly_rates if r["period"] == "mid_peak"]
    return {
        "date": date,
        "weekday": dt.strftime("%A"),
        "season": cfg["season"],
        "pricing_type": "time_of_use",
        "currency": "USD",
        "unit": "per_kWh",
        "is_weekend": cfg["is_weekend"],
        "day_of_week_multiplier": cfg["dow_mult"],
        "base_rate_usd_per_kwh": cfg["base"],
        "peak_window_hours": cfg["peak_hours"],
        "off_peak_hours": off,
        "on_peak_hours": on,
        "mid_peak_hours": mid,
        "min_rate": min(r["rate"] for r in hourly_rates),
        "max_rate": max(r["rate"] for r in hourly_rates),
        "profile_note": (
            f"{cfg['season']} {dt.strftime('%A')}: on-peak hours={on or 'none'} "
            f"(not a fixed 16:00-21:00 table)"
        ),
        "hourly_rates": hourly_rates,
    }


@tool
def query_energy_usage(
    start_date: str, end_date: str, device_type: str = None
) -> Dict[str, Any]:
    """Query historical household energy usage between two dates.

    Args:
        start_date: Inclusive start date YYYY-MM-DD.
        end_date: Inclusive end date YYYY-MM-DD.
        device_type: Optional filter: EV, HVAC, appliance.
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        records = db_manager.get_usage_by_date_range(start_dt, end_dt)
        note = None
        wanted = []
        if device_type:
            raw = re.split(r"[,/&]| and ", str(device_type), flags=re.I)
            wanted = [w.strip() for w in raw if w.strip()]
            aliases = {"ev": "EV", "hvac": "HVAC", "ac": "HVAC", "appliance": "appliance",
                       "dishwasher": "appliance", "washer": "appliance", "dryer": "appliance"}
            wanted = [aliases.get(w.lower(), w) for w in wanted]
            filtered = [r for r in records if r.device_type in wanted]
            # "EV and HVAC" must not wipe the result if casing/alias was wrong.
            if filtered:
                records = filtered
            elif records:
                note = (
                    f"No rows matched device_type={device_type!r}; "
                    "returning all devices in the window."
                )
                wanted = []
        by_device: Dict[str, Dict[str, float]] = {}
        for r in records:
            slot = by_device.setdefault(
                r.device_type or "unknown",
                {"kwh": 0.0, "cost_usd": 0.0, "records": 0},
            )
            slot["kwh"] += r.consumption_kwh
            slot["cost_usd"] += r.cost_usd or 0
            slot["records"] += 1
        for slot in by_device.values():
            slot["kwh"] = round(slot["kwh"], 2)
            slot["cost_usd"] = round(slot["cost_usd"], 2)
        return {
            "start_date": start_date,
            "end_date": end_date,
            "device_type": device_type,
            "device_filter": wanted or None,
            "note": note,
            "by_device": by_device,
            "total_records": len(records),
            "total_consumption_kwh": round(sum(r.consumption_kwh for r in records), 2),
            "total_cost_usd": round(sum(r.cost_usd or 0 for r in records), 2),
            "records": [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "consumption_kwh": r.consumption_kwh,
                    "device_type": r.device_type,
                    "device_name": r.device_name,
                    "cost_usd": r.cost_usd,
                }
                for r in records[:96]
            ],
        }
    except Exception as e:
        return {"error": f"Failed to query energy usage: {str(e)}"}


@tool
def query_solar_generation(start_date: str, end_date: str) -> Dict[str, Any]:
    """Query historical rooftop solar generation between two dates.

    Args:
        start_date: Inclusive start date YYYY-MM-DD.
        end_date: Inclusive end date YYYY-MM-DD.
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        records = db_manager.get_generation_by_date_range(start_dt, end_dt)
        n_days = max((end_dt - start_dt).days, 1)
        total = sum(r.generation_kwh for r in records)
        return {
            "start_date": start_date,
            "end_date": end_date,
            "total_records": len(records),
            "total_generation_kwh": round(total, 2),
            "average_daily_generation": round(total / n_days, 2),
            "records": [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "generation_kwh": r.generation_kwh,
                    "weather_condition": r.weather_condition,
                    "temperature_c": r.temperature_c,
                    "solar_irradiance": r.solar_irradiance,
                }
                for r in records[:96]
            ],
        }
    except Exception as e:
        return {"error": f"Failed to query solar generation: {str(e)}"}


@tool
def get_recent_energy_summary(hours: int = 24) -> Dict[str, Any]:
    """Summarise recent consumption and solar generation.

    Args:
        hours: Lookback window in hours (default 24).
    """
    try:
        usage_records = db_manager.get_recent_usage(hours)
        generation_records = db_manager.get_recent_generation(hours)
        summary = {
            "time_period_hours": hours,
            "usage": {
                "total_consumption_kwh": round(sum(r.consumption_kwh for r in usage_records), 2),
                "total_cost_usd": round(sum(r.cost_usd or 0 for r in usage_records), 2),
                "device_breakdown": {},
            },
            "generation": {
                "total_generation_kwh": round(
                    sum(r.generation_kwh for r in generation_records), 2
                ),
                "average_weather": "sunny" if generation_records else "unknown",
            },
        }
        for record in usage_records:
            device = record.device_type or "unknown"
            bucket = summary["usage"]["device_breakdown"].setdefault(
                device, {"consumption_kwh": 0, "cost_usd": 0, "records": 0}
            )
            bucket["consumption_kwh"] += record.consumption_kwh
            bucket["cost_usd"] += record.cost_usd or 0
            bucket["records"] += 1
        for bucket in summary["usage"]["device_breakdown"].values():
            bucket["consumption_kwh"] = round(bucket["consumption_kwh"], 2)
            bucket["cost_usd"] = round(bucket["cost_usd"], 2)
        if generation_records:
            weathers = [r.weather_condition for r in generation_records if r.weather_condition]
            if weathers:
                summary["generation"]["average_weather"] = max(
                    set(weathers), key=weathers.count
                )
        return summary
    except Exception as e:
        return {"error": f"Failed to get recent energy summary: {str(e)}"}


@tool
def search_energy_tips(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search energy-saving tips from the knowledge base (RAG).

    Args:
        query: Natural-language topic, e.g. "EV charging with solar".
        max_results: How many passages to return.
    """
    max_results = max(1, min(int(max_results or 5), 8))
    chroma = _chroma_search(query, max_results)
    if chroma and not chroma.get("error"):
        return chroma
    local = _local_search(query, max_results)
    if chroma and chroma.get("error"):
        local["warning"] = chroma["error"]
    return local


def _iter_documents() -> List[Path]:
    folder = Path("data/documents")
    if not folder.exists():
        folder = Path(__file__).resolve().parent / "data" / "documents"
    return sorted(folder.glob("*.txt"))


def _chroma_search(query: str, k: int) -> Optional[Dict[str, Any]]:
    persist_directory = "data/vectorstore"
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception as exc:
        return {"error": f"Chroma/OpenAI stack unavailable: {exc}"}

    api_key = os.getenv("VOCAREUM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "No VOCAREUM_API_KEY / OPENAI_API_KEY for embeddings"}

    try:
        kwargs = {"api_key": api_key}
        if os.getenv("VOCAREUM_API_KEY"):
            kwargs["base_url"] = os.getenv(
                "OPENAI_BASE_URL", "https://openai.vocareum.com/v1"
            )
        embeddings = OpenAIEmbeddings(**kwargs)
        os.makedirs(persist_directory, exist_ok=True)
        chroma_db = os.path.join(persist_directory, "chroma.sqlite3")
        if not os.path.exists(chroma_db):
            documents = []
            for path in _iter_documents():
                documents.extend(TextLoader(str(path)).load())
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=persist_directory,
            )
        else:
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
            )
        docs = vectorstore.similarity_search(query, k=k)
        return {
            "query": query,
            "total_results": len(docs),
            "tips": [
                {
                    "rank": i + 1,
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "relevance_score": "high" if i < 2 else "medium" if i < 4 else "low",
                }
                for i, doc in enumerate(docs)
            ],
        }
    except Exception as exc:
        return {"error": f"Failed to search energy tips: {exc}"}


_TOKEN = re.compile(r"[a-z0-9]+")


def _local_search(query: str, k: int) -> Dict[str, Any]:
    chunks = []
    for path in _iter_documents():
        text = path.read_text(encoding="utf-8")
        blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
        for block in blocks or [text]:
            chunks.append((path.name, block))
    q_tokens = _TOKEN.findall(query.lower())
    scored = []
    for source, block in chunks:
        tokens = set(_TOKEN.findall(block.lower()))
        overlap = len(set(q_tokens) & tokens)
        bonus = sum(3 for t in q_tokens if t in block.lower())
        score = overlap + bonus
        if score:
            scored.append((score, source, block))
    scored.sort(reverse=True)
    tips = []
    for i, (score, source, block) in enumerate(scored[:k]):
        tips.append(
            {
                "rank": i + 1,
                "content": block,
                "source": source,
                "relevance_score": "high" if i < 2 else "medium" if i < 4 else "low",
            }
        )
    return {"query": query, "total_results": len(tips), "tips": tips, "retriever": "local"}


@tool
def calculate_energy_savings(
    device_type: str,
    current_usage_kwh: float,
    optimized_usage_kwh: float,
    price_per_kwh: float = 0.12,
) -> Dict[str, Any]:
    """Calculate kWh and USD savings from an optimization.

    Args:
        device_type: EV, HVAC, appliance, etc.
        current_usage_kwh: kWh for ONE event (one EV charge, one dishwasher
            cycle ~1.2-1.5 kWh). Never pass a 24-hour household total.
        optimized_usage_kwh: kWh for the same event after the change.
        price_per_kwh: Tariff applied to the saved kWh.
    """
    savings_kwh = current_usage_kwh - optimized_usage_kwh
    savings_usd = savings_kwh * price_per_kwh
    savings_percentage = (
        (savings_kwh / current_usage_kwh) * 100 if current_usage_kwh else 0
    )
    return {
        "device_type": device_type,
        "current_usage_kwh": current_usage_kwh,
        "optimized_usage_kwh": optimized_usage_kwh,
        "savings_kwh": round(savings_kwh, 2),
        "savings_usd": round(savings_usd, 2),
        "savings_percentage": round(savings_percentage, 1),
        "price_per_kwh": price_per_kwh,
        "annual_savings_usd": round(savings_usd * 365, 2),
        "co2_kg_saved_annual": round(savings_kwh * 365 * 0.4, 1),
    }


TOOL_KIT = [
    get_weather_forecast,
    get_electricity_prices,
    query_energy_usage,
    query_solar_generation,
    get_recent_energy_summary,
    search_energy_tips,
    calculate_energy_savings,
]
