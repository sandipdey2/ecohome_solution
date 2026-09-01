"""Home profile and runtime configuration for the EcoHome Energy Advisor."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTOR_DIR = DATA_DIR / "vectorstore"


def _writable_db_path() -> Path:
    explicit = os.getenv("ECOHOME_DB_PATH")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            Path("/tmp/ecohome_energy_data.db"),
            DATA_DIR / "energy_data.db",
        ]
    )
    for path in candidates:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            probe = path.with_suffix(path.suffix + ".probe")
            probe.write_text("ok")
            probe.unlink(missing_ok=True)
            return path
        except OSError:
            continue
    return Path("/tmp/ecohome_energy_data.db")


DB_PATH = _writable_db_path()


@dataclass(frozen=True)
class HomeProfile:
    """Default household the advisor personalizes against.

    Values are representative of a mid-size Bengaluru home with rooftop
    solar and an EV — the kinds of loads EcoHome is designed to optimize.
    Currency is INR so savings read naturally for an Indian household;
    TOU bands are a realistic *illustrative* tariff (BESCOM residential
    supply is still mostly slab-based; TOU is what EcoHome is selling).
    """

    location: str = "Bengaluru, IN"
    latitude: float = 12.9716
    longitude: float = 77.5946
    timezone: str = "Asia/Kolkata"
    currency: str = "INR"
    currency_symbol: str = "₹"

    solar_capacity_kw: float = 5.0
    solar_performance_ratio: float = 0.78
    ev_battery_kwh: float = 40.0
    ev_charge_rate_kw: float = 3.3
    hvac_typical_kw: float = 1.8
    grid_emission_factor_kg_per_kwh: float = 0.71  # India grid-average-ish

    # Illustrative time-of-use tariff (₹ / kWh)
    off_peak_rate: float = 5.50
    mid_peak_rate: float = 8.20
    on_peak_rate: float = 12.40
    weekend_discount: float = 0.90


HOME = HomeProfile()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY") or os.getenv(
    "OPENWEATHERMAP_API_KEY", ""
)
LLM_MODEL = os.getenv("ECOHOME_LLM_MODEL", "gpt-4o-mini")


# Device catalogue used by the seeder and the optimizer
DEVICE_CATALOGUE: dict[str, dict] = {
    "EV": {
        "name": "EV Charger (7.2 kW / 3.3 kW typical)",
        "typical_kwh": 12.0,
        "flexible": True,
        "preferred": "solar_then_offpeak",
    },
    "HVAC": {
        "name": "Split AC (living + bedroom)",
        "typical_kwh": 8.0,
        "flexible": False,
        "preferred": "precool_before_peak",
    },
    "appliance": {
        "name": "Wet appliances (dishwasher / washer / dryer)",
        "typical_kwh": 1.6,
        "flexible": True,
        "preferred": "offpeak_or_solar",
    },
    "water_heater": {
        "name": "Storage water heater",
        "typical_kwh": 2.4,
        "flexible": True,
        "preferred": "solar_midday",
    },
    "pool": {
        "name": "Pool pump",
        "typical_kwh": 4.5,
        "flexible": True,
        "preferred": "solar",
    },
    "other": {
        "name": "Always-on + lighting + electronics",
        "typical_kwh": 6.0,
        "flexible": False,
        "preferred": "efficiency",
    },
}


def tou_period(hour: int) -> str:
    """Map an hour-of-day onto the home's TOU band."""
    if hour >= 18 and hour < 23:
        return "on_peak"
    if hour >= 23 or hour < 6:
        return "off_peak"
    if 10 <= hour < 16:
        return "mid_peak"  # solar-rich window, mid rate
    return "mid_peak"


def tou_rate(hour: int, is_weekend: bool = False) -> float:
    period = tou_period(hour)
    rate = {
        "off_peak": HOME.off_peak_rate,
        "mid_peak": HOME.mid_peak_rate,
        "on_peak": HOME.on_peak_rate,
    }[period]
    if is_weekend:
        rate *= HOME.weekend_discount
    return round(rate, 4)
