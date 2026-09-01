#!/usr/bin/env python3
"""Create tables and load 60 days of household + solar history."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecohome.seed import seed_database


if __name__ == "__main__":
    info = seed_database(days=60, seed=42, reset=True)
    print("EcoHome database ready.")
    for k, v in info.items():
        print(f"  {k}: {v}")
