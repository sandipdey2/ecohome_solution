#!/usr/bin/env python3
"""Run the Energy Advisor against the four canonical project questions."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecohome.agent import EnergyAdvisor
from ecohome.database import DatabaseManager
from ecohome.seed import seed_database


QUESTIONS = [
    "When should I charge my electric car tomorrow to minimize cost and maximize solar power?",
    "What temperature should I set my thermostat on Wednesday afternoon if electricity prices spike?",
    "Suggest three ways I can reduce energy use based on my usage history.",
    "How much can I save by running my dishwasher during off-peak hours?",
    "What's the best time to run my pool pump this week based on the weather forecast?",
]


def main() -> None:
    db = DatabaseManager()
    if db.usage_count() == 0:
        print("Seeding empty database...")
        seed_database()

    advisor = EnergyAdvisor(use_llm=False)
    for i, q in enumerate(QUESTIONS, 1):
        print("=" * 78)
        print(f"Q{i}. {q}")
        print("-" * 78)
        result = advisor.ask(q)
        print(result["answer"])
        print()


if __name__ == "__main__":
    main()
