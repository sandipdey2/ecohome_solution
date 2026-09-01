"""python -m ecohome "When should I charge the EV tomorrow?" """

from __future__ import annotations

import argparse
import sys

from ecohome.agent import EnergyAdvisor
from ecohome.database import DatabaseManager
from ecohome.seed import seed_database


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EcoHome Energy Advisor")
    parser.add_argument("question", nargs="*", help="Question to ask the advisor")
    parser.add_argument("--seed", action="store_true", help="(Re)seed the demo database")
    parser.add_argument("--llm", action="store_true", help="Use the ReAct LLM agent if an API key is set")
    args = parser.parse_args(argv)

    db = DatabaseManager()
    if args.seed or db.usage_count() == 0:
        print("Seeding demo household history...", file=sys.stderr)
        seed_database()

    question = " ".join(args.question).strip()
    if not question:
        question = (
            "When should I charge my electric car tomorrow "
            "to minimize cost and maximize solar power?"
        )

    advisor = EnergyAdvisor(use_llm=args.llm)
    result = advisor.ask(question)
    print(result["answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
