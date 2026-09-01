#!/usr/bin/env python3
"""Lightweight evaluation harness for the EcoHome advisor.

Checks the five rubric axes used in the project brief:
accuracy, relevance, completeness, tool usage, reasoning.
Runs without an LLM so results are deterministic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecohome.agent import EnergyAdvisor
from ecohome.database import DatabaseManager
from ecohome.seed import seed_database


CASES = [
    {
        "id": "ev_schedule",
        "question": "When should I charge my electric car tomorrow to minimize cost and maximize solar power?",
        "expect_intent": "schedule",
        "expect_device": "EV",
        "must_tools": {"weather", "prices", "schedule", "tips"},
        "answer_must_include": ["solar", "peak"],
        "custom": "ev_not_peak",
    },
    {
        "id": "hvac_setpoint",
        "question": "What temperature should I set my thermostat tomorrow afternoon if electricity prices spike?",
        "expect_intent": "hvac",
        "expect_device": "HVAC",
        "must_tools": {"weather", "prices", "schedule"},
        "answer_must_include": ["°C", "peak"],
        "custom": None,
    },
    {
        "id": "history_tips",
        "question": "Suggest three ways I can reduce energy use based on my usage history.",
        "expect_intent": "history",
        "expect_device": "",
        "must_tools": {"summary_7d", "tips"},
        "answer_must_include": ["kWh"],
        "custom": None,
    },
    {
        "id": "dishwasher_savings",
        "question": "How much can I save by running my dishwasher during off-peak hours?",
        "expect_intent": "savings",
        "expect_device": "appliance",
        "must_tools": {"prices", "schedule"},
        "answer_must_include": ["₹"],
        "custom": "positive_savings",
    },
    {
        "id": "pool_week",
        "question": "What's the best time to run my pool pump this week based on the weather forecast?",
        "expect_intent": "schedule",
        "expect_device": "pool",
        "must_tools": {"weather", "schedule", "tips"},
        "answer_must_include": ["pool"],
        "custom": "positive_savings",
    },
    {
        "id": "hvac_wednesday",
        "question": "What temperature should I set my thermostat on Wednesday afternoon if electricity prices spike?",
        "expect_intent": "hvac",
        "expect_device": "HVAC",
        "must_tools": {"weather", "prices", "schedule"},
        "answer_must_include": ["°C"],
        "custom": None,
    },
]


def evaluate_case(advisor: EnergyAdvisor, case: dict) -> dict:
    result = advisor.ask(case["question"])
    tools = set((result.get("tool_results") or {}).keys())
    answer = result.get("answer") or ""
    checks = []

    checks.append(
        _check(
            "intent",
            result.get("intent") == case["expect_intent"],
            f"intent={result.get('intent')} expected {case['expect_intent']}",
        )
    )
    if case["expect_device"]:
        checks.append(
            _check(
                "device",
                result.get("device") == case["expect_device"],
                f"device={result.get('device')} expected {case['expect_device']}",
            )
        )
    missing = set(case["must_tools"]) - tools
    checks.append(_check("tool_usage", not missing, f"missing tools {missing}"))

    missing_words = [
        w
        for w in case["answer_must_include"]
        if w.lower() not in answer.lower() and w not in answer
    ]
    checks.append(_check("relevance", not missing_words, f"answer missing {missing_words}"))
    checks.append(_check("completeness", len(answer) > 350, f"answer only {len(answer)} chars"))
    checks.append(
        _check(
            "reasoning",
            any(s in answer.lower() for s in ("versus", "because", "peak", "solar")),
            "no explicit rationale",
        )
    )

    if case["custom"] == "ev_not_peak":
        sched = (result.get("tool_results") or {}).get("schedule") or {}
        hours = sched.get("hours") or []
        overlap = [h for h in hours if 18 <= int(h) <= 21]
        checks.append(
            _check("accuracy", not overlap, f"EV schedule still hits peak hours {overlap}")
        )
        checks.append(
            _check(
                "savings_positive",
                float(sched.get("savings_vs_evening") or 0) > 0,
                f"savings_vs_evening={sched.get('savings_vs_evening')}",
            )
        )
    if case["custom"] == "positive_savings":
        sched = (result.get("tool_results") or {}).get("schedule") or {}
        checks.append(
            _check(
                "accuracy",
                float(sched.get("savings_vs_evening") or 0) > 0,
                f"dishwasher savings={sched.get('savings_vs_evening')}",
            )
        )

    passed = sum(1 for c in checks if c["pass"])
    return {
        "id": case["id"],
        "passed": passed,
        "total": len(checks),
        "score": round(passed / len(checks), 3),
        "checks": checks,
        "intent": result.get("intent"),
        "preview": answer[:240].replace("\n", " "),
    }


def _check(name: str, ok: bool, detail: str) -> dict:
    return {"name": name, "pass": bool(ok), "detail": detail}


def main() -> None:
    db = DatabaseManager()
    if db.usage_count() == 0:
        seed_database()
    advisor = EnergyAdvisor(use_llm=False)
    reports = [evaluate_case(advisor, c) for c in CASES]
    overall = sum(r["passed"] for r in reports) / max(sum(r["total"] for r in reports), 1)
    print(json.dumps({"overall": round(overall, 3), "cases": reports}, indent=2))
    if overall < 0.8:
        sys.exit(1)


if __name__ == "__main__":
    main()
