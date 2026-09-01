"""Structured evaluation report for notebook 03.

generate_evaluation_report() builds the document.
display_evaluation_report() prints it.
Recommendations are produced by an LLM from the actual case outcomes.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def _judge_llm():
    key = os.getenv("VOCAREUM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set VOCAREUM_API_KEY or OPENAI_API_KEY")
    from langchain_openai import ChatOpenAI

    kwargs = {
        "model": os.getenv("ECOHOME_LLM_MODEL", "gpt-4o-mini"),
        "temperature": 0,
        "api_key": key,
    }
    base = os.getenv("OPENAI_BASE_URL")
    if os.getenv("VOCAREUM_API_KEY") or base:
        kwargs["base_url"] = base or "https://openai.vocareum.com/v1"
    return ChatOpenAI(**kwargs)


def _parse_json(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    start, end = raw.find("{"), raw.rfind("}")
    return json.loads(raw[start : end + 1])


def llm_recommendations(summary: Dict[str, Any], cases: List[Dict[str, Any]]) -> List[str]:
    """Case-specific improvement list. Not threshold strings."""
    llm = _judge_llm()
    payload = {
        "overall_scores": summary,
        "cases": [
            {
                "id": c.get("id"),
                "question": c.get("question"),
                "response_overall": (c.get("response_metrics") or {}).get("overall"),
                "tool_overall": (c.get("tool_metrics") or {}).get("overall"),
                "missing_tools": (c.get("tool_metrics") or {}).get("missing_tools"),
                "used_tools": (c.get("tool_metrics") or {}).get("used_tools"),
                "feedback": (c.get("response_metrics") or {}).get("feedback"),
                "preview": c.get("preview"),
            }
            for c in cases
        ],
    }
    msg = llm.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are the course reviewer for EcoHome. "
                    "Read the overall scores and each case (missing tools, "
                    "feedback, preview). Return ONLY JSON of the form "
                    '{"recommendations": ["...", "..."]} with 3 to 5 items. '
                    "Each item must name the failing case id and the fix. "
                    "Do not emit generic slogans."
                ),
            },
            {"role": "user", "content": json.dumps(payload, default=str)},
        ]
    )
    data = _parse_json(getattr(msg, "content", "") or "")
    recs = []
    for item in data.get("recommendations") or []:
        if isinstance(item, dict):
            cid = item.get("case_id") or item.get("id") or ""
            fix = item.get("fix") or item.get("recommendation") or item.get("text") or item
            recs.append(f"{cid}: {fix}".strip(": "))
        else:
            recs.append(str(item))
    return recs


def generate_evaluation_report(test_results: List[Dict[str, Any]],
                               evaluate_response,
                               evaluate_tool_usage) -> Dict[str, Any]:
    """Build the rubric report: scores, strengths, weaknesses, recommendations."""
    rows: List[Dict[str, Any]] = []
    for result in test_results:
        resp = result.get("response") or {}
        messages = resp.get("messages") if isinstance(resp, dict) else []
        final = ""
        if messages:
            final = getattr(messages[-1], "content", "") or str(messages[-1])
        elif not isinstance(resp, dict):
            final = str(resp)
        r_eval = evaluate_response(result.get("question"), final, result.get("expected_response", ""))
        t_eval = evaluate_tool_usage(messages, result.get("expected_tools", []))
        rows.append(
            {
                "id": result.get("test_id"),
                "question": result.get("question"),
                "response_metrics": r_eval,
                "tool_metrics": t_eval,
                "preview": (final or "")[:200].replace("\n", " "),
                "error": result.get("error"),
            }
        )

    n = max(len(rows), 1)

    def mean(path_a, path_b):
        return round(sum((r[path_a] or {}).get(path_b, 0) for r in rows) / n, 3)

    overall_scores = {
        "n_tests": len(rows),
        "mean_response": mean("response_metrics", "overall"),
        "mean_tool_usage": mean("tool_metrics", "overall"),
        "accuracy": mean("response_metrics", "accuracy"),
        "relevance": mean("response_metrics", "relevance"),
        "completeness": mean("response_metrics", "completeness"),
        "usefulness": mean("response_metrics", "usefulness"),
        "tool_appropriateness": mean("tool_metrics", "appropriateness"),
        "tool_completeness": mean("tool_metrics", "completeness"),
        "composite": 0.0,
    }
    overall_scores["composite"] = round(
        (overall_scores["mean_response"] + overall_scores["mean_tool_usage"]) / 2, 3
    )

    strengths, weaknesses = [], []
    labels = [
        ("accuracy", overall_scores["accuracy"]),
        ("relevance", overall_scores["relevance"]),
        ("completeness", overall_scores["completeness"]),
        ("usefulness", overall_scores["usefulness"]),
        ("tool appropriateness", overall_scores["tool_appropriateness"]),
        ("tool completeness", overall_scores["tool_completeness"]),
    ]
    for label, val in labels:
        bucket = strengths if val >= 0.7 else weaknesses
        bucket.append({"metric": label, "score": val})

    for row in rows:
        missing = (row.get("tool_metrics") or {}).get("missing_tools") or []
        if missing:
            weaknesses.append(
                {
                    "metric": f"tools:{row['id']}",
                    "score": (row.get("tool_metrics") or {}).get("overall"),
                    "detail": f"{row['id']} missed {missing}",
                }
            )

    recommendations = llm_recommendations(overall_scores, rows)

    return {
        "title": "EcoHome Energy Advisor — Evaluation Report",
        "overall_scores": overall_scores,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "recommendations": recommendations,
        "cases": rows,
    }


def display_evaluation_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Separate display function required by the rubric."""
    s = report["overall_scores"]
    print("=" * 72)
    print(report.get("title") or "EVALUATION REPORT")
    print("=" * 72)
    print("\n1. Overall scores")
    print(f"   Tests                 : {s['n_tests']}")
    print(f"   Composite             : {s['composite']:.2f}")
    print(f"   Mean response         : {s['mean_response']:.2f}")
    print(f"   Mean tool usage       : {s['mean_tool_usage']:.2f}")
    print(f"   accuracy={s['accuracy']:.2f}  relevance={s['relevance']:.2f}  "
          f"completeness={s['completeness']:.2f}  usefulness={s['usefulness']:.2f}")
    print(f"   tool_appropriateness={s['tool_appropriateness']:.2f}  "
          f"tool_completeness={s['tool_completeness']:.2f}")

    print("\n2. Strengths")
    for item in report["strengths"]:
        extra = f" — {item['detail']}" if item.get("detail") else ""
        print(f"   + {item['metric']}={item['score']:.2f}{extra}")

    print("\n3. Weaknesses")
    weak = report["weaknesses"] or [{"metric": "none flagged", "score": 1.0}]
    for item in weak:
        extra = f" — {item['detail']}" if item.get("detail") else ""
        print(f"   - {item['metric']}={item.get('score', 0):.2f}{extra}")

    print("\n4. Recommendations for improvement (LLM, from this run)")
    for i, rec in enumerate(report["recommendations"], 1):
        print(f"   {i}. {rec}")

    print("\n5. Per-case appendix")
    for row in report["cases"]:
        rm, tm = row["response_metrics"], row["tool_metrics"]
        print(f"\n   [{row['id']}] response={rm.get('overall')} tools={tm.get('overall')}")
        print("      Q:", (row.get("question") or "")[:90])
        print("      used:", tm.get("used_tools"), " missing:", tm.get("missing_tools"))
        for line in (rm.get("feedback") or [])[:4]:
            print("     ", line)
        print("      preview:", row.get("preview", "")[:140])
    print("=" * 72)
    return report
