"""LLM-as-judge helpers used by 03_run_and_evaluate.ipynb."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def _chat():
    api_key = os.getenv("VOCAREUM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    from langchain_openai import ChatOpenAI

    kwargs = {
        "model": os.getenv("ECOHOME_LLM_MODEL", "gpt-4o-mini"),
        "temperature": 0.0,
        "api_key": api_key,
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
            raw = raw[4:]
        raw = raw.strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    return json.loads(raw)


def _ask(system: str, user: str) -> Dict[str, Any]:
    llm = _chat()
    if llm is None:
        raise RuntimeError("No VOCAREUM_API_KEY / OPENAI_API_KEY for LLM judge")
    msg = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    return _parse_json(getattr(msg, "content", "") or "")


def evaluate_response(question: str, final_response: str, expected_response: str) -> Dict[str, Any]:
    """LLM-as-a-judge: accuracy, relevance, completeness, usefulness."""
    system = (
        "You are a strict evaluator for a home-energy advisor. "
        "Score each metric from 0 to 1. Return ONLY compact JSON with keys "
        "accuracy, relevance, completeness, usefulness, overall, feedback "
        "(feedback is a list of short strings). "
        "Accuracy = facts and expected concepts are correct. "
        "Relevance = stays on the asked device/topic. "
        "Completeness = hours, numbers, and a why. "
        "Usefulness = customer can act today. "
        "overall is the mean of the four scores."
    )
    user = (
        f"QUESTION:\n{question}\n\n"
        f"EXPECTED CONCEPTS:\n{expected_response}\n\n"
        f"AGENT ANSWER:\n{final_response}\n"
    )
    data = _ask(system, user)
    for key in ("accuracy", "relevance", "completeness", "usefulness"):
        data[key] = float(max(0.0, min(1.0, float(data.get(key, 0)))))
    data["overall"] = round(
        sum(data[k] for k in ("accuracy", "relevance", "completeness", "usefulness")) / 4, 3
    )
    if not isinstance(data.get("feedback"), list):
        data["feedback"] = [str(data.get("feedback") or "")]
    return data


def evaluate_tool_usage(messages_list, expected_tools: List[str]) -> Dict[str, Any]:
    """Rubric metrics from messages_list + expected_tools.

    1. Tool Appropriateness — were the right tools selected for the task?
    2. Tool Completeness — were all necessary tools used?
    Plus comprehensive, case-specific feedback for each metric.
    Scores come from an LLM judge and are independent (not copies).
    """
    used = _used_tools(messages_list)
    expected = list(expected_tools or [])
    system = (
        "You examine which tools an energy-advisor agent called.\n"
        "Score TWO independent metrics from 0 to 1. Never set them equal "
        "just to tidy the math.\n"
        "1) appropriateness: Were the tools that actually ran a good fit "
        "for the task? Wrong-family tools score low even if completeness "
        "is also low. Extra relevant tools may raise this score.\n"
        "2) completeness: Were all necessary/expected tools used? Name "
        "each missing tool and what the answer lost without it. If every "
        "expected tool ran, completeness is 1.0 even when extra tools also ran.\n"
        "Return ONLY JSON with keys appropriateness, completeness, and "
        "feedback (object with those two keys, each 2-4 sentences)."
    )
    user = (
        f"EXPECTED / NECESSARY TOOLS: {expected}\n"
        f"TOOLS ACTUALLY USED: {used}\n"
        f"Transcript sketch: {_message_sketch(messages_list)}\n"
    )
    data = _ask(system, user)
    appropriateness = float(max(0.0, min(1.0, float(data.get("appropriateness", 0)))))
    completeness = float(max(0.0, min(1.0, float(data.get("completeness", 0)))))
    fb = data.get("feedback") or {}
    if isinstance(fb, dict):
        feedback = [
            f"Tool Appropriateness ({appropriateness:.2f}): {fb.get('appropriateness', '')}",
            f"Tool Completeness ({completeness:.2f}): {fb.get('completeness', '')}",
        ]
    elif isinstance(fb, list):
        feedback = [str(x) for x in fb]
    else:
        feedback = [str(fb)]
    return {
        "appropriateness": appropriateness,
        "completeness": completeness,
        "overall": round((appropriateness + completeness) / 2, 3),
        "used_tools": used,
        "expected_tools": expected,
        "missing_tools": sorted(set(expected) - set(used)),
        "extra_tools": sorted(set(used) - set(expected)),
        "feedback": feedback,
    }


def recommend_improvements(summary: Dict[str, Any], cases: List[Dict[str, Any]]) -> List[str]:
    """LLM-authored improvement recommendations for the evaluation report."""
    system = (
        "You are a course reviewer for an energy-advisor agent. "
        "Given scores and per-case tool gaps, return JSON "
        '{"recommendations": ["...", "..."]} with 3 to 5 specific actions.'
    )
    compact = [
        {
            "id": c.get("id"),
            "response": (c.get("response_metrics") or {}).get("overall"),
            "tools": (c.get("tool_metrics") or {}).get("overall"),
            "missing": (c.get("tool_metrics") or {}).get("missing_tools"),
            "feedback": ((c.get("response_metrics") or {}).get("feedback") or [])[:2],
        }
        for c in cases
    ]
    data = _ask(system, json.dumps({"summary": summary, "cases": compact}, default=str))
    recs = data.get("recommendations") or []
    return [str(r) for r in recs][:6]


def _used_tools(messages) -> List[str]:
    used: List[str] = []
    for msg in messages or []:
        name = getattr(msg, "name", None)
        dump = msg.model_dump() if hasattr(msg, "model_dump") else {}
        if dump.get("tool_call_id") and name:
            used.append(name)
        calls = getattr(msg, "tool_calls", None) or dump.get("tool_calls") or []
        for call in calls:
            n = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if n:
                used.append(n)
    out, seen = [], set()
    for n in used:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _message_sketch(messages) -> List[str]:
    sketch = []
    for msg in messages or []:
        name = getattr(msg, "name", None) or type(msg).__name__
        sketch.append(str(name))
    return sketch[:24]
