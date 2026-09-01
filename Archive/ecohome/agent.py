"""EcoHome Energy Advisor — LangGraph orchestration.

Two graphs ship in this module:

1. `build_advisor_graph()` — an explicit StateGraph that classifies the
   question, calls the right tools, runs the optimizer, retrieves RAG
   tips and composes a grounded answer. This path does **not** need an
   LLM API key; it is the default, and it is what "make data-driven
   decisions" actually means.

2. `EnergyAdvisor.as_react_agent()` — the course-classic ReAct loop via
   `langgraph.prebuilt.create_react_agent`, used when OPENAI_API_KEY is
   set. Same tools, more fluent prose.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from ecohome.config import HOME, LLM_MODEL, OPENAI_API_KEY
from ecohome.tools import TOOL_BY_NAME, TOOL_KIT


SYSTEM_PROMPT = f"""You are the EcoHome Energy Advisor for a home in {HOME.location}.
You help the household cut electricity cost and grid carbon by shifting
flexible loads onto rooftop solar and off-peak tariff hours.

Tools you can call:
- get_home_profile: household configuration (solar kW, EV, tariff)
- get_weather_forecast(location, days): hourly weather + irradiance
- get_electricity_prices(date): hourly TOU rates
- query_energy_usage(start_date, end_date, device_type): history
- query_solar_generation(start_date, end_date): PV history
- get_recent_energy_summary(hours): recent usage + generation
- search_energy_tips(query): RAG over the energy-saving knowledge base
- calculate_energy_savings(...): money / kWh / CO2 arithmetic
- optimize_device_schedule(device_type, kwh): the actual scheduler

How to work:
1. Prefer tools over guesses. For any "when should I run X" question,
   call optimize_device_schedule plus weather and prices.
2. For "based on my usage" questions, call get_recent_energy_summary
   or query_energy_usage for the last 7–30 days.
3. Cite knowledge-base tips when you recommend a behaviour change.
4. Give concrete hours, ₹ amounts and a one-line why.
5. Currency is {HOME.currency}. Peak is 18:00–23:00, off-peak 23:00–06:00,
   solar plateau typically 10:00–15:00.
6. If a tool errors, say so briefly and fall back to best practices.

Never invent meter readings. Never recommend charging an EV between
18:00 and 22:00 unless the battery is critically low and you say so.

Knowledge-base topics you should retrieve when relevant:
HVAC strategies, smart-home automation, renewable self-consumption,
seasonal (monsoon vs dry) tactics, and home-battery dispatch.
"""


class AdvisorState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    intent: str
    device: str
    target_date: str
    horizon_days: int
    tool_results: dict[str, Any]
    answer: str
    error: str


def _today() -> datetime:
    return datetime.now()


def _lookback_dates(days: int = 14) -> tuple[str, str]:
    end = _today()
    start = end - timedelta(days=days)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def classify(state: AdvisorState) -> AdvisorState:
    q = (state.get("question") or _last_human(state) or "").lower()
    device = _detect_device(q)
    intent: str
    if any(w in q for w in ("when", "schedule", "best time", "charge", "run my", "start my")):
        intent = "schedule"
    elif any(w in q for w in ("save", "saving", "how much", "cost", "roi", "₹", "rupee")):
        intent = "savings"
    elif any(w in q for w in ("history", "usage", "pattern", "last week", "my data", "summary")):
        intent = "history"
    elif any(w in q for w in ("thermostat", "temperature", "ac ", "a/c", "hvac", "pre-cool", "precool")):
        intent = "hvac"
    elif any(w in q for w in ("tip", "advice", "best practice", "how can i", "ways to", "reduce")):
        intent = "tips"
    elif any(w in q for w in ("weather", "solar", "sun", "forecast", "irradiance")):
        intent = "solar"
    else:
        intent = "general"
    return {
        "intent": intent,
        "device": device,
        "target_date": _parse_target_date(q),
        "horizon_days": 7 if "week" in q else 3 if "wednesday" in q or "tomorrow" in q else 2,
    }


def gather(state: AdvisorState) -> AdvisorState:
    intent = state.get("intent") or "general"
    device = state.get("device") or ""
    question = state.get("question") or _last_human(state)
    target_date = state.get("target_date") or ""
    horizon = int(state.get("horizon_days") or 2)
    results: dict[str, Any] = {}
    errors: list[str] = []

    def call(name: str, args: dict) -> Any:
        tool = TOOL_BY_NAME[name]
        try:
            raw = tool.invoke(args)
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(parsed, dict) and parsed.get("error"):
                errors.append(f"{name}: {parsed['error']}")
            return parsed
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            return {"error": str(exc), "tool": name}

    results["profile"] = call("get_home_profile", {})
    start, end = _lookback_dates(14)

    if intent in {"schedule", "savings", "hvac", "general", "solar"}:
        results["weather"] = call(
            "get_weather_forecast",
            {"location": HOME.location, "days": horizon},
        )
        results["prices"] = call("get_electricity_prices", {"date": target_date})

    if intent in {"history", "savings", "general", "tips"}:
        results["summary_7d"] = call("get_recent_energy_summary", {"hours": 24 * 7})
        results["usage_14d"] = call(
            "query_energy_usage",
            {"start_date": start, "end_date": end, "device_type": device},
        )
        results["solar_14d"] = call(
            "query_solar_generation", {"start_date": start, "end_date": end}
        )

    if intent in {"schedule", "savings", "hvac"} and device:
        results["schedule"] = call(
            "optimize_device_schedule",
            {"device_type": device, "kwh": 0.0, "date": target_date, "days": horizon},
        )

    if intent == "schedule" and not device:
        # Default to EV — the highest-value flexible load
        results["schedule"] = call(
            "optimize_device_schedule",
            {"device_type": "EV", "kwh": 0.0, "date": target_date, "days": horizon},
        )

    rag_query = question or device or "energy savings time of use solar"
    results["tips"] = call("search_energy_tips", {"query": rag_query, "max_results": 3})

    if errors:
        results["tool_errors"] = errors

    if intent == "savings" and results.get("schedule") and not results["schedule"].get("error"):
        sched = results["schedule"]
        event = float(sched.get("savings_vs_evening") or 0)
        kg_event = float(sched.get("grid_kwh") or 0) * HOME.grid_emission_factor_kg_per_kwh
        # Naive evening plan uses almost all grid energy; optimized uses solar.
        naive_grid = float(sched.get("kwh_requested") or 0)
        kg_saved = max(naive_grid - float(sched.get("grid_kwh") or 0), 0) * HOME.grid_emission_factor_kg_per_kwh
        results["savings"] = {
            "savings_per_event": round(event, 2),
            "savings_per_year": round(event * 365, 2),
            "co2_kg_saved_per_event": round(kg_saved, 3),
            "co2_kg_saved_per_year": round(kg_saved * 365, 1),
            "currency": HOME.currency,
        }

    return {"tool_results": results, "error": "; ".join(errors) if errors else ""}


def compose(state: AdvisorState) -> AdvisorState:
    intent = state.get("intent") or "general"
    device = state.get("device") or "device"
    q = state.get("question") or _last_human(state)
    tr = state.get("tool_results") or {}
    parts: list[str] = []

    parts.append(f"**EcoHome Energy Advisor** — {HOME.location}")
    parts.append(f"_Question:_ {q}")
    parts.append("")

    if weather := tr.get("weather"):
        friendly = weather.get("solar_friendly_hours") or []
        src = weather.get("source", "forecast")
        if friendly:
            day0 = friendly[0]
            parts.append(
                f"**Solar window** ({src}): {day0.get('date')} looks best around "
                f"{_hours_label(day0.get('best_hours', []))} "
                f"(peak irradiance {day0.get('peak_irradiance')} W/m²)."
            )
        if weather.get("warning"):
            parts.append(f"_Note:_ {weather['warning']}")
        parts.append("")

    if prices := tr.get("prices"):
        parts.append(
            f"**Tariff today** ({prices.get('date')}): "
            f"off-peak {HOME.currency_symbol}{prices.get('off_peak_rate')}/kWh "
            f"at {_hours_label(prices.get('off_peak_hours', []))} · "
            f"on-peak {HOME.currency_symbol}{prices.get('on_peak_rate')}/kWh "
            f"at {_hours_label(prices.get('on_peak_hours', []))}."
        )
        parts.append("")

    if summary := tr.get("summary_7d"):
        parts.append("**Last 7 days**")
        parts.append(
            f"- Used {summary.get('usage_kwh')} kWh "
            f"({HOME.currency_symbol}{summary.get('usage_cost')}) · "
            f"solar produced {summary.get('solar_kwh')} kWh · "
            f"self-sufficiency {summary.get('self_sufficiency_pct')}%."
        )
        parts.append(
            f"- On-peak consumption {summary.get('on_peak_usage_kwh')} kWh. "
            f"Biggest load: {summary.get('biggest_load')}."
        )
        if bd := summary.get("by_device"):
            ranked = sorted(bd.items(), key=lambda kv: kv[1]["kwh"], reverse=True)
            bits = [f"{k} {v['kwh']} kWh ({HOME.currency_symbol}{v['cost']})" for k, v in ranked]
            parts.append("- By device: " + " · ".join(bits))
        parts.append("")

    if sched := tr.get("schedule"):
        if sched.get("error"):
            parts.append(f"Scheduler error: {sched['error']}")
        else:
            label = sched.get("device_type") or device or "device"
            parts.append(f"**Recommended {label} schedule**")
            parts.append(
                f"- Run at {_hours_label(sched.get('hours', []))} "
                f"({sched.get('style')})."
            )
            parts.append(
                f"- Planned cost {HOME.currency_symbol}{sched.get('cost')} · "
                f"{sched.get('solar_kwh')} kWh from solar, "
                f"{sched.get('grid_kwh')} kWh from grid."
            )
            parts.append(
                f"- Versus a naïve 18:00+ start "
                f"({_hours_label(sched.get('naive_plan_hours', []))}): "
                f"you avoid {HOME.currency_symbol}{sched.get('savings_vs_evening')} "
                f"on this event."
            )
            if therm := sched.get("thermostat"):
                for day in therm.get("days", [])[:2]:
                    parts.append(
                        f"- {day['date']}: pre-cool to {day['precool_setpoint_c']}°C "
                        f"at {_hours_label(day['precool_window'])}, then hold "
                        f"{day['recommended_setpoint_c']}°C through the peak. "
                        f"{day['rationale']}"
                    )
            parts.append("")

    if sav := tr.get("savings"):
        parts.append("**Annualised impact** (assuming this event repeats daily):")
        parts.append(
            f"- {HOME.currency_symbol}{sav.get('savings_per_year')}/year · "
            f"{sav.get('co2_kg_saved_per_year')} kg CO₂e/year."
        )
        parts.append("")

    if tips := (tr.get("tips") or {}).get("tips"):
        parts.append("**From the knowledge base**")
        for tip in tips[:3]:
            snippet = _first_sentences(tip.get("content", ""), 2)
            parts.append(f"- ({tip.get('source')}) {snippet}")
        parts.append("")

    parts.append("**Next step:** set a delay-start / wallbox schedule for the hours above. "
                 "Ask me about another device if you want a whole-home plan.")

    answer = "\n".join(parts).strip()
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


def build_advisor_graph():
    """Compile the explicit EcoHome StateGraph."""
    g = StateGraph(AdvisorState)
    g.add_node("classify", classify)
    g.add_node("gather", gather)
    g.add_node("compose", compose)
    g.add_edge(START, "classify")
    g.add_edge("classify", "gather")
    g.add_edge("gather", "compose")
    g.add_edge("compose", END)
    return g.compile()


def _parse_target_date(q: str) -> str:
    """Resolve 'tomorrow' / weekday names to YYYY-MM-DD."""
    today = _today().date()
    q = q.lower()
    if "today" in q:
        return today.isoformat()
    if "tomorrow" in q:
        return (today + timedelta(days=1)).isoformat()
    weekdays = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]
    for i, name in enumerate(weekdays):
        if name in q:
            delta = (i - today.weekday()) % 7
            if delta == 0:
                delta = 7
            return (today + timedelta(days=delta)).isoformat()
    return today.isoformat()


def _detect_device(q: str) -> str:
    if any(w in q for w in ("ev", "car", "vehicle", "charger", "tesla", "nexon")):
        return "EV"
    if any(w in q for w in ("thermostat", "hvac", "ac ", "a/c", "air con", "cool")):
        return "HVAC"
    if any(w in q for w in ("pool", "pump")):
        return "pool"
    if any(w in q for w in ("geyser", "water heater", "hot water")):
        return "water_heater"
    if any(w in q for w in ("dishwasher", "washer", "washing", "dryer", "laundry", "appliance")):
        return "appliance"
    return ""


def _last_human(state: AdvisorState) -> str:
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
        if isinstance(msg, dict) and msg.get("role") in {"user", "human"}:
            return str(msg.get("content", ""))
    return ""


def _hours_label(hours: list) -> str:
    if not hours:
        return "n/a"
    uniq = []
    for h in hours:
        try:
            hv = int(h)
        except (TypeError, ValueError):
            continue
        if hv not in uniq:
            uniq.append(hv)
    return ", ".join(f"{h:02d}:00" for h in uniq)


def _first_sentences(text: str, n: int = 2) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return " ".join(parts[:n]).strip()


class EnergyAdvisor:
    """Public façade used by the CLI, notebooks and tests."""

    def __init__(self, use_llm: Optional[bool] = None, model: str = LLM_MODEL) -> None:
        self.use_llm = OPENAI_API_KEY != "" if use_llm is None else use_llm
        self.model_name = model
        self.graph = build_advisor_graph()
        self.react = None
        self.history: list[tuple[str, str]] = []
        if self.use_llm and OPENAI_API_KEY:
            self.react = self._build_react(model)

    def _build_react(self, model: str):
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import create_react_agent

        llm = ChatOpenAI(model=model, temperature=0, api_key=OPENAI_API_KEY)
        return create_react_agent(
            model=llm,
            tools=TOOL_KIT,
            prompt=SystemMessage(content=SYSTEM_PROMPT),
        )

    def ask(self, question: str, context: str = "") -> dict[str, Any]:
        """Run one turn. Returns answer text plus raw state/messages."""
        q = question.strip()
        if context:
            q = f"{q}\n\nContext: {context}"

        if self.react is not None:
            messages: list[BaseMessage] = []
            for role, content in self.history[-6:]:
                messages.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
            messages.append(HumanMessage(content=q))
            raw = self.react.invoke({"messages": messages})
            answer = _message_text(raw["messages"][-1]) if isinstance(raw, dict) else str(raw)
            self.history.append(("user", question))
            self.history.append(("assistant", answer))
            return {"answer": answer, "mode": "react", "raw": raw}

        raw = self.graph.invoke(
            {
                "question": q,
                "messages": [HumanMessage(content=q)],
                "intent": "",
                "device": "",
                "tool_results": {},
                "answer": "",
            }
        )
        answer = raw.get("answer") or _message_text((raw.get("messages") or [AIMessage("")])[-1])
        self.history.append(("user", question))
        self.history.append(("assistant", answer))
        return {
            "answer": answer,
            "mode": "graph",
            "intent": raw.get("intent"),
            "device": raw.get("device"),
            "tool_results": raw.get("tool_results"),
            "raw": raw,
        }

    def invoke(
        self,
        question: str,
        context: Optional[str] = None,
        reset_history: bool = False,
    ) -> dict[str, Any]:
        """Course-compatible entry point used by the evaluation notebook."""
        if reset_history:
            self.reset()
        extra = context or ""
        if self.history and not extra:
            extra = "Prior turns:\n" + "\n".join(
                f"{role}: {content[:400]}" for role, content in self.history[-4:]
            )
        try:
            return self.ask(question, context=extra)
        except Exception as exc:
            fallback = (
                "I could not finish a full data-driven pass "
                f"({type(exc).__name__}: {exc}). "
                " Generic fallback: shift EV, dishwasher, laundry and the pool "
                "pump off the 18:00–23:00 peak and onto midday solar or 23:00–06:00 "
                f"off-peak. Peak tariff is {HOME.currency_symbol}{HOME.on_peak_rate}/kWh."
            )
            return {"answer": fallback, "mode": "error", "error": str(exc)}

    def reset(self) -> None:
        self.history.clear()


class Agent(EnergyAdvisor):
    """Alias matching the Udacity starter `Agent` class name."""

    def __init__(self, instructions: Optional[str] = None, model: str = LLM_MODEL) -> None:
        super().__init__(use_llm=None, model=model)
        if instructions:
            self.system_instructions = instructions


def _message_text(msg: Any) -> str:
    if msg is None:
        return ""
    if isinstance(msg, BaseMessage):
        c = msg.content
        return c if isinstance(c, str) else str(c)
    return str(msg)
