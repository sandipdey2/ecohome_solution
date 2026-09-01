"""EcoHome Energy Advisor — LangGraph agent (schema, nodes, edges)."""

from __future__ import annotations

import json
import os
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from tools import TOOL_KIT, TOOL_KIT as _TOOLS

load_dotenv()

TOOL_BY_NAME = {t.name: t for t in TOOL_KIT}


DEFAULT_SYSTEM_INSTRUCTIONS = """
Who you are
You are the EcoHome Energy Advisor, a smart-home energy optimization agent.
Your role is to help a household with rooftop solar, an EV, HVAC and
appliances cut electricity cost and carbon.

What you should do (steps)
1. Read the question and any Location context.
2. Decide which tools you need. Never guess weather, prices or meter data.
3. Call the tools. For scheduling questions always call get_weather_forecast
   AND get_electricity_prices. For "my usage / history" questions call
   get_recent_energy_summary or query_energy_usage. For "how much can I save"
   also call calculate_energy_savings. Always call search_energy_tips when
   giving behaviour advice.
4. Analyse the tool results (peak vs off-peak rates, solar-friendly hours,
   device totals).
5. Write a recommendation.

Key capabilities
- Weather integration and solar-window prediction
- Time-of-use price optimization
- Historical usage analysis
- RAG over the energy-saving knowledge base
- Multi-device plans (EV, HVAC, appliances, pool, battery)
- Savings and simple ROI arithmetic

Recommendation format
- Concrete clock hours (e.g. 11:00–14:00), not "sometime in the afternoon"
- A dollar figure when a tariff shift is involved
- A one-line why that cites the tool data
- A knowledge-base citation (source filename)
- A next step the customer can take today
Never invent meter readings. Never put EV charging on 16:00–21:00 peak
unless the battery is critically low and you say so.

Example questions you handle
- "When should I charge my electric car tomorrow to minimize cost and maximize solar power?"
- "What temperature should I set my thermostat on Wednesday afternoon if electricity prices spike?"
- "Suggest three ways I can reduce energy use based on my usage history."
- "How much can I save by running my dishwasher during off-peak hours?"
- "What's the best time to run my pool pump this week based on the weather forecast?"

Tariff shape to assume unless tools say otherwise
- Off-peak 23:00–06:00, on-peak 16:00–21:00, solar plateau 10:00–15:00.
""".strip()


class AgentState(TypedDict, total=False):
    """LangGraph state schema."""

    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    context: str
    used_tools: List[str]


def _llm():
    api_key = os.getenv("VOCAREUM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    from langchain_openai import ChatOpenAI

    kwargs = {"model": os.getenv("ECOHOME_LLM_MODEL", "gpt-4o-mini"), "temperature": 0.0, "api_key": api_key}
    if os.getenv("VOCAREUM_API_KEY"):
        kwargs["base_url"] = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
    return ChatOpenAI(**kwargs)


def build_energy_advisor_graph(instructions: str, model: str = "gpt-4o-mini"):
    """Explicit StateGraph: agent node → tools node → agent node → END."""

    llm = _llm()
    if llm is not None:
        llm = llm.bind_tools(TOOL_KIT)

    def agent_node(state: AgentState) -> AgentState:
        if llm is None:
            return _offline_agent_node(state, instructions)
        sys = [SystemMessage(content=instructions)]
        if state.get("context"):
            sys.append(SystemMessage(content=f"Additional user context: {state['context']}"))
        result = llm.invoke(sys + list(state.get("messages") or []))
        if not (getattr(result, "tool_calls", None) or []):
            pending = _pending_tools(state)
            if pending:
                result = AIMessage(
                    content="Calling required energy tools before recommending.",
                    tool_calls=[
                        {"name": n, "args": _default_args(n, state), "id": n} for n in pending
                    ],
                )
        return {"messages": [result]}

    def tools_node(state: AgentState) -> AgentState:
        last = (state.get("messages") or [None])[-1]
        calls = getattr(last, "tool_calls", None) or []
        outbound: List[BaseMessage] = []
        used = list(state.get("used_tools") or [])
        for call in calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {}) or {}
            call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", name)
            tool = TOOL_BY_NAME.get(name)
            if tool is None:
                payload = json.dumps({"error": f"unknown tool {name}"})
            else:
                try:
                    raw = tool.invoke(args)
                    payload = raw if isinstance(raw, str) else json.dumps(raw, default=str)
                except Exception as exc:
                    payload = json.dumps({"error": str(exc)})
            used.append(name)
            outbound.append(ToolMessage(content=payload, tool_call_id=call_id, name=name))
        return {"messages": outbound, "used_tools": used}

    def route(state: AgentState) -> Literal["tools", "end"]:
        last = (state.get("messages") or [None])[-1]
        calls = getattr(last, "tool_calls", None) or []
        return "tools" if calls else "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def _offline_agent_node(state: AgentState, instructions: str) -> AgentState:
    """Planner used when no LLM key is present — still a graph node."""
    question = state.get("question") or ""
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, HumanMessage):
            question = msg.content if isinstance(msg.content, str) else str(msg.content)
            break
    already = {getattr(m, "name", "") for m in state.get("messages") or []}
    needed = _plan_tools(question)
    pending = [n for n in needed if n not in already]
    if pending:
        # Ask the tools node to run them via synthetic tool_calls
        calls = [{"name": n, "args": _default_args(n, state), "id": n} for n in pending]
        msg = AIMessage(content="Calling tools for a data-driven plan.", tool_calls=calls)
        return {"messages": [msg]}
    answer = _compose_offline(question, state)
    return {"messages": [AIMessage(content=answer)]}


def _plan_tools(question: str) -> List[str]:
    q = question.lower()
    tools: List[str] = []

    meter_history = any(w in q for w in ("meter", "kwh did", "how many kwh", "last 7 days", "last seven"))
    if meter_history and any(w in q for w in ("ev", "hvac", "appliance", "used")):
        return ["query_energy_usage"]

    if any(w in q for w in ("history", "usage history", "reduce energy use", "three ways")):
        tools.extend(["get_recent_energy_summary", "search_energy_tips"])
    if any(w in q for w in ("solar generation", "self-consumption", "maximize solar", "maximize self")):
        tools.extend(["get_weather_forecast", "query_solar_generation"])
    if any(w in q for w in ("save", "saving", "how much can i save")):
        tools.extend(["get_electricity_prices", "calculate_energy_savings"])
    if any(w in q for w in ("charge", "ev", "car", "dishwasher", "washer", "dryer", "pool", "thermostat", "pre-cool", "precool", "battery", "peak")):
        tools.extend(["get_weather_forecast", "get_electricity_prices"])
    if any(w in q for w in ("battery", "storage", "tip", "best practice")):
        tools.append("search_energy_tips")
    if "weather" in q or "forecast" in q:
        tools.append("get_weather_forecast")

    # unique, stable order
    out, seen = [], set()
    for name in tools:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out or ["get_weather_forecast", "get_electricity_prices"]


def _pending_tools(state: AgentState) -> List[str]:
    question = state.get("question") or ""
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, HumanMessage):
            question = msg.content if isinstance(msg.content, str) else str(msg.content)
            break
    already = set()
    for msg in state.get("messages") or []:
        name = getattr(msg, "name", None)
        if name in TOOL_BY_NAME:
            already.add(name)
        dump = msg.model_dump() if hasattr(msg, "model_dump") else {}
        if dump.get("tool_call_id") and name:
            already.add(name)
    return [n for n in _plan_tools(question) if n not in already]


def _default_args(name: str, state: AgentState) -> Dict[str, Any]:
    from datetime import datetime, timedelta

    location = "San Francisco, CA"
    ctx = state.get("context") or ""
    if "location:" in ctx.lower():
        location = ctx.split(":", 1)[-1].strip()
    today = datetime.now().strftime("%Y-%m-%d")
    week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    mapping = {
        "get_weather_forecast": {"location": location, "days": 3},
        "get_electricity_prices": {"date": today},
        "search_energy_tips": {"query": state.get("question") or "energy savings", "max_results": 3},
        "get_recent_energy_summary": {"hours": 168},
        "query_solar_generation": {"start_date": week_ago, "end_date": today},
        "query_energy_usage": {
            "start_date": week_ago,
            "end_date": today,
            # omit device_type so EV + HVAC + appliances all come back
        },
        "calculate_energy_savings": {
            "device_type": "EV",
            "current_usage_kwh": 12.0,
            "optimized_usage_kwh": 5.0,
            "price_per_kwh": 0.35,
        },
    }
    q = (state.get("question") or "").lower()
    if name == "calculate_energy_savings" and "dishwasher" in q:
        mapping[name] = {
            "device_type": "appliance",
            "current_usage_kwh": 1.4,
            "optimized_usage_kwh": 0.55,
            "price_per_kwh": 0.35,
        }
    return mapping.get(name, {})


def _tool_payloads(state: AgentState) -> Dict[str, Any]:
    out = {}
    for msg in state.get("messages") or []:
        name = getattr(msg, "name", None)
        content = getattr(msg, "content", None)
        if name and content:
            try:
                out[name] = json.loads(content) if isinstance(content, str) else content
            except Exception:
                out[name] = {"raw": content}
    return out


def _fmt_hours(hours) -> str:
    vals = []
    for h in hours or []:
        try:
            hv = int(h)
        except (TypeError, ValueError):
            continue
        if hv not in vals:
            vals.append(hv)
    return ", ".join(f"{h:02d}:00" for h in vals)


def _compose_offline(question: str, state: AgentState) -> str:
    data = _tool_payloads(state)
    prices = data.get("get_electricity_prices") or {}
    weather = data.get("get_weather_forecast") or {}
    tips = data.get("search_energy_tips") or {}
    summary = data.get("get_recent_energy_summary") or {}
    savings = data.get("calculate_energy_savings") or {}
    off = prices.get("off_peak_hours", [23, 0, 1, 2, 3, 4, 5])
    peak = prices.get("on_peak_hours", [16, 17, 18, 19, 20, 21])
    friendly = []
    for day in weather.get("solar_friendly_hours") or []:
        friendly.extend(day.get("best_hours") or [])
    solar_hours = sorted(set(friendly))[:6] or list(range(10, 16))
    q = question.lower()
    if "ev" in q or "car" in q or "charg" in q:
        body = (
            f"Charge in the solar plateau ({_fmt_hours(solar_hours)}) first, "
            f"then finish overnight off-peak ({_fmt_hours(off)}). "
            f"Avoid on-peak {_fmt_hours(peak)}."
        )
    elif "thermostat" in q or "temperature" in q or "pre-cool" in q or "hvac" in q:
        body = (
            f"Pre-cool 1°F between 14:00 and 16:00, then hold 76–78°F through "
            f"on-peak {_fmt_hours(peak)}."
        )
    elif "dishwasher" in q or "washer" in q or "dryer" in q:
        body = (
            f"Delay-start the appliance so the main cycle hits "
            f"{_fmt_hours(solar_hours) or _fmt_hours(off)}, not {_fmt_hours(peak)}."
        )
    elif "pool" in q:
        body = f"Run the pool pump under the solar window {_fmt_hours(solar_hours)} this week."
    elif "battery" in q or "storage" in q:
        body = (
            f"Charge the battery from solar at {_fmt_hours(solar_hours)} and discharge "
            f"into on-peak {_fmt_hours(peak)}. Do not charge from on-peak grid."
        )
    else:
        body = (
            f"Move flexible loads off {_fmt_hours(peak)} onto solar "
            f"{_fmt_hours(solar_hours)} or off-peak {_fmt_hours(off)}."
        )
    if savings:
        body += (
            f" Estimated saving ${savings.get('savings_usd')} per event "
            f"(${savings.get('annual_savings_usd')}/year)."
        )
    if summary and not summary.get("error"):
        u = summary.get("usage") or {}
        g = summary.get("generation") or {}
        body += (
            f" Last 7 days: {u.get('total_consumption_kwh')} kWh "
            f"(${u.get('total_cost_usd')}), solar {g.get('total_generation_kwh')} kWh."
        )
    cites = []
    for tip in (tips.get("tips") or [])[:2]:
        cites.append(f"- ({tip.get('source')}) {str(tip.get('content', ''))[:220]}")
    if cites:
        body += "\n\nFrom the knowledge base:\n" + "\n".join(cites)
    body += "\n\nNext step: set a delay-start / wallbox schedule for the hours above."
    return body


class Agent:
    """Course contract: constructor(instructions, model) + invoke(question, context)."""

    def __init__(self, instructions: str = "", model: str = "gpt-4o-mini"):
        self.instructions = (instructions or DEFAULT_SYSTEM_INSTRUCTIONS).strip()
        self.model_name = model
        self.history: List[tuple] = []
        self.graph = build_energy_advisor_graph(self.instructions, model)

    def invoke(self, question: str, context: str = None, reset_history: bool = False) -> Dict[str, Any]:
        if reset_history:
            self.history = []
        messages: List[BaseMessage] = []
        if context:
            messages.append(SystemMessage(content=f"Additional user context: {context}"))
        for role, content in self.history[-6:]:
            messages.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
        messages.append(HumanMessage(content=question))
        result = self.graph.invoke(
            {
                "messages": messages,
                "question": question,
                "context": context or "",
                "used_tools": [],
            }
        )
        answer = _last_text(result)
        self.history.append(("user", question))
        self.history.append(("assistant", answer))
        result["answer"] = answer
        return result

    def get_agent_tools(self):
        return [t.name for t in TOOL_KIT]


def _last_text(result: Dict[str, Any]) -> str:
    msgs = result.get("messages") or []
    if not msgs:
        return ""
    last = msgs[-1]
    return getattr(last, "content", None) or str(last)
