# EcoHome Energy Advisor

LangGraph agent that schedules EV charging, HVAC, appliances, pool pumps and home batteries against rooftop solar, weather and time-of-use prices.


## Changes in this revision

Reviewer items that were previously marked fail, plus follow-up quality fixes:

1. **`get_electricity_prices` is date-dynamic** (`tools.py`). Peak *hours* move with season and weekday (summer 15–21, winter dual peak 7–10 + 17–20, spring weekend has no on-peak). Weekday weights and `random.Random(f"tou-{date}")` jitter change the dollars. This is not a fixed 16:00–21:00 table at $0.22 × 0.9.
2. **`evaluate_response()` is LLM-as-a-judge** (notebook 03 + `llm_judge.py`). Accuracy, relevance, completeness and usefulness are scored from meaning, with 2–4 sentence critiques. No token overlap.
3. **`evaluate_tool_usage()` scores appropriateness and completeness independently.** Extra tools do not force completeness to 0 if every expected tool ran.
4. **`generate_evaluation_report()` recommendations are LLM-authored** (`evaluation_report.py`). No `if avg_tool < 0.8` strings. Dict-shaped recs are flattened to `"case_id: fix"`. Display is a separate `display_evaluation_report()`.
5. **Test case `usage_query_1`** with `"expected_tools": ["query_energy_usage"]`. Suite is 11 cases. All seven tools appear as expected at least once.
6. **Meter query no longer returns 0 kWh** when the model passes `"EV and HVAC"`. The tool splits aliases, always returns `by_device`, and falls back to all devices if the filter matches nothing.
7. **Solar seed includes today** (`range(31)` in notebook 01) so a last-24h generation query is not empty after sunset-only windows.
8. **Dishwasher savings use ~1.2–1.5 kWh per cycle**, not the 24-hour appliance total.

Latest executed 03 run (LLM judge): 11 tests, composite **0.88**, mean response **0.90**, mean tools **0.86**, appropriateness **1.00**, completeness **0.72**.

## Environment

| Item | Value |
|------|--------|
| Local Python | 3.13 / 3.12 / 3.11 |
| Course kernel | 3.11.x |
| Packages | `requirements.txt` |
| LLM (agent + judges) | Vocareum `gpt-4o-mini` |
| Embeddings | Vocareum / OpenAI embeddings into Chroma; lexical fallback if embeddings fail |

```bash
cd ecohome_solution
pip install -r requirements.txt
```

### API key

`.env` in this folder (gitignored):

```
OPENAI_API_KEY=your_vocareum_key
OPENAI_BASE_URL=https://openai.vocareum.com/v1
VOCAREUM_API_KEY=your_vocareum_key
```

Vocareum does not implement `GET /v1/models`. Test with chat completions:

```bash
set -a && source .env && set +a
curl -sS https://openai.vocareum.com/v1/chat/completions \
  -H "Authorization: Bearer $VOCAREUM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"ping"}],"max_tokens":16}'
```

Start Jupyter from this directory after sourcing `.env`:

```bash
cd ecohome_solution
set -a && source .env && set +a
jupyter notebook
```

Anaconda Navigator does not inherit terminal `export`s. Call `load_dotenv(".env")` in the first notebook cell.

## Run order

Working directory must be `ecohome_solution/`.

1. `01_db_setup.ipynb` — SQLite `EnergyUsage` / `SolarGeneration`, sample data, tool smoke tests.
2. `02_rag_setup.ipynb` — load `data/documents/`, split, persist Chroma under `data/vectorstore/`, test `search_energy_tips`.
3. `03_run_and_evaluate.ipynb` — `ECOHOME_SYSTEM_PROMPT`, 11 test cases (`reset_history=True`), LLM judges, structured report.
4. `04_agent_run.ipynb` — optional pass over the brief example questions.

Restart the kernel on 03 after 01/02. The report cell calls the LLM about twice per case plus once for recommendations.

```python
from agent import Agent
agent = Agent(instructions=ECOHOME_SYSTEM_PROMPT, model="gpt-4o-mini")
result = agent.invoke(
    question="When should I charge my electric car tomorrow to minimize cost and maximize solar power?",
    context="Location: San Francisco, CA",
    reset_history=True,
)
print(result["messages"][-1].content)
```

## Layout

```
ecohome_solution/
  agent.py                 # Agent + StateGraph; forces planned tools if the LLM skips them
  tools.py                 # @tool functions + dynamic TOU + robust meter query
  llm_judge.py             # LLM-as-judge helpers
  evaluation_report.py     # generate_evaluation_report + display_evaluation_report
  models/energy.py         # SQLAlchemy EnergyUsage, SolarGeneration, DatabaseManager
  01_db_setup.ipynb
  02_rag_setup.ipynb
  03_run_and_evaluate.ipynb
  04_agent_run.ipynb
  requirements.txt
  data/documents/          # knowledge base (13 txt files)
  data/energy_data.db      # created by notebook 01
  data/vectorstore/        # created by notebook 02
```

## Agent

`agent.py` defines an explicit LangGraph:

- State: `messages`, `question`, `context`, `used_tools`
- Nodes: `agent` (LLM with tools, or offline planner), `tools`
- Edges: `START → agent → (tools | END)`, `tools → agent`

If the model returns a final answer without the tools the question needs, the agent node injects those tool calls before answering.

Contract:

- `Agent(instructions=..., model=...)`
- `invoke(question, context=None, reset_history=False)` → graph result with `messages` and `answer`

## Tools

| Tool | Role |
|------|------|
| `get_weather_forecast(location, days)` | Hourly weather + solar-friendly hours (Open-Meteo, synthetic fallback) |
| `get_electricity_prices(date)` | Dynamic TOU: season + weekday peak windows + date-seeded jitter |
| `query_energy_usage(start_date, end_date, device_type)` | Historical consumption; `by_device` totals; multi-device filters |
| `query_solar_generation(start_date, end_date)` | Historical PV |
| `get_recent_energy_summary(hours)` | Recent usage + generation rollup |
| `search_energy_tips(query, max_results)` | Chroma RAG, lexical fallback |
| `calculate_energy_savings(...)` | Per-event kWh / USD / annual CO2 (one dishwasher cycle ≠ 24h total) |

## Knowledge base

Starter files plus the five required topics and extras (13 documents under `data/documents/`).

## Evaluation (notebook 03)

Eleven cases: EV ×2, thermostat ×2, appliances ×2, solar, usage history, pool pump, **meter history via `query_energy_usage`**, battery dispatch.

| Function | What it does |
|----------|----------------|
| `evaluate_response` | LLM judge: accuracy, relevance, completeness, usefulness + feedback |
| `evaluate_tool_usage` | LLM judge: appropriateness and completeness scored separately |
| `generate_evaluation_report` | Aggregates scores; recommendations from an LLM reviewer |
| `display_evaluation_report` | Prints sections 1–5 (scores, strengths, weaknesses, recs, appendix) |

## Example questions

- When should I charge my electric car tomorrow to minimize cost and maximize solar power?
- What temperature should I set my thermostat on Wednesday afternoon if electricity prices spike?
- Suggest three ways I can reduce energy use based on my usage history.
- How much can I save by running my dishwasher during off-peak hours?
- What is the best time to run my pool pump this week based on the weather forecast?
- How many kWh did my EV and HVAC use over the last 7 days according to the meter history?

Typical policy: solar window ~10:00–15:00 first, finish flexible load overnight off-peak, avoid on-peak late afternoon/evening.
