# Agent-Based Contract Prediction - Design Document

## Overview

Replace the sklearn regression models with an LLM agent that predicts MLB player contracts across all three cost-control phases (pre-arb, arbitration, free agency). The agent's predictions must be grounded in evidence, cite where every figure came from, be reproducible via persisted traces, and fluctuate over time as inputs (eventually live stats) change.

Tools are introduced in phases. **Phase 0 (implemented)** is deliberately tool-less: the LLM predicts from its own knowledge. Accuracy is lower and knowledge-cutoff-bound — that is accepted; it establishes the architecture (schema, citations, traces, evaluation harness) that later phases build on.

## Status (July 2026)

Phase 0 is implemented and verified end-to-end (see Design Decisions below). The sklearn-era models and their design docs have been archived to `archive/v3/` — `agent/` no longer depends on them (`agent/service_time.py` and `agent/metrics.py` replace the two functions it used to import from `models/`).

The Roadmap below was revised this month: the original plan added tools first (live stats, then comps/heuristics). After using the CLI hands-on, grounding correctness and request ergonomics turned out to matter more than new tools — specifically, the agent would "predict" a number for a player who is already under a known signed contract, and every invocation required already knowing the internal `player_id`. Phase 1 addresses those two gaps before any new tool is added; the live-stats tool moves to Phase 2.

**Phase 1's natural-language front door is now implemented** (see Design Decision #7): a persistent orchestrator agent, talking to the user only in natural language, calls two sub-agents as tools — intake (resolves player/year/mode, asking clarifying questions through the orchestrator) and predict (the unchanged Phase 0 predictor). Intake can look up a player's full projected phase timeline (`agent/phase.py:project_phase_timeline`) rather than a single year, and a `hypothetical_free_agent` mode supports "what would they get in free agency right now" requests. The `known`/`projected` contract-status split described in Design Decision #4 is **still open** — this pass adds the timeline *lookup* tool, not new prediction-skipping behavior — and remains the next Phase 1 item.

## Design Decisions

### 1. Framework: Strands SDK + OpenAI ✅
- `strands-agents` with `OpenAIModel`, matching the pattern established by `review-queue-agent` (the repo's existing agent format)
- Default model `gpt-5-mini` (configurable via `--model` / `AGENT_MODEL_ID`); gpt-5* models reject sampling params, so `agent/config.py:model_params()` gates `temperature` by model family
- **Rationale**: one agent framework across the repo; structured output is first-class (`agent(prompt, structured_output_model=ContractPrediction)`)

### 2. Phase 0 is LLM-only ✅
- No tools attached; the agent reasons from training knowledge given player/phase context
- **Rationale**: cheapest possible baseline that exercises the full architecture end-to-end. Each later tool addition can be measured against this baseline.
- Known consequence (observed in verification): the model's knowledge cutoff makes it badly undervalue post-cutoff breakouts (e.g. predicted $10M for Tarik Skubal's 2026 arb year; actual $32M). This is precisely the gap the Phase 2 stats tool closes.

### 3. Citations from day one ✅
- Every material figure requires a `Citation` in the structured output
- Schema is forward-compatible: Phase 0 emits `source_type="model_knowledge"` (claim + basis, including how current the knowledge is); Phase 2+ adds `source_type="tool"` with `tool_name` and `tool_call_ref` pointing into the trace
- **Rationale**: grounding is a product requirement, and citation quality is measurable even when accuracy is contaminated (see Validation)

### 4. Deterministic phase resolution, harness-side ✅
- `agent/phase.py` resolves pre-arb/arb/free-agent from the Spotrac contract history — never asked of the LLM
- Observed years use Spotrac's own categorization; other years project the latest known service time forward (+1.0/season; thresholds 3.0/6.0 normalized years)
- Sentinels handled: `service_time = -1` (free agents), `age = -1`; mid-contract gap years resolve via the covering deal
- Known limitation: super-two players (top ~22% of 2+ service time) are arb-eligible a year early and will be classified pre-arb; flagged in the resolution notes
- **Rationale**: phase is a matter of record/rule, not judgment.
- **Update (Roadmap Phase 1)**: the resolver currently collapses two different situations into `method="projected"` — a genuine future forecast, and a year that falls inside an already-signed, currently-active multi-year deal (it notes "under contract through YYYY" but the agent still "predicts" a number that is, in fact, already known). Splitting this into a `known` outcome that returns the actual figure with no LLM call is **still open** — not yet built.
- **Added (July 2026)**: `project_phase_timeline()` answers "what phase applies in every future year" (e.g. `{"pre-arb": [2025, 2027], "arb": [2028, 2030], "free-agent": [2031, null]}`, `null` end = open-ended) rather than one year at a time, reusing the same +1.0/season projection. Exposed to the intake sub-agent as a tool (`agent/intake/tools.py:get_contract_phase_timeline`) — this is the one piece of phase logic that *is* now agent-facing, since "which years does this timeline cover" is closer to a lookup the agent reasons over than a routing decision the harness must make deterministically. `resolve_phase()` itself, which decides what to feed the predictor for one specific year, stays harness-side and untouched. Known caveats (not solved): super-two eligibility; assumes uninterrupted active-roster time (no adjustment for minor-league options or injury stints pausing service-time accrual).

### 5. Fresh agent per prediction ✅
- `agent/predictor.py` constructs a new Strands `Agent` per run (the review-queue agent reuses one across items)
- **Rationale**: no conversation-state bleed between players; each trace's messages describe exactly one run

### 6. Reproducibility via traces + history ✅
- Every run writes `predictions/traces/{run_id}.json`: model id, `PROMPT_VERSION`, phase resolution (with method + notes), system + user prompts, full messages, structured output, token usage, latency
- Every run appends to `predictions/history.csv` (predicted AAV/duration/total, range, actual AAV when known, model, prompt version, trace path) — repeated runs for the same player over time show projection fluctuation
- LLM outputs are not bit-identical across runs; reproducibility means every number is auditable back to its run context
- Traces are git-tracked (revisit if noisy)

### 7. Three-tier agent architecture for the NL front door ✅ (July 2026)
- A persistent **orchestrator** agent (`agent/orchestrator/`) is the only thing that talks to the user, every turn, in natural language (`OrchestratorTurn{message, done}` — never raw JSON shown to the user). It calls two sub-agents as tools:
  - **intake** (`agent/intake/`): resolves a request into `{player_id, target_year, mode}`. Stateless per call — the orchestrator always passes the full accumulated context (original request + every clarifying answer so far), so `resolve_intake()` builds a fresh Agent each call rather than holding its own conversation. Uses `find_player` (structured `first_name`/`last_name`/`position` filters, not a single free-text query — the agent reasons over *all* matches, never a silent first-match pick) and `get_contract_phase_timeline`.
  - **predict** (`agent/predict/`): the existing, unchanged Phase 0 predictor, wrapped as `predict_tool` and additionally exposed as `predict_for()` for direct reuse. `mode="hypothetical_free_agent"` constructs a synthetic `PhaseResolution(phase="free-agent", method="hypothetical", ...)` directly in `predict_for()`, without touching `resolve_phase()`.
- Each package owns its schema/prompts/logic and exports the tool interface a parent invokes (`agent/intake/resolver.py:intake_tool`, `agent/predict/tools.py:predict_tool`); shared logic (`config.py`, `phase.py`, `trace.py`, `service_time.py`, `metrics.py`) stays directly under `agent/`.
- **Rationale**: a Strands `structured_output_model` is just one more callable tool internally (confirmed by reading `strands/tools/structured_output/structured_output_tool.py`), so an agent with both `tools=[...]` and a `structured_output_model` can call its own tools across multiple turns before finalizing structured output — this is what makes "sub-agent as a tool" work with plain `@tool`-decorated functions that invoke another `Agent` internally.
- **Deliberate exception to Decision #5**: the orchestrator is the one persistent (non-fresh) agent in the system, scoped to its own back-and-forth with the user. The underlying prediction still gets its own fresh agent via the untouched `agent/predict/predictor.py`, so prediction reproducibility (Decision #6) is unaffected — the orchestrator's own conversation is traced separately (`predictions/conversations/{run_id}.json`, linking to the prediction's trace path) precisely because it's variable-length in a way the prediction trace deliberately isn't.
- **Open design question from the original Phase 1 proposal resolved in practice**: rather than a strict one-shot parse or strict incremental slot-filling, intake extracts everything it can from the given context in one pass and the orchestrator iterates only on what's actually missing or ambiguous — closer to slot-filling in effect (only asks about gaps) without being mechanically one-field-at-a-time.
- **Fixed (July 2026): follow-up questions.** `run_conversation()` used to `return` the instant a turn had `done=True`, so the first delivered prediction ended the whole session — there was no way to ask "why so low?" or "what about 2027 instead?" afterward. `done` now means "this turn's answer is complete," not "stop talking to me": the loop asks for a follow-up (`agent/orchestrator/agent.py:EXIT_INPUTS` — blank or an exit-like reply ends it) and, if the user keeps going, feeds it back into the *same* persistent orchestrator agent, so prior turns (including the prediction's reasoning/citations) stay in context and the system-prompt-plus-history prefix is reused turn to turn (OpenAI's automatic prompt caching applies for free — no separate caching work needed). The system prompt tells the orchestrator to answer follow-ups from context when possible and only re-invoke `predict_tool` (a fresh predict-agent call per Decision #5, so the number can genuinely change) when the follow-up alters the actual request. A separate `MAX_CONVERSATION_TURNS` safety cap bounds the whole session; `MAX_TURNS` still caps *consecutive* clarification rounds specifically.

## Dataset

**Sources**: `dataset/contracts_spotrac.csv` (phase resolution, backtest sampling, actuals), `dataset/players.csv` (name/position lookup)

Notes: rows deduped by `contract_id`; `service_time`/`age` use `-1` sentinels; multi-year deals have no rows for covered years. Service time normalization (`years + days/172`) lives in `agent/service_time.py:normalize_service_time`.

## Citation Schema

```python
class Citation(BaseModel):
    source_type: Literal["model_knowledge", "tool"]
    claim: str                      # the specific figure/fact supported
    basis: str                      # what it's known from, and how current
    tool_name: Optional[str]        # populated once tool-sourced citations arrive (Roadmap Phase 2+)
    tool_call_ref: Optional[str]    # ref into trace messages

class ContractPrediction(BaseModel):
    aav_millions: float
    duration_years: int             # 1 for pre-arb/arb
    total_value_millions: float
    aav_low_millions: float
    aav_high_millions: float
    reasoning: str
    citations: list[Citation]       # min 1; one per material figure
    confidence: Literal["low", "medium", "high"]
```

Arithmetic consistency (`total ≈ aav × duration`) is checked but recorded in the trace rather than rejected — LLMs slip on arithmetic and the rest of the output is still valuable.

## File Structure

```
agent/
├── config.py          # dotenv, DEFAULT_MODEL_ID, model-gated params, paths
├── phase.py            # deterministic phase resolver + project_phase_timeline()
├── trace.py             # trace JSON + history.csv + conversation-trace persistence
├── service_time.py       # normalize_service_time
├── metrics.py             # backtest scoring metrics
├── backtest.py             # CLI: python -m agent.backtest / make backtest-agent
├── ask.py                   # CLI: python -m agent.ask "..." / make ask REQUEST="..."
│
├── predict/                  # Phase 0 predictor, packaged as a sub-agent
│   ├── __init__.py             # re-exports run_prediction, lookup_player, actual_aav_for, predict_for
│   ├── __main__.py              # python -m agent.predict --player-id ... --year ...
│   ├── predict.py                 # lookup_player, run_prediction, predict_for(player_id, year, mode)
│   ├── predictor.py                 # fresh-agent-per-run prediction (structured output)
│   ├── prompts.py                    # PROMPT_VERSION, SYSTEM_PROMPT, build_prediction_prompt
│   ├── schema.py                       # Citation, ContractPrediction
│   ├── tools.py                         # predict_tool — exported for the orchestrator
│   └── tests/
│
├── intake/                    # NL request -> {player_id, target_year, mode}
│   ├── schema.py                # IntakeResult
│   ├── prompts.py                 # INTAKE_SYSTEM_PROMPT
│   ├── tools.py                     # find_player, get_contract_phase_timeline
│   ├── resolver.py                    # resolve_intake() + intake_tool (exported for the orchestrator)
│   └── tests/
│
└── orchestrator/               # the one agent that talks to the user
    ├── schema.py                 # OrchestratorTurn{message, done}
    ├── prompts.py                  # ORCHESTRATOR_SYSTEM_PROMPT
    ├── agent.py                      # create_orchestrator_agent(), run_conversation()
    └── tests/

predictions/
├── traces/            # one JSON per prediction run
├── conversations/       # one JSON per orchestrator conversation, links to a trace path
├── backtests/             # backtest summary JSONs
└── history.csv               # append-only prediction log
```

## Validation Strategy

- **Offline unit tests** (`make test-agent`): phase resolver against Max Scherzer's real career (pre-arb 2011 → arb 2012-14 → FA 2015+, sentinels, mid-contract years) plus synthetic crossover cases; schema round-trip and citation constraints; trace/history writing
- **Live checks**: one prediction per phase; verify citations per material figure, trace + history artifacts
- **Backtest harness** (`make backtest-agent`): seeded N-per-phase sample of historical contracts, scored on AAV with `agent/metrics.py` metrics at phase-scaled tolerances (±$0.25M pre-arb / ±$1M arb / ±$5M FA)

**Training-data contamination caveat**: in Phase 0 the LLM has likely memorized many historical contract outcomes, so backtests on past contracts measure recall as much as prediction. They are harness scaffolding and a citation-quality check. The honest test is predictions for upcoming contracts accumulating in `predictions/history.csv` and scored once actuals land.

## Roadmap

*Revised July 2026. Original plan led with tools (stats, then comps/heuristics); reordered because grounding known contracts correctly and accepting a natural-language request matter more right now than adding new tools, and neither depends on a new data source.*

### Phase 1 — natural-language front door + contract-status routing
- **Natural-language request parsing — done.** `agent/ask.py` / `make ask REQUEST="..."` replaces the `--player-id`/`--name` + `--year` CLI contract with a single free-text request, handled by the orchestrator/intake/predict architecture in Design Decision #7. Intake resolves player-name ambiguity against `players.csv` via structured `find_player(first_name, last_name, position)` filters and reports back plainly (through the orchestrator) when it can't disambiguate, rather than guessing. `hypothetical_free_agent` mode is supported for "what would this player get in free agency right now" requests.
- **Contract-status routing — still open.** The resolver's `projected` outcome still collapses "genuine forecast" and "year covered by an already-signed multi-year deal" together; splitting out a `known` outcome that returns the actual figure with **no LLM call** has not been built. `project_phase_timeline()` (Design Decision #4) gives intake visibility into a player's known/projected years, but `predict_tool` still always calls the LLM today, even for years that are actually already known.
- Ships against the current dataset; no new data source required.

### Phase 2 — live data sourcing
- **Stats**: live MLB Stats API tool (`statsapi.mlb.com`) — the agent fetches a player's recent seasons on demand instead of us maintaining stats CSVs. Covers every standard batting/pitching stat the old pipeline consumed (G, PA, HR, RBI, AVG/OBP/SLG/OPS, SO, BB, ERA, WHIP, IP, K/BB, saves, holds...). Does **not** provide FanGraphs-proprietary metrics: WAR, wRC+, FIP/xFIP/SIERA, or Statcast-derived rates (HardHit%, Barrel%, GB%/FB%). Expected effect: estimates improve sharply for post-cutoff performance (the Skubal case), and citations become `source_type="tool"` with specific stat lines as evidence.
- **Contracts**: no public API for actual signed dollar figures is known to exist — Spotrac (what we already scrape) remains close to the only free source. "Don't rely on our dataset" most likely means moving from a batch-scraped CSV to on-demand/cached live fetches of the same source, not switching to a different one. Scope as its own spike, separate from the stats swap.
- **Player identity**: same caveat as contracts — replacing `players.csv` lookups with a live source needs its own investigation before Phase 1's NL front door can drop the dataset dependency entirely.
- System prompt grows: tool-usage guidance, instruction to prefer tool evidence over memory.

### Phase 3 — multi-year forecasting
- Extend the schema/prompt to accept a target-year range and optional user-supplied assumptions (stat trajectories, injury status, etc.) instead of a single target year.
- Wants Phase 2's live stats in place first — forecasting several years out purely from training memory isn't much better than Phase 0.

### Phase 4 — team conditioning + guardrails
- Comparable-contracts tool over the historical contracts dataset; league-minimum/CBA schedule and arb-raise-pattern heuristic tools; richer system prompt carrying business rules (counting stats drive arb salaries; aging curves for FA duration).
- Scenario support: team-specific predictions and user-defined constraints/guardrails on the prediction, as first-class request inputs.
- Deferred last — benefits most from a stable multi-year forecasting core (Phase 3) and comps/team-context data (this phase) to condition on; building it earlier would mean guessing at an interface that's still moving.

### Appendix A — FanGraphs dataset-refresh fallback (not implemented)

pybaseball's FanGraphs fetch (`leaders-legacy.aspx`) returns HTTP 403 behind a captcha, with no upstream fix. If the historical stats CSVs ever need refreshing (e.g. to feed a comps tool with WAR), the verified fallback is a small `data_generation/fangraphs_api.py` exposing pybaseball-compatible `batting_stats`/`pitching_stats` backed by FanGraphs' current JSON API:

```
https://www.fangraphs.com/api/leaders/major-league/data
  ?pos=all&stats=bat|pit&lg=all&qual=0&season=YYYY&season1=YYYY
  &ind=0|1&month=0&team=0&pageitems=2000000000&pagenum=1&type=8
```

Verified working unauthenticated (2026-07) with field names matching pybaseball's exactly, except `playerid` → rename to `IDfg`. Supports the season-range + `ind=0` aggregation the rolling-window collection in `data_generation/stats.py` relies on. Unofficial API — pair with browser-like headers, politeness delays, and in-module memoization.

## Files to Reference (Existing Patterns)

- `git show review-queue-agent:data_generation/review_queue_agent.py` — Strands agent conventions this package mirrors
- `agent/service_time.py` — service time normalization
- `agent/metrics.py` — shared metrics (used by backtest)
- `agent/predict/predictor.py` — fresh-agent-per-call pattern (Decision #5)
- `agent/orchestrator/agent.py` — the one persistent-agent exception (Decision #7), and the injectable-agent pattern `agent/orchestrator/tests/test_agent.py` uses to test the conversation loop without a live LLM call
- `archive/v3/docs_pre_arb/` — the sklearn-era design docs this system supersedes (moved from `docs/pre_arb/`; sklearn models archived to `archive/v3/models/`)
