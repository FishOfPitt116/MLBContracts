# Agent-Based Contract Prediction - Design Document

## Overview

Replace the sklearn regression models with an LLM agent that predicts MLB player contracts across all three cost-control phases (pre-arb, arbitration, free agency). The agent's predictions must be grounded in evidence, cite where every figure came from, be reproducible via persisted traces, and fluctuate over time as inputs (eventually live stats) change.

Tools are introduced in phases. **Phase 0 (implemented)** is deliberately tool-less: the LLM predicts from its own knowledge. Accuracy is lower and knowledge-cutoff-bound — that is accepted; it establishes the architecture (schema, citations, traces, evaluation harness) that later phases build on.

## Design Decisions

### 1. Framework: Strands SDK + OpenAI ✅
- `strands-agents` with `OpenAIModel`, matching the pattern established by `review-queue-agent` (the repo's existing agent format)
- Default model `gpt-5-mini` (configurable via `--model` / `AGENT_MODEL_ID`); gpt-5* models reject sampling params, so `agent/config.py:model_params()` gates `temperature` by model family
- **Rationale**: one agent framework across the repo; structured output is first-class (`agent(prompt, structured_output_model=ContractPrediction)`)

### 2. Phase 0 is LLM-only ✅
- No tools attached; the agent reasons from training knowledge given player/phase context
- **Rationale**: cheapest possible baseline that exercises the full architecture end-to-end. Each later tool addition can be measured against this baseline.
- Known consequence (observed in verification): the model's knowledge cutoff makes it badly undervalue post-cutoff breakouts (e.g. predicted $10M for Tarik Skubal's 2026 arb year; actual $32M). This is precisely the gap the Phase 1 stats tool closes.

### 3. Citations from day one ✅
- Every material figure requires a `Citation` in the structured output
- Schema is forward-compatible: Phase 0 emits `source_type="model_knowledge"` (claim + basis, including how current the knowledge is); Phase 1+ adds `source_type="tool"` with `tool_name` and `tool_call_ref` pointing into the trace
- **Rationale**: grounding is a product requirement, and citation quality is measurable even when accuracy is contaminated (see Validation)

### 4. Deterministic phase resolution, harness-side ✅
- `agent/phase.py` resolves pre-arb/arb/free-agent from the Spotrac contract history — never asked of the LLM
- Observed years use Spotrac's own categorization; other years project the latest known service time forward (+1.0/season; thresholds 3.0/6.0 normalized years)
- Sentinels handled: `service_time = -1` (free agents), `age = -1`; mid-contract gap years resolve via the covering deal
- Known limitation: super-two players (top ~22% of 2+ service time) are arb-eligible a year early and will be classified pre-arb; flagged in the resolution notes
- **Rationale**: phase is a matter of record/rule, not judgment. In Phase 1 this becomes an agent-facing tool, which also enables scenario-style requests ("what would this contract look like at N years / with options / on team X") to flow through the agent while phase stays deterministic.

### 5. Fresh agent per prediction ✅
- `agent/predictor.py` constructs a new Strands `Agent` per run (the review-queue agent reuses one across items)
- **Rationale**: no conversation-state bleed between players; each trace's messages describe exactly one run

### 6. Reproducibility via traces + history ✅
- Every run writes `predictions/traces/{run_id}.json`: model id, `PROMPT_VERSION`, phase resolution (with method + notes), system + user prompts, full messages, structured output, token usage, latency
- Every run appends to `predictions/history.csv` (predicted AAV/duration/total, range, actual AAV when known, model, prompt version, trace path) — repeated runs for the same player over time show projection fluctuation
- LLM outputs are not bit-identical across runs; reproducibility means every number is auditable back to its run context
- Traces are git-tracked (revisit if noisy)

## Dataset

**Sources**: `dataset/contracts_spotrac.csv` (phase resolution, backtest sampling, actuals), `dataset/players.csv` (name/position lookup)

Notes: rows deduped by `contract_id`; `service_time`/`age` use `-1` sentinels; multi-year deals have no rows for covered years. Service time normalization (`years + days/172`) reuses `models/preprocessing.py:normalize_service_time`.

## Citation Schema

```python
class Citation(BaseModel):
    source_type: Literal["model_knowledge", "tool"]
    claim: str                      # the specific figure/fact supported
    basis: str                      # what it's known from, and how current
    tool_name: Optional[str]        # Phase 1+
    tool_call_ref: Optional[str]    # Phase 1+: ref into trace messages

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
├── config.py        # dotenv, DEFAULT_MODEL_ID, model-gated params, paths
├── schema.py        # Citation, ContractPrediction
├── phase.py         # deterministic phase resolver
├── prompts.py       # PROMPT_VERSION, SYSTEM_PROMPT, build_prediction_prompt
├── predictor.py     # fresh-agent-per-run prediction (structured output)
├── trace.py         # trace JSON + history.csv persistence
├── predict.py       # CLI: python -m agent.predict / make predict
├── backtest.py      # CLI: python -m agent.backtest / make backtest-agent
└── tests/           # offline tests (no API key needed): make test-agent
predictions/
├── traces/          # one JSON per run
├── backtests/       # backtest summary JSONs
└── history.csv      # append-only prediction log
```

## Validation Strategy

- **Offline unit tests** (`make test-agent`): phase resolver against Max Scherzer's real career (pre-arb 2011 → arb 2012-14 → FA 2015+, sentinels, mid-contract years) plus synthetic crossover cases; schema round-trip and citation constraints; trace/history writing
- **Live checks**: one prediction per phase; verify citations per material figure, trace + history artifacts
- **Backtest harness** (`make backtest-agent`): seeded N-per-phase sample of historical contracts, scored on AAV with `models/evaluation.py` metrics at phase-scaled tolerances (±$0.25M pre-arb / ±$1M arb / ±$5M FA)

**Training-data contamination caveat**: in Phase 0 the LLM has likely memorized many historical contract outcomes, so backtests on past contracts measure recall as much as prediction. They are harness scaffolding and a citation-quality check. The honest test is predictions for upcoming contracts accumulating in `predictions/history.csv` and scored once actuals land.

## Roadmap

### Phase 1 — stats tool + phase tool
- **Live MLB Stats API tool** (`statsapi.mlb.com`): the agent fetches a player's recent seasons on demand instead of us maintaining stats CSVs. Covers every standard batting/pitching stat the old pipeline consumed (G, PA, HR, RBI, AVG/OBP/SLG/OPS, SO, BB, ERA, WHIP, IP, K/BB, saves, holds...). Does **not** provide FanGraphs-proprietary metrics: WAR, wRC+, FIP/xFIP/SIERA, or Statcast-derived rates (HardHit%, Barrel%, GB%/FB%). Expected effect: estimates improve sharply for post-cutoff performance (the Skubal case), and citations become `source_type="tool"` with specific stat lines as evidence.
- **Phase resolver exposed as a tool**, so the agent can reason about eligibility timelines (e.g. "arb-2 in 2027, FA after 2029") and support scenario requests.
- System prompt grows: tool-usage guidance, instruction to prefer tool evidence over memory.

### Phase 2+ — research tools and business logic
- Comparable-contracts tool over the historical contracts dataset
- League-minimum/CBA schedule and arb-raise-pattern heuristic tools (encoding decision logic that today lives in analysis docs)
- Richer system prompt carrying business rules (counting stats drive arb salaries; aging curves for FA duration)
- Scenario support: user-directed variations (contract length, options, team context) as first-class prompt inputs
- Revisit archiving `models/` to `archive/v3` once the agent baseline is established

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
- `models/preprocessing.py` — service time normalization (reused)
- `models/evaluation.py` — shared metrics (reused by backtest)
- `docs/pre_arb/` — the sklearn-era design docs this system supersedes
