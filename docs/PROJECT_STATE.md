# MLBContracts — Project State (July 2026)

## 1. High-Level Goal

A web application that projects MLB player contract value and updates those projections daily based on in-season performance. The system covers all three phases of player cost control: pre-arbitration, arbitration, and free agency.

---

## 2. What's Been Implemented

### Data Pipeline

Three-stage pipeline that produces the training/prediction dataset:

| Stage | Module | Output |
|-------|--------|--------|
| Contract scraper | `data_generation/spotrac.py` | `dataset/contracts_spotrac.csv`, `dataset/players.csv` |
| Stats aggregator | `data_generation/stats.py` | `dataset/batter_stats.csv`, `dataset/pitcher_stats.csv` |
| Join | `data_generation/join.py` | `dataset/contracts_with_stats.csv` (186 columns, all window sizes) |

Stats are collected via **pybaseball** for 1-, 3-, 5-, and 10-year rolling windows. Contracts are scraped from Spotrac and mapped to FanGraphs player IDs.

### Pre-Arbitration Model (`archive/v3/models/pre_arb/`) — **archived, superseded by the agent**

Predicts single-year salaries for players with < 3 years service time. These salaries cluster tightly around the CBA-mandated league minimum.

- **Algorithm:** Ridge Regression (Random Forest achieved lower training error but cannot extrapolate to future CBA minimum increases; Ridge correctly projects ~$30K/year growth)
- **Features:** `contract_year`, `age`, `service_time`, `position` — performance stats add essentially no predictive value for pre-arb salaries
- **Performance:**

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| MAE | $39K | ≤ $150K | PASS |
| Within ±$250K | 99.3% | ≥ 95% | PASS |
| CV MAE | $36K ± $0.6K | — | — |

The $39K MAE is effectively the practical floor — remaining errors are unpredictable outliers (goodwill raises, mislabeled extensions, prorated call-up salaries). Archived to `archive/v3/` (July 2026) alongside `models/preprocessing.py` and `models/evaluation.py`, whose functions now live in `agent/service_time.py` and `agent/metrics.py` so the agent has no dependency on archived code. `docs/pre_arb/` moved to `archive/v3/docs_pre_arb/`.

### Arbitration Model (`models/arb/`) — **on `arb-model` branch, not yet merged**

Predicts single-year salaries for players in their arbitration years (service time 2–6 years). Separate models for pitchers and batters because their stat profiles are completely different.

- **Algorithm:** Random Forest with 200 estimators
- **Features:** WAR (1y/3y/5y), strikeouts (pitchers), home runs + RBIs (batters), plus age, service time, contract year, position
- **Key finding:** Counting stats (WAR, HR, RBI, K) correlate far better with salary (+0.73–0.87) than rate stats (AVG, OPS, wRC+, +0.31–0.58) because counting stats capture both performance and playing time durability
- **Performance:**

| Model | MAE | RMSE | R² | CV MAE |
|-------|-----|------|----|--------|
| Pitcher | $893K | $1.44M | 0.68 | $925K ± $90K |
| Batter | $978K | $1.73M | 0.84 | $858K ± $47K |

Per-tier performance vs. ±10% tolerance target (95% threshold):

| Tier | Pitcher | Batter | Target |
|------|---------|--------|--------|
| Arb Year 1 (avg $2.25M, tol ±$230K) | 39.9% | 39.8% | 95% — FAIL |
| Arb Year 2 (avg $4.00M, tol ±$400K) | 28.4% | 35.8% | 95% — FAIL |
| Arb Year 3 (avg $6.38M, tol ±$640K) | 39.7% | 33.3% | 95% — FAIL |

The overall MAE (~$900K) is solid, but the ±10% per-tier targets are not met. The large errors come from predictable-but-unmodeled categories: elite closer premiums, injury discounts, and bounce-back contracts.

### Agent-Based Prediction, Phase 0 (`agent/`) — **merged to main, July 2026**

The project direction shifted from sklearn models to an LLM agent that predicts contracts across all three phases (see `docs/agent/DESIGN.md`). Phase 0 is deliberately tool-less — the LLM (Strands SDK + OpenAI `gpt-5-mini`) predicts from its own knowledge — and establishes the architecture later phases build on:

- **Citations from day one**: structured output requires a citation (claim + basis) per material figure; schema is forward-compatible with tool-sourced citations
- **Deterministic phase resolution** (`agent/phase.py`): pre-arb/arb/FA resolved from Spotrac contract history, never by the LLM (super-two is a known caveat)
- **Reproducibility**: every run persists a full trace JSON (`predictions/traces/`) and an append-only `predictions/history.csv` row, so projections fluctuate visibly across runs
- **Harness**: `make predict PLAYER=... YEAR=...`, `make backtest-agent`, `make test-agent` (18 offline tests)

Preliminary Phase 0 baseline (13 scored predictions, partial seeded backtest):

| Phase | n | MAE | Notes |
|-------|---|-----|-------|
| pre-arb | 6 | $0.052M | Near the archived Ridge model's $0.039M — minimums are memorizable |
| arb | 6 | $4.95M | Dominated by post-knowledge-cutoff breakouts (predicted $10M for Skubal 2026; actual $32M) |
| free-agent | 1 | $0.00M | Single run (Scherzer 2026) — no meaningful sample yet |

**Caveat**: Phase 0 backtests are training-data-contaminated (the LLM memorized many historical outcomes), so these numbers measure recall as much as prediction. The Skubal-style misses are the gap the Phase 1 stats tool targets.

### GitHub Actions Workflow

A daily workflow (`daily-dataset.yml`) was set up to run the full data pipeline and commit updated CSVs automatically. **Currently disabled** (schedule commented out) due to the stats collection failure described below.

---

## 3. Open Issues

### Critical: Stats Data Collection is Broken

**pybaseball** fetches stats from `https://www.fangraphs.com/leaders-legacy.aspx`, which now returns HTTP 403 and presents a human verification captcha. Every year (2016–2026) fails with this error. This blocks:

- Updating the stats dataset with current-season performance
- The daily update workflow (core requirement for in-season projections)
- Retraining models on fresh data

**Decision (July 2026)**: with the shift to the agent architecture, current-season stats will be consumed as a **live MLB Stats API tool** in agent Phase 1 rather than by maintaining our own stats CSVs. That covers every standard stat the pipeline consumed but not the FanGraphs-proprietary metrics (WAR, wRC+, FIP/xFIP/SIERA, Statcast rates). If the historical stats CSVs ever need refreshing (e.g. for a comps tool with WAR), a verified fallback exists: FanGraphs' current JSON API with pybaseball-compatible fields — see `docs/agent/DESIGN.md` Appendix A.

### Arb Model Tier Accuracy Targets Not Met

The ±10% per-tier tolerance at 95% is very aggressive. The best per-tier result is 39.9% vs. the 95% target. Root causes identified in the model docs:

- High inherent salary variance within tiers (Year 1 ranges $600K–$7.9M despite similar service time)
- No features for team context (payroll, market size, agent leverage)
- No arbitration hearing history or comparable contract data
- Specific outlier archetypes (elite closers, injury bounce-backs) that are structurally mispredicted

Potential improvements: interaction terms (service_time × WAR shows +0.60–0.65 correlation), wider tolerance targets (±15–20%), or adding more features if a better data source resolves the collection issue.

### arb-model Branch Not Merged

The `arb-model` branch contains the completed arbitration model implementation but has not been merged to main. This should be merged once the team decides whether to address the tier accuracy issues first or merge and iterate.

### No Free Agency Model

Per the pre-arb README, the planned three-model system is:
1. Pre-arb model — **done**
2. Arb model — **implemented, not merged**
3. FA model — **not started**

Free agency contracts are structurally different: multi-year deals, open market bidding, no CBA floor/ceiling. This is likely the hardest prediction problem of the three.

### No Web Application

The entire front-end and API layer doesn't exist yet. The archived sklearn pre-arb model is a pipeline serialized to a `.pkl` file with a programmatic Python interface (`archive/v3/models/pre_arb/inspect.py`); the arb model has the same shape on the unmerged `arb-model` branch. Building the web app requires:

- A serving layer (REST API wrapping the model `.pkl` files)
- A front-end for browsing player projections
- A database or storage layer for current-season rolling stats
- A trigger mechanism to refresh projections as stats update (replacement for the now-broken daily workflow)

---

## 4. What's Next

In rough priority order (see the roadmap in `docs/agent/DESIGN.md`):

1. **Agent Phase 1** — Add the live MLB Stats API tool + expose the phase resolver as an agent tool. Expected to close the post-knowledge-cutoff misses (the $10M-vs-$32M Skubal case) and upgrade citations from model memory to tool evidence.

2. **Finish the Phase 0 baseline** — Complete a full seeded backtest (the first run was cut short at 13 predictions) and record the baseline to compare Phase 1 against.

3. **Agent Phase 2+** — Comparable-contracts tool, league-minimum/CBA and arb-raise heuristic tools, business logic in the system prompt, scenario support (alternate lengths, options, teams).

4. **Web application** — Serving layer + front end over the agent's predictions and history (unchanged, still furthest out).

5. **Daily update workflow** — Revisit once Phase 1 makes projections respond to live stats; the workflow's stats-collection step is superseded by the live-tool decision above.

Done: sklearn models (`models/`) and their design docs (`docs/pre_arb/`) archived to `archive/v3/` (July 2026); the unmerged `arb-model` branch stays a branch.
