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

### Agent-Based Prediction, Phase 0 + Phase 2 (`agent/`) — **merged to main, July 2026**

The project direction shifted from sklearn models to an LLM agent that predicts contracts across all three phases (see `docs/agent/DESIGN.md`). Phase 0 was deliberately tool-less — the LLM (Strands SDK + OpenAI `gpt-5-mini`) predicted from its own knowledge — and established the architecture later phases build on. Phase 2 has since added the agent's first tools, comparable contracts and live stats:

- **Citations from day one**: structured output requires a citation (claim + basis) per material figure; schema is forward-compatible with tool-sourced citations
- **Deterministic phase resolution** (`agent/phase.py`): pre-arb/arb/FA resolved from Spotrac contract history, never by the LLM (super-two is a known caveat)
- **Reproducibility**: every run persists a full trace JSON (`predictions/traces/`) and an append-only `predictions/history.csv` row, so projections fluctuate visibly across runs
- **Harness**: `make predict PLAYER=... YEAR=...` (direct flags), `make ask REQUEST="..."` (natural-language front door, Phase 1), `make backtest-agent`, `make test-agent` (120 offline tests)
- **`query_comparable_contracts`** (`agent/predict/comparables.py`, Phase 2): grounds comparable-contract facts and a player's own contract history in real records instead of memory — see `docs/agent/DESIGN.md`'s Phase 2 entry for the full design, including a real backtest-discovered no-lookahead leak (the tool could return the target player's own real contract for the very year being predicted, since backtest targets are historical rows in the same CSV the tool reads) and its fix (a hard `year < target_year` cutoff baked into the tool per-prediction, invisible to and un-overridable by the model).
- **`query_batting_stats`/`query_pitching_stats`** (`agent/predict/mlb_stats_api.py`, Phase 2): live `statsapi.mlb.com` calls (not the stalled local stats CSVs) for both the target player and any comparable players surfaced by `query_comparable_contracts` — see `docs/agent/DESIGN.md`'s Phase 2 entry for the endpoint details, the player-identity crosswalk that made this possible, and the retry/no-lookahead design.

**Phase 0 baseline** (July 27, 2026, pre-comparables-tool — first complete, non-degenerate seeded backtest: `--n-per-phase 5 --seed 42`, model `gpt-5-mini`, `predictions/backtests/backtest_20260727T221344Z.json`; supersedes the earlier 13-prediction partial run):

| Phase | n | MAE | R² | % within tolerance |
|-------|---|-----|----|---------------------|
| pre-arb | 5 | $0.006M | 0.923 | 100% (±$0.25M) |
| arb | 5 | $1.655M | 0.452 | 40% (±$1.0M) |
| free-agent | 5 | $2.130M | 0.874 | 20% (±20%, relative) |

**Phase 2 baseline** (July 27, 2026, comparables tool + no-lookahead fix — same seed/sample/model, `predictions/backtests/backtest_20260727T231718Z.json`):

| Phase | n | MAE | R² | % within tolerance | vs. Phase 0 |
|-------|---|-----|----|---------------------|-------|
| pre-arb | 5 | $0.009M | 0.827 | 100% (±$0.25M) | flat — already near-perfectly memorized, comparables add little |
| arb | 5 | $1.295M | 0.594 | **60%** (±$1.0M) | improved — biggest tolerance-pass jump (40%→60%) |
| free-agent | 5 | $1.870M | 0.894 | 20% (±20%, relative) | modestly improved MAE/R², tolerance-pass rate unchanged |

A genuine but modest improvement, concentrated in arb — grounding comparable-contract *facts* helps somewhat, but doesn't fix blindness to current-season performance (Richards 54% low, Andrus 58% low in this run — still real misses). That gap is what the Phase 2 stats tool (not yet built) targets next.

**Caveats**: (1) Phase 0 backtests are training-data-contaminated (the LLM memorized many historical outcomes), so these numbers measure recall as much as prediction — the honest test is `predictions/history.csv` entries for genuinely future seasons, scored once actuals land. (2) `n=5`/phase is small; treat these as directional, not precise, and expect meaningful swings run to run (an earlier same-seed Phase 0 re-run had arb at 60%/free-agent MAE $2.58M — same sampled contracts, different LLM guesses, before any tool existed). The free-agent under-prediction pattern (and the Skubal-style post-cutoff misses, e.g. predicted $10M for 2026 vs. actual $32M in earlier runs) is the gap the Phase 2 stats tool targets.

### GitHub Actions Workflow

A daily workflow (`daily-dataset.yml`) was set up to run the full data pipeline and commit updated CSVs automatically. **Currently disabled** (schedule commented out) due to the stats collection failure described below.

---

## 3. Open Issues

### Critical: Stats Data Collection is Broken

**pybaseball** fetches stats from `https://www.fangraphs.com/leaders-legacy.aspx`, which now returns HTTP 403 and presents a human verification captcha. Every year (2016–2026) fails with this error. This blocks:

- Updating the stats dataset with current-season performance
- The daily update workflow (core requirement for in-season projections)
- Retraining models on fresh data

**Decision (July 2026), done**: with the shift to the agent architecture, current-season stats are now consumed as a **live MLB Stats API tool** (`agent/predict/mlb_stats_api.py:query_batting_stats`/`query_pitching_stats`, July 27, 2026) rather than by maintaining our own stats CSVs — this issue no longer blocks the agent (the local CSVs above are still stale and still affected by the pybaseball/FanGraphs 403, but the agent doesn't depend on them anymore). Covers every standard stat the old pipeline consumed but not the FanGraphs-proprietary metrics (WAR, wRC+, FIP/xFIP/SIERA, Statcast rates) — confirmed genuinely absent from the live API too, not just our old scrape. Player identity (mapping our `player_id`/`fangraphs_id` to the API's MLBAM id) was the open risk on this plan — solved via `agent/predict/player_id_crosswalk.py`, joining `players.csv`'s existing `fangraphs_id` against `pybaseball.chadwick_register()` (reused only for its id-crosswalk table, not its broken FanGraphs stats scrape). If the historical stats CSVs ever need refreshing anyway (e.g. for FanGraphs-proprietary WAR), a verified fallback still exists: FanGraphs' current JSON API with pybaseball-compatible fields — see `docs/agent/DESIGN.md` Appendix A.

### `contracts_spotrac.csv`: service_time is never tracked for free agents

Discovered live (July 2026) while testing the agent's `query_comparable_contracts` tool: **100% of free-agent rows** (2,084 of 2,084) carry Spotrac's `service_time = -1` "not tracked" sentinel — not sometimes-missing, always missing. Spotrac apparently stops recording it once a player clears the point where it no longer gates arbitration/free-agency eligibility.

Consequence: any query filtering free-agent contracts by service time returns zero matches unconditionally, regardless of how many real comparables exist — confirmed live: a search for free-agent SPs aged 29-31 with a service-time bound found nothing, but dropping only the service-time bound surfaced 161 real matches, several at $22-30M AAV. This isn't a bug in the query logic; the underlying data simply has no signal there for free agents. `query_comparable_contracts` (`agent/predict/comparables.py`) now raises an error immediately if `phase="free-agent"` is combined with a service_time bound, rather than silently returning nothing — service_time filtering still works normally for pre-arb/arb (confirmed: 120 real matches on a real 5.0-6.0 service-time band), and age/position/phase/year remain the comparability dimensions for free agents. `service_time` is still returned per match for context either way (correctly `null` for free agents).

**Not yet investigated**: whether this is fixable (a different Spotrac field, or a computable proxy — e.g. debut year plus active-year count) or is a genuine gap in the source data that has to stay a known limitation.

### `contracts_spotrac.csv`: club/player option years are never scraped — breaks phase resolution for affected players

Discovered live (July 2026) via a real `make ask` request for Brandon Lowe's projected 2027 salary, which the agent answered as "pre-arb" — badly wrong. Lowe's real timeline (user-supplied ground truth) is pre-arb 2019-2021, arb 2022-2024, then two **accepted club options**: 2025 ($10.5M salary / $9.5M AAV, service time 6.058) and 2026 ($11.5M / $11.0M, service time 7.058) — he's several years past arbitration, not still in it.

Two compounding bugs, one shallow and one structural:

1. **`agent/phase.py` bug (not yet fixed)**: when the latest contract row on file has `service_time = -1` (unknown), `resolve_phase()`'s fallback just echoes that row's `type` forward with no bound — for Lowe, whose *last row with a real service_time* is years stale, this reads as "still pre-arb" indefinitely. `project_phase_timeline()` has the opposite failure mode in the same situation: it gives up and returns early with a caveat rather than projecting anything. Two fix directions were scoped (search backward for the nearest row with a real service_time and anchor the projection there; or surface explicit uncertainty instead of guessing) but neither is implemented yet.
2. **Deeper, structural root cause**: `data_generation/spotrac.py` has **zero handling for club/player option contract rows** — confirmed via direct code inspection, not inference. The scraper only visits four bulk category-list URLs (pre-arb, arb, free-agent, extensions); it never visits per-player contract-detail pages, which is where option-year rows actually live on Spotrac. Lowe's real 2025/2026 rows are simply **absent** from `contracts_spotrac.csv` — not misclassified, not present-with-wrong-values, just missing entirely. No phase-resolution logic, however clever, can recover a contractual fact (an exercised option) that was never scraped in the first place.

**This is likely bigger than a `phase.py` patch.** The shallow fix (a smarter fallback) is worth doing regardless — it improves behavior for *any* player with a stale service_time anchor, not just option-year cases — but it cannot fully solve Lowe's case, because the underlying row doesn't exist. Fixing that requires the scraper to actually collect option-year contract data, which likely means visiting per-player detail pages (a materially different scraping approach than the current four-bulk-list design) rather than a small patch. Whether that means *adding* per-player detail scraping alongside the existing bulk lists, or *restructuring* contract collection around per-player timelines entirely, is an open design question — flagged here as something to reconsider, not yet decided.

**Not yet investigated**: how common option years are across the dataset (Lowe may not be an isolated case — any player with a multi-year deal containing option years likely has the same gap), and whether a cheaper interim signal (e.g. detecting the gap itself and flagging `method="unknown"` rather than guessing wrong) is worth doing before the full scraper fix.

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

1. **Agent Phase 1** — Natural-language front door: **done** (July 2026). `agent/ask.py` / `make ask REQUEST="..."` resolves a free-text request via a three-tier orchestrator/intake/predict agent architecture (see `docs/agent/DESIGN.md` Design Decision #7), reporting ambiguity through natural-language clarifying questions instead of guessing. `hypothetical_free_agent` mode supports "what would they get in free agency right now." `make ask` is also now a genuine multi-turn conversation rather than one Q&A — the orchestrator stays open for follow-ups on the same persistent agent after delivering an answer. Contract-status routing is also done: intake attaches `known`/`forecast` status right after resolving player/year, so the orchestrator skips `predict_tool` (and the LLM) entirely for a year already covered by a signed contract, instead of "predicting" a number that's already public record.

2. **Finish the Phase 0 baseline** — **done** (July 27, 2026). A complete, non-degenerate seeded backtest is recorded above in section 2 (the first run was cut short at 13 predictions and had a schema gap that let a degenerate 0/0 prediction slip through; both are fixed now). Later phases compare against this.

3. **Agent Phase 2** — Live data sourcing. **Comparable contracts — done** (July 27, 2026): `query_comparable_contracts` grounds comparable-contract facts and a player's own history in real records; see section 2 above for the baseline comparison and the no-lookahead fix. **Stats — done** (July 27, 2026): `query_batting_stats`/`query_pitching_stats` (live `statsapi.mlb.com`, not the stalled local CSVs) ground performance for both the target player and any comparable players, with the player-identity mapping problem solved via a `players.csv` fangraphs_id -> MLBAM id crosswalk. **Proposed, not yet built**: a market value references/trends tool (`query_market_value_trends`) returning percentile summaries (p90/max/min/median AAV) for a given year or year-range/phase/position, rather than individual comparable rows — see `docs/agent/DESIGN.md`'s Phase 2 Roadmap entry for the sketch. **Still open**: contract-data and player-identity live sourcing for `contracts_spotrac.csv`/`players.csv` themselves (no known public API for signed-contract dollar figures — Spotrac remains the source, likely fetched live/cached rather than replaced) — now entangled with the option-year gap below, since any live-sourcing spike should address both.

4. **Fix the club/player option contract gap** — newly found (July 2026), see this doc's Open Issues above and `docs/agent/DESIGN.md` Design Decision #4's update. Two parts, likely two separate efforts: (a) a bounded/anchored fix to `agent/phase.py`'s unknown-service_time fallback (smaller, worth doing regardless of (b)), and (b) a scraper change so `data_generation/spotrac.py` actually captures option-year contract rows (larger — likely requires per-player detail-page scraping, not just a patch to the existing four-bulk-list approach). Not yet scheduled against the phase numbering below; flagged as high priority given it produces confidently wrong answers, not just missing data.

5. **Agent Phase 3** — Multi-year forecasting: target-year ranges and user-supplied stat/context assumptions.

6. **Agent Phase 4** — CBA/league-minimum schedule and arb-raise-pattern heuristic tools, business logic in the system prompt, team-conditioned scenario support and user guardrails.

7. **Web application** — Serving layer + front end over the agent's predictions and history (unchanged, still furthest out).

8. **Daily update workflow** — Revisit once Phase 2 makes projections respond to live stats; the workflow's stats-collection step is superseded by the live-tool decision above.

Done: sklearn models (`models/`) and their design docs (`docs/pre_arb/`) archived to `archive/v3/` (July 2026); the unmerged `arb-model` branch stays a branch.
