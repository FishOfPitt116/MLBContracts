# MLBContracts

MLBContracts collects MLB player contract data, related player metadata, and season statistics; provides tooling to assemble datasets; and includes analysis and modeling code to explore and predict contract values.

This README explains repository layout, how to install dependencies, how to run data collection and analysis, and where outputs are stored. Previous code versions live under `archive/` and are kept for reference only.

## Table of contents
- [Project overview](#project-overview)
- [Repository layout](#repository-layout-key-files--dirs)
- [Requirements & setup](#requirements--setup)
- [Data collection](#data-collection)
- [Analysis & visualization](#analysis--visualization)
- [Contract prediction agent](#contract-prediction-agent)
- [Troubleshooting & tips](#troubleshooting--tips)
- [Current state & remaining work](#current-state--remaining-work)
- [Archive](#archive)
- [Contributing](#contributing)

## Project overview
- Source data is primarily scraped/assembled into CSV datasets and then used for EDA, plotting, and simple ML experiments.
- Analysis scripts generate plots under `analysis/graphs/`.
- Scripts call small modules that implement analysis pipelines (e.g., arbitration, pre-arbitration, free agent analyses).
- The active prediction system is an LLM agent (`agent/`, Strands SDK + OpenAI) that replaces the earlier sklearn regression models (archived under `archive/v3/`) — see [Contract prediction agent](#contract-prediction-agent) below and `docs/agent/DESIGN.md` for the full design.

## Repository layout (key files / dirs)
- data_generation/ — scrapers and dataset assembly tools (spotrac, helpers, save/read utilities)
- dataset/ — CSVs produced/consumed by analysis and the agent (e.g., `dataset/contracts_spotrac.csv`, `dataset/players.csv`, `dataset/mlbam_id_crosswalk.csv`)
- analysis/ — plotting and analysis scripts (e.g., `analysis/contract_analysis.py`, analysis helpers, `analysis/graphs/`)
- agent/ — LLM-based contract prediction agent (orchestrator/intake/predict sub-agents, phase resolution, backtest harness); see `docs/agent/DESIGN.md`
- predictions/ — agent run artifacts: `traces/` (one JSON per prediction), `conversations/` (one JSON per `make ask` session), `history.csv` (append-only prediction log); gitignored (unbounded growth)
- docs/ — `docs/agent/DESIGN.md` (agent design/roadmap) and `docs/PROJECT_STATE.md` (overall project status, open issues, what's next)
- archive/ — previous repository versions and experiments (for historical reference only)
- README.md — this file

## Requirements & setup
- macOS (development was performed on macOS; commands below assume macOS shell)
- Python 3.8+ recommended
- Create a virtual environment and install dependencies:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
  If a `requirements.txt` is not present, install the common data/plotting packages:
  ```bash
  pip install pandas seaborn matplotlib scikit-learn pytest
  ```

## Data collection
- Data collection is a two-step process: **contracts** (via spotrac scraping) + **statistics** (via pybaseball)
- Primary scrapers/assemblers:
  - `data_generation/spotrac.py` — Scrapes MLB contract data from spotrac.com
  - `data_generation/stats.py` — Collects player season statistics from pybaseball
- Typical usage:
  ```bash
  # Run both contract and stats collection (recommended)
  make dataset
  # or run separately
  python -m data_generation.spotrac --start-year 2011 --end-year 2025
  python -m data_generation.stats
  ```
- Contract collection optimization: Historical years (not current year) are cached and skipped on subsequent runs
  - Use `--overwrite` flag to force re-fetch: `python -m data_generation.spotrac --start-year 2011 --end-year 2025 --overwrite`
- Stats collection optimization: Players without current-year data are skipped on subsequent runs
- Non-interactive mode for automation:
  ```bash
  python -m data_generation.spotrac --start-year 2011 --end-year 2025 --non-interactive
  ```
  - Ambiguous player name matches are queued to `dataset/review_queue.csv` instead of prompting
  - Process the queue later with `make review-queue` or `python -m data_generation.review_queue`
- Output datasets:
  - `dataset/contracts_spotrac.csv` — Main contract table used by analysis
  - `dataset/batter_stats.csv` — Batting statistics (individual years and rolling windows)
  - `dataset/pitcher_stats.csv` — Pitching statistics (individual years and rolling windows)

## Analysis & visualization
- Main analysis entrypoint: `analysis/contract_analysis.py`
  - This script loads `dataset/contracts_spotrac.csv`, computes AAV (average annual value), and runs a set of analyses:
    - `contract_value_distribution()` — boxplots of AAV by age and contract type (saved to `analysis/graphs/contract_value_distribution.png`)
    - Pre-arbitration, arbitration, and free-agent analysis functions are invoked via `pre_arb.main()`, `arb.main()`, and `free_agents.main()` when run as `__main__`.
  - Run it with:
    ```bash
    make analyze
    # or
    python analysis/contract_analysis.py
    ```
  - The script creates `analysis/graphs/` (if missing) and writes PNG files there.

- **Note on datasets**: Before running analysis, ensure both contracts and stats datasets are current:
  - `dataset/contracts_spotrac.csv` — Player contracts (required)
  - `dataset/batter_stats.csv` — Batting stats with rolling windows (optional for current analyses)
  - `dataset/pitcher_stats.csv` — Pitching stats with rolling windows (optional for current analyses)
  - Run `make dataset` to refresh all datasets.

- Arbitration service-time vs contract value plot:
  - The arbitration scatter plot function (e.g., `arbitration_service_time_vs_contract_value`) generates a scatter of service time vs AAV and overlays a dotted best-fit regression line (via seaborn/matplotlib). Look for the generated PNGs in `analysis/graphs/`.

- Notes:
  - If plots appear clipped, increase `plt.figure(figsize=(..., ...))` or adjust `plt.tight_layout()`.
  - Ensure `dataset/contracts_spotrac.csv` exists and is up to date before running analysis.

## Contract prediction agent
- The active prediction system is `agent/` — an LLM agent (Strands SDK + OpenAI, default model `gpt-5-mini`) that predicts contracts across all three cost-control phases (pre-arb, arbitration, free agency). It supersedes the sklearn models in `archive/v3/models/`.
- Setup: requires `OPENAI_API_KEY` in a `.env` file at the repo root.
- Usage:
  ```bash
  # Natural-language front door (recommended) — resolves player/year/mode, asks clarifying
  # questions if ambiguous, and supports multi-turn follow-ups
  make ask REQUEST="what will Scherzer make in 2026?"

  # Direct flags, one prediction, no conversation
  make predict PLAYER=Scherzer_5166 YEAR=2026

  # Seeded backtest against historical contracts (5 samples per phase)
  make backtest-agent

  # Offline unit tests, no API key needed
  make test-agent
  ```
- Every prediction persists a full trace (`predictions/traces/{run_id}.json`: prompts, tool calls, structured output, citations) and appends a row to `predictions/history.csv`. Every `make ask` conversation persists a linked transcript under `predictions/conversations/`.
- The agent grounds its predictions with tools — `query_comparable_contracts` (real historical contract records) and `query_batting_stats`/`query_pitching_stats` (live `statsapi.mlb.com` performance data) — rather than relying solely on the model's training knowledge. See `docs/agent/DESIGN.md` for the full tool/architecture design and `docs/PROJECT_STATE.md` for current status and open issues.

## Archive
- `archive/` contains earlier project snapshots and experimental code. These are preserved for reference only and are not part of the active pipeline. Do not modify files under `archive/` when working on the main pipeline.

## Troubleshooting & tips
- **Missing datasets**: Run `make dataset` to generate all datasets (contracts + stats)
  - Contracts dataset: `dataset/contracts_spotrac.csv`
  - Batter stats dataset: `dataset/batter_stats.csv`
  - Pitcher stats dataset: `dataset/pitcher_stats.csv`
- **Players pending review**: After non-interactive runs, check and process queued players:
  ```bash
  make review-queue                                  # Interactive processing
  python -m data_generation.review_queue --status   # View queue without processing
  ```
- **Data collection optimizations**:
  - Contract collection caches historical years—only current year (2026) is fetched on subsequent runs
  - Stats collection skips players not active in current year—only updates for current-year players
  - Use `--overwrite` flag with spotrac scraper to force full re-fetch: `python -m data_generation.spotrac --start-year 2011 --end-year 2025 --overwrite`
- **Plot files not appearing**: Confirm `analysis/contract_analysis.py` created `analysis/graphs/` (it does by default) and check file permissions.
- **Reproducibility**: Use a virtual environment and pin dependencies in `requirements.txt`.

## Current state & remaining work
- The agent (Phase 0 baseline, Phase 1 natural-language front door, and Phase 2's comparable-contracts + live-stats tools) is implemented and covered by ~120 offline tests. Full status, baseline backtest numbers, and open issues live in `docs/PROJECT_STATE.md`; the design rationale and roadmap live in `docs/agent/DESIGN.md`.
- Notable open items (see those two docs for detail):
  - A dataset-completeness gap: `data_generation/spotrac.py` never scrapes club-option contract years, which breaks phase resolution for affected players (e.g. a player years into accepted club options is still read as "pre-arb") — may require rethinking how `contracts_spotrac.csv` is assembled, not just a small patch.
  - No market-value distribution/percentile tool yet (e.g. "what's the p90/max/min AAV for a free-agent SP in a given year or across several years") — the agent currently only sees individual comparable contracts, not broader market context.
  - Live sourcing for `contracts_spotrac.csv`/`players.csv` themselves (currently static, batch-scraped files) is still open.
  - Multi-year forecasting (Phase 3) and team-conditioned scenario support / CBA & arb-raise heuristic tools (Phase 4) are designed but not yet built.

## Contributing
- Open a PR with a clear description and tests for new behavior.
- Keep changes small and focused; update `README.md` or inline docstrings for any behavior changes.

## License
- No license file included by default. Add a LICENSE to clarify reuse and distribution terms.
