# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Data collection (contracts + stats)
make dataset

# Run separately if needed
python -m data_generation.spotrac --start-year 2011 --end-year 2025
python -m data_generation.stats

# Force re-fetch all data (bypasses caching)
python -m data_generation.spotrac --start-year 2011 --end-year 2025 --overwrite

# Non-interactive mode (for automation/cron jobs)
python -m data_generation.spotrac --start-year 2011 --end-year 2025 --non-interactive

# Process queued players after non-interactive runs
make review-queue                               # Interactive processing
python -m data_generation.review_queue --status # View queue without processing

# Join contracts with player stats (for modeling)
make join                                       # Uses 1-year window (prior season)
python -m data_generation.join --window 3       # Use 3-year rolling window

# Run analysis and generate plots
make analyze
```

## Architecture

### Data Pipeline

1. **Spotrac Scraper** (`data_generation/spotrac.py`): Scrapes MLB contract data from spotrac.com
   - Collects pre-arbitration, arbitration, and free-agent contracts (plus extensions)
   - Maps player names to FanGraphs IDs via pybaseball's `playerid_lookup()`
   - Output: `dataset/contracts_spotrac.csv`, `dataset/players.csv`

2. **Stats Aggregator** (`data_generation/stats.py`): Collects player statistics via pybaseball
   - Creates individual-year and rolling-window stats (1, 3, 5, 10-year windows)
   - Output: `dataset/batter_stats.csv`, `dataset/pitcher_stats.csv`

3. **Dataset Join** (`data_generation/join.py`): Joins contracts with prior-season stats
   - For contract in year Y, attaches stats from year Y-1
   - Includes both batting (`bat_*`) and pitching (`pit_*`) stats for all players
   - Supports multiple window sizes (1, 3, 5, 10-year)
   - Output: `dataset/contracts_with_stats.csv`

4. **Analysis Pipeline** (`analysis/contract_analysis.py`): Main analysis entrypoint
   - Computes AAV (Average Annual Value) = value / duration
   - Generates plots to `analysis/graphs/`
   - Calls sub-analyses: `pre_arb.main()`, `arb.main()`, `free_agents.main()`

### Key Data Structures (in `data_generation/records.py`)

- **Player**: ID, names, position, birth date, spotrac/baseball-reference links
- **Salary**: Contract value (millions), duration, type, age, service time, year
- **BatterStats/PitcherStats**: Comprehensive stats with window_years for rolling accumulations
- **ReviewQueueItem**: Players needing manual name resolution (used by `--non-interactive` mode)

### Data Identity

- **Player ID**: `{last_name}_{spotrac_link_id}` (e.g., "Smith_12345")
- **Contract ID**: `{player_id}_{year}` (e.g., "Smith_12345_2025")
- **Stats Key**: (player_id, year, window_years) tuple

## Key Conventions

### Caching & Performance

- Player/Contract objects cached in module-level dicts, initialized from CSV at import
- Historical years (non-current) are cached—only current year re-fetched on subsequent runs
- Stats collection skips players not active in current year
- pybaseball caching enabled via `cache.enable()` in stats.py

### CSV-First Approach

- All analysis reads from CSV files via pandas, not direct API calls
- CSV I/O handled by `data_generation/save.py`
- Datasets stored in `dataset/` directory

### Python Environment

- Makefile sets `PYTHONPATH=$(pwd)` to enable imports like `from data_generation.spotrac import ...`
- Always activate `.venv` before running

### Archive

- `archive/` contains v1 and v2.1 snapshots—do not modify or integrate into active pipeline
