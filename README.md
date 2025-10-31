# MLBContracts

MLBContracts collects MLB player contract data, related player metadata, and season statistics; provides tooling to assemble datasets; and includes analysis and modeling code to explore and predict contract values.

This README explains repository layout, how to install dependencies, how to run data collection and analysis, and where outputs are stored. Previous code versions live under `archive/` and are kept for reference only.

## Table of contents
- Project overview
- Repository layout
- Requirements & setup
- Data collection
- Analysis & visualization
- Modeling & experiments
- Tests
- Archive
- Contributing & license

## Project overview
- Source data is primarily scraped/assembled into CSV datasets and then used for EDA, plotting, and simple ML experiments.
- Analysis scripts generate plots under `analysis/graphs/`.
- Scripts call small modules that implement analysis pipelines (e.g., arbitration, pre-arbitration, free agent analyses).

## Repository layout (key files / dirs)
- data_generation/ — scrapers and dataset assembly tools (spotrac, helpers, save/read utilities)
- dataset/ — CSVs produced/consumed by analysis (e.g., `dataset/contracts_spotrac.csv`)
- analysis/ — plotting and analysis scripts (e.g., `analysis/contract_analysis.py`, analysis helpers, `analysis/graphs/`)
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
- Primary scraper/assembler: `data_generation/spotrac.py`
- Typical usage:
  ```bash
  # run the spotrac data collection entrypoint
  make dataset
  # or
  python -m data_generation.spotrac
  # or
  python data_generation/spotrac.py
  ```
- Output dataset examples:
  - `dataset/contracts_spotrac.csv` — main contract table used by analysis

## Analysis & visualization
- Main analysis entrypoint: `analysis/contract_analysis.py`
  - This script loads `dataset/contracts_spotrac.csv`, computes AAV (average annual value), and runs a set of analyses:
    - `contract_value_distribution()` — boxplots of AAV by age and contract type (saved to `analysis/graphs/contract_value_distribution.png`)
    - Pre-arbitration, arbitration, and free-agent analysis functions are invoked via `pre_arb.main()`, `arb.main()`, and `free_agents.main()` when run as `__main__`.
  - Run it with:
    ```bash
    python analysis/contract_analysis.py
    ```
  - The script creates `analysis/graphs/` (if missing) and writes PNG files there.

- Arbitration service-time vs contract value plot:
  - The arbitration scatter plot function (e.g., `arbitration_service_time_vs_contract_value`) generates a scatter of service time vs AAV and overlays a dotted best-fit regression line (via seaborn/matplotlib). Look for the generated PNGs in `analysis/graphs/`.

- Notes:
  - If plots appear clipped, increase `plt.figure(figsize=(..., ...))` or adjust `plt.tight_layout()`.
  - Ensure `dataset/contracts_spotrac.csv` exists and is up to date before running analysis.

## Archive
- `archive/` contains earlier project snapshots and experimental code. These are preserved for reference only and are not part of the active pipeline. Do not modify files under `archive/` when working on the main pipeline.

## Troubleshooting & tips
- Missing dataset: re-run `make dataset` or locate CSVs in `dataset/`.
- Plot files not appearing: confirm `analysis/contract_analysis.py` created `analysis/graphs/` (it does by default) and check file permissions.
- Reproducibility: use a virtual environment and pin dependencies in `requirements.txt`.

## Contributing
- Open a PR with a clear description and tests for new behavior.
- Keep changes small and focused; update `README.md` or inline docstrings for any behavior changes.

## License
- No license file included by default. Add a LICENSE to clarify reuse and distribution terms.
