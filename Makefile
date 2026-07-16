# Python interpreter (uses venv if available, otherwise system python)
PYTHON_BIN = $(shell if [ -f .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
PYTHON = PYTHONPATH=$(shell pwd) $(PYTHON_BIN)

# files
CONTRACTS_DATASET_FILE = data_generation.spotrac
STATS_DATASET_FILE = data_generation.stats
ANALYSIS_FILE = analysis/contract_analysis.py

.PHONY: dataset dataset-auto analyze review-queue join predict backtest-agent test-agent

build: dataset

dataset:
	@echo "Assembling players and contracts dataset..."
	$(PYTHON) -m $(CONTRACTS_DATASET_FILE) --start-year 2011 --end-year $(shell date +%Y)
	@echo "Assembling stats dataset..."
	$(PYTHON) -m $(STATS_DATASET_FILE)
	@echo "Full dataset assembly complete."

dataset-auto:
	@echo "Assembling players and contracts dataset (non-interactive)..."
	$(PYTHON) -m $(CONTRACTS_DATASET_FILE) --start-year 2011 --end-year $(shell date +%Y) --non-interactive
	@echo "Assembling stats dataset..."
	$(PYTHON) -m $(STATS_DATASET_FILE)
	@echo "Joining contracts with stats..."
	$(PYTHON) -m data_generation.join
	@echo "Full dataset assembly complete."

analyze:
	@echo "Running contract analysis..."
	$(PYTHON) $(ANALYSIS_FILE)
	@echo "Contract analysis complete!"

review-queue:
	@echo "Processing player review queue..."
	$(PYTHON) -m data_generation.review_queue

join:
	@echo "Joining contracts with player stats..."
	$(PYTHON) -m data_generation.join
	@echo "Join complete!"

# Agent-based contract prediction (see docs/agent/DESIGN.md)
# Usage: make predict PLAYER=Scherzer_5166 YEAR=2026
predict:
	$(PYTHON) -m agent.predict --player-id $(PLAYER) --year $(YEAR)

backtest-agent:
	$(PYTHON) -m agent.backtest --n-per-phase 5 --seed 42

test-agent:
	$(PYTHON) -m pytest agent/tests -q