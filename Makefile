# Python interpreter (uses venv if available, otherwise system python)
PYTHON_BIN = $(shell if [ -f .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
PYTHON = PYTHONPATH=$(shell pwd) $(PYTHON_BIN)

# files
# TODO: add more test directories here as they get created
# TODO: change `data_generation/spotrac.py` to the actual main dataset generation file once stats are added
CONTRACTS_DATASET_FILE = data_generation.spotrac
STATS_DATASET_FILE = data_generation.stats
ANALYSIS_FILE = analysis/contract_analysis.py

.PHONY: dataset dataset-auto analyze review-queue join

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

review-queue-agent:
	@echo "Processing player review queue with agent..."
	$(PYTHON) -m data_generation.review_queue_agent

review-queue-agent-dry:
	@echo "Processing player review queue with agent (dry run)..."
	$(PYTHON) -m data_generation.review_queue_agent --dry-run --limit 10

join:
	@echo "Joining contracts with player stats..."
	$(PYTHON) -m data_generation.join
	@echo "Join complete!"