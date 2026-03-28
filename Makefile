# Python interpreter (uses venv)
PYTHON = PYTHONPATH=$(shell pwd) .venv/bin/python

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

join:
	@echo "Joining contracts with player stats..."
	$(PYTHON) -m data_generation.join
	@echo "Join complete!"