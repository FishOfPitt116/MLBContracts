# Python interpreter
PYTHON = PYTHONPATH=$(shell pwd) python3 

# files
# TODO: add more test directories here as they get created
# TODO: change `data_generation/spotrac.py` to the actual main dataset generation file once stats are added
CONTRACTS_DATASET_FILE = data_generation.spotrac
STATS_DATASET_FILE = data_generation.stats
ANALYSIS_FILE = analysis/contract_analysis.py

.PHONY: dataset analyze

build: dataset

dataset:
	@echo "Assembling players and contracts dataset..."
	$(PYTHON) -m $(CONTRACTS_DATASET_FILE) --start-year 2011 --end-year 2026
	@echo "Assembling stats dataset..."
	$(PYTHON) -m $(STATS_DATASET_FILE)
	@echo "Full dataset assembly complete."

analyze:
	@echo "Running contract analysis..."
	$(PYTHON) $(ANALYSIS_FILE)
	@echo "Contract analysis complete!"