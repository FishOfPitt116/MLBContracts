# Python interpreter
PYTHON = PYTHONPATH=$(shell pwd) python3 

# files
# TODO: add more test directories here as they get created
# TODO: change `data_generation/spotrac.py` to the actual main dataset generation file once stats are added
DATASET_FILE = data_generation.spotrac
ANALYSIS_FILE = analysis/contract_analysis.py

.PHONY: dataset analyze

build: dataset

dataset:
	@echo "Assembling dataset..."
	$(PYTHON) -m $(DATASET_FILE) --start-year 2011 --end-year 2025

analyze:
	@echo "Running contract analysis..."
	$(PYTHON) $(ANALYSIS_FILE)
	@echo "Contract analysis complete!"