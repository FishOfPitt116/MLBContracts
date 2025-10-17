# Python interpreter
PYTHON = PYTHONPATH=$(shell pwd) python3 

# files
# TODO: add more test directories here as they get created
# TEST_DIRS = tst/parse
MAIN_FILE = src/main.py
PREDICT_FILE = src/predict.py
# TODO: change `data_generation/spotrac.py` to the actual main dataset generation file once stats are added
DATASET_FILE = data_generation.spotrac
ANALYSIS_FILE = analysis/contract_analysis.py

.PHONY: build run predict dataset

build: run

# test:
# 	@echo "Running unit tests..."
# 	$(foreach dir, $(TEST_DIRS), $(PYTHON) -m unittest discover -s $(dir))
# 	@echo "All tests passed!"

run:
	@echo "Running main application..."
	$(PYTHON) $(MAIN_FILE)

predict:
	@echo "Running prediction using best models..."
	$(PYTHON) $(PREDICT_FILE)

dataset:
	@echo "Assembling dataset..."
	$(PYTHON) -m $(DATASET_FILE) --start-year 2011 --end-year 2025

analyze:
	@echo "Running contract analysis..."
	$(PYTHON) $(ANALYSIS_FILE)
	@echo "Contract analysis complete!"