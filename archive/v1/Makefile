# Python interpreter
PYTHON = PYTHONPATH=$(shell pwd) python3 

# files
MAIN_FILE = src/main.py
PREDICT_FILE = src/predict.py

.PHONY: run predict

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
