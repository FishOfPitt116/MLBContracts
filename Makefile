# Python interpreter
PYTHON = PYTHONPATH=$(shell pwd) python3

# files
# add more test directories here as they get created
# TEST_DIRS = tst/parse
MAIN_FILE = src/main.py

build: run

# test:
# 	@echo "Running unit tests..."
# 	$(foreach dir, $(TEST_DIRS), $(PYTHON) -m unittest discover -s $(dir))
# 	@echo "All tests passed!"

run:
	@echo "Running main application..."
	$(PYTHON) $(MAIN_FILE)