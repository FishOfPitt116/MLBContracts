# Python interpreter (uses venv if available, otherwise system python)
PYTHON_BIN = $(shell if [ -f .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
PYTHON = PYTHONPATH=$(shell pwd) $(PYTHON_BIN)

# files
# TODO: add more test directories here as they get created
# TODO: change `data_generation/spotrac.py` to the actual main dataset generation file once stats are added
CONTRACTS_DATASET_FILE = data_generation.spotrac
STATS_DATASET_FILE = data_generation.stats
ANALYSIS_FILE = analysis/contract_analysis.py

.PHONY: dataset dataset-auto analyze review-queue-human review-queue-status review-queue-agent join

build: dataset

# =============================================================================
# Dataset Generation
# =============================================================================

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

join:
	@echo "Joining contracts with player stats..."
	$(PYTHON) -m data_generation.join
	@echo "Join complete!"

# =============================================================================
# Review Queue (Player Matching)
# =============================================================================

# Show review queue statistics
review-queue-status:
	@$(PYTHON) -m data_generation.review_queue_agent --status

# Run AI agent to process pending items (requires OPENAI_API_KEY)
review-queue-agent:
	@echo "Running AI agent on review queue..."
	$(PYTHON) -m data_generation.review_queue_agent

# Dry run - show what agent would do without making changes
review-queue-agent-dry:
	@echo "Running AI agent on review queue (dry run)..."
	$(PYTHON) -m data_generation.review_queue_agent --dry-run --limit 5 --verbose

# Human review of items the agent couldn't match
review-queue-human:
	@echo "Processing player review queue (human review)..."
	$(PYTHON) -m data_generation.review_queue_human

# =============================================================================
# Analysis
# =============================================================================

analyze:
	@echo "Running contract analysis..."
	$(PYTHON) $(ANALYSIS_FILE)
	@echo "Contract analysis complete!"