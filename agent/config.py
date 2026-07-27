"""Configuration for the contract prediction agent."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Repo root is the parent of the agent/ package
REPO_ROOT = Path(__file__).resolve().parent.parent

# Load OPENAI_API_KEY (and any overrides) from the repo-root .env
load_dotenv(REPO_ROOT / ".env")

# Dataset paths
CONTRACTS_CSV = REPO_ROOT / "dataset" / "contracts_spotrac.csv"
PLAYERS_CSV = REPO_ROOT / "dataset" / "players.csv"

# Prediction output paths
PREDICTIONS_DIR = REPO_ROOT / "predictions"
TRACES_DIR = PREDICTIONS_DIR / "traces"
BACKTESTS_DIR = PREDICTIONS_DIR / "backtests"
CONVERSATIONS_DIR = PREDICTIONS_DIR / "conversations"
HISTORY_CSV = PREDICTIONS_DIR / "history.csv"

# Model configuration. gpt-5* reasoning models reject sampling params like
# temperature; only pass temperature to non-reasoning (gpt-4o*) models.
DEFAULT_MODEL_ID = os.environ.get("AGENT_MODEL_ID", "gpt-5-mini")

# Output-length ceilings: a guardrail against a single runaway response (cost/latency),
# not a tuned-for-quality budget. gpt-5* reasoning models count reasoning tokens against
# this too, hence the larger number; gpt-4o* has no hidden reasoning overhead.
GPT5_MAX_COMPLETION_TOKENS = 4000
GPT4O_MAX_TOKENS = 2000


def model_params(model_id):
    """Return OpenAI params appropriate for the model family."""
    if model_id.startswith("gpt-4o"):
        return {"temperature": 0.1, "max_tokens": GPT4O_MAX_TOKENS}
    if model_id.startswith("gpt-5"):
        # Reasoning models use max_completion_tokens, not max_tokens (OpenAI API).
        return {"reasoning_effort": "low", "max_completion_tokens": GPT5_MAX_COMPLETION_TOKENS}
    return {}
