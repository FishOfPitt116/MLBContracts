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
HISTORY_CSV = PREDICTIONS_DIR / "history.csv"

# Model configuration. gpt-5* reasoning models reject sampling params like
# temperature; only pass temperature to non-reasoning (gpt-4o*) models.
DEFAULT_MODEL_ID = os.environ.get("AGENT_MODEL_ID", "gpt-5-mini")


def model_params(model_id):
    """Return OpenAI params appropriate for the model family."""
    if model_id.startswith("gpt-4o"):
        return {"temperature": 0.1}
    return {}
