"""Prediction agent: structured output, now with Phase 2 tools.

Phase 0 was deliberately tool-less. Phase 2 adds, one category at a time:
query_comparable_contracts (agent/predict/comparables.py) for comparable-
contract facts and a player's own contract history; query_batting_stats /
query_pitching_stats (agent/predict/mlb_stats_api.py, live statsapi.mlb.com
calls) for performance grounding. Anything not yet covered by a tool still
relies on model knowledge, same as Phase 0.

A fresh Agent is created per prediction (unlike the review-queue agent, which
reuses one agent across queue items) so conversation state never bleeds
between players and each trace's messages describe exactly one run. Each
fresh agent also gets its own tool instances scoped to that prediction's
target_year (see comparables.py's and mlb_stats_api.py's NO LOOKAHEAD notes)
-- target_year is required here specifically so that cutoff can never be
forgotten for a real prediction call.
"""

import os
import sys
import time

from agent.config import DEFAULT_MODEL_ID, model_params
from agent.predict.comparables import make_comparable_contracts_tool
from agent.predict.mlb_stats_api import make_batting_stats_tool, make_pitching_stats_tool
from agent.predict.prompts import SYSTEM_PROMPT
from agent.predict.schema import ContractPrediction
from agent.tool_logging import ToolCallLogger

# One retry on structured-output failure before giving up
MAX_ATTEMPTS = 2


def _import_strands():
    try:
        from strands import Agent
        from strands.models.openai import OpenAIModel
        from strands.types.exceptions import StructuredOutputException
    except ImportError:
        print("Error: strands-agents package not installed.")
        print("Install with: pip install 'strands-agents[openai]'")
        print("")
        print("Note: strands-agents requires Python >= 3.10")
        print(f"Current Python version: {sys.version}")
        sys.exit(1)
    return Agent, OpenAIModel, StructuredOutputException


def create_agent(target_year, model_id=None):
    """Create a fresh Strands agent for one prediction, targeting target_year.

    target_year is required (not defaulted) so the comparable-contracts tool's
    no-lookahead cutoff is never accidentally omitted for a real prediction.
    """
    Agent, OpenAIModel, _ = _import_strands()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set (checked .env).")
        sys.exit(1)

    model_id = model_id or DEFAULT_MODEL_ID
    model = OpenAIModel(model_id=model_id, params=model_params(model_id))
    return Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[
            make_comparable_contracts_tool(target_year),
            make_batting_stats_tool(target_year),
            make_pitching_stats_tool(target_year),
        ],
        hooks=[ToolCallLogger()],
    )


def predict_contract(user_prompt, target_year, model_id=None):
    """Run one prediction. Returns (prediction, messages, usage, latency_seconds).

    target_year: the season being predicted -- passed straight through to
    create_agent() so the comparable-contracts tool can never see a contract
    dated that year or later (see comparables.py's NO LOOKAHEAD note).

    Raises StructuredOutputException if the model fails to produce a valid
    ContractPrediction after MAX_ATTEMPTS.
    """
    _, _, StructuredOutputException = _import_strands()

    last_error = None
    for attempt in range(MAX_ATTEMPTS):
        agent = create_agent(target_year, model_id)
        start = time.monotonic()
        try:
            result = agent(user_prompt, structured_output_model=ContractPrediction)
        except StructuredOutputException as error:
            last_error = error
            continue
        latency = time.monotonic() - start
        prediction = result.structured_output
        usage = dict(result.metrics.accumulated_usage) if result.metrics else None
        return prediction, agent.messages, usage, latency
    raise last_error
