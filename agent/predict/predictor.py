"""Prediction agent: structured output, now with its first Phase 2 tool.

Phase 0 was deliberately tool-less; Phase 2 adds query_comparable_contracts
(agent/predict/comparables.py) so comparable-contract reasoning is grounded in
real records instead of training memory. Stat-line grounding is still Phase 0
(model knowledge) until a stats data source is wired in as a later tool.

A fresh Agent is created per prediction (unlike the review-queue agent, which
reuses one agent across queue items) so conversation state never bleeds
between players and each trace's messages describe exactly one run.
"""

import os
import sys
import time

from agent.config import DEFAULT_MODEL_ID, model_params
from agent.predict.comparables import query_comparable_contracts
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


def create_agent(model_id=None):
    """Create a fresh Strands agent for one prediction."""
    Agent, OpenAIModel, _ = _import_strands()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set (checked .env).")
        sys.exit(1)

    model_id = model_id or DEFAULT_MODEL_ID
    model = OpenAIModel(model_id=model_id, params=model_params(model_id))
    return Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[query_comparable_contracts],
        hooks=[ToolCallLogger()],
    )


def predict_contract(user_prompt, model_id=None):
    """Run one prediction. Returns (prediction, messages, usage, latency_seconds).

    Raises StructuredOutputException if the model fails to produce a valid
    ContractPrediction after MAX_ATTEMPTS.
    """
    _, _, StructuredOutputException = _import_strands()

    last_error = None
    for attempt in range(MAX_ATTEMPTS):
        agent = create_agent(model_id)
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
