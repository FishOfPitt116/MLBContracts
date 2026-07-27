"""Intake sub-agent: resolves free text into a concrete (player, year, mode).

Stateless per call — the caller (the orchestrator) always passes the full
accumulated context, so a fresh Agent is built each time rather than reusing
conversation state here.
"""

import os
import sys

from strands import tool

from agent.config import DEFAULT_MODEL_ID, model_params
from agent.intake.prompts import build_intake_system_prompt
from agent.intake.schema import IntakeResult
from agent.intake.tools import find_player, get_contract_phase_timeline
from agent.phase import resolve_phase


def _attach_contract_status(intake_result):
    """Deterministically label a ready 'predict' result known/forecast.

    Never left to the LLM: phase resolution is harness-side (agent/phase.py),
    and so is this. Done here, as early as intake, rather than waiting until
    predict_tool runs, so the orchestrator can skip calling predict_tool (and
    the LLM call inside it) entirely for a year that's already on record.
    "hypothetical_free_agent" and not-yet-ready results are left untouched —
    contract_status only applies to a resolved real-world prediction request.
    """
    if intake_result.status != "ready" or intake_result.mode != "predict":
        return intake_result

    resolution = resolve_phase(intake_result.player_id, intake_result.target_year)
    if resolution.known_value is not None:
        return intake_result.model_copy(
            update={"contract_status": "known", "known_contract": resolution.known_value}
        )
    return intake_result.model_copy(update={"contract_status": "forecast"})


def _import_strands():
    try:
        from strands import Agent
        from strands.models.openai import OpenAIModel
    except ImportError:
        print("Error: strands-agents package not installed.")
        print("Install with: pip install 'strands-agents[openai]'")
        print("")
        print("Note: strands-agents requires Python >= 3.10")
        print(f"Current Python version: {sys.version}")
        sys.exit(1)
    return Agent, OpenAIModel


def resolve_intake(context, model_id=None):
    """Run one intake pass over the given context. Returns an IntakeResult."""
    Agent, OpenAIModel = _import_strands()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set (checked .env).")
        sys.exit(1)

    model_id = model_id or DEFAULT_MODEL_ID
    model = OpenAIModel(model_id=model_id, params=model_params(model_id))
    agent = Agent(
        model=model,
        tools=[find_player, get_contract_phase_timeline],
        system_prompt=build_intake_system_prompt(),
    )
    result = agent(context, structured_output_model=IntakeResult)
    return _attach_contract_status(result.structured_output)


@tool
def intake_tool(context: str) -> dict:
    """Resolve a contract-prediction request into {player, year, mode}, or ask for more.

    Pass the FULL accumulated context each call — the original request plus any
    clarifying answers the user has already given, concatenated together. This
    tool is stateless per call and re-derives its answer from context alone.

    Args:
        context: The full conversation so far, as plain text.

    Returns:
        The intake result: status ("ready" or "needs_clarification"), player_id,
        player_name, target_year, mode, clarifying_question, and notes.
    """
    return resolve_intake(context).model_dump()
