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


# Bound on how many years forward _attach_contract_status will look for a
# genuine forecast year when auto-advancing past an expiring known deal.
_MAX_ADVANCE_YEARS = 5


def _attach_contract_status(intake_result):
    """Deterministically label a ready 'predict' result known/forecast.

    Never left to the LLM: phase resolution is harness-side (agent/phase.py),
    and so is this. Done here, as early as intake, rather than waiting until
    predict_tool runs, so the orchestrator can skip calling predict_tool (and
    the LLM call inside it) entirely for a year that's already on record.
    "hypothetical_free_agent" and not-yet-ready results are left untouched —
    contract_status only applies to a resolved real-world prediction request.

    Balances two things a user asking for a "projected contract" cares about:
    a player deep into a stable multi-year deal (e.g. signed through 2030) has
    nothing more interesting to project, so the known deal IS the answer. But
    a player whose defaulted year lands on the last year of a short/expiring
    deal (e.g. a one-year arb salary ending this season) is a case where
    "projected" clearly means their NEXT contract — so when the year was
    defaulted (not stated by the user) and the request wants a forecast, this
    advances target_year past the expiring deal to the next real forecast
    year, carrying the expiring deal along as prior_known_contract so the
    assistant can mention both instead of only one or the other.
    """
    if intake_result.status != "ready" or intake_result.mode != "predict":
        return intake_result

    player_id = intake_result.player_id
    year = intake_result.target_year
    resolution = resolve_phase(player_id, year)

    if resolution.known_value is None:
        return intake_result.model_copy(update={"contract_status": "forecast"})

    deal_ends_this_year = resolution.known_value["end_year"] <= year
    if intake_result.year_was_defaulted and intake_result.wants_forecast and deal_ends_this_year:
        next_year, advanced = year, resolution
        for _ in range(_MAX_ADVANCE_YEARS):
            next_year += 1
            advanced = resolve_phase(player_id, next_year)
            if advanced.known_value is None:
                break
        note = (
            f"current {resolution.known_value['contract_id']} deal ends after {year}; "
            f"since a projection was requested, defaulting target_year to {next_year} instead"
        )
        update = {
            "target_year": next_year,
            "notes": intake_result.notes + [note],
            "prior_known_contract": resolution.known_value,
        }
        if advanced.known_value is not None:
            update["contract_status"] = "known"
            update["known_contract"] = advanced.known_value
        else:
            update["contract_status"] = "forecast"
        return intake_result.model_copy(update=update)

    return intake_result.model_copy(
        update={"contract_status": "known", "known_contract": resolution.known_value}
    )


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
