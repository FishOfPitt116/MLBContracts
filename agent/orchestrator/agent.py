"""The orchestrator: the one persistent, conversational agent the user talks to.

Unlike agent/predict/predictor.py's fresh-agent-per-prediction design, one
Agent instance persists for the whole conversation so it remembers earlier
clarification turns. The underlying prediction (via predict_tool) still gets
its own fresh agent internally, so prediction reproducibility is unaffected.
"""

import os
import sys

from agent.config import DEFAULT_MODEL_ID, model_params
from agent.intake.resolver import intake_tool
from agent.orchestrator.prompts import ORCHESTRATOR_SYSTEM_PROMPT
from agent.orchestrator.schema import OrchestratorTurn
from agent.predict.tools import predict_tool
from agent.tool_logging import ToolCallLogger

# Consecutive clarification rounds (done=False) before giving up on one request.
MAX_TURNS = 8

# Total turns overall (clarifications + delivered answers + follow-ups), as a
# safety valve against a runaway loop — not expected to bind in normal use.
MAX_CONVERSATION_TURNS = 50

# Follow-up prompt replies that end the conversation instead of starting a new turn.
EXIT_INPUTS = {"", "quit", "exit", "no", "nope", "nothing", "no thanks", "bye", "done"}


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


def create_orchestrator_agent(model_id=None):
    """Create the one persistent Agent for a conversation."""
    Agent, OpenAIModel = _import_strands()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set (checked .env).")
        sys.exit(1)

    model_id = model_id or DEFAULT_MODEL_ID
    model = OpenAIModel(model_id=model_id, params=model_params(model_id))
    return Agent(
        model=model,
        tools=[intake_tool, predict_tool],
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        hooks=[ToolCallLogger()],
    )


def run_conversation(initial_request, model_id=None, ask_fn=input, agent=None):
    """Drive the conversation to completion, including follow-up questions.

    A delivered answer (done=True) no longer ends the call outright: the
    caller is asked for a follow-up, and anything other than a blank/exit-like
    reply (see EXIT_INPUTS) feeds back in as the next turn on the same
    persistent agent, so it keeps the full conversation history (and gets the
    benefit of prompt caching on that shared prefix). Returns the turn
    transcript covering every round, clarifications and follow-ups alike.

    agent: an injectable stand-in for the real Strands Agent (must support
    `agent(text, structured_output_model=OrchestratorTurn) -> result` where
    `result.structured_output` is an OrchestratorTurn) — used by tests to
    avoid a real LLM call. Defaults to a fresh create_orchestrator_agent().
    """
    agent = agent if agent is not None else create_orchestrator_agent(model_id)
    text = initial_request
    turns = []
    clarification_streak = 0

    for _ in range(MAX_CONVERSATION_TURNS):
        result = agent(text, structured_output_model=OrchestratorTurn)
        turn = result.structured_output
        turns.append({"user": text, "message": turn.message, "done": turn.done})
        print(turn.message)

        if not turn.done:
            clarification_streak += 1
            if clarification_streak >= MAX_TURNS:
                turns.append(
                    {
                        "user": None,
                        "message": f"Sorry, I couldn't resolve this after {MAX_TURNS} turns.",
                        "done": True,
                    }
                )
                print(turns[-1]["message"])
                return turns
            text = ask_fn("> ")
            continue

        clarification_streak = 0
        reply = ask_fn("\nAnything else? (press Enter to finish)\n> ")
        if reply.strip().lower() in EXIT_INPUTS:
            return turns
        text = reply

    return turns
