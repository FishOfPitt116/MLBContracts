"""The orchestrator: the one persistent, conversational agent the user talks to.

Unlike agent/predict/predictor.py's fresh-agent-per-prediction design, one
Agent instance persists for the whole conversation so it remembers earlier
clarification turns. The underlying prediction (via predict_tool) still gets
its own fresh agent internally, so prediction reproducibility is unaffected.
"""

import os
import re
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

# Matches a `message` that starts like raw JSON rather than prose.
_JSON_LIKE_RE = re.compile(r"^\s*[\{\[]")

_JSON_LEAK_NUDGE = (
    "Your last reply's `message` looked like raw JSON, not natural language. Rewrite it as a "
    "normal conversational sentence/paragraph for a person to read -- no braces, no key: value "
    "syntax, no code fences -- describing the same information in words."
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


def create_orchestrator_agent(model_id=None, extra_hooks=None):
    """Create the one persistent Agent for a conversation.

    extra_hooks: additional HookProviders appended after the default
    ToolCallLogger (e.g. web/status.py's StatusHook, for UI progress
    reporting) -- optional, so the CLI's behavior is unchanged.
    """
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
        hooks=[ToolCallLogger(), *(extra_hooks or [])],
    )


def get_turn(agent, text, _retried=False):
    """Run one orchestrator turn and return its OrchestratorTurn.

    Guards against a real failure mode observed live (July 2026, gpt-5-mini):
    despite the system prompt's explicit "never show raw JSON" instruction,
    the model occasionally serializes predict_tool's structured result
    straight into `message` instead of paraphrasing it in prose. One
    corrective retry on the same persistent agent (so full context is kept)
    before giving up and returning whatever we got -- prompt text alone
    wasn't reliable enough here, same lesson as the no-lookahead/service_time
    guards elsewhere in this codebase: correctness-critical behavior gets a
    code-level check, not just a prompt request.

    Shared by both the CLI loop (run_conversation, below) and web/server.py's
    one-turn-per-request handler, so the guard applies in both places.
    """
    result = agent(text, structured_output_model=OrchestratorTurn)
    turn = result.structured_output
    if not _retried and _JSON_LIKE_RE.match(turn.message):
        return get_turn(agent, _JSON_LEAK_NUDGE, _retried=True)
    return turn


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
        turn = get_turn(agent, text)
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
