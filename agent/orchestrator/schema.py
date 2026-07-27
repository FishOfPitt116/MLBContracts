"""Structured output schema for the orchestrator agent.

Every turn, the orchestrator emits one of these — never raw JSON shown to the
user. `message` is always natural language: either a clarifying question or
an answer. `done` tells the harness whether this turn's answer is complete,
not whether the conversation itself is over — a follow-up question can always
come next (see agent/orchestrator/agent.py:run_conversation), answered on the
same persistent agent with the prior turns still in context.
"""

from pydantic import BaseModel, Field


class OrchestratorTurn(BaseModel):
    message: str = Field(
        description="Natural-language text to show the user — a clarifying question or an answer."
    )
    done: bool = Field(
        description="False while still gathering info for the current request (message is a "
        "clarifying question). True once this turn's answer is complete (message is the answer) — "
        "does not end the conversation; a follow-up may still come next."
    )
