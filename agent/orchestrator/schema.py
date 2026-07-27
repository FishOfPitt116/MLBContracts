"""Structured output schema for the orchestrator agent.

Every turn, the orchestrator emits one of these — never raw JSON shown to the
user. `message` is always natural language: either a clarifying question or
the final answer. `done` tells the harness whether to keep the conversation
going or stop.
"""

from pydantic import BaseModel, Field


class OrchestratorTurn(BaseModel):
    message: str = Field(
        description="Natural-language text to show the user — a clarifying question or the final answer."
    )
    done: bool = Field(
        description="False while still gathering info (message is a question). "
        "True once a prediction has been delivered (message is the final answer)."
    )
