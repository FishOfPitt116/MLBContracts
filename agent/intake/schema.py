"""Structured output schema for the intake sub-agent.

Intake resolves a free-text request into a concrete (player_id, target_year,
mode) the predict sub-agent can act on, asking for clarification instead of
guessing when something is missing or ambiguous.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class IntakeResult(BaseModel):
    status: Literal["ready", "needs_clarification"] = Field(
        description="'ready' once player_id, target_year, and mode are all resolved."
    )
    player_id: Optional[str] = Field(
        default=None, description="Resolved player_id (from find_player), or None."
    )
    player_name: Optional[str] = Field(
        default=None, description="Human-readable name of the resolved player, or None."
    )
    target_year: Optional[int] = Field(default=None, description="Resolved target season.")
    mode: Literal["predict", "hypothetical_free_agent"] = Field(
        default="predict",
        description=(
            "'predict' resolves the player's actual contract phase for target_year. "
            "'hypothetical_free_agent' is for requests like 'what would they get in "
            "free agency right now', ignoring the player's real current contract status."
        ),
    )
    clarifying_question: Optional[str] = Field(
        default=None,
        description="A single focused question to ask the user, set only when needs_clarification.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Any context worth surfacing (e.g. the phase timeline looked up for this player).",
    )
