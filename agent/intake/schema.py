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
    year_was_defaulted: bool = Field(
        default=False,
        description=(
            "True when you filled in target_year yourself (current season, or a phase-"
            "milestone year) rather than the user stating one explicitly. False when the "
            "user gave an explicit year — always honor their explicit choice as-is."
        ),
    )
    wants_forecast: bool = Field(
        default=True,
        description=(
            "True (the default) unless the request is clearly asking about the player's "
            "actual/current/already-signed status rather than a future projection — e.g. "
            "'what is his current salary', 'how much is he making this year', 'what's he "
            "actually signed for'. Leave True for anything phrased as a prediction/forecast "
            "('projected contract', 'what will he get', 'predict his next deal'), which is "
            "the common case for this assistant."
        ),
    )
    contract_status: Optional[Literal["known", "forecast"]] = Field(
        default=None,
        description=(
            "Set by the harness after the LLM call, not by the model — leave null. "
            "'known' means target_year's contract is already on record (a signed deal "
            "covers it); 'forecast' means it genuinely needs a prediction. Only set when "
            "status='ready' and mode='predict'."
        ),
    )
    known_contract: Optional[dict] = Field(
        default=None,
        description=(
            "Set by the harness, not by the model — leave null. When contract_status="
            "'known': {contract_id, aav_millions, duration_years, total_value_millions, "
            "start_year, end_year} for the deal that covers target_year."
        ),
    )
    prior_known_contract: Optional[dict] = Field(
        default=None,
        description=(
            "Set by the harness, not by the model — leave null. Populated when a defaulted, "
            "forecast-seeking target_year got auto-advanced past an already-expiring known "
            "deal (see year_was_defaulted/wants_forecast): the on-record figure for the year "
            "just before target_year, so the assistant can mention it alongside the "
            "projection instead of only giving one or the other."
        ),
    )
