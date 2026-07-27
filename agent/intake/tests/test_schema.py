"""IntakeResult schema validation (no LLM calls)."""

import pytest
from pydantic import ValidationError

from agent.intake.schema import IntakeResult


def test_ready_result_round_trips():
    result = IntakeResult(
        status="ready",
        player_id="Scherzer_5166",
        player_name="Max Scherzer",
        target_year=2026,
        mode="predict",
    )
    assert IntakeResult.model_validate(result.model_dump()) == result


def test_needs_clarification_allows_missing_fields():
    result = IntakeResult(
        status="needs_clarification",
        clarifying_question="Did you mean Max Scherzer or someone else?",
    )
    assert result.player_id is None
    assert result.target_year is None
    assert result.mode == "predict"


def test_hypothetical_free_agent_mode():
    result = IntakeResult(
        status="ready",
        player_id="Burnes_1",
        player_name="Corbin Burnes",
        target_year=2026,
        mode="hypothetical_free_agent",
    )
    assert result.mode == "hypothetical_free_agent"


def test_invalid_status_rejected():
    with pytest.raises(ValidationError):
        IntakeResult(status="not_a_real_status")


def test_invalid_mode_rejected():
    with pytest.raises(ValidationError):
        IntakeResult(status="ready", mode="not_a_real_mode")
