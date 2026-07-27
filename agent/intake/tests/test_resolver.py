"""Tests for _attach_contract_status (no LLM calls).

Contract-status routing is deterministic and harness-side, same principle as
resolve_phase() itself: attached right after intake resolves player+year, so
the orchestrator can skip predict_tool entirely for an already-known year.
"""

from agent.intake.resolver import _attach_contract_status
from agent.intake.schema import IntakeResult


def _ready(target_year, mode="predict"):
    return IntakeResult(
        status="ready",
        player_id="Scherzer_5166",
        player_name="Max Scherzer",
        target_year=target_year,
        mode=mode,
    )


def test_year_covered_by_a_signed_deal_is_known():
    # 2016 falls inside Scherzer's 2015 7yr/$210M deal
    result = _attach_contract_status(_ready(2016))
    assert result.contract_status == "known"
    assert result.known_contract == {
        "contract_id": "Scherzer_5166_2015",
        "aav_millions": 30.0,
        "duration_years": 7,
        "total_value_millions": 210.0,
    }


def test_genuine_future_year_is_forecast():
    result = _attach_contract_status(_ready(2030))
    assert result.contract_status == "forecast"
    assert result.known_contract is None


def test_hypothetical_free_agent_mode_is_left_untouched():
    result = _attach_contract_status(_ready(2026, mode="hypothetical_free_agent"))
    assert result.contract_status is None
    assert result.known_contract is None


def test_needs_clarification_is_left_untouched():
    result = IntakeResult(status="needs_clarification", clarifying_question="Which year?")
    attached = _attach_contract_status(result)
    assert attached.contract_status is None
    assert attached == result
