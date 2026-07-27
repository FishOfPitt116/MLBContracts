"""Tests for _attach_contract_status (no LLM calls).

Contract-status routing is deterministic and harness-side, same principle as
resolve_phase() itself: attached right after intake resolves player+year, so
the orchestrator can skip predict_tool entirely for an already-known year.
"""

from agent.intake.resolver import _attach_contract_status
from agent.intake.schema import IntakeResult


def _ready(
    target_year,
    mode="predict",
    player_id="Scherzer_5166",
    player_name="Max Scherzer",
    year_was_defaulted=False,
    wants_forecast=True,
):
    return IntakeResult(
        status="ready",
        player_id=player_id,
        player_name=player_name,
        target_year=target_year,
        mode=mode,
        year_was_defaulted=year_was_defaulted,
        wants_forecast=wants_forecast,
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
        "start_year": 2015,
        "end_year": 2021,
    }


def test_stable_long_term_deal_is_not_advanced_even_when_defaulted_and_forecast_wanted():
    # Scherzer's 2015 deal still has years left after 2016 -> nothing to project yet,
    # the known deal itself IS the right answer even for a "projected contract" ask.
    result = _attach_contract_status(_ready(2016, year_was_defaulted=True, wants_forecast=True))
    assert result.target_year == 2016
    assert result.contract_status == "known"
    assert result.prior_known_contract is None


def test_genuine_future_year_is_forecast():
    result = _attach_contract_status(_ready(2030))
    assert result.contract_status == "forecast"
    assert result.known_contract is None


def test_expiring_deal_advances_to_a_real_projection_when_defaulted_and_forecast_wanted():
    # Skubal's 2026 arb deal ends after 2026; his service time crosses into
    # free agency in 2027, a genuine forecast year.
    result = _attach_contract_status(
        _ready(
            2026,
            player_id="Skubal_26337",
            player_name="Tarik Skubal",
            year_was_defaulted=True,
            wants_forecast=True,
        )
    )
    assert result.target_year == 2027
    assert result.contract_status == "forecast"
    assert result.known_contract is None
    assert result.prior_known_contract["contract_id"] == "Skubal_26337_2026"
    assert result.prior_known_contract["aav_millions"] == 32.0
    assert any("2027" in n for n in result.notes)


def test_expiring_deal_not_advanced_when_year_was_explicit():
    # The user explicitly asked about 2026 -> honor it exactly, no rollforward.
    result = _attach_contract_status(
        _ready(
            2026,
            player_id="Skubal_26337",
            player_name="Tarik Skubal",
            year_was_defaulted=False,
            wants_forecast=True,
        )
    )
    assert result.target_year == 2026
    assert result.contract_status == "known"
    assert result.prior_known_contract is None


def test_expiring_deal_not_advanced_when_current_status_was_wanted():
    # Defaulted year, but the request wasn't asking for a projection at all.
    result = _attach_contract_status(
        _ready(
            2026,
            player_id="Skubal_26337",
            player_name="Tarik Skubal",
            year_was_defaulted=True,
            wants_forecast=False,
        )
    )
    assert result.target_year == 2026
    assert result.contract_status == "known"
    assert result.prior_known_contract is None


def test_hypothetical_free_agent_mode_is_left_untouched():
    result = _attach_contract_status(_ready(2026, mode="hypothetical_free_agent"))
    assert result.contract_status is None
    assert result.known_contract is None


def test_needs_clarification_is_left_untouched():
    result = IntakeResult(status="needs_clarification", clarifying_question="Which year?")
    attached = _attach_contract_status(result)
    assert attached.contract_status is None
    assert attached == result
