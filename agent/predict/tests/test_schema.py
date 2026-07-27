"""Tests for the prediction/citation schema."""

import pytest
from pydantic import ValidationError

from agent.predict.schema import Citation, ContractPrediction


def make_prediction(**overrides):
    fields = dict(
        aav_millions=10.0,
        duration_years=3,
        total_value_millions=30.0,
        aav_low_millions=8.0,
        aav_high_millions=12.0,
        reasoning="test",
        citations=[
            Citation(
                source_type="model_knowledge",
                claim="AAV around $10M",
                basis="comparable mid-rotation starters signed 2023-2025",
            )
        ],
        confidence="medium",
    )
    fields.update(overrides)
    return ContractPrediction(**fields)


def make_no_contract_prediction(**overrides):
    fields = dict(
        no_contract=True,
        reasoning="Player has not appeared on an MLB roster since 2011 per model knowledge.",
        citations=[
            Citation(
                source_type="model_knowledge",
                claim="No MLB appearances since 2011",
                basis="model training knowledge of player career timeline",
            )
        ],
        confidence="low",
    )
    fields.update(overrides)
    return ContractPrediction(**fields)


def test_round_trip():
    pred = make_prediction()
    restored = ContractPrediction.model_validate_json(pred.model_dump_json())
    assert restored == pred


def test_requires_at_least_one_citation():
    with pytest.raises(ValidationError):
        make_prediction(citations=[])


def test_rejects_zero_duration():
    # A real backtest run once returned a degenerate 0yr/$0M "contract" for a
    # real player (Hinske_852_2013) that slipped through with no constraint.
    with pytest.raises(ValidationError):
        make_prediction(duration_years=0, aav_millions=0.0, total_value_millions=0.0)


def test_rejects_non_positive_aav():
    with pytest.raises(ValidationError):
        make_prediction(aav_millions=0.0)


def test_rejects_non_positive_total_value():
    with pytest.raises(ValidationError):
        make_prediction(total_value_millions=0.0)


def test_tool_citation_fields_optional_in_phase_0():
    citation = Citation(
        source_type="model_knowledge", claim="x", basis="y"
    )
    assert citation.tool_name is None
    assert citation.tool_call_ref is None


def test_arithmetic_note_flags_drift():
    pred = make_prediction(total_value_millions=45.0)  # 10 * 3 = 30 expected
    note = pred.arithmetic_note()
    assert note is not None
    assert "disagrees" in note


def test_arithmetic_note_passes_consistent_total():
    assert make_prediction().arithmetic_note() is None


def test_no_contract_is_valid_with_all_numeric_fields_unset():
    pred = make_no_contract_prediction()
    assert pred.no_contract is True
    assert pred.aav_millions is None
    assert pred.duration_years is None
    assert pred.total_value_millions is None
    assert pred.aav_low_millions is None
    assert pred.aav_high_millions is None


def test_no_contract_arithmetic_note_is_none():
    # Nothing to check arithmetic-wise when there's no predicted contract at all.
    assert make_no_contract_prediction().arithmetic_note() is None


def test_no_contract_rejects_numeric_fields_present():
    # no_contract=True must not also carry a placeholder/real number -- one or the
    # other, never both (this is exactly the Hinske_852_2013 failure mode: a
    # confused 0/0/0 "prediction" that looked like a real, if degenerate, answer).
    with pytest.raises(ValidationError):
        make_no_contract_prediction(aav_millions=0.5)


def test_missing_numeric_fields_rejected_when_not_no_contract():
    # The inverse: a "real" prediction (no_contract left False) can't leave the
    # numeric fields unset either -- the sum type has to pick one shape.
    with pytest.raises(ValidationError):
        ContractPrediction(
            reasoning="test",
            citations=[
                Citation(source_type="model_knowledge", claim="x", basis="y")
            ],
            confidence="low",
        )
