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


def test_round_trip():
    pred = make_prediction()
    restored = ContractPrediction.model_validate_json(pred.model_dump_json())
    assert restored == pred


def test_requires_at_least_one_citation():
    with pytest.raises(ValidationError):
        make_prediction(citations=[])


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
