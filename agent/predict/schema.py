"""Structured output schema for contract predictions.

The citation schema is designed to evolve across phases:
- Phase 0 (LLM-only): every citation has source_type="model_knowledge" with a
  free-text basis describing what the model believes it knows and from when.
- Phase 1+ (tools): citations gain source_type="tool" with tool_name and a
  tool_call_ref pointing into the trace messages, so every figure can be
  traced back to a specific tool result.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

# Predicted total is allowed to drift from aav * duration by this fraction
# before we flag it (LLMs get arithmetic wrong; we record, not reject).
ARITHMETIC_TOLERANCE = 0.02


class Citation(BaseModel):
    source_type: Literal["model_knowledge", "tool"] = Field(
        description="Where this claim comes from. Phase 0 agents only emit model_knowledge."
    )
    claim: str = Field(description="The specific figure or fact being supported.")
    basis: str = Field(
        description=(
            "For model_knowledge: what the model believes it knows and from when "
            "(e.g. 'CBA league minimum was $740K in 2024, per 2022 CBA schedule'). "
            "For tool: a summary of the tool result relied on."
        )
    )
    tool_name: Optional[str] = Field(
        default=None, description="Phase 1+: name of the tool that produced the evidence."
    )
    tool_call_ref: Optional[str] = Field(
        default=None, description="Phase 1+: reference to the tool call in the run trace."
    )


class ContractPrediction(BaseModel):
    no_contract: bool = Field(
        default=False,
        description=(
            "True if you believe the player will have NO MLB contract at all for the target "
            "season (retired, released and off any MLB roster, non-tendered and not re-signed, "
            "out of affiliated baseball, etc.). This is a real, valid outcome, not a fallback for "
            "uncertainty — if you're merely unsure, predict a number with low confidence instead. "
            "When True, leave aav_millions/duration_years/total_value_millions/aav_low_millions/"
            "aav_high_millions unset rather than inventing a placeholder like 0."
        ),
    )
    aav_millions: Optional[float] = Field(
        default=None,
        gt=0,
        description="Predicted average annual value in millions of USD. Required unless no_contract=True.",
    )
    duration_years: Optional[int] = Field(
        default=None,
        ge=1,
        description="Predicted contract length in years (1 for pre-arb/arb). Required unless no_contract=True.",
    )
    total_value_millions: Optional[float] = Field(
        default=None,
        gt=0,
        description="Predicted total contract value in millions of USD. Required unless no_contract=True.",
    )
    aav_low_millions: Optional[float] = Field(
        default=None,
        description="Low end of the plausible AAV range, millions of USD. Required unless no_contract=True.",
    )
    aav_high_millions: Optional[float] = Field(
        default=None,
        description="High end of the plausible AAV range, millions of USD. Required unless no_contract=True.",
    )
    reasoning: str = Field(description="Concise explanation of how the prediction was reached.")
    citations: list[Citation] = Field(
        min_length=1,
        description="One citation per material figure used (salary anchors, comparables, stats).",
    )
    confidence: Literal["low", "medium", "high"]

    @model_validator(mode="after")
    def _check_no_contract_consistency(self):
        """Enforce the sum-type split: either a real prediction, or no_contract, never both/neither.

        A flat schema can't express "one of two shapes" natively, so no_contract is a
        discriminator flag instead — this validator is what actually enforces the split,
        since Optional alone would let a confused mix of "no_contract=True but also gave
        numbers" or "no_contract=False but left numbers unset" through silently.
        """
        numeric_fields = (
            self.aav_millions,
            self.duration_years,
            self.total_value_millions,
            self.aav_low_millions,
            self.aav_high_millions,
        )
        if self.no_contract:
            if any(value is not None for value in numeric_fields):
                raise ValueError(
                    "no_contract=True must leave aav/duration/total/range fields unset"
                )
        elif any(value is None for value in numeric_fields):
            raise ValueError(
                "aav_millions/duration_years/total_value_millions/aav_low_millions/"
                "aav_high_millions are all required unless no_contract=True"
            )
        return self

    def arithmetic_note(self):
        """Return a note if total value disagrees with aav * duration, else None.

        Recorded in the trace rather than raised: rejecting the whole
        prediction over a rounding slip would lose the rest of the output.
        Always None for no_contract=True — there's nothing to check.
        """
        if self.no_contract:
            return None
        expected = self.aav_millions * self.duration_years
        if expected == 0:
            return None
        drift = abs(self.total_value_millions - expected) / abs(expected)
        if drift > ARITHMETIC_TOLERANCE:
            return (
                f"total_value_millions={self.total_value_millions} disagrees with "
                f"aav*duration={expected:.3f} (drift {drift:.1%})"
            )
        return None
