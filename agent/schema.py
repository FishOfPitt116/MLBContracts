"""Structured output schema for contract predictions.

The citation schema is designed to evolve across phases:
- Phase 0 (LLM-only): every citation has source_type="model_knowledge" with a
  free-text basis describing what the model believes it knows and from when.
- Phase 1+ (tools): citations gain source_type="tool" with tool_name and a
  tool_call_ref pointing into the trace messages, so every figure can be
  traced back to a specific tool result.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

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
    aav_millions: float = Field(description="Predicted average annual value in millions of USD.")
    duration_years: int = Field(description="Predicted contract length in years (1 for pre-arb/arb).")
    total_value_millions: float = Field(description="Predicted total contract value in millions of USD.")
    aav_low_millions: float = Field(description="Low end of the plausible AAV range, millions of USD.")
    aav_high_millions: float = Field(description="High end of the plausible AAV range, millions of USD.")
    reasoning: str = Field(description="Concise explanation of how the prediction was reached.")
    citations: list[Citation] = Field(
        min_length=1,
        description="One citation per material figure used (salary anchors, comparables, stats).",
    )
    confidence: Literal["low", "medium", "high"]

    def arithmetic_note(self):
        """Return a note if total value disagrees with aav * duration, else None.

        Recorded in the trace rather than raised: rejecting the whole
        prediction over a rounding slip would lose the rest of the output.
        """
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
