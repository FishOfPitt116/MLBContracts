"""predict's exported tool interface, for a parent (the orchestrator) to invoke."""

from strands import tool

from agent.predict.predict import predict_for


@tool
def predict_tool(player_id: str, year: int, mode: str = "predict") -> dict:
    """Predict an MLB player's contract for a season and return a summary.

    Runs the full Phase 0 prediction pipeline (deterministic phase resolution,
    LLM prediction with citations, trace + history persistence) and returns
    just the headline figures.

    Args:
        player_id: Resolved player id, e.g. "Scherzer_5166" (from intake_tool).
        year: Target season.
        mode: "predict" resolves the player's actual contract phase for `year`.
            "hypothetical_free_agent" ignores current contract status and
            forces a free-agent valuation.

    Returns:
        {aav_millions, duration_years, total_value_millions, aav_low_millions,
         aav_high_millions, confidence, reasoning, trace_path}
    """
    return predict_for(player_id, year, mode=mode)
