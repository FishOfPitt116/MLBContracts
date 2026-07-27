"""Tools available to the intake sub-agent, plus intake's own exported tool.

find_player and get_contract_phase_timeline are pure lookups (no LLM calls) the
intake agent uses to resolve a request. intake_tool wraps the intake agent
itself so a parent (the orchestrator) can invoke it as a single tool call.
"""

import pandas as pd
from strands import tool

from agent.config import PLAYERS_CSV
from agent.phase import project_phase_timeline


@tool
def find_player(first_name: str = "", last_name: str = "", position: str = "") -> dict:
    """Search players.csv for candidates matching the given fields.

    Each provided field is a case-insensitive filter (first_name/last_name match
    as a substring, position matches exactly); omitted fields are wildcards. At
    least one of first_name/last_name is required. Returns every match found so
    the caller can reason about ambiguity rather than picking one silently.

    Args:
        first_name: Player's first name (or part of it), e.g. "Max".
        last_name: Player's last name (or part of it), e.g. "Scherzer".
        position: Exact position, e.g. "SP", "C", "1B".

    Returns:
        {"matches": [{"player_id", "first_name", "last_name", "position"}, ...]}
    """
    if not first_name and not last_name:
        return {"matches": [], "note": "At least one of first_name or last_name is required."}

    players = pd.read_csv(PLAYERS_CSV)
    mask = pd.Series(True, index=players.index)
    if first_name:
        mask &= players["first_name"].str.contains(first_name, case=False, na=False, regex=False)
    if last_name:
        mask &= players["last_name"].str.contains(last_name, case=False, na=False, regex=False)
    if position:
        mask &= players["position"].str.lower() == position.lower()

    matches = players[mask]
    return {
        "matches": [
            {
                "player_id": row["player_id"],
                "first_name": row["first_name"],
                "last_name": row["last_name"],
                "position": row["position"],
            }
            for _, row in matches.iterrows()
        ]
    }


@tool
def get_contract_phase_timeline(player_id: str) -> dict:
    """Look up which contract phase (pre-arb/arb/free-agent) applies in every future year.

    Args:
        player_id: Spotrac-derived player id, e.g. "Scherzer_5166" (from find_player).

    Returns:
        {"pre-arb": [start,end]|None, "arb": [start,end]|None,
         "free-agent": [start,None]|None (null end = open-ended, indefinite),
         "caveats": [str, ...]}
    """
    return project_phase_timeline(player_id)
