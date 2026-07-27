"""Deterministic pre-arb / arb / free-agent phase resolution.

Resolves a player's contract phase for a target year from the Spotrac
contract history in dataset/contracts_spotrac.csv, without asking the LLM.
For years with an observed contract row, the row's Spotrac-sourced type is
authoritative. For other years the phase is projected from the latest known
service time (+1.0 service year per season).

Known limitation (super-two): each year the top ~22% of 2+ service-time
players qualify for arbitration a year early. Service-time thresholds alone
will classify those players as pre-arb. Acceptable for Phase 0; a richer
eligibility tool can refine this in Phase 1.

Roadmap: this resolver is harness-side in Phase 0 and becomes an agent-facing
tool in Phase 1, so scenario-style requests ("what if he signed for N years")
can flow through the agent with the phase still resolved deterministically.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from agent.config import CONTRACTS_CSV
from agent.service_time import normalize_service_time

# Service time sentinel Spotrac uses when it stops tracking (free agents)
UNKNOWN = -1

# Normalized service-time thresholds
ARB_THRESHOLD = 3.0
FA_THRESHOLD = 6.0

PHASE_PRE_ARB = "pre-arb"
PHASE_ARB = "arb"
PHASE_FA = "free-agent"

_contracts_cache = None


@dataclass
class PhaseResolution:
    player_id: str
    year: int
    phase: str
    method: str  # "observed" (contract row for that year) or "projected"
    service_time_estimate: Optional[float] = None  # normalized (172-day year)
    age_estimate: Optional[int] = None
    notes: list = field(default_factory=list)


def load_contract_history(contracts_csv=None):
    """Load the Spotrac contracts CSV, deduplicated by contract_id."""
    global _contracts_cache
    if contracts_csv is not None:
        df = pd.read_csv(contracts_csv)
        return df.drop_duplicates(subset="contract_id")
    if _contracts_cache is None:
        _contracts_cache = pd.read_csv(CONTRACTS_CSV).drop_duplicates(subset="contract_id")
    return _contracts_cache


def _phase_from_service_time(normalized_st):
    if normalized_st < ARB_THRESHOLD:
        return PHASE_PRE_ARB
    if normalized_st < FA_THRESHOLD:
        return PHASE_ARB
    return PHASE_FA


def _year_range(years):
    """[min, max] of a year collection, or None if empty."""
    if not years:
        return None
    ordered = sorted(years)
    return [ordered[0], ordered[-1]]


def _estimate_age(rows, year):
    """Latest valid observed age, shifted to the target year."""
    valid = rows[rows["age"] != UNKNOWN]
    if valid.empty:
        return None
    latest = valid.iloc[-1]
    return int(latest["age"]) + (year - int(latest["year"]))


def resolve_phase(player_id, year, contracts_df=None):
    """Resolve a player's contract phase for a target year.

    Args:
        player_id: Spotrac-derived player id (e.g. "Scherzer_5166")
        year: Target season
        contracts_df: Optional pre-loaded contracts DataFrame (for tests)

    Returns:
        PhaseResolution

    Raises:
        ValueError: if the player has no contract history at all
    """
    df = contracts_df if contracts_df is not None else load_contract_history()
    rows = (
        df[df["player_id"] == player_id]
        .drop_duplicates(subset="contract_id")
        .sort_values("year")
        .reset_index(drop=True)
    )
    if rows.empty:
        raise ValueError(f"No contract history found for player_id={player_id}")

    notes = []
    age = _estimate_age(rows, year)

    # 1. Observed: a contract row exists for the target year
    exact = rows[rows["year"] == year]
    if not exact.empty:
        row = exact.iloc[0]
        st = row["service_time"]
        st_norm = normalize_service_time(st) if st != UNKNOWN else None
        return PhaseResolution(
            player_id=player_id,
            year=year,
            phase=row["type"],
            method="observed",
            service_time_estimate=st_norm,
            age_estimate=age,
            notes=notes,
        )

    # 2. Mid-contract: target year covered by an earlier multi-year deal
    covering = rows[(rows["year"] < year) & (rows["year"] + rows["duration"] > year)]
    if not covering.empty:
        row = covering.iloc[-1]
        end_year = int(row["year"]) + int(row["duration"]) - 1
        notes.append(
            f"under contract through {end_year} "
            f"({row['contract_id']}: {int(row['duration'])}yr/${row['value']}M {row['type']})"
        )
        st = row["service_time"]
        st_norm = None
        if st != UNKNOWN:
            st_norm = normalize_service_time(st) + (year - int(row["year"]))
        return PhaseResolution(
            player_id=player_id,
            year=year,
            phase=row["type"],
            method="projected",
            service_time_estimate=st_norm,
            age_estimate=age,
            notes=notes,
        )

    prior = rows[rows["year"] < year]
    if prior.empty:
        # Target year predates the recorded career; project backwards
        earliest = rows.iloc[0]
        notes.append(
            f"target year {year} predates first recorded contract ({int(earliest['year'])})"
        )
        st = earliest["service_time"]
        if st != UNKNOWN:
            st_norm = max(0.0, normalize_service_time(st) - (int(earliest["year"]) - year))
            return PhaseResolution(
                player_id=player_id,
                year=year,
                phase=_phase_from_service_time(st_norm),
                method="projected",
                service_time_estimate=st_norm,
                age_estimate=age,
                notes=notes,
            )
        return PhaseResolution(
            player_id=player_id,
            year=year,
            phase=PHASE_PRE_ARB,
            method="projected",
            service_time_estimate=None,
            age_estimate=age,
            notes=notes,
        )

    latest = prior.iloc[-1]

    # 3. Once a free agent, always a free agent
    if latest["type"] == PHASE_FA:
        notes.append(f"latest known contract ({int(latest['year'])}) was free-agent")
        return PhaseResolution(
            player_id=player_id,
            year=year,
            phase=PHASE_FA,
            method="projected",
            service_time_estimate=None,
            age_estimate=age,
            notes=notes,
        )

    # 4. Project service time forward: +1.0 service year per season
    st = latest["service_time"]
    if st == UNKNOWN:
        notes.append(
            f"latest contract ({int(latest['year'])}) has unknown service time; "
            f"falling back to its type '{latest['type']}'"
        )
        return PhaseResolution(
            player_id=player_id,
            year=year,
            phase=latest["type"],
            method="projected",
            service_time_estimate=None,
            age_estimate=age,
            notes=notes,
        )

    st_norm = normalize_service_time(st) + (year - int(latest["year"]))
    phase = _phase_from_service_time(st_norm)
    notes.append(
        f"projected from {int(latest['year'])} service time "
        f"{st} -> {st_norm:.2f} normalized service years"
    )
    if phase == PHASE_PRE_ARB and st_norm >= 2.0:
        notes.append("super-two caveat: player may be arb-eligible a year early")
    return PhaseResolution(
        player_id=player_id,
        year=year,
        phase=phase,
        method="projected",
        service_time_estimate=st_norm,
        age_estimate=age,
        notes=notes,
    )


# Caveats always attached to project_phase_timeline's output — known gaps in the
# simple year-by-year projection, left for future refinement rather than solved now.
TIMELINE_CAVEATS = [
    "does not account for super-two eligibility (may enter arbitration a year early)",
    "assumes uninterrupted active-roster time each season; does not adjust for "
    "minor-league options or injury stints that would pause service-time accrual",
]


def project_phase_timeline(player_id, contracts_df=None, horizon_years=10):
    """Project which years fall in which contract phase, from known history onward.

    Returns a dict of phase -> [start_year, end_year] (end_year is None for an
    open-ended free-agent tail — once a free agent, always a free agent) plus a
    "caveats" list. Phases with no known or projected years are None.

    Known years come straight from contract history (grouped by Spotrac's own
    `type` per row); years beyond the latest known row are projected forward
    with the same +1.0-service-year-per-season heuristic resolve_phase() uses.
    Simple by design — see TIMELINE_CAVEATS for edge cases not handled.
    """
    df = contracts_df if contracts_df is not None else load_contract_history()
    rows = (
        df[df["player_id"] == player_id]
        .drop_duplicates(subset="contract_id")
        .sort_values("year")
        .reset_index(drop=True)
    )
    if rows.empty:
        raise ValueError(f"No contract history found for player_id={player_id}")

    phase_years = {PHASE_PRE_ARB: set(), PHASE_ARB: set(), PHASE_FA: set()}
    for _, row in rows.iterrows():
        phase_years[row["type"]].add(int(row["year"]))

    latest = rows.iloc[-1]

    if latest["type"] == PHASE_FA:
        # Once a free agent, always a free agent - no forward projection needed.
        return {
            PHASE_PRE_ARB: _year_range(phase_years[PHASE_PRE_ARB]),
            PHASE_ARB: _year_range(phase_years[PHASE_ARB]),
            PHASE_FA: [min(phase_years[PHASE_FA]), None],
            "caveats": list(TIMELINE_CAVEATS),
        }

    st = latest["service_time"]
    if st == UNKNOWN:
        return {
            PHASE_PRE_ARB: _year_range(phase_years[PHASE_PRE_ARB]),
            PHASE_ARB: _year_range(phase_years[PHASE_ARB]),
            PHASE_FA: _year_range(phase_years[PHASE_FA]),
            "caveats": [
                *TIMELINE_CAVEATS,
                f"latest known contract ({int(latest['year'])}) has unknown service "
                "time; cannot project a timeline beyond known years",
            ],
        }

    base_year = int(latest["year"])
    base_st = normalize_service_time(st)
    end_of_known = base_year + int(latest["duration"]) - 1

    for offset in range(1, horizon_years + 1):
        year = end_of_known + offset
        st_norm = base_st + (year - base_year)
        phase = _phase_from_service_time(st_norm)
        phase_years[phase].add(year)
        if phase == PHASE_FA:
            break  # once a free agent, always a free agent

    fa_years = phase_years[PHASE_FA]
    return {
        PHASE_PRE_ARB: _year_range(phase_years[PHASE_PRE_ARB]),
        PHASE_ARB: _year_range(phase_years[PHASE_ARB]),
        PHASE_FA: [min(fa_years), None] if fa_years else None,
        "caveats": list(TIMELINE_CAVEATS),
    }
