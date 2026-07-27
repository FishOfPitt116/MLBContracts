"""Comparable-contracts lookup: the predict agent's first Phase 2 tool.

Queries dataset/contracts_spotrac.csv -- the same contracts-only dataset
agent/phase.py already resolves phases from -- joined with dataset/players.csv
for position, so the predict agent can ground comparable-contract reasoning
and a player's own contract history in real records instead of training
memory. Deliberately NOT backed by the stats-joined dataset: stat-line
comparability (WAR, HR, ERA, etc.) is a later Phase 2 tool once there's an
actual live/verified stats source wired in, not this one.

NO LOOKAHEAD: every prediction gets its own tool instance (make_comparable_contracts_tool),
built with the target year baked in and hard-excluding any contract dated that year or
later -- from ANY player, not just the target one. Discovered via a backtest run that
scored a perfect (fabricated) 100%: query_comparable_contracts(player_id=<target>) with no
year filter returns a player's full history, including the exact contract being predicted,
since backtest targets are drawn from already-observed rows in the same CSV the tool reads.
The model wasn't predicting, it was reading the answer out of its own tool call. The cutoff
is enforced in the tool itself, baked in via a per-prediction closure, not a caller-supplied
filter -- the model is never given a parameter that could bypass it.

KNOWN DATASET ISSUE (tracked in docs/PROJECT_STATE.md's Open Issues): Spotrac
never records service_time for free-agent rows -- 100% of them carry the -1
"not tracked" sentinel, not just some (pre-arb/arb are fine: 99.3%/93.8% of
rows have real values). A service_time filter combined with phase="free-agent"
therefore can't ever match anything -- confirmed live: a query for free-agent
SPs aged 29-31 with a service-time bound found nothing, but dropping only the
service-time bound surfaced 161 real matches, some at $22-30M AAV. A plain
docstring warning wasn't enough to prevent that combination -- that's exactly
the mistake that surfaced this -- so phase="free-agent" + a service_time bound
now raises ValueError outright instead of silently/misleadingly returning
zero matches; service_time filtering elsewhere (pre-arb/arb, or no phase
filter) works normally and just excludes untracked rows.
"""

from typing import Optional

import pandas as pd
from strands import tool

from agent.config import CONTRACTS_CSV, PLAYERS_CSV
from agent.service_time import normalize_service_time

# Spotrac's sentinel for "service time not tracked" (free agents)
UNKNOWN_SERVICE_TIME = -1


def _load_contracts():
    return pd.read_csv(CONTRACTS_CSV).drop_duplicates(subset="contract_id")


def _load_players():
    return pd.read_csv(PLAYERS_CSV)


def _query_comparable_contracts(
    contracts_df,
    players_df,
    player_id="",
    position="",
    phase="",
    min_age=None,
    max_age=None,
    min_service_time=None,
    max_service_time=None,
    min_year=None,
    max_year=None,
    exclude_player_id="",
    limit=15,
    before_year=None,
):
    """Pure filtering logic, injectable dataframes for testing.

    service_time filtering only ever matches pre-arb/arb rows -- see the
    module docstring's KNOWN DATASET ISSUE. phase="free-agent" combined with a
    service_time bound is a self-contradictory request (raises ValueError,
    not a silent empty result) since it can never match anything; service_time
    bounds without an explicit free-agent phase just exclude untracked rows.

    before_year: hard cutoff (year < before_year) applied unconditionally,
    on top of whatever the caller's own min_year/max_year say -- the
    no-lookahead guard described in the module docstring. None here means
    "no cutoff," used only by tests; real callers always go through
    make_comparable_contracts_tool(), which always supplies it.
    """
    has_service_time_bound = min_service_time is not None or max_service_time is not None
    if phase == "free-agent" and has_service_time_bound:
        raise ValueError(
            "service_time is never tracked for free-agent contracts (Spotrac doesn't "
            "record it once a player reaches free agency), so phase='free-agent' "
            "combined with min_service_time/max_service_time can never match anything. "
            "Drop the service_time bound and use age/position/phase instead for "
            "free-agent comparables."
        )

    df = contracts_df.merge(
        players_df[["player_id", "first_name", "last_name", "position"]],
        on="player_id",
        how="left",
    )
    df["service_time_normalized"] = df["service_time"].apply(
        lambda st: None if st == UNKNOWN_SERVICE_TIME else normalize_service_time(st)
    )

    mask = pd.Series(True, index=df.index)
    if before_year is not None:
        mask &= df["year"] < before_year
    if player_id:
        mask &= df["player_id"] == player_id
    if exclude_player_id:
        mask &= df["player_id"] != exclude_player_id
    if position:
        mask &= df["position"].str.contains(position, case=False, na=False, regex=False)
    if phase:
        mask &= df["type"] == phase
    if min_age is not None:
        mask &= df["age"] >= min_age
    if max_age is not None:
        mask &= df["age"] <= max_age
    if min_year is not None:
        mask &= df["year"] >= min_year
    if max_year is not None:
        mask &= df["year"] <= max_year

    if has_service_time_bound:
        # Not free-agent-restricted (that case already raised above), but rows
        # with no tracked service time (all free-agent rows, if phase is unset)
        # still can't be meaningfully compared -- exclude rather than error,
        # since this call didn't specifically ask for free-agent rows.
        mask &= df["service_time_normalized"].notna()
        if min_service_time is not None:
            mask &= df["service_time_normalized"] >= min_service_time
        if max_service_time is not None:
            mask &= df["service_time_normalized"] <= max_service_time

    matches = df[mask].sort_values("year", ascending=False)
    total_before_limit = len(matches)
    matches = matches.head(limit)

    records = []
    for _, row in matches.iterrows():
        has_name = pd.notna(row.get("first_name")) and pd.notna(row.get("last_name"))
        records.append(
            {
                "contract_id": row["contract_id"],
                "player_id": row["player_id"],
                "player_name": f"{row['first_name']} {row['last_name']}" if has_name else None,
                "position": row["position"] if pd.notna(row["position"]) else None,
                "age": None if pd.isna(row["age"]) or row["age"] == -1 else int(row["age"]),
                # pandas coerces the None from the lambda above to NaN in a float
                # Series -- normalize back to None here (same pattern as age above).
                "service_time": (
                    None
                    if pd.isna(row["service_time_normalized"])
                    else row["service_time_normalized"]
                ),
                "year": int(row["year"]),
                "duration_years": int(row["duration"]),
                "value_millions": float(row["value"]),
                "aav_millions": round(float(row["value"]) / int(row["duration"]), 4),
                "phase": row["type"],
            }
        )

    return {"matches": records, "n_matches_before_limit": total_before_limit}


def make_comparable_contracts_tool(before_year):
    """Build a query_comparable_contracts tool scoped to one prediction's no-lookahead cutoff.

    before_year is baked into the closure, NOT exposed as a parameter the model
    can set or see -- every prediction (agent/predict/predictor.py:create_agent)
    builds its own tool instance via this factory with the target year, so the
    cutoff can't be bypassed, only every match filtered by it. See the module
    docstring's NO LOOKAHEAD note for why this exists.
    """

    @tool
    def query_comparable_contracts(
        player_id: str = "",
        position: str = "",
        phase: str = "",
        min_age: Optional[float] = None,
        max_age: Optional[float] = None,
        min_service_time: Optional[float] = None,
        max_service_time: Optional[float] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        exclude_player_id: str = "",
        limit: int = 15,
    ) -> dict:
        """Search real historical MLB contracts, filtered on whichever fields you give.

        Use this for TWO things, both required rather than relied on from memory:
        1. The target player's OWN contract history: pass player_id alone (or with a
           year range) to see their actual past deals before projecting the next one.
        2. Comparable OTHER players' contracts: filter by position/phase/age/service
           time (and pass exclude_player_id=the target player's id so they don't show
           up as their own comparable) to ground a projection in real market data
           instead of guessing at what similar players got paid.

        Every result is already restricted to contracts signed before the season
        you're predicting -- you cannot see this season's or any future season's
        deals for ANY player through this tool, by design.

        Not covered here: performance/stat-line comparability (WAR, HR, ERA, etc.) --
        no stats data source is wired into this tool yet, so that judgment still
        relies on your own knowledge for now, per the system prompt.

        Args:
            player_id: Exact player_id (e.g. "Skubal_26337"). Sole filter for "this
                player's own history"; combine with min_year/max_year to narrow it.
            position: Substring match against the position field (e.g. "SP" also
                matches "SP/SP2"; "1B" also matches "1B/3B"). Case-insensitive.
            phase: Exact match: "pre-arb", "arb", or "free-agent".
            min_age / max_age: Player's age in the contract year (inclusive bounds).
            min_service_time / max_service_time: NORMALIZED MLB service time (e.g. 2.5
                = 2 years plus half of a third), NOT raw years.days format. ONLY
                MEANINGFUL FOR pre-arb/arb — Spotrac never tracks service time for
                free agents. Combining either bound with phase="free-agent" RAISES
                AN ERROR (it can never match anything, so this is treated as an
                invalid request, not a silent empty result) — use age/position/phase
                instead when searching free-agent comparables. Without an explicit
                phase="free-agent", a service_time bound just excludes untracked rows.
            min_year / max_year: Contract signing year (inclusive bounds). Dollar
                values are era-relative -- a $10M deal in 2012 is not the same market
                as $10M in 2025, so prefer comparing within a similar year range, or
                explicitly account for era in your reasoning if you don't.
            exclude_player_id: Drop this player_id from results (typically the target
                player, when searching for comparable OTHER players).
            limit: Max rows to return (default 15) -- this is grounding evidence for
                one prediction, not a full data dump; keep it small and relevant.

        Returns:
            {"matches": [{contract_id, player_id, player_name, position, age,
             service_time, year, duration_years, value_millions, aav_millions,
             phase}, ...], "n_matches_before_limit": int}. n_matches_before_limit
            shows whether `limit` actually truncated anything. Raises an error
            instead of returning if phase="free-agent" is combined with a
            service_time bound (see min_service_time above).
        """
        return _query_comparable_contracts(
            _load_contracts(),
            _load_players(),
            player_id=player_id,
            position=position,
            phase=phase,
            min_age=min_age,
            max_age=max_age,
            min_service_time=min_service_time,
            max_service_time=max_service_time,
            min_year=min_year,
            max_year=max_year,
            exclude_player_id=exclude_player_id,
            limit=limit,
            before_year=before_year,
        )

    return query_comparable_contracts
