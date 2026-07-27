"""Live MLB Stats API tools: query_batting_stats / query_pitching_stats.

The predict agent's stats tools (Phase 2), replacing the earlier plan to
build these off the local batter_stats.csv/pitcher_stats.csv -- those can't
be refreshed (see docs/PROJECT_STATE.md's "Stats Data Collection is Broken"
issue) and are stalled several months behind "now." statsapi.mlb.com is live,
free, and unauthenticated -- verified directly (July 2026):

    GET /api/v1/people/{mlbamId}/stats?stats=yearByYear&group=hitting|pitching

returns every season for a player in one call, including the current
in-progress one. Confirmed it does NOT provide FanGraphs-proprietary metrics
(WAR, wRC+, FIP, xFIP, SIERA, Statcast rates) -- matches what DESIGN.md
already expected; that judgment still relies on model knowledge until (if
ever) a FanGraphs-backed source is wired in separately. Rate stats (avg, era,
whip, ...) come back from the API as strings ("*.313*"), coerced to float
here. Pitcher-group responses can carry stray batting fields for pitchers
who've batted (interleague/NL) -- ignored by only extracting the curated
pitching field list below, not everything the API returns.

Two separate tools, not one combined one, specifically so a two-way player
(Ohtani) can be looked up under both without a merged/ambiguous schema.

Player identity: resolved via agent/predict/player_id_crosswalk.py (our
existing fangraphs_id -> a pybaseball-Chadwick-backed MLBAM id), not a live
name search per call.

No lookahead: same closure-based design as comparables.py -- every prediction
gets its own tool instances via make_batting_stats_tool/make_pitching_stats_tool
with the target year baked in as a hard season < before_year cutoff, enforced
in our own code after the API responds, invisible to and un-overridable by
the model.

Retries: transient failures (timeouts, connection errors, 5xx, 429) retry
with exponential backoff (3 attempts: 0.5s/1s/2s) before raising; a
definitive client error (e.g. 404) raises immediately, no wasted retries.
Either way the model sees a real tool failure, not a silent empty result --
same principle as the free-agent+service_time guard in comparables.py.
"""

import time
from typing import Optional

import requests
from strands import tool

from agent.predict.player_id_crosswalk import resolve_mlbam_id

STATS_API_BASE = "https://statsapi.mlb.com/api/v1"
REQUEST_TIMEOUT_SECONDS = 10
MAX_HTTP_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 0.5

# Curated fields, not everything the API returns -- these are the standard
# counting/rate stats the old local pipeline consumed too (see CLAUDE.md).
BATTING_FIELDS = [
    "gamesPlayed", "plateAppearances", "atBats", "hits", "doubles", "triples",
    "homeRuns", "runs", "rbi", "stolenBases", "baseOnBalls", "strikeOuts",
    "avg", "obp", "slg", "ops",
]
PITCHING_FIELDS = [
    "gamesPlayed", "gamesStarted", "inningsPitched", "wins", "losses",
    "saves", "holds", "strikeOuts", "baseOnBalls", "era", "whip",
    "strikeoutsPer9Inn", "walksPer9Inn",
]


def _is_transient(error):
    if isinstance(error, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(error, requests.HTTPError):
        status = error.response.status_code if error.response is not None else None
        return status is not None and (status >= 500 or status == 429)
    return False


def _get_json(url, params):
    """GET with retry+backoff on transient failures; raises immediately otherwise."""
    last_error = None
    for attempt in range(MAX_HTTP_ATTEMPTS):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
                headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            last_error = error
            if not _is_transient(error) or attempt == MAX_HTTP_ATTEMPTS - 1:
                raise RuntimeError(f"MLB Stats API request failed: {error}") from error
            time.sleep(BACKOFF_BASE_SECONDS * (2**attempt))
    raise RuntimeError(f"MLB Stats API request failed: {last_error}") from last_error  # pragma: no cover


def _coerce(value):
    """API rate stats come back as strings (e.g. ".313"); numbers everywhere else."""
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _fetch_year_by_year(mlbam_id, group, fields, before_year, min_year, max_year):
    data = _get_json(
        f"{STATS_API_BASE}/people/{mlbam_id}/stats",
        params={"stats": "yearByYear", "group": group},
    )
    stats_blocks = data.get("stats", [])
    if not stats_blocks:
        return []

    rows = []
    for split in stats_blocks[0].get("splits", []):
        year = int(split["season"])
        if before_year is not None and year >= before_year:
            continue
        if min_year is not None and year < min_year:
            continue
        if max_year is not None and year > max_year:
            continue
        stat = split.get("stat", {})
        row = {"year": year, "team": (split.get("team") or {}).get("name")}
        for field in fields:
            row[field] = _coerce(stat.get(field))
        rows.append(row)
    return sorted(rows, key=lambda r: r["year"])


def _query_stats(player_ids, group, fields, before_year, min_year, max_year):
    """Shared lookup logic for both tools. Pure aside from the network call,
    which _fetch_year_by_year/_get_json own -- tests monkeypatch those."""
    stats = {}
    unmapped = []
    for player_id in player_ids:
        mlbam_id = resolve_mlbam_id(player_id)
        if mlbam_id is None:
            unmapped.append(player_id)
            continue
        stats[player_id] = _fetch_year_by_year(mlbam_id, group, fields, before_year, min_year, max_year)
    result = {"stats": stats}
    if unmapped:
        result["unmapped_player_ids"] = unmapped
    return result


def make_batting_stats_tool(before_year):
    """Build a query_batting_stats tool scoped to one prediction's no-lookahead cutoff.

    Verified live (July 2026): @tool reads the docstring at decoration time to
    build the schema description sent to the model -- assigning __doc__
    afterward on a shared inner function does NOT reach it (tested directly;
    the model would see a useless one-word fallback). So batting and pitching
    each get their own fully-written @tool function here, not a shared factory.
    """

    @tool
    def query_batting_stats(
        player_ids: list, min_year: Optional[int] = None, max_year: Optional[int] = None
    ) -> dict:
        """Look up real year-by-year MLB batting stats for one or more players.

        Use this for the target player (if they hit) AND for any comparable
        players surfaced by query_comparable_contracts -- a "comparable
        contract" is only meaningful if grounded in comparable PERFORMANCE
        too, not just similar age/position/service time. Pass multiple
        player_ids in one call to compare several players at once. For a
        two-way player (e.g. Shohei Ohtani), call this AND query_pitching_stats
        -- they're separate tools specifically so neither schema has to guess
        which fields apply.

        Every result is already restricted to seasons before the one you're
        predicting -- you cannot see this season's or any future season's
        stats for ANY player through this tool, by design.

        Live data (statsapi.mlb.com), not a static dataset -- current-season
        numbers are real. Does NOT include WAR, wRC+, or Statcast rates
        (HardHit%, Barrel%) -- those are FanGraphs-proprietary and unavailable
        here; that judgment still relies on your own knowledge.

        Args:
            player_ids: One or more player_id values (e.g. ["Skubal_26337"]).
            min_year: Earliest season to include (inclusive). Omit for full
                available history up to the no-lookahead cutoff.
            max_year: Latest season to include (inclusive).

        Returns:
            {"stats": {player_id: [{year, team, gamesPlayed, plateAppearances,
             atBats, hits, doubles, triples, homeRuns, runs, rbi, stolenBases,
             baseOnBalls, strikeOuts, avg, obp, slg, ops}, ...]}}. A player_id
            this tool can't map to a live MLB player id (no fangraphs_id on
            record, or not found in the id crosswalk) appears in a separate
            top-level "unmapped_player_ids" list instead of "stats" -- that's
            a real gap, not "the player had no stats."
        """
        return _query_stats(player_ids, "hitting", BATTING_FIELDS, before_year, min_year, max_year)

    return query_batting_stats


def make_pitching_stats_tool(before_year):
    """Build a query_pitching_stats tool scoped to one prediction's no-lookahead cutoff.

    See make_batting_stats_tool's docstring for why this is a fully separate
    @tool function rather than sharing one via a factory.
    """

    @tool
    def query_pitching_stats(
        player_ids: list, min_year: Optional[int] = None, max_year: Optional[int] = None
    ) -> dict:
        """Look up real year-by-year MLB pitching stats for one or more players.

        Use this for the target player (if they pitch) AND for any comparable
        players surfaced by query_comparable_contracts -- a "comparable
        contract" is only meaningful if grounded in comparable PERFORMANCE
        too, not just similar age/position/service time. Pass multiple
        player_ids in one call to compare several players at once. For a
        two-way player (e.g. Shohei Ohtani), call this AND query_batting_stats
        -- they're separate tools specifically so neither schema has to guess
        which fields apply.

        Every result is already restricted to seasons before the one you're
        predicting -- you cannot see this season's or any future season's
        stats for ANY player through this tool, by design.

        Live data (statsapi.mlb.com), not a static dataset -- current-season
        numbers are real. Does NOT include WAR, FIP, xFIP, or SIERA -- those
        are FanGraphs-proprietary and unavailable here; that judgment still
        relies on your own knowledge.

        Args:
            player_ids: One or more player_id values (e.g. ["Skubal_26337"]).
            min_year: Earliest season to include (inclusive). Omit for full
                available history up to the no-lookahead cutoff.
            max_year: Latest season to include (inclusive).

        Returns:
            {"stats": {player_id: [{year, team, gamesPlayed, gamesStarted,
             inningsPitched, wins, losses, saves, holds, strikeOuts,
             baseOnBalls, era, whip, strikeoutsPer9Inn, walksPer9Inn}, ...]}}.
            A player_id this tool can't map to a live MLB player id (no
            fangraphs_id on record, or not found in the id crosswalk) appears
            in a separate top-level "unmapped_player_ids" list instead of
            "stats" -- that's a real gap, not "the player had no stats."
        """
        return _query_stats(player_ids, "pitching", PITCHING_FIELDS, before_year, min_year, max_year)

    return query_pitching_stats
