import numpy as np
import pandas as pd

from agent.service_time import normalize_service_time
from data_generation.player_lookup import get_player_by_id
from data_generation.save import read_batter_stats, read_pitcher_stats
from models.arb.config import (
    ARB_3,
    ARB_BAND_THRESHOLDS,
    CONTRACT_TYPE,
    CONTRACTS_WITH_STATS_CSV,
    EXCLUDED_PLAYER_IDS,
    MAX_DURATION,
    SP_SPLIT_THRESHOLD,
    STATS_WINDOW_YEARS,
)

_training_frame_cache = None


def arb_band(normalized_service_time):
    if normalized_service_time is None or pd.isna(normalized_service_time):
        return None
    for threshold, band in ARB_BAND_THRESHOLDS:
        if normalized_service_time < threshold:
            return band
    return ARB_3


def _classify_role(position, pit_games, pit_games_started):
    is_pitcher = position.str.contains("SP|RP", regex=True) | (position == "P")
    label_is_sp = position.str.contains("SP", regex=True)
    label_is_rp = position.str.contains("RP", regex=True) & ~label_is_sp
    gs_ratio = pit_games_started / pit_games.replace(0, np.nan)
    has_workload = gs_ratio.notna()
    workload_sp = gs_ratio >= SP_SPLIT_THRESHOLD
    role = np.select(
        [
            ~is_pitcher,
            is_pitcher & has_workload & workload_sp,
            is_pitcher & has_workload & ~workload_sp,
            is_pitcher & ~has_workload & label_is_sp,
            is_pitcher & ~has_workload & label_is_rp,
        ],
        ["batter", "SP", "RP", "SP", "RP"],
        default="P(other)",
    )
    return role, is_pitcher


def _compute_training_frame(df):
    role, is_pitcher = _classify_role(df["position"], df["pit_games_3y"], df["pit_games_started_3y"])
    df = df.assign(role=role)

    mask = (
        (df["contract_type"] == CONTRACT_TYPE)
        & (df["duration"] == MAX_DURATION)
        & ~df["player_id"].isin(EXCLUDED_PLAYER_IDS)
        & (df["role"] != "P(other)")
    )
    broken_stats = ((~is_pitcher) & (df["bat_games_3y"] > 0) & (df["bat_plate_appearances_3y"] == 0)) | (
        is_pitcher & (df["pit_games_3y"] > 0) & (df["pit_innings_pitched_3y"] == 0)
    )
    frame = df[mask & ~broken_stats].copy()
    row_is_pitcher = is_pitcher.loc[frame.index]

    frame["war_3y"] = frame["pit_war_3y"].where(row_is_pitcher, frame["bat_war_3y"])
    frame["volume_3y"] = frame["pit_innings_pitched_3y"].where(row_is_pitcher, frame["bat_plate_appearances_3y"])
    frame["service_time"] = frame["service_time"].apply(normalize_service_time)
    frame["war_x_service"] = frame["war_3y"] * frame["service_time"]
    frame["log_aav"] = np.log(frame["value"] / frame["duration"])
    return frame[
        ["player_id", "contract_year", "role", "log_aav", "war_3y", "volume_3y", "service_time", "war_x_service"]
    ]


def build_training_frame(contracts_with_stats_df=None):
    if contracts_with_stats_df is not None:
        return _compute_training_frame(contracts_with_stats_df)
    global _training_frame_cache
    if _training_frame_cache is None:
        _training_frame_cache = _compute_training_frame(pd.read_csv(CONTRACTS_WITH_STATS_CSV))
    return _training_frame_cache


def player_year_features(player_id, year, batter_stats=None, pitcher_stats=None, player=None):
    player = player if player is not None else get_player_by_id(player_id)
    if player is None:
        raise ValueError(f"Unknown player_id={player_id!r}")
    if not player.position:
        raise ValueError(f"player_id={player_id!r} has no position on record")

    position = player.position
    is_pitcher = ("SP" in position) or ("RP" in position) or (position == "P")
    label_is_sp = "SP" in position
    label_is_rp = ("RP" in position) and not label_is_sp
    stats_year = year - 1

    if is_pitcher:
        rows = pitcher_stats if pitcher_stats is not None else read_pitcher_stats()
        match = next(
            (
                s
                for s in rows
                if s.player_id == player_id and s.year == stats_year and s.window_years == STATS_WINDOW_YEARS
            ),
            None,
        )
        if match is None:
            raise ValueError(f"No pitcher stats for player_id={player_id!r}, stats_year={stats_year}, window=3")
        if match.games_started is None or not match.games:
            role = "SP" if label_is_sp else ("RP" if label_is_rp else None)
            if role is None:
                raise ValueError(f"Cannot resolve pitcher role for player_id={player_id!r}")
        else:
            gs_ratio = match.games_started / match.games
            role = "SP" if gs_ratio >= SP_SPLIT_THRESHOLD else "RP"
        return {"role": role, "war_3y": match.war, "volume_3y": match.innings_pitched}

    rows = batter_stats if batter_stats is not None else read_batter_stats()
    match = next(
        (s for s in rows if s.player_id == player_id and s.year == stats_year and s.window_years == STATS_WINDOW_YEARS),
        None,
    )
    if match is None:
        raise ValueError(f"No batter stats for player_id={player_id!r}, stats_year={stats_year}, window=3")
    return {"role": "batter", "war_3y": match.war, "volume_3y": match.plate_appearances}
