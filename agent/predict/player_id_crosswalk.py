"""player_id -> MLBAM id crosswalk, for the live MLB Stats API tools.

Our internal player_id is Spotrac-derived; players.csv already carries a
fangraphs_id per player. The MLB Stats API identifies players by its own
numeric MLBAM id, unrelated to either. pybaseball's Chadwick register bridges
all of these (key_fangraphs, key_mlbam, key_bbref, key_retro) -- reused here
only for that crosswalk, not for pybaseball's actual stats fetch, which is
the part that's broken (see docs/PROJECT_STATE.md's Open Issues and
docs/agent/DESIGN.md Appendix A).

Verified live (July 2026): playerid_lookup('skubal','tarik') returns
key_mlbam=669373, which exactly matches what GET /api/v1/people/search?
names=Tarik Skubal returns from the live API -- confirms the join is sound.

Cached to dataset/mlbam_id_crosswalk.csv (player_id, mlbam_id) after the
first build, since these ids don't change -- avoids re-downloading the full
Chadwick register (all of pro baseball history, a slow one-time fetch) on
every run. Delete the file to force a rebuild (e.g. after players.csv gains
new players not yet in the cached crosswalk).
"""

import pandas as pd

from agent.config import PLAYERS_CSV, REPO_ROOT

CROSSWALK_CSV = REPO_ROOT / "dataset" / "mlbam_id_crosswalk.csv"


def build_crosswalk(players_df, register_df):
    """Join players_df.fangraphs_id -> register_df.key_mlbam. Pure, for testing.

    register_df is shaped like pybaseball.chadwick_register()'s output:
    must have key_fangraphs and key_mlbam columns. Players with no
    fangraphs_id, or no matching register row, get mlbam_id=NaN -- callers
    treat that as "unmapped," not an error.
    """
    register = register_df.dropna(subset=["key_fangraphs"]).copy()
    register["key_fangraphs"] = register["key_fangraphs"].astype(int)
    register = register.drop_duplicates(subset="key_fangraphs")

    merged = players_df.merge(
        register[["key_fangraphs", "key_mlbam"]],
        left_on="fangraphs_id",
        right_on="key_fangraphs",
        how="left",
    )
    crosswalk = merged[["player_id", "key_mlbam"]].rename(columns={"key_mlbam": "mlbam_id"})
    crosswalk["mlbam_id"] = crosswalk["mlbam_id"].astype("Int64")  # nullable int
    return crosswalk


def _load_or_build_crosswalk():
    if CROSSWALK_CSV.exists():
        return pd.read_csv(CROSSWALK_CSV)

    import pybaseball  # deferred: slow first-time register download

    print("Building player_id -> MLBAM id crosswalk (one-time; caches to "
          f"{CROSSWALK_CSV})...")
    players_df = pd.read_csv(PLAYERS_CSV)
    register_df = pybaseball.chadwick_register()
    crosswalk = build_crosswalk(players_df, register_df)
    CROSSWALK_CSV.parent.mkdir(parents=True, exist_ok=True)
    crosswalk.to_csv(CROSSWALK_CSV, index=False)
    return crosswalk


def resolve_mlbam_id(player_id, crosswalk_df=None):
    """player_id -> MLBAM id, or None if unmapped. Loads/builds the cache if needed."""
    crosswalk = crosswalk_df if crosswalk_df is not None else _load_or_build_crosswalk()
    row = crosswalk[crosswalk["player_id"] == player_id]
    if row.empty or pd.isna(row.iloc[0]["mlbam_id"]):
        return None
    return int(row.iloc[0]["mlbam_id"])
