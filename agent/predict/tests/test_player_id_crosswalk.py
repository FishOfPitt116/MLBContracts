"""Tests for the player_id -> MLBAM id crosswalk (no pybaseball/network calls)."""

import numpy as np
import pandas as pd

from agent.predict.player_id_crosswalk import build_crosswalk, resolve_mlbam_id


def _players():
    return pd.DataFrame(
        [
            {"player_id": "Skubal_26337", "fangraphs_id": 22267, "first_name": "Tarik", "last_name": "Skubal"},
            {"player_id": "Scherzer_5166", "fangraphs_id": 3137, "first_name": "Max", "last_name": "Scherzer"},
            {"player_id": "NoFgId_1", "fangraphs_id": np.nan, "first_name": "No", "last_name": "Id"},
            {"player_id": "NotInRegister_2", "fangraphs_id": 99999, "first_name": "Not", "last_name": "Found"},
        ]
    )


def _register():
    # Shaped like pybaseball.chadwick_register()'s real output columns.
    return pd.DataFrame(
        [
            {"name_last": "skubal", "name_first": "tarik", "key_mlbam": 669373, "key_fangraphs": 22267.0},
            {"name_last": "scherzer", "name_first": "max", "key_mlbam": 453286, "key_fangraphs": 3137.0},
            # A duplicate key_fangraphs row (e.g. a data quirk) to prove dedup works.
            {"name_last": "scherzer", "name_first": "max", "key_mlbam": 453286, "key_fangraphs": 3137.0},
            {"name_last": "someone", "name_first": "else", "key_mlbam": 1, "key_fangraphs": np.nan},
        ]
    )


def test_resolves_known_players_to_verified_real_mlbam_ids():
    # These exact ids were confirmed live against statsapi.mlb.com.
    crosswalk = build_crosswalk(_players(), _register())
    assert resolve_mlbam_id("Skubal_26337", crosswalk) == 669373
    assert resolve_mlbam_id("Scherzer_5166", crosswalk) == 453286


def test_unmapped_when_no_fangraphs_id():
    crosswalk = build_crosswalk(_players(), _register())
    assert resolve_mlbam_id("NoFgId_1", crosswalk) is None


def test_unmapped_when_fangraphs_id_not_in_register():
    crosswalk = build_crosswalk(_players(), _register())
    assert resolve_mlbam_id("NotInRegister_2", crosswalk) is None


def test_unknown_player_id_is_unmapped_not_an_error():
    crosswalk = build_crosswalk(_players(), _register())
    assert resolve_mlbam_id("Ghost_9", crosswalk) is None
