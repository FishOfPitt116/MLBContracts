import pandas as pd
import pytest

from data_generation.records import BatterStats, PitcherStats, Player
from models.arb.features import arb_band, build_training_frame, player_year_features


def contract_row(**overrides):
    base = dict(
        player_id="X",
        contract_year=2020,
        contract_type="arb",
        duration=1,
        value=3.0,
        position="1B",
        service_time=3.050,
        bat_games_3y=400,
        bat_plate_appearances_3y=1200,
        bat_war_3y=5.0,
        pit_games_3y=0,
        pit_games_started_3y=0,
        pit_innings_pitched_3y=0.0,
        pit_war_3y=0.0,
    )
    base.update(overrides)
    return base


def synthetic_contracts_with_stats(rows):
    return pd.DataFrame(rows)


def make_player(position, player_id="X"):
    return Player(player_id=player_id, fangraphs_id=1, first_name="A", last_name="B", position=position, spotrac_link="")


def make_pitcher_stats(player_id="X", year=2019, window_years=3, games=50, games_started=45, war=3.0, innings_pitched=180.0):
    return PitcherStats(
        player_id=player_id,
        year=year,
        window_years=window_years,
        games=games,
        games_started=games_started,
        innings_pitched=innings_pitched,
        war=war,
    )


def make_batter_stats(player_id="X", year=2019, window_years=3, war=4.0, plate_appearances=1500):
    return BatterStats(
        player_id=player_id, year=year, window_years=window_years, war=war, plate_appearances=plate_appearances
    )


class TestArbBand:
    def test_super_two_below_three(self):
        assert arb_band(2.5) == "Super Two"

    def test_three_is_arb_1_boundary(self):
        assert arb_band(3.0) == "Arb-1"

    def test_below_four_is_arb_1(self):
        assert arb_band(3.5) == "Arb-1"

    def test_four_is_arb_2_boundary(self):
        assert arb_band(4.0) == "Arb-2"

    def test_below_five_is_arb_2(self):
        assert arb_band(4.5) == "Arb-2"

    def test_five_and_above_is_arb_3(self):
        assert arb_band(5.0) == "Arb-3"
        assert arb_band(6.2) == "Arb-3"

    def test_none_and_nan_are_none(self):
        assert arb_band(None) is None
        assert arb_band(float("nan")) is None


class TestBuildTrainingFrame:
    def test_multi_year_extension_excluded(self):
        df = synthetic_contracts_with_stats([contract_row(player_id="Ext_1", duration=2)])
        frame = build_training_frame(df)
        assert "Ext_1" not in frame["player_id"].values

    def test_broken_batter_stats_join_excluded(self):
        df = synthetic_contracts_with_stats(
            [contract_row(player_id="Broken_1", bat_games_3y=100, bat_plate_appearances_3y=0)]
        )
        frame = build_training_frame(df)
        assert "Broken_1" not in frame["player_id"].values

    def test_broken_pitcher_stats_join_excluded(self):
        df = synthetic_contracts_with_stats(
            [
                contract_row(
                    player_id="Broken_2",
                    position="SP",
                    pit_games_3y=50,
                    pit_games_started_3y=45,
                    pit_innings_pitched_3y=0.0,
                )
            ]
        )
        frame = build_training_frame(df)
        assert "Broken_2" not in frame["player_id"].values

    def test_ohtani_excluded(self):
        df = synthetic_contracts_with_stats(
            [contract_row(player_id="Ohtani_24661", position="SP", pit_games_3y=50, pit_games_started_3y=45)]
        )
        frame = build_training_frame(df)
        assert len(frame) == 0

    def test_mislabeled_position_workload_wins(self):
        # Burnes-style bug: labeled RP but full starter workload
        df = synthetic_contracts_with_stats(
            [
                contract_row(
                    player_id="Burnes_like",
                    position="RP",
                    pit_games_3y=80,
                    pit_games_started_3y=70,
                    pit_innings_pitched_3y=420.0,
                    pit_war_3y=6.0,
                )
            ]
        )
        frame = build_training_frame(df)
        row = frame[frame["player_id"] == "Burnes_like"].iloc[0]
        assert row["role"] == "SP"

    def test_missing_workload_falls_back_to_label(self):
        df = synthetic_contracts_with_stats(
            [
                contract_row(player_id="LabelSP", position="SP", pit_games_3y=0, pit_games_started_3y=0),
                contract_row(player_id="LabelRP", position="RP", pit_games_3y=0, pit_games_started_3y=0),
            ]
        )
        frame = build_training_frame(df)
        assert frame[frame["player_id"] == "LabelSP"].iloc[0]["role"] == "SP"
        assert frame[frame["player_id"] == "LabelRP"].iloc[0]["role"] == "RP"

    def test_unresolvable_role_dropped(self):
        df = synthetic_contracts_with_stats(
            [contract_row(player_id="Other_1", position="P", pit_games_3y=0, pit_games_started_3y=0)]
        )
        frame = build_training_frame(df)
        assert "Other_1" not in frame["player_id"].values

    def test_batter_always_role_batter(self):
        df = synthetic_contracts_with_stats([contract_row(player_id="Batter_1", position="1B")])
        frame = build_training_frame(df)
        assert frame[frame["player_id"] == "Batter_1"].iloc[0]["role"] == "batter"


class TestPlayerYearFeatures:
    def test_clean_sp(self):
        result = player_year_features(
            "X",
            2020,
            pitcher_stats=[make_pitcher_stats(games=50, games_started=48, war=5.0, innings_pitched=190.0)],
            player=make_player("SP"),
        )
        assert result == {"role": "SP", "war_3y": 5.0, "volume_3y": 190.0}

    def test_clean_rp(self):
        result = player_year_features(
            "X",
            2020,
            pitcher_stats=[make_pitcher_stats(games=50, games_started=5, war=1.5, innings_pitched=60.0)],
            player=make_player("RP"),
        )
        assert result["role"] == "RP"

    def test_mislabeled_pitcher_workload_wins(self):
        result = player_year_features(
            "X",
            2020,
            pitcher_stats=[make_pitcher_stats(games=50, games_started=48)],
            player=make_player("RP"),
        )
        assert result["role"] == "SP"

    def test_missing_games_falls_back_to_label(self):
        result = player_year_features(
            "X",
            2020,
            pitcher_stats=[make_pitcher_stats(games=0, games_started=0)],
            player=make_player("SP"),
        )
        assert result["role"] == "SP"

    def test_batter(self):
        result = player_year_features(
            "X",
            2020,
            batter_stats=[make_batter_stats(war=4.2, plate_appearances=1600)],
            player=make_player("1B"),
        )
        assert result == {"role": "batter", "war_3y": 4.2, "volume_3y": 1600}

    def test_unknown_player_raises(self):
        with pytest.raises(ValueError):
            player_year_features("Nobody_0", 2020)

    def test_no_position_raises(self):
        with pytest.raises(ValueError):
            player_year_features("X", 2020, player=make_player(""))

    def test_no_matching_stats_row_raises(self):
        with pytest.raises(ValueError):
            player_year_features(
                "X",
                2020,
                pitcher_stats=[make_pitcher_stats(year=2015)],
                player=make_player("SP"),
            )
