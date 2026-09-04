import pandas as pd
import pytest

from agent.phase import PHASE_ARB, UNKNOWN
from data_generation.records import BatterStats, PitcherStats, Player
from models.config import CBA_END_YEAR
from models.exceptions import PhaseMismatchError, UnsupportedSegmentError
from models.arb.model import predict_arb_aav

TRAIN_YEARS = range(2015, 2021)  # 2015-2020, all strictly before every target year used below


def synthetic_contracts(rows):
    return pd.DataFrame(
        rows,
        columns=["contract_id", "player_id", "age", "service_time", "year", "duration", "value", "type"],
    )


def make_player(position, player_id="Target"):
    return Player(player_id=player_id, fangraphs_id=1, first_name="A", last_name="B", position=position, spotrac_link="")


def make_batter_stats(player_id, year, war, plate_appearances):
    return BatterStats(player_id=player_id, year=year, window_years=3, war=war, plate_appearances=plate_appearances)


def make_pitcher_stats(player_id, year, war, innings_pitched, games=32, games_started=30):
    return PitcherStats(
        player_id=player_id,
        year=year,
        window_years=3,
        games=games,
        games_started=games_started,
        innings_pitched=innings_pitched,
        war=war,
    )


def make_role_rows(role, value_scale=1.0):
    position = {"batter": "1B", "SP": "SP", "RP": "RP"}[role]
    rows = []
    for year in TRAIN_YEARS:
        for k in range(4):
            war = 1.0 + k * 0.7 + (year - 2015) * 0.1
            service_time = 1.0 + (year % 4) + k * 0.020  # raw years.days format, days < 172
            value = round((0.8 + 0.5 * war + 0.2 * service_time + 0.05 * k) * value_scale, 3)
            row = dict(
                player_id=f"{role}_{year}_{k}",
                contract_year=year,
                contract_type="arb",
                duration=1,
                value=value,
                position=position,
                service_time=service_time,
                bat_games_3y=0,
                bat_plate_appearances_3y=0,
                bat_war_3y=0.0,
                pit_games_3y=0,
                pit_games_started_3y=0,
                pit_innings_pitched_3y=0.0,
                pit_war_3y=0.0,
            )
            if role == "batter":
                row.update(bat_games_3y=140, bat_plate_appearances_3y=400 + k * 50, bat_war_3y=war)
            else:
                gs = 30 if role == "SP" else 5
                row.update(pit_games_3y=32, pit_games_started_3y=gs, pit_innings_pitched_3y=150.0 + k * 20, pit_war_3y=war)
            rows.append(row)
    return rows


def training_frame(*roles, value_scale=1.0):
    rows = []
    for role in roles:
        rows.extend(make_role_rows(role, value_scale=value_scale))
    return pd.DataFrame(rows)


class TestEndToEndPredictions:
    def test_predicts_super_two_batter(self):
        contracts_df = synthetic_contracts(
            [["Target_Batter_2021", "Target_Batter", 25, 1.100, 2021, 1, 1.0, "arb"]]
        )
        pred = predict_arb_aav(
            "Target_Batter",
            2021,
            contracts_df=contracts_df,
            training_df=training_frame("batter", "SP"),
            batter_stats=[make_batter_stats("Target_Batter", 2020, war=3.5, plate_appearances=1300)],
            player=make_player("1B", "Target_Batter"),
        )
        assert pred.phase == PHASE_ARB
        assert pred.method == "regression"
        assert pred.role == "batter"
        assert pred.arb_band == "Super Two"
        assert pred.training_n > 0
        assert pred.training_years[1] < 2021
        assert pred.aav_millions > 0

    def test_predicts_arb1_pitcher(self):
        contracts_df = synthetic_contracts([["Target_SP_2021", "Target_SP", 27, 3.050, 2021, 1, 3.0, "arb"]])
        pred = predict_arb_aav(
            "Target_SP",
            2021,
            contracts_df=contracts_df,
            training_df=training_frame("batter", "SP"),
            pitcher_stats=[make_pitcher_stats("Target_SP", 2020, war=4.0, innings_pitched=180.0)],
            player=make_player("SP", "Target_SP"),
        )
        assert pred.role == "SP"
        assert pred.arb_band == "Arb-1"
        assert pred.aav_millions > 0


class TestBandGating:
    def test_arb2_raises_unsupported_segment(self):
        contracts_df = synthetic_contracts([["Target_2021", "Target", 28, 4.050, 2021, 1, 5.0, "arb"]])
        with pytest.raises(UnsupportedSegmentError) as exc_info:
            predict_arb_aav("Target", 2021, contracts_df=contracts_df, training_df=training_frame("batter"))
        assert exc_info.value.segment == "Arb-2"

    def test_arb3_raises_unsupported_segment(self):
        contracts_df = synthetic_contracts([["Target_2021", "Target", 30, 5.050, 2021, 1, 8.0, "arb"]])
        with pytest.raises(UnsupportedSegmentError) as exc_info:
            predict_arb_aav("Target", 2021, contracts_df=contracts_df, training_df=training_frame("batter"))
        assert exc_info.value.segment == "Arb-3"


class TestPhaseMismatch:
    def test_pre_arb_year_raises(self):
        contracts_df = synthetic_contracts([["Target_2021", "Target", 23, 1.100, 2021, 1, 0.6, "pre-arb"]])
        with pytest.raises(PhaseMismatchError):
            predict_arb_aav("Target", 2021, contracts_df=contracts_df, training_df=training_frame("batter"))

    def test_free_agent_year_raises(self):
        contracts_df = synthetic_contracts([["Target_2021", "Target", 32, 7.000, 2021, 1, 20.0, "free-agent"]])
        with pytest.raises(PhaseMismatchError):
            predict_arb_aav("Target", 2021, contracts_df=contracts_df, training_df=training_frame("batter"))


class TestCbaBoundary:
    def test_within_and_beyond_cba(self):
        contracts_df = synthetic_contracts(
            [
                ["Target_CBA_2026", "Target_CBA", 26, 1.100, CBA_END_YEAR, 1, 1.0, "arb"],
                ["Target_CBA_2027", "Target_CBA", 27, 1.100, CBA_END_YEAR + 1, 1, 1.0, "arb"],
            ]
        )
        batter_stats = [
            make_batter_stats("Target_CBA", CBA_END_YEAR - 1, war=3.0, plate_appearances=1200),
            make_batter_stats("Target_CBA", CBA_END_YEAR, war=3.2, plate_appearances=1250),
        ]
        player = make_player("1B", "Target_CBA")
        train_df = training_frame("batter")

        within = predict_arb_aav(
            "Target_CBA", CBA_END_YEAR, contracts_df=contracts_df, training_df=train_df,
            batter_stats=batter_stats, player=player,
        )
        assert within.within_current_cba is True
        assert within.notes == []

        beyond = predict_arb_aav(
            "Target_CBA", CBA_END_YEAR + 1, contracts_df=contracts_df, training_df=train_df,
            batter_stats=batter_stats, player=player,
        )
        assert beyond.within_current_cba is False
        assert any("CBA" in note for note in beyond.notes)


class TestEdgeCases:
    def test_insufficient_training_rows_raises(self):
        contracts_df = synthetic_contracts([["Target_2021", "Target", 25, 1.100, 2021, 1, 1.0, "arb"]])
        tiny_training_df = pd.DataFrame(make_role_rows("batter")[:2])
        with pytest.raises(ValueError):
            predict_arb_aav(
                "Target",
                2021,
                contracts_df=contracts_df,
                training_df=tiny_training_df,
                batter_stats=[make_batter_stats("Target", 2020, war=3.0, plate_appearances=1000)],
                player=make_player("1B", "Target"),
            )

    def test_unknown_player_propagates_value_error(self):
        contracts_df = synthetic_contracts([["Other_2021", "Other", 25, 1.100, 2021, 1, 1.0, "arb"]])
        with pytest.raises(ValueError):
            predict_arb_aav("Nobody_0", 2021, contracts_df=contracts_df, training_df=training_frame("batter"))
        # not our own exceptions:
        with pytest.raises(ValueError) as exc_info:
            predict_arb_aav("Nobody_0", 2021, contracts_df=contracts_df, training_df=training_frame("batter"))
        assert not isinstance(exc_info.value, (PhaseMismatchError, UnsupportedSegmentError))

    def test_unknown_service_time_raises(self):
        contracts_df = synthetic_contracts([["Target_2021", "Target", 25, UNKNOWN, 2021, 1, 1.0, "arb"]])
        with pytest.raises(ValueError):
            predict_arb_aav("Target", 2021, contracts_df=contracts_df, training_df=training_frame("batter"))


class TestInjectedTrainingDfBypassesCache:
    def test_no_cache_pollution_across_calls(self):
        contracts_df = synthetic_contracts([["Target_2021", "Target", 25, 1.100, 2021, 1, 1.0, "arb"]])
        batter_stats = [make_batter_stats("Target", 2020, war=3.0, plate_appearances=1000)]
        player = make_player("1B", "Target")

        pred_low = predict_arb_aav(
            "Target", 2021, contracts_df=contracts_df, training_df=training_frame("batter", value_scale=1.0),
            batter_stats=batter_stats, player=player,
        )
        pred_high = predict_arb_aav(
            "Target", 2021, contracts_df=contracts_df, training_df=training_frame("batter", value_scale=5.0),
            batter_stats=batter_stats, player=player,
        )
        assert pred_low.aav_millions != pred_high.aav_millions
