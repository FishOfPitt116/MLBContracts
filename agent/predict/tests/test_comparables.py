"""Tests for query_comparable_contracts' filtering logic (no LLM calls, no CSV I/O).

Uses synthetic contracts/players frames matching the real dataset/*.csv shapes,
injected directly into _query_comparable_contracts (the pure function the
@tool-decorated wrapper calls after loading the real CSVs).
"""

import pandas as pd
import pytest

from agent.predict.comparables import _query_comparable_contracts

CONTRACT_COLUMNS = ["contract_id", "player_id", "age", "service_time", "year", "duration", "value", "type"]
PLAYER_COLUMNS = ["player_id", "fangraphs_id", "first_name", "last_name", "position", "spotrac_link"]


def contracts(rows):
    return pd.DataFrame(rows, columns=CONTRACT_COLUMNS)


def players(rows):
    return pd.DataFrame(rows, columns=PLAYER_COLUMNS)


def _players_fixture():
    return players(
        [
            ["Skubal_1", 1, "Tarik", "Skubal", "SP", "x"],
            ["Cole_2", 2, "Gerrit", "Cole", "SP/SP1", "x"],
            ["Devers_3", 3, "Rafael", "Devers", "3B/DH", "x"],
            ["Gore_4", 4, "MacKenzie", "Gore", "SP", "x"],
        ]
    )


def _contracts_fixture():
    return contracts(
        [
            ["Skubal_1_2024", "Skubal_1", 27, 3.114, 2024, 1, 2.65, "arb"],
            ["Skubal_1_2026", "Skubal_1", 29, 5.114, 2026, 1, 32.0, "arb"],
            ["Cole_2_2020", "Cole_2", 29, -1, 2020, 9, 324.0, "free-agent"],
            ["Devers_3_2023", "Devers_3", 26, -1, 2023, 11, 331.5, "free-agent"],
            ["Gore_4_2025", "Gore_4", 26, 4.100, 2025, 1, 5.5, "arb"],
        ]
    )


def test_player_id_returns_only_that_players_history_sorted_by_year_desc():
    result = _query_comparable_contracts(
        _contracts_fixture(), _players_fixture(), player_id="Skubal_1"
    )
    years = [m["year"] for m in result["matches"]]
    assert years == [2026, 2024]
    assert all(m["player_id"] == "Skubal_1" for m in result["matches"])


def test_position_substring_match():
    result = _query_comparable_contracts(_contracts_fixture(), _players_fixture(), position="SP")
    ids = {m["player_id"] for m in result["matches"]}
    assert ids == {"Skubal_1", "Cole_2", "Gore_4"}  # "SP" matches "SP", "SP/SP1", and "SP"


def test_phase_filter():
    result = _query_comparable_contracts(
        _contracts_fixture(), _players_fixture(), phase="free-agent"
    )
    assert {m["contract_id"] for m in result["matches"]} == {"Cole_2_2020", "Devers_3_2023"}


def test_age_range_filter():
    result = _query_comparable_contracts(
        _contracts_fixture(), _players_fixture(), min_age=28, max_age=29
    )
    assert {m["contract_id"] for m in result["matches"]} == {"Skubal_1_2026", "Cole_2_2020"}


def test_service_time_is_normalized_not_raw_and_always_reported():
    # 3.114 normalizes to 3 + 114/172; 5.114 -> 5 + 114/172. Reported for every
    # match regardless of whether it was used as a filter.
    result = _query_comparable_contracts(_contracts_fixture(), _players_fixture(), player_id="Skubal_1")
    by_id = {m["contract_id"]: m["service_time"] for m in result["matches"]}
    assert by_id["Skubal_1_2024"] == pytest.approx(3 + 114 / 172)
    assert by_id["Skubal_1_2026"] == pytest.approx(5 + 114 / 172)


def test_free_agent_service_time_is_none_not_minus_one():
    result = _query_comparable_contracts(_contracts_fixture(), _players_fixture(), phase="free-agent")
    assert all(m["service_time"] is None for m in result["matches"])


def test_service_time_filter_works_for_arb():
    # Gore (4.58 normalized) is in range; Skubal's 3.66 and 5.66 are not.
    result = _query_comparable_contracts(
        _contracts_fixture(), _players_fixture(), phase="arb", min_service_time=4.0, max_service_time=5.0
    )
    assert [m["contract_id"] for m in result["matches"]] == ["Gore_4_2025"]


def test_service_time_bound_excludes_untracked_rows_without_explicit_free_agent_phase():
    # No phase filter -- service_time bound should just quietly exclude the
    # free-agent rows (which have no tracked service time) rather than error,
    # since this call didn't specifically ask for free-agent comparables.
    result = _query_comparable_contracts(
        _contracts_fixture(), _players_fixture(), min_service_time=0.0
    )
    ids = {m["player_id"] for m in result["matches"]}
    assert "Cole_2" not in ids
    assert "Devers_3" not in ids


def test_service_time_plus_free_agent_phase_raises():
    # This combination can never match anything (Spotrac never tracks service
    # time for free agents) -- must be a loud error, not a silent empty result.
    with pytest.raises(ValueError, match="service_time"):
        _query_comparable_contracts(
            _contracts_fixture(), _players_fixture(), phase="free-agent", min_service_time=3.0
        )
    with pytest.raises(ValueError, match="service_time"):
        _query_comparable_contracts(
            _contracts_fixture(), _players_fixture(), phase="free-agent", max_service_time=8.0
        )


def test_exclude_player_id_removes_the_target_player():
    result = _query_comparable_contracts(
        _contracts_fixture(), _players_fixture(), phase="arb", exclude_player_id="Skubal_1"
    )
    ids = {m["player_id"] for m in result["matches"]}
    assert "Skubal_1" not in ids
    assert ids == {"Gore_4"}


def test_year_range_filter():
    result = _query_comparable_contracts(
        _contracts_fixture(), _players_fixture(), min_year=2023, max_year=2024
    )
    assert {m["contract_id"] for m in result["matches"]} == {"Skubal_1_2024", "Devers_3_2023"}


def test_limit_truncates_but_reports_true_total():
    result = _query_comparable_contracts(_contracts_fixture(), _players_fixture(), limit=2)
    assert len(result["matches"]) == 2
    assert result["n_matches_before_limit"] == 5


def test_aav_computed_from_value_and_duration():
    result = _query_comparable_contracts(
        _contracts_fixture(), _players_fixture(), player_id="Cole_2"
    )
    assert result["matches"][0]["aav_millions"] == 36.0  # 324.0 / 9


def test_unknown_player_in_contracts_but_missing_from_players_df_does_not_crash():
    orphan_contracts = contracts([["Ghost_9_2020", "Ghost_9", 30, 5.0, 2020, 1, 1.0, "arb"]])
    result = _query_comparable_contracts(orphan_contracts, _players_fixture(), player_id="Ghost_9")
    assert result["matches"][0]["player_name"] is None
    assert result["matches"][0]["position"] is None
