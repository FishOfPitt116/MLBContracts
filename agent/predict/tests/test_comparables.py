"""Tests for query_comparable_contracts' filtering logic (no LLM calls, no CSV I/O).

Uses synthetic contracts/players frames matching the real dataset/*.csv shapes,
injected directly into _query_comparable_contracts (the pure function the
@tool-decorated wrapper calls after loading the real CSVs).
"""

import pandas as pd
import pytest

from agent.predict.comparables import _query_comparable_contracts, make_comparable_contracts_tool

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


class TestNoLookahead:
    """Regression coverage for the backtest leak: player_id="Skubal_1" with no year
    filter used to return the target player's OWN target-year contract -- the exact
    answer -- since backtest targets are drawn from already-observed rows in the
    same CSV the tool reads. before_year fixes this; these tests prove it holds."""

    def test_before_year_excludes_the_target_players_own_target_year_contract(self):
        # This is the exact leak: querying Skubal's own history used to return his
        # real 2026 contract when 2026 was the very year being predicted.
        result = _query_comparable_contracts(
            _contracts_fixture(), _players_fixture(), player_id="Skubal_1", before_year=2026
        )
        assert [m["contract_id"] for m in result["matches"]] == ["Skubal_1_2024"]

    def test_before_year_excludes_other_players_contracts_from_that_year_too(self):
        # Not just the target player -- ANY contract dated the target year or later
        # is anachronistic evidence for a genuine forecast and must be excluded.
        result = _query_comparable_contracts(
            _contracts_fixture(), _players_fixture(), before_year=2025
        )
        years = {m["year"] for m in result["matches"]}
        assert max(years) < 2025
        assert "Skubal_1_2026" not in {m["contract_id"] for m in result["matches"]}
        assert "Gore_4_2025" not in {m["contract_id"] for m in result["matches"]}

    def test_before_year_none_means_no_cutoff(self):
        # Only true for direct pure-function calls in tests -- real callers always
        # go through make_comparable_contracts_tool(), which always supplies it.
        result = _query_comparable_contracts(_contracts_fixture(), _players_fixture())
        assert result["n_matches_before_limit"] == 5

    def test_tool_schema_never_exposes_before_year_to_the_model(self):
        # The model must have no parameter through which to see or override the
        # cutoff -- it's a closure variable, not a tool argument.
        tool = make_comparable_contracts_tool(before_year=2026)
        exposed_params = set(tool.tool_spec["inputSchema"]["json"]["properties"].keys())
        assert "before_year" not in exposed_params
        assert exposed_params == {
            "player_id", "position", "phase", "min_age", "max_age",
            "min_service_time", "max_service_time", "min_year", "max_year",
            "exclude_player_id", "limit",
        }

    def test_tool_factory_produces_a_working_cutoff(self, monkeypatch):
        monkeypatch.setattr(
            "agent.predict.comparables._load_contracts", _contracts_fixture
        )
        monkeypatch.setattr(
            "agent.predict.comparables._load_players", _players_fixture
        )
        tool = make_comparable_contracts_tool(before_year=2026)
        result = tool(player_id="Skubal_1")
        assert [m["contract_id"] for m in result["matches"]] == ["Skubal_1_2024"]
