"""Tests for intake's own tools (no LLM calls)."""

import pandas as pd
import pytest

from agent.intake.tools import find_player, get_contract_phase_timeline
from agent.phase import PHASE_ARB, PHASE_FA, PHASE_PRE_ARB


@pytest.fixture
def players(monkeypatch, tmp_path):
    csv_path = tmp_path / "players.csv"
    pd.DataFrame(
        [
            {
                "player_id": "Scherzer_5166",
                "fangraphs_id": 3137,
                "first_name": "Max",
                "last_name": "Scherzer",
                "position": "SP",
                "spotrac_link": "https://example.com/scherzer",
            },
            {
                "player_id": "Smith_1",
                "fangraphs_id": 1,
                "first_name": "John",
                "last_name": "Smith",
                "position": "C",
                "spotrac_link": "https://example.com/smith1",
            },
            {
                "player_id": "Smith_2",
                "fangraphs_id": 2,
                "first_name": "Jane",
                "last_name": "Smith",
                "position": "SS",
                "spotrac_link": "https://example.com/smith2",
            },
        ]
    ).to_csv(csv_path, index=False)
    monkeypatch.setattr("agent.intake.tools.PLAYERS_CSV", csv_path)
    return csv_path


class TestFindPlayer:
    def test_no_name_field_returns_empty_with_note(self, players):
        result = find_player(position="SP")
        assert result["matches"] == []
        assert "required" in result["note"]

    def test_single_exact_match(self, players):
        result = find_player(first_name="Max", last_name="Scherzer")
        assert [m["player_id"] for m in result["matches"]] == ["Scherzer_5166"]

    def test_ambiguous_last_name_returns_all_matches(self, players):
        result = find_player(last_name="Smith")
        ids = {m["player_id"] for m in result["matches"]}
        assert ids == {"Smith_1", "Smith_2"}

    def test_position_narrows_ambiguous_match(self, players):
        result = find_player(last_name="Smith", position="SS")
        assert [m["player_id"] for m in result["matches"]] == ["Smith_2"]

    def test_substring_match(self, players):
        result = find_player(last_name="cherz")
        assert [m["player_id"] for m in result["matches"]] == ["Scherzer_5166"]

    def test_no_match(self, players):
        result = find_player(last_name="Nobody")
        assert result["matches"] == []


class TestGetContractPhaseTimeline:
    def test_wraps_project_phase_timeline(self):
        # Scherzer_5166 is present in the real dataset/contracts_spotrac.csv.
        timeline = get_contract_phase_timeline("Scherzer_5166")
        assert timeline[PHASE_PRE_ARB] == [2011, 2011]
        assert timeline[PHASE_ARB] == [2012, 2014]
        assert timeline[PHASE_FA][1] is None
