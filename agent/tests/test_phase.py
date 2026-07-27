"""Tests for the deterministic phase resolver.

Uses the real contracts CSV for Max Scherzer (a career spanning all three
phases) plus synthetic frames for projection edge cases.
"""

import pandas as pd
import pytest

from agent.phase import (
    PHASE_ARB,
    PHASE_FA,
    PHASE_PRE_ARB,
    load_contract_history,
    project_phase_timeline,
    resolve_phase,
)

SCHERZER = "Scherzer_5166"


@pytest.fixture(scope="module")
def contracts():
    return load_contract_history()


def synthetic_history(rows):
    return pd.DataFrame(
        rows,
        columns=["contract_id", "player_id", "age", "service_time", "year", "duration", "value", "type"],
    )


class TestObservedYears:
    def test_scherzer_2011_pre_arb(self, contracts):
        res = resolve_phase(SCHERZER, 2011, contracts)
        assert res.phase == PHASE_PRE_ARB
        assert res.method == "observed"
        # 2.079 in years.days -> 2 + 79/172
        assert res.service_time_estimate == pytest.approx(2 + 79 / 172)
        # 2011 was a 1yr/$0.6M deal -> on-record AAV is just that $0.6M
        assert res.known_value == {
            "contract_id": "Scherzer_5166_2011",
            "aav_millions": 0.6,
            "duration_years": 1,
            "total_value_millions": 0.6,
            "start_year": 2011,
            "end_year": 2011,
        }

    def test_scherzer_2013_arb(self, contracts):
        res = resolve_phase(SCHERZER, 2013, contracts)
        assert res.phase == PHASE_ARB
        assert res.method == "observed"
        assert res.known_value["aav_millions"] == pytest.approx(6.725)

    def test_scherzer_2015_free_agent(self, contracts):
        res = resolve_phase(SCHERZER, 2015, contracts)
        assert res.phase == PHASE_FA
        assert res.method == "observed"
        # service_time is the -1 sentinel for free agents
        assert res.service_time_estimate is None
        # 2015 was the 7yr/$210M deal -> on-record AAV is $30M, not $210M
        assert res.known_value["aav_millions"] == pytest.approx(30.0)
        assert res.known_value["duration_years"] == 7

    def test_scherzer_2026_age_sentinel(self, contracts):
        # The 2026 row has age = -1; age must come from an earlier valid row
        res = resolve_phase(SCHERZER, 2026, contracts)
        assert res.phase == PHASE_FA
        assert res.age_estimate == 41  # age 40 in 2025


class TestProjectedYears:
    def test_scherzer_2016_covered_by_multi_year_deal(self, contracts):
        # 2015 7-year deal covers 2015-2021; no row exists for 2016, but the
        # dollar figure is still on record -> method="known", not "projected"
        res = resolve_phase(SCHERZER, 2016, contracts)
        assert res.phase == PHASE_FA
        assert res.method == "known"
        assert any("under contract through 2021" in n for n in res.notes)
        assert res.known_value == {
            "contract_id": "Scherzer_5166_2015",
            "aav_millions": 30.0,
            "duration_years": 7,
            "total_value_millions": 210.0,
            "start_year": 2015,
            "end_year": 2021,
        }

    def test_scherzer_future_year_stays_free_agent(self, contracts):
        # 2030 is beyond the last known deal -> a genuine forecast, no known_value
        res = resolve_phase(SCHERZER, 2030, contracts)
        assert res.phase == PHASE_FA
        assert res.method == "projected"
        assert res.known_value is None

    def test_projection_crosses_pre_arb_to_arb(self):
        df = synthetic_history(
            [
                ["Doe_1_2020", "Doe_1", 23, 1.100, 2020, 1, 0.58, "pre-arb"],
                ["Doe_1_2021", "Doe_1", 24, 2.100, 2021, 1, 0.60, "pre-arb"],
            ]
        )
        # 2021 service 2.100 -> 2 + 100/172 = 2.581 normalized
        res_2022 = resolve_phase("Doe_1", 2022, df)
        assert res_2022.phase == PHASE_ARB
        assert res_2022.service_time_estimate == pytest.approx(3 + 100 / 172)

        res_2025 = resolve_phase("Doe_1", 2025, df)
        assert res_2025.phase == PHASE_FA
        assert res_2025.age_estimate == 28

    def test_super_two_caveat_noted(self):
        df = synthetic_history(
            [["Doe_2_2021", "Doe_2", 24, 1.150, 2021, 1, 0.60, "pre-arb"]]
        )
        # 2022: 1 + 150/172 + 1 = 2.87 -> pre-arb, but in super-two territory
        res = resolve_phase("Doe_2", 2022, df)
        assert res.phase == PHASE_PRE_ARB
        assert any("super-two" in n for n in res.notes)

    def test_year_before_recorded_career(self):
        df = synthetic_history(
            [["Doe_3_2020", "Doe_3", 25, 3.050, 2020, 1, 1.5, "arb"]]
        )
        res = resolve_phase("Doe_3", 2018, df)
        assert res.phase == PHASE_PRE_ARB
        assert res.method == "projected"


class TestEdgeCases:
    def test_unknown_player_raises(self, contracts):
        with pytest.raises(ValueError):
            resolve_phase("Nobody_0", 2024, contracts)

    def test_duplicate_rows_deduped(self):
        row = ["Doe_4_2020", "Doe_4", 23, 1.000, 2020, 1, 0.58, "pre-arb"]
        df = synthetic_history([row, row])
        res = resolve_phase("Doe_4", 2020, df)
        assert res.phase == PHASE_PRE_ARB


class TestPhaseTimeline:
    def test_scherzer_matches_real_career(self, contracts):
        timeline = project_phase_timeline(SCHERZER, contracts)
        assert timeline[PHASE_PRE_ARB] == [2011, 2011]
        assert timeline[PHASE_ARB] == [2012, 2014]
        assert timeline[PHASE_FA] == [2015, None]
        assert timeline["caveats"]

    def test_second_year_pre_arb_projects_full_trajectory(self):
        # Mirrors the "2nd year of pre-arb" example: known 2025+2026 pre-arb
        # rows, then projected forward through arb into free agency.
        df = synthetic_history(
            [
                ["Doe_5_2025", "Doe_5", 23, 1.100, 2025, 1, 0.60, "pre-arb"],
                ["Doe_5_2026", "Doe_5", 24, 1.150, 2026, 1, 0.62, "pre-arb"],
            ]
        )
        timeline = project_phase_timeline("Doe_5", df)
        assert timeline[PHASE_PRE_ARB] == [2025, 2027]
        assert timeline[PHASE_ARB] == [2028, 2030]
        assert timeline[PHASE_FA] == [2031, None]

    def test_already_free_agent_is_open_ended(self, contracts):
        timeline = project_phase_timeline(SCHERZER, contracts)
        assert timeline[PHASE_FA][1] is None

    def test_unknown_service_time_stops_projection(self):
        df = synthetic_history(
            [["Doe_6_2020", "Doe_6", 30, -1, 2020, 1, 5.0, "arb"]]
        )
        timeline = project_phase_timeline("Doe_6", df)
        assert timeline[PHASE_ARB] == [2020, 2020]
        assert timeline[PHASE_FA] is None
        assert any("unknown service time" in c for c in timeline["caveats"])

    def test_unknown_player_raises(self, contracts):
        with pytest.raises(ValueError):
            project_phase_timeline("Nobody_0", contracts)
