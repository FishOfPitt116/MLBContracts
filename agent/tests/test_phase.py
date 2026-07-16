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

    def test_scherzer_2013_arb(self, contracts):
        res = resolve_phase(SCHERZER, 2013, contracts)
        assert res.phase == PHASE_ARB
        assert res.method == "observed"

    def test_scherzer_2015_free_agent(self, contracts):
        res = resolve_phase(SCHERZER, 2015, contracts)
        assert res.phase == PHASE_FA
        assert res.method == "observed"
        # service_time is the -1 sentinel for free agents
        assert res.service_time_estimate is None

    def test_scherzer_2026_age_sentinel(self, contracts):
        # The 2026 row has age = -1; age must come from an earlier valid row
        res = resolve_phase(SCHERZER, 2026, contracts)
        assert res.phase == PHASE_FA
        assert res.age_estimate == 41  # age 40 in 2025


class TestProjectedYears:
    def test_scherzer_2016_covered_by_multi_year_deal(self, contracts):
        # 2015 7-year deal covers 2015-2021; no row exists for 2016
        res = resolve_phase(SCHERZER, 2016, contracts)
        assert res.phase == PHASE_FA
        assert res.method == "projected"
        assert any("under contract through 2021" in n for n in res.notes)

    def test_scherzer_future_year_stays_free_agent(self, contracts):
        res = resolve_phase(SCHERZER, 2030, contracts)
        assert res.phase == PHASE_FA
        assert res.method == "projected"

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
