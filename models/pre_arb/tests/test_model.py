"""Tests for the deterministic pre-arb trend-line model.

Uses the real contracts CSV for Max Scherzer's known arb year (a
phase-mismatch case) plus synthetic frames for the trend fit itself, mirroring
agent/tests/test_phase.py's fixture/synthetic_history pattern.
"""

import numpy as np
import pandas as pd
import pytest

from agent.phase import load_contract_history
from models.exceptions import PhaseMismatchError
from models.pre_arb.config import CBA_END_YEAR, TREND_START_YEAR
from models.pre_arb.model import _compute_trend, predict_pre_arb_aav

SCHERZER = "Scherzer_5166"

# Real yearly medians from analysis/2_pre_arb_deterministic_model.ipynb Section 1
TREND_YEARS = list(range(2016, 2026))
TREND_VALUES = [0.508, 0.535, 0.545, 0.555, 0.565, 0.570, 0.700, 0.720, 0.740, 0.760]


@pytest.fixture(scope="module")
def contracts():
    return load_contract_history()


def synthetic_history(rows):
    return pd.DataFrame(
        rows,
        columns=["contract_id", "player_id", "age", "service_time", "year", "duration", "value", "type"],
    )


def trend_rows():
    return [
        [f"Trend_{year}", f"Trend_{year}", 24, 1.100, year, 1, value, "pre-arb"]
        for year, value in zip(TREND_YEARS, TREND_VALUES)
    ]


class TestTrendFit:
    def test_recovers_notebook_scale_slope(self):
        df = synthetic_history(trend_rows())
        slope, _, fit_years = _compute_trend(df)
        assert 0.020 <= slope <= 0.035
        assert fit_years == [TREND_START_YEAR, 2025]

    def test_ignores_multi_year_and_outlier_rows(self):
        rows = trend_rows() + [
            ["Ext_2020", "Ext_1", 25, 2.000, 2020, 2, 4.0, "pre-arb"],  # duration > 1
            ["Out_2021", "Out_1", 26, 2.500, 2021, 1, 6.0, "pre-arb"],  # value >= 5.0
        ]
        df = synthetic_history(rows)
        slope, intercept, fit_years = _compute_trend(df)
        expected_slope, expected_intercept = np.polyfit(TREND_YEARS, TREND_VALUES, 1)
        assert slope == pytest.approx(expected_slope)
        assert intercept == pytest.approx(expected_intercept)
        assert fit_years == [TREND_START_YEAR, 2025]

    def test_insufficient_years_raises(self):
        df = synthetic_history([["Doe_9_2016", "Doe_9", 23, 1.000, 2016, 1, 0.5, "pre-arb"]])
        with pytest.raises(ValueError):
            _compute_trend(df)


class TestPredictPreArbAav:
    def test_2026_prediction_matches_notebook(self):
        rows = trend_rows() + [["Doe_7_2025", "Doe_7", 24, 1.100, 2025, 1, 0.760, "pre-arb"]]
        df = synthetic_history(rows)
        pred = predict_pre_arb_aav("Doe_7", 2026, df)
        assert pred.phase == "pre-arb"
        assert pred.method == "trend"
        assert pred.aav_millions == pytest.approx(0.787, abs=0.03)
        assert pred.within_current_cba is True
        assert pred.notes == []
        assert pred.trend_fit_years == [TREND_START_YEAR, 2025]

    def test_cba_boundary_flag(self):
        rows = trend_rows() + [["Doe_8_2026", "Doe_8", 24, 1.050, 2026, 1, 0.78, "pre-arb"]]
        df = synthetic_history(rows)

        within = predict_pre_arb_aav("Doe_8", CBA_END_YEAR, df)
        assert within.within_current_cba is True
        assert within.notes == []

        beyond = predict_pre_arb_aav("Doe_8", CBA_END_YEAR + 1, df)
        assert beyond.within_current_cba is False
        assert any("CBA" in note for note in beyond.notes)

    def test_phase_mismatch_raises_for_known_arb_year(self, contracts):
        with pytest.raises(PhaseMismatchError) as exc_info:
            predict_pre_arb_aav(SCHERZER, 2013, contracts)
        err = exc_info.value
        assert err.expected_phase == "pre-arb"
        assert err.actual_phase == "arb"
        assert err.resolution.method == "observed"

    def test_unknown_player_propagates_value_error(self, contracts):
        with pytest.raises(ValueError):
            predict_pre_arb_aav("Nobody_0", 2024, contracts)
