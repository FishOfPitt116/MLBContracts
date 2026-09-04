from dataclasses import dataclass, field

import numpy as np

from agent.phase import PHASE_PRE_ARB, load_contract_history, resolve_phase
from models.exceptions import PhaseMismatchError
from models.pre_arb.config import CBA_END_YEAR, CONTRACT_TYPE, MAX_DURATION, MAX_VALUE, TREND_START_YEAR
from models.types import YearPrediction

_trend_cache = None


@dataclass
class PreArbPrediction(YearPrediction):
    trend_fit_years: list = field(default_factory=list)


def _compute_trend(df):
    filtered = df[
        (df["type"] == CONTRACT_TYPE) & (df["duration"] == MAX_DURATION) & (df["value"] < MAX_VALUE)
    ]
    yearly_median = filtered.groupby("year")["value"].median()
    recent = yearly_median[yearly_median.index >= TREND_START_YEAR]
    if len(recent) < 2:
        raise ValueError("not enough distinct years to fit a trend line")
    slope, intercept = np.polyfit(recent.index, recent.values, 1)
    return slope, intercept, [int(recent.index.min()), int(recent.index.max())]


def _fitted_trend(contracts_df=None):
    if contracts_df is not None:
        return _compute_trend(contracts_df)
    global _trend_cache
    if _trend_cache is None:
        _trend_cache = _compute_trend(load_contract_history())
    return _trend_cache


def predict_pre_arb_aav(player_id, year, contracts_df=None):
    resolution = resolve_phase(player_id, year, contracts_df)
    if resolution.phase != PHASE_PRE_ARB:
        raise PhaseMismatchError(player_id, year, PHASE_PRE_ARB, resolution.phase, resolution)

    slope, intercept, fit_years = _fitted_trend(contracts_df)
    aav = round(float(slope * year + intercept), 4)
    within_current_cba = year <= CBA_END_YEAR
    notes = (
        []
        if within_current_cba
        else [
            f"year {year} is beyond the current CBA's known term "
            f"({TREND_START_YEAR}-{CBA_END_YEAR}); trend extrapolated, not validated"
        ]
    )
    return PreArbPrediction(
        player_id=player_id,
        year=year,
        phase=PHASE_PRE_ARB,
        method="trend",
        aav_millions=aav,
        within_current_cba=within_current_cba,
        phase_resolution=resolution,
        notes=notes,
        trend_fit_years=fit_years,
    )
