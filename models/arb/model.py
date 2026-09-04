from dataclasses import dataclass, field

import numpy as np

from agent.phase import PHASE_ARB, resolve_phase
from models.arb.config import MIN_TRAINING_ROWS, ROLE_FEATURE_SPECS, SUPPORTED_ARB_BANDS
from models.arb.features import arb_band, build_training_frame, player_year_features
from models.config import CBA_END_YEAR
from models.exceptions import PhaseMismatchError, UnsupportedSegmentError
from models.types import YearPrediction

_coef_cache = {}


@dataclass
class ArbPrediction(YearPrediction):
    role: str = ""
    arb_band: str = ""
    war_3y: float = 0.0
    volume_3y: float = None
    training_years: list = field(default_factory=list)
    training_n: int = 0


def _fit_role(frame, role, as_of_year):
    xcols = ROLE_FEATURE_SPECS[role]
    sub = frame[(frame["role"] == role) & (frame["contract_year"] < as_of_year)]
    sub_tr = sub[["contract_year", "log_aav"] + xcols].dropna(subset=["log_aav"] + xcols)
    if len(sub_tr) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"only {len(sub_tr)} training rows for role={role!r} before {as_of_year} "
            f"(need >= {MIN_TRAINING_ROWS})"
        )
    n = len(sub_tr)
    X = np.column_stack([sub_tr[c].values for c in xcols] + [np.ones(n)])
    y = sub_tr["log_aav"].values
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit_years = [int(sub_tr["contract_year"].min()), int(sub_tr["contract_year"].max())]
    return coefs, fit_years, n


def _fitted_coefficients(role, as_of_year, training_df=None):
    if training_df is not None:
        frame = build_training_frame(training_df)
        return _fit_role(frame, role, as_of_year)
    key = (role, as_of_year)
    if key not in _coef_cache:
        _coef_cache[key] = _fit_role(build_training_frame(), role, as_of_year)
    return _coef_cache[key]


def predict_arb_aav(
    player_id,
    year,
    contracts_df=None,
    training_df=None,
    batter_stats=None,
    pitcher_stats=None,
    player=None,
):
    resolution = resolve_phase(player_id, year, contracts_df)
    if resolution.phase != PHASE_ARB:
        raise PhaseMismatchError(player_id, year, PHASE_ARB, resolution.phase, resolution)

    if resolution.service_time_estimate is None:
        raise ValueError(
            f"{player_id} in {year}: arb phase resolved but service_time_estimate is "
            "unknown; cannot determine arbitration band"
        )
    band = arb_band(resolution.service_time_estimate)
    if band not in SUPPORTED_ARB_BANDS:
        raise UnsupportedSegmentError(player_id, year, band, list(SUPPORTED_ARB_BANDS))

    feats = player_year_features(player_id, year, batter_stats, pitcher_stats, player)
    role = feats["role"]
    war_3y = feats["war_3y"]
    volume_3y = feats["volume_3y"]
    service_time = resolution.service_time_estimate
    war_x_service = war_3y * service_time

    coefs, fit_years, n = _fitted_coefficients(role, year, training_df)
    feature_values = {
        "war_3y": war_3y,
        "volume_3y": volume_3y,
        "service_time": service_time,
        "war_x_service": war_x_service,
    }
    xcols = ROLE_FEATURE_SPECS[role]
    x = np.array([feature_values[c] for c in xcols] + [1.0])
    aav = round(float(np.exp(coefs @ x)), 4)

    within_current_cba = year <= CBA_END_YEAR
    notes = (
        []
        if within_current_cba
        else [
            f"year {year} is beyond the current CBA's known term (through {CBA_END_YEAR}); "
            "coefficients extrapolated, not validated"
        ]
    )

    return ArbPrediction(
        player_id=player_id,
        year=year,
        phase=PHASE_ARB,
        method="regression",
        aav_millions=aav,
        within_current_cba=within_current_cba,
        phase_resolution=resolution,
        notes=notes,
        role=role,
        arb_band=band,
        war_3y=war_3y,
        volume_3y=volume_3y,
        training_years=fit_years,
        training_n=n,
    )
