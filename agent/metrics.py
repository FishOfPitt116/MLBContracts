"""Evaluation metrics for scoring predicted vs. actual contract values."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_mae(y_true, y_pred):
    """Calculate Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_r2(y_true, y_pred):
    """Calculate R-squared score."""
    return r2_score(y_true, y_pred)


def calculate_pct_within_tolerance(y_true, y_pred, tolerance):
    """
    Calculate percentage of predictions within an absolute tolerance of actual values.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        tolerance: Maximum acceptable absolute error (same units as y)

    Returns:
        float: Percentage (0-1) of predictions within tolerance
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    errors = np.abs(y_true - y_pred)
    within_tolerance = errors <= tolerance
    return np.mean(within_tolerance)


def calculate_pct_within_relative_tolerance(y_true, y_pred, tolerance):
    """
    Calculate percentage of predictions within a relative tolerance of actual values.

    An absolute tolerance doesn't scale when actual values span orders of
    magnitude within one sample (free agency: a $1.35M veteran deal and a
    $20M ace deal in the same backtest) — a flat $5M band is meaningless for
    the former and reports "within tolerance" on errors that are actually
    huge in relative terms. Use this instead wherever that's the case.

    Args:
        y_true: Actual values (must be nonzero)
        y_pred: Predicted values
        tolerance: Maximum acceptable relative error, as a fraction (e.g. 0.20 for +/-20%)

    Returns:
        float: Percentage (0-1) of predictions within tolerance
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    relative_errors = np.abs(y_true - y_pred) / np.abs(y_true)
    within_tolerance = relative_errors <= tolerance
    return np.mean(within_tolerance)


def calculate_all_metrics(y_true, y_pred, tolerance=0.25, relative=False):
    """
    Calculate all evaluation metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        tolerance: Tolerance for the pct_within_tolerance metric — an absolute dollar
            amount by default, or a fraction (e.g. 0.20) when relative=True
        relative: Use a relative (percentage) tolerance instead of an absolute one

    Returns:
        dict: Dictionary of all metrics
    """
    if relative:
        pct_within_tolerance = calculate_pct_within_relative_tolerance(y_true, y_pred, tolerance)
    else:
        pct_within_tolerance = calculate_pct_within_tolerance(y_true, y_pred, tolerance)
    return {
        "mae": calculate_mae(y_true, y_pred),
        "rmse": calculate_rmse(y_true, y_pred),
        "r2": calculate_r2(y_true, y_pred),
        "pct_within_tolerance": pct_within_tolerance,
        "tolerance": tolerance,
        "relative": relative,
        "n_samples": len(y_true),
    }


def format_metrics_report(metrics):
    """
    Format metrics dictionary as human-readable report.

    Args:
        metrics: Dictionary of evaluation metrics

    Returns:
        str: Formatted report string
    """
    if metrics.get("relative"):
        tolerance_line = (
            f"% Within ±{metrics['tolerance'] * 100:.0f}%: "
            f"{metrics['pct_within_tolerance'] * 100:.2f}%"
        )
    else:
        tolerance_line = (
            f"% Within ±${metrics['tolerance']}M: {metrics['pct_within_tolerance'] * 100:.2f}%"
        )
    lines = [
        "=" * 50,
        "Model Evaluation Metrics",
        "=" * 50,
        f"Mean Absolute Error (MAE): ${metrics['mae']:.4f}M (${metrics['mae'] * 1000:.1f}K)",
        f"Root Mean Squared Error (RMSE): ${metrics['rmse']:.4f}M",
        f"R² Score: {metrics['r2']:.4f}",
        tolerance_line,
        f"Sample Size: {metrics['n_samples']:,}",
        "=" * 50,
    ]
    return "\n".join(lines)
