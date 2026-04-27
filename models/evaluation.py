"""Shared evaluation metrics for contract prediction models."""

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
    Calculate percentage of predictions within tolerance of actual values.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        tolerance: Maximum acceptable error (same units as y)

    Returns:
        float: Percentage (0-1) of predictions within tolerance
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    errors = np.abs(y_true - y_pred)
    within_tolerance = errors <= tolerance
    return np.mean(within_tolerance)


def calculate_all_metrics(y_true, y_pred, tolerance=0.25):
    """
    Calculate all evaluation metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        tolerance: Tolerance for pct_within_tolerance metric

    Returns:
        dict: Dictionary of all metrics
    """
    return {
        "mae": calculate_mae(y_true, y_pred),
        "rmse": calculate_rmse(y_true, y_pred),
        "r2": calculate_r2(y_true, y_pred),
        "pct_within_tolerance": calculate_pct_within_tolerance(y_true, y_pred, tolerance),
        "tolerance": tolerance,
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
    lines = [
        "=" * 50,
        "Model Evaluation Metrics",
        "=" * 50,
        f"Mean Absolute Error (MAE): ${metrics['mae']:.4f}M (${metrics['mae'] * 1000:.1f}K)",
        f"Root Mean Squared Error (RMSE): ${metrics['rmse']:.4f}M",
        f"R² Score: {metrics['r2']:.4f}",
        f"% Within ±${metrics['tolerance']}M: {metrics['pct_within_tolerance'] * 100:.2f}%",
        f"Sample Size: {metrics['n_samples']:,}",
        "=" * 50,
    ]
    return "\n".join(lines)
