"""Tests for agent/metrics.py, in particular the relative-tolerance path.

An absolute dollar tolerance is fine for pre-arb/arb (salaries cluster tightly)
but meaningless for free-agent, where deal sizes span nearly two orders of
magnitude in one sample. See agent/backtest.py's PHASE_TOLERANCE_IS_RELATIVE.
"""

from agent.metrics import (
    calculate_all_metrics,
    calculate_pct_within_relative_tolerance,
    calculate_pct_within_tolerance,
    format_metrics_report,
)


def test_absolute_tolerance_is_flat_regardless_of_scale():
    # A $1M miss passes on a $20M deal but fails on a $1.35M deal -- same
    # absolute tolerance, wildly different real-world significance.
    pct = calculate_pct_within_tolerance(y_true=[20.0, 1.35], y_pred=[19.5, 0.0], tolerance=1.0)
    assert pct == 0.5  # only the $20M one passes an absolute $1M band


def test_relative_tolerance_scales_with_actual_value():
    # Same two contracts, judged by percentage instead: the $20M miss (2.5%) passes
    # a 20% band; the $1.35M miss (100% off) correctly still fails.
    pct = calculate_pct_within_relative_tolerance(
        y_true=[20.0, 1.35], y_pred=[19.5, 0.0], tolerance=0.20
    )
    assert pct == 0.5


def test_relative_tolerance_catches_the_hinske_style_miss():
    # Predicted $0 against an actual $1.35M deal is a 100% miss -- should never
    # pass any reasonable relative tolerance, unlike the old flat $5M band.
    pct = calculate_pct_within_relative_tolerance(y_true=[1.35], y_pred=[0.0], tolerance=0.20)
    assert pct == 0.0


def test_calculate_all_metrics_relative_flag():
    metrics = calculate_all_metrics([10.0], [8.0], tolerance=0.20, relative=True)
    assert metrics["relative"] is True
    assert metrics["pct_within_tolerance"] == 1.0  # 20% off, right at the edge of a 20% band

    metrics = calculate_all_metrics([10.0], [8.0], tolerance=0.10, relative=True)
    assert metrics["pct_within_tolerance"] == 0.0  # 20% off fails a 10% band


def test_calculate_all_metrics_defaults_to_absolute():
    metrics = calculate_all_metrics([10.0], [8.0], tolerance=1.0)
    assert metrics["relative"] is False


def test_format_metrics_report_shows_percent_for_relative():
    metrics = calculate_all_metrics([10.0], [9.0], tolerance=0.20, relative=True)
    report = format_metrics_report(metrics)
    assert "±20%" in report
    assert "$0.2M" not in report


def test_format_metrics_report_shows_dollars_for_absolute():
    metrics = calculate_all_metrics([10.0], [9.0], tolerance=1.0)
    report = format_metrics_report(metrics)
    assert "±$1.0M" in report
