"""Tests for post-calibration validation metrics."""

from __future__ import annotations

import pytest

from pupil_tracker.calibration import (
    ValidationSample,
    ValidationTarget,
    compute_validation_metrics,
    validation_pattern,
)
from pupil_tracker.models import GazeSample


def gaze(*, x: float, y: float, valid: bool = True) -> GazeSample:
    return GazeSample(
        timestamp=1.0,
        x=x,
        y=y,
        confidence=0.9,
        valid=valid,
        region_id=None,
    )


def test_validation_pattern_returns_stable_intermediate_targets() -> None:
    targets = validation_pattern()

    assert [(target.id, target.x, target.y) for target in targets] == [
        ("v0", 0.25, 0.25),
        ("v1", 0.75, 0.25),
        ("v2", 0.50, 0.50),
        ("v3", 0.25, 0.75),
        ("v4", 0.75, 0.75),
    ]


def test_exact_gaze_at_validation_target_returns_zero_error() -> None:
    target = validation_pattern()[0]
    metrics = compute_validation_metrics(
        [
            ValidationSample(
                target=target,
                gaze_sample=gaze(x=250.0, y=200.0),
            )
        ],
        screen_width=1000.0,
        screen_height=800.0,
    )

    assert metrics.sample_count == 1
    assert metrics.mean_error_px == 0.0
    assert metrics.median_error_px == 0.0
    assert metrics.max_error_px == 0.0
    assert metrics.per_target_error_px == {"v0": 0.0}
    assert metrics.mean_abs_x_error_px == 0.0
    assert metrics.mean_abs_y_error_px == 0.0
    assert metrics.mean_signed_y_error_px == 0.0
    assert metrics.per_target_signed_y_error_px == {"v0": 0.0}
    assert metrics.grid_cell_accuracy == 1.0
    assert metrics.per_target_grid_cell_accuracy == {"v0": 1.0}
    assert metrics.recommendation == "excellent"


def test_validation_metrics_compute_mean_median_max_and_per_target_error() -> None:
    target_a, target_b = validation_pattern()[:2]

    metrics = compute_validation_metrics(
        [
            ValidationSample(target=target_a, gaze_sample=gaze(x=250.0, y=200.0)),
            ValidationSample(target=target_a, gaze_sample=gaze(x=310.0, y=200.0)),
            ValidationSample(target=target_b, gaze_sample=gaze(x=870.0, y=200.0)),
        ],
        screen_width=1000.0,
        screen_height=800.0,
    )

    assert metrics.sample_count == 3
    assert metrics.mean_error_px == pytest.approx(60.0)
    assert metrics.median_error_px == pytest.approx(60.0)
    assert metrics.max_error_px == pytest.approx(120.0)
    assert metrics.per_target_error_px == {"v0": pytest.approx(30.0), "v1": pytest.approx(120.0)}
    assert metrics.recommendation == "excellent"


def test_validation_metrics_report_per_axis_error() -> None:
    target_a, target_b = validation_pattern()[:2]

    metrics = compute_validation_metrics(
        [
            ValidationSample(target=target_a, gaze_sample=gaze(x=260.0, y=230.0)),
            ValidationSample(target=target_a, gaze_sample=gaze(x=230.0, y=210.0)),
            ValidationSample(target=target_b, gaze_sample=gaze(x=780.0, y=160.0)),
        ],
        screen_width=1000.0,
        screen_height=800.0,
    )

    assert metrics.mean_abs_x_error_px == pytest.approx(20.0)
    assert metrics.mean_abs_y_error_px == pytest.approx(26.6666667)
    assert metrics.mean_signed_y_error_px == pytest.approx(0.0)
    assert metrics.per_target_signed_y_error_px == {
        "v0": pytest.approx(20.0),
        "v1": pytest.approx(-40.0),
    }


def test_validation_metrics_report_grid_cell_accuracy_for_window_selection() -> None:
    target_top_left, target_top_right, target_center = validation_pattern()[:3]

    metrics = compute_validation_metrics(
        [
            ValidationSample(
                target=target_top_left,
                gaze_sample=gaze(x=260.0, y=180.0),
            ),
            ValidationSample(
                target=target_top_left,
                gaze_sample=gaze(x=700.0, y=180.0),
            ),
            ValidationSample(
                target=target_top_right,
                gaze_sample=gaze(x=770.0, y=180.0),
            ),
            ValidationSample(
                target=target_center,
                gaze_sample=gaze(x=510.0, y=390.0),
            ),
        ],
        screen_width=900.0,
        screen_height=600.0,
    )

    assert metrics.grid_cell_accuracy == pytest.approx(0.75)
    assert metrics.grid_columns == 3
    assert metrics.grid_rows == 3
    assert metrics.per_target_grid_cell_accuracy == {
        "v0": pytest.approx(0.5),
        "v1": pytest.approx(1.0),
        "v2": pytest.approx(1.0),
    }


def test_validation_metrics_accept_configurable_grid_dimensions() -> None:
    target = ValidationTarget(id="custom", x=0.25, y=0.50)

    three_by_three = compute_validation_metrics(
        [ValidationSample(target=target, gaze_sample=gaze(x=300.0, y=450.0))],
        screen_width=1600.0,
        screen_height=900.0,
    )
    four_by_three = compute_validation_metrics(
        [ValidationSample(target=target, gaze_sample=gaze(x=300.0, y=450.0))],
        screen_width=1600.0,
        screen_height=900.0,
        grid_columns=4,
        grid_rows=3,
    )

    assert three_by_three.grid_cell_accuracy == 1.0
    assert four_by_three.grid_cell_accuracy == 0.0
    assert four_by_three.grid_columns == 4
    assert four_by_three.grid_rows == 3


@pytest.mark.parametrize(
    ("grid_columns", "grid_rows"),
    [(0, 3), (3, 0), (-1, 3), (3, -1)],
)
def test_validation_metrics_reject_invalid_grid_dimensions(
    grid_columns: int,
    grid_rows: int,
) -> None:
    target = validation_pattern()[0]

    with pytest.raises(ValueError, match="grid dimensions must be positive"):
        compute_validation_metrics(
            [ValidationSample(target=target, gaze_sample=gaze(x=250.0, y=200.0))],
            screen_width=1000.0,
            screen_height=800.0,
            grid_columns=grid_columns,
            grid_rows=grid_rows,
        )


@pytest.mark.parametrize(
    ("offset_px", "expected"),
    [
        (100.0, "good"),
        (150.0, "usable"),
        (250.0, "retry"),
    ],
)
def test_validation_recommendation_thresholds(offset_px: float, expected: str) -> None:
    target = validation_pattern()[2]

    metrics = compute_validation_metrics(
        [
            ValidationSample(
                target=target,
                gaze_sample=gaze(x=500.0 + offset_px, y=400.0),
            )
        ],
        screen_width=1000.0,
        screen_height=800.0,
    )

    assert metrics.recommendation == expected


def test_validation_metrics_reject_empty_or_invalid_sample_sets() -> None:
    target = validation_pattern()[0]

    with pytest.raises(ValueError, match="at least one valid validation sample is required"):
        compute_validation_metrics([], screen_width=1000.0, screen_height=800.0)

    with pytest.raises(ValueError, match="at least one valid validation sample is required"):
        compute_validation_metrics(
            [ValidationSample(target=target, gaze_sample=gaze(x=0.0, y=0.0, valid=False))],
            screen_width=1000.0,
            screen_height=800.0,
        )
