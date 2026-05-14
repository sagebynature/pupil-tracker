"""Tests for post-calibration validation metrics."""

from __future__ import annotations

import pytest

from pupil_tracker.calibration import (
    ValidationSample,
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
