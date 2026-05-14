"""Tests for scalar calibration feature diagnostics."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from pupil_tracker.calibration import summarize_feature_diagnostics
from pupil_tracker.models import CalibrationSample, CalibrationTarget, RawObservation


def _target(target_id: str, *, x: float = 0.5, y: float = 0.2) -> CalibrationTarget:
    return CalibrationTarget(id=target_id, x=x, y=y)


def _sample(
    target: CalibrationTarget,
    *,
    timestamp: float = 1.0,
    valid: bool = True,
    feature_vector: tuple[float, ...] = (1.0, 2.0),
) -> CalibrationSample:
    return CalibrationSample(
        target=target,
        observation=RawObservation(
            timestamp=timestamp,
            valid=valid,
            confidence=1.0 if valid else 0.0,
            feature_vector=feature_vector,
        ),
    )


def test_feature_diagnostics_report_mean_and_std_per_target() -> None:
    target = _target("top", x=0.5, y=0.2)
    samples = (
        _sample(target, timestamp=1.0, feature_vector=(1.0, 2.0)),
        _sample(target, timestamp=2.0, feature_vector=(3.0, 4.0)),
    )

    summary = summarize_feature_diagnostics(samples)

    assert summary.feature_count == 2
    assert tuple(summary.target_summaries) == ("top",)
    target_summary = summary.target_summaries["top"]
    assert target_summary.target_id == "top"
    assert target_summary.target_x == pytest.approx(0.5)
    assert target_summary.target_y == pytest.approx(0.2)
    assert target_summary.accepted_count == 2
    assert target_summary.feature_mean == pytest.approx((2.0, 3.0))
    assert target_summary.feature_std == pytest.approx((1.0, 1.0))


def test_feature_diagnostics_group_multiple_targets_independently() -> None:
    top = _target("top", x=0.5, y=0.2)
    bottom = _target("bottom", x=0.5, y=0.8)

    summary = summarize_feature_diagnostics(
        (
            _sample(top, timestamp=1.0, feature_vector=(1.0, 2.0)),
            _sample(bottom, timestamp=2.0, feature_vector=(5.0, 7.0)),
            _sample(top, timestamp=3.0, feature_vector=(3.0, 4.0)),
        )
    )

    assert tuple(summary.target_summaries) == ("top", "bottom")
    assert summary.target_summaries["top"].feature_mean == pytest.approx((2.0, 3.0))
    assert summary.target_summaries["bottom"].feature_mean == pytest.approx((5.0, 7.0))
    assert summary.target_summaries["bottom"].feature_std == pytest.approx((0.0, 0.0))


def test_feature_diagnostics_skip_invalid_and_empty_feature_observations() -> None:
    target = _target("center", x=0.5, y=0.5)

    summary = summarize_feature_diagnostics(
        (
            _sample(target, timestamp=1.0, feature_vector=(2.0, 4.0)),
            _sample(target, timestamp=2.0, valid=False, feature_vector=(100.0, 100.0)),
            _sample(target, timestamp=3.0, feature_vector=()),
        )
    )

    assert summary.feature_count == 2
    assert summary.target_summaries["center"].accepted_count == 1
    assert summary.target_summaries["center"].feature_mean == pytest.approx((2.0, 4.0))
    assert summary.target_summaries["center"].feature_std == pytest.approx((0.0, 0.0))


def test_feature_diagnostics_reject_inconsistent_feature_lengths() -> None:
    target = _target("center")

    with pytest.raises(ValueError, match="feature vector length"):
        summarize_feature_diagnostics(
            (
                _sample(target, timestamp=1.0, feature_vector=(1.0, 2.0)),
                _sample(target, timestamp=2.0, feature_vector=(1.0, 2.0, 3.0)),
            )
        )


def test_feature_diagnostics_return_empty_summary_when_no_valid_features() -> None:
    target = _target("center")

    summary = summarize_feature_diagnostics(
        (
            _sample(target, timestamp=1.0, valid=False, feature_vector=(1.0, 2.0)),
            _sample(target, timestamp=2.0, feature_vector=()),
        )
    )

    assert summary.feature_count == 0
    assert summary.target_summaries == {}


def test_feature_diagnostics_are_json_serializable() -> None:
    target = _target("top", x=0.25, y=0.2)
    summary = summarize_feature_diagnostics(
        (
            _sample(target, timestamp=1.0, feature_vector=(1.0, 2.0)),
            _sample(target, timestamp=2.0, feature_vector=(3.0, 4.0)),
        )
    )

    encoded = json.dumps(asdict(summary), sort_keys=True)

    assert '"feature_count": 2' in encoded
    assert '"target_id": "top"' in encoded
