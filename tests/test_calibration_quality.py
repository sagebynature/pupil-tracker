"""Tests for calibration sample quality decisions."""

from __future__ import annotations

import pytest

from pupil_tracker.calibration import (
    CalibrationQualityFilter,
    FeatureStabilityConfig,
    summarize_target_quality,
)
from pupil_tracker.models import RawObservation


def observation(
    *,
    valid: bool = True,
    confidence: float = 0.9,
    feature_vector: tuple[float, ...] = (0.1, 0.2, 0.3),
    reason: str | None = None,
) -> RawObservation:
    return RawObservation(
        timestamp=1.0,
        valid=valid,
        confidence=confidence,
        feature_vector=feature_vector,
        reason=reason,
    )


def test_accepts_valid_high_confidence_observation() -> None:
    quality_filter = CalibrationQualityFilter(min_confidence=0.6)

    decision = quality_filter.decide(observation(confidence=0.8))

    assert decision.accepted is True
    assert decision.reason == "accepted"


def test_rejects_invalid_observation() -> None:
    quality_filter = CalibrationQualityFilter(min_confidence=0.6)

    decision = quality_filter.decide(
        observation(valid=False, confidence=0.0, reason="no face detected")
    )

    assert decision.accepted is False
    assert decision.reason == "invalid observation: no face detected"


def test_rejects_low_confidence_observation() -> None:
    quality_filter = CalibrationQualityFilter(min_confidence=0.6)

    decision = quality_filter.decide(observation(confidence=0.59))

    assert decision.accepted is False
    assert decision.reason == "confidence below 0.60"


def test_rejects_empty_feature_vector() -> None:
    quality_filter = CalibrationQualityFilter(min_confidence=0.6)

    decision = quality_filter.decide(observation(feature_vector=()))

    assert decision.accepted is False
    assert decision.reason == "missing feature vector"


def test_rejects_feature_count_mismatch() -> None:
    quality_filter = CalibrationQualityFilter(
        min_confidence=0.6,
        expected_feature_count=4,
    )

    decision = quality_filter.decide(observation(feature_vector=(0.1, 0.2, 0.3)))

    assert decision.accepted is False
    assert decision.reason == "feature vector length 3 != expected 4"


def test_accepts_observation_within_feature_stability_delta() -> None:
    quality_filter = CalibrationQualityFilter(
        min_confidence=0.6,
        stability_config=FeatureStabilityConfig(
            feature_indices=(1, 2),
            max_delta=0.05,
        ),
    )

    decision = quality_filter.decide(
        observation(feature_vector=(0.1, 0.23, 0.33)),
        reference_features=(0.1, 0.2, 0.3),
    )

    assert decision.accepted is True
    assert decision.reason == "accepted"


def test_rejects_observation_with_unstable_feature_delta() -> None:
    quality_filter = CalibrationQualityFilter(
        min_confidence=0.6,
        stability_config=FeatureStabilityConfig(
            feature_indices=(1, 2),
            max_delta=0.05,
        ),
    )

    decision = quality_filter.decide(
        observation(feature_vector=(0.1, 0.28, 0.3)),
        reference_features=(0.1, 0.2, 0.3),
    )

    assert decision.accepted is False
    assert decision.reason == "feature 1 drift 0.080 exceeds 0.050"


def test_feature_stability_config_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="feature_indices must not be empty"):
        FeatureStabilityConfig(feature_indices=(), max_delta=0.05)
    with pytest.raises(ValueError, match="feature indices must be non-negative"):
        FeatureStabilityConfig(feature_indices=(-1,), max_delta=0.05)
    with pytest.raises(ValueError, match="max_delta must be positive"):
        FeatureStabilityConfig(feature_indices=(1,), max_delta=0.0)


@pytest.mark.parametrize("min_confidence", [-0.1, 1.1])
def test_quality_filter_rejects_invalid_confidence_threshold(
    min_confidence: float,
) -> None:
    with pytest.raises(ValueError, match="min_confidence must be between 0 and 1"):
        CalibrationQualityFilter(min_confidence=min_confidence)


def test_target_quality_summary_advances_with_enough_accepted_samples() -> None:
    summary = summarize_target_quality(
        target_id="r0c0",
        accepted_observations=(
            observation(confidence=0.8),
            observation(confidence=0.9),
        ),
        rejected_count=1,
        min_samples=2,
    )

    assert summary.target_id == "r0c0"
    assert summary.accepted_count == 2
    assert summary.rejected_count == 1
    assert summary.mean_confidence == pytest.approx(0.85)
    assert summary.meets_min_samples is True
    assert summary.recommendation == "advance"


def test_target_quality_summary_retries_with_too_few_accepted_samples() -> None:
    summary = summarize_target_quality(
        target_id="r0c0",
        accepted_observations=(observation(confidence=0.8),),
        rejected_count=3,
        min_samples=2,
    )

    assert summary.accepted_count == 1
    assert summary.rejected_count == 3
    assert summary.mean_confidence == pytest.approx(0.8)
    assert summary.meets_min_samples is False
    assert summary.recommendation == "retry"


def test_target_quality_summary_reports_zero_confidence_for_no_accepted_samples() -> None:
    summary = summarize_target_quality(
        target_id="r0c0",
        accepted_observations=(),
        rejected_count=5,
        min_samples=2,
    )

    assert summary.accepted_count == 0
    assert summary.rejected_count == 5
    assert summary.mean_confidence == 0.0
    assert summary.recommendation == "retry"


def test_target_quality_summary_rejects_invalid_min_samples() -> None:
    with pytest.raises(ValueError, match="min_samples must be positive"):
        summarize_target_quality(
            target_id="r0c0",
            accepted_observations=(),
            rejected_count=0,
            min_samples=0,
        )
