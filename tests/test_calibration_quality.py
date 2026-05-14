"""Tests for calibration sample quality decisions."""

from __future__ import annotations

import pytest

from pupil_tracker.calibration import CalibrationQualityFilter
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


@pytest.mark.parametrize("min_confidence", [-0.1, 1.1])
def test_quality_filter_rejects_invalid_confidence_threshold(
    min_confidence: float,
) -> None:
    with pytest.raises(ValueError, match="min_confidence must be between 0 and 1"):
        CalibrationQualityFilter(min_confidence=min_confidence)
