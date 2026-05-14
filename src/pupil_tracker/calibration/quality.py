"""Calibration sample quality decisions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pupil_tracker.models import RawObservation


@dataclass(frozen=True)
class CalibrationSampleDecision:
    """Decision for whether one tracker observation is usable for calibration."""

    accepted: bool
    reason: str


@dataclass(frozen=True)
class TargetQualitySummary:
    """Per-target calibration capture quality summary."""

    target_id: str
    accepted_count: int
    rejected_count: int
    mean_confidence: float
    meets_min_samples: bool
    recommendation: Literal["advance", "retry"]


class CalibrationQualityFilter:
    """Filter raw tracker observations before storing calibration samples."""

    def __init__(
        self,
        *,
        min_confidence: float,
        expected_feature_count: int | None = None,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            msg = "min_confidence must be between 0 and 1"
            raise ValueError(msg)
        if expected_feature_count is not None and expected_feature_count <= 0:
            msg = "expected_feature_count must be positive"
            raise ValueError(msg)
        self.min_confidence = min_confidence
        self.expected_feature_count = expected_feature_count

    def decide(self, observation: RawObservation) -> CalibrationSampleDecision:
        """Return whether `observation` should be accepted for calibration."""

        if not observation.valid:
            reason = observation.reason if observation.reason is not None else "unknown"
            return CalibrationSampleDecision(
                accepted=False,
                reason=f"invalid observation: {reason}",
            )
        if observation.confidence < self.min_confidence:
            return CalibrationSampleDecision(
                accepted=False,
                reason=f"confidence below {self.min_confidence:.2f}",
            )
        if not observation.feature_vector:
            return CalibrationSampleDecision(
                accepted=False,
                reason="missing feature vector",
            )
        feature_count = len(observation.feature_vector)
        if (
            self.expected_feature_count is not None
            and feature_count != self.expected_feature_count
        ):
            return CalibrationSampleDecision(
                accepted=False,
                reason=(
                    f"feature vector length {feature_count}"
                    f" != expected {self.expected_feature_count}"
                ),
            )
        return CalibrationSampleDecision(accepted=True, reason="accepted")


def summarize_target_quality(
    *,
    target_id: str,
    accepted_observations: Sequence[RawObservation],
    rejected_count: int,
    min_samples: int,
) -> TargetQualitySummary:
    """Summarize whether a calibration target has enough accepted samples."""

    if min_samples <= 0:
        msg = "min_samples must be positive"
        raise ValueError(msg)
    if rejected_count < 0:
        msg = "rejected_count must be non-negative"
        raise ValueError(msg)

    accepted_count = len(accepted_observations)
    mean_confidence = (
        sum(observation.confidence for observation in accepted_observations)
        / accepted_count
        if accepted_count > 0
        else 0.0
    )
    meets_min_samples = accepted_count >= min_samples
    return TargetQualitySummary(
        target_id=target_id,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        mean_confidence=mean_confidence,
        meets_min_samples=meets_min_samples,
        recommendation="advance" if meets_min_samples else "retry",
    )
