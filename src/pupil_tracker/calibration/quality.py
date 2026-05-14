"""Calibration sample quality decisions."""

from __future__ import annotations

from dataclasses import dataclass

from pupil_tracker.models import RawObservation


@dataclass(frozen=True)
class CalibrationSampleDecision:
    """Decision for whether one tracker observation is usable for calibration."""

    accepted: bool
    reason: str


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
