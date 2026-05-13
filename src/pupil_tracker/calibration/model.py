"""Polynomial/ridge calibration model for mapping observations to screen gaze."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import hypot
from typing import Any, Protocol, cast

from pupil_tracker.models import CalibrationSample, GazeSample, RawObservation


class _CalibrationPipeline(Protocol):
    """Minimal fitted estimator surface used by this module."""

    def fit(
        self,
        x_train: Sequence[tuple[float, ...]],
        y_train: Sequence[tuple[float, float]],
    ) -> Any:
        """Fit the estimator and return the estimator."""

    def predict(self, features: Sequence[tuple[float, ...]]) -> Any:
        """Predict screen-space coordinates for feature vectors."""


@dataclass(frozen=True)
class CalibrationFitResult:
    """Summary metrics for a fitted calibration model."""

    sample_count: int
    mean_error_px: float
    max_error_px: float


class PolynomialRidgeCalibrationModel:
    """Fit screen coordinates from raw tracker feature vectors."""

    def __init__(self, degree: int = 2, alpha: float = 1.0) -> None:
        if degree < 1:
            msg = "degree must be at least 1"
            raise ValueError(msg)
        if alpha < 0:
            msg = "alpha must be non-negative"
            raise ValueError(msg)

        self._degree = degree
        self._alpha = alpha
        self._pipeline: _CalibrationPipeline | None = None

    def fit(
        self,
        samples: Sequence[CalibrationSample],
        screen_width: float,
        screen_height: float,
    ) -> CalibrationFitResult:
        """Fit the calibration model from valid calibration samples."""

        self._validate_screen_dimensions(screen_width, screen_height)
        valid_samples = [sample for sample in samples if sample.observation.valid]
        if len(valid_samples) < 3:
            msg = "calibration requires at least 3 valid samples"
            raise ValueError(msg)

        feature_count = len(valid_samples[0].observation.feature_vector)
        if feature_count == 0:
            msg = "calibration samples must include feature vectors"
            raise ValueError(msg)
        if any(len(sample.observation.feature_vector) != feature_count for sample in valid_samples):
            msg = "calibration sample feature vectors must have matching lengths"
            raise ValueError(msg)

        x_train = [sample.observation.feature_vector for sample in valid_samples]
        y_train = [
            (sample.target.x * screen_width, sample.target.y * screen_height)
            for sample in valid_samples
        ]

        pipeline = self._make_pipeline()
        pipeline.fit(x_train, y_train)
        self._pipeline = pipeline

        predictions = pipeline.predict(x_train)
        errors = [
            hypot(prediction[0] - expected[0], prediction[1] - expected[1])
            for prediction, expected in zip(predictions, y_train, strict=True)
        ]
        return CalibrationFitResult(
            sample_count=len(valid_samples),
            mean_error_px=sum(errors) / len(errors),
            max_error_px=max(errors),
        )

    def predict(
        self,
        observation: RawObservation,
        screen_width: float,
        screen_height: float,
    ) -> GazeSample:
        """Predict a calibrated screen-space gaze sample for `observation`."""

        self._validate_screen_dimensions(screen_width, screen_height)
        if self._pipeline is None:
            msg = "calibration model is not fitted"
            raise RuntimeError(msg)
        if not observation.valid:
            return GazeSample(
                timestamp=observation.timestamp,
                x=0.0,
                y=0.0,
                confidence=0.0,
                valid=False,
            )
        if not observation.feature_vector:
            msg = "observation must include a feature vector"
            raise ValueError(msg)

        prediction = self._pipeline.predict([observation.feature_vector])[0]
        return GazeSample(
            timestamp=observation.timestamp,
            x=float(prediction[0]),
            y=float(prediction[1]),
            confidence=observation.confidence,
            valid=True,
        )

    def _make_pipeline(self) -> _CalibrationPipeline:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures

        return cast(
            _CalibrationPipeline,
            make_pipeline(PolynomialFeatures(degree=self._degree), Ridge(alpha=self._alpha)),
        )

    @staticmethod
    def _validate_screen_dimensions(screen_width: float, screen_height: float) -> None:
        if screen_width <= 0 or screen_height <= 0:
            msg = "screen dimensions must be positive"
            raise ValueError(msg)
