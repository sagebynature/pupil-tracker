"""Calibrated gaze runtime for the desktop demo."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from pupil_tracker.models import GazeSample, RawObservation
from pupil_tracker.screen import region_3x3
from pupil_tracker.smoothing import EmaGazeSmoother


class GazeCalibrationModelLike(Protocol):
    """Calibration model surface used by the demo gaze runtime."""

    def predict(
        self,
        observation: RawObservation,
        screen_width: float,
        screen_height: float,
    ) -> GazeSample:
        """Predict a screen-space gaze sample for a raw observation."""


class GazeSmootherLike(Protocol):
    """Smoother surface used by the demo gaze runtime."""

    def update(self, sample: GazeSample) -> GazeSample:
        """Update smoother state and return the smoothed sample."""


class GazeRuntime:
    """Convert valid tracker observations into smoothed, region-mapped gaze."""

    def __init__(
        self,
        *,
        model: GazeCalibrationModelLike,
        smoother: GazeSmootherLike | None = None,
    ) -> None:
        self.model = model
        self.smoother = smoother if smoother is not None else EmaGazeSmoother()

    def update(
        self,
        observation: RawObservation,
        *,
        screen_width: float,
        screen_height: float,
    ) -> GazeSample | None:
        """Return one calibrated gaze sample, or None when gaze is unavailable."""

        if screen_width <= 0 or screen_height <= 0:
            msg = "screen dimensions must be positive"
            raise ValueError(msg)
        if not observation.valid:
            return None
        try:
            predicted = self.model.predict(observation, screen_width, screen_height)
        except RuntimeError:
            return None
        smoothed = self.smoother.update(predicted)
        if not smoothed.valid:
            return None
        return replace(
            smoothed,
            region_id=region_3x3(smoothed.x, smoothed.y, screen_width, screen_height),
        )
