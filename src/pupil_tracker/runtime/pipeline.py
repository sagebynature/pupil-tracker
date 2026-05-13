"""Synchronous pull-based runtime gaze pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from pupil_tracker.models import GazeSample, RawObservation
from pupil_tracker.screen import region_3x3
from pupil_tracker.tracking import Frame, TrackerBackend


class CameraSource(Protocol):
    """Camera source that yields one frame per call."""

    def read(self) -> Frame:
        """Read the next camera frame."""


class CalibrationModel(Protocol):
    """Calibration model that maps raw observations to screen gaze."""

    def predict(
        self,
        observation: RawObservation,
        screen_width: float,
        screen_height: float,
    ) -> GazeSample:
        """Predict a calibrated gaze sample."""


class GazeSmoother(Protocol):
    """Smoother for calibrated gaze samples."""

    def update(self, sample: GazeSample) -> GazeSample:
        """Update smoother state and return a sample."""


@dataclass(frozen=True)
class RuntimeStepResult:
    """Result from a single synchronous runtime step."""

    observation: RawObservation
    gaze_sample: GazeSample | None


class RuntimePipeline:
    """Compose camera, tracker backend, calibration, smoothing, and regions."""

    def __init__(
        self,
        camera: CameraSource,
        backend: TrackerBackend,
        calibration_model: CalibrationModel,
        smoother: GazeSmoother,
        screen_width: float,
        screen_height: float,
    ) -> None:
        if screen_width <= 0 or screen_height <= 0:
            msg = "screen dimensions must be positive"
            raise ValueError(msg)

        self._camera = camera
        self._backend = backend
        self._calibration_model = calibration_model
        self._smoother = smoother
        self._screen_width = screen_width
        self._screen_height = screen_height

    def step(self) -> RuntimeStepResult:
        """Run one synchronous tracking pipeline step."""

        frame = self._camera.read()
        observation = self._backend.process(frame)
        if not observation.valid:
            return RuntimeStepResult(observation=observation, gaze_sample=None)

        gaze_sample = self._calibration_model.predict(
            observation,
            self._screen_width,
            self._screen_height,
        )
        smoothed = self._smoother.update(gaze_sample)
        if not smoothed.valid:
            return RuntimeStepResult(observation=observation, gaze_sample=None)

        region_id = region_3x3(
            smoothed.x,
            smoothed.y,
            self._screen_width,
            self._screen_height,
        )
        return RuntimeStepResult(
            observation=observation,
            gaze_sample=replace(smoothed, region_id=region_id),
        )
