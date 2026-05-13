"""Tracking runtime seam for the desktop demo."""

from __future__ import annotations

from dataclasses import dataclass

from pupil_tracker.models import Point2D, RawObservation, Rect
from pupil_tracker.tracking import Frame, TrackerBackend


@dataclass(frozen=True)
class TrackingStatus:
    """User-facing tracking status for one processed frame."""

    observation: RawObservation
    message: str

    @property
    def valid(self) -> bool:
        """Return whether the underlying observation is valid."""

        return self.observation.valid

    @property
    def confidence(self) -> float:
        """Return the underlying observation confidence."""

        return self.observation.confidence

    @property
    def face_bounds(self) -> Rect | None:
        """Return detected face bounds when available."""

        return self.observation.face_bounds

    @property
    def left_iris(self) -> Point2D | None:
        """Return detected left iris center when available."""

        return self.observation.left_iris

    @property
    def right_iris(self) -> Point2D | None:
        """Return detected right iris center when available."""

        return self.observation.right_iris


class TrackingRuntime:
    """Process frames through an injected tracker backend."""

    def __init__(self, *, backend: TrackerBackend) -> None:
        self._backend = backend

    def process(self, frame: Frame) -> TrackingStatus:
        """Process one frame and return a UI-friendly status object."""

        observation = self._backend.process(frame)
        message = "face tracked" if observation.valid else (
            observation.reason or "tracker observation invalid"
        )
        return TrackingStatus(observation=observation, message=message)

    def close(self) -> None:
        """Release tracker backend resources."""

        self._backend.close()
