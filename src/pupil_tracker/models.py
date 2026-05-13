"""Core data models shared by the tracking, calibration, and demo layers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Point2D:
    """A two-dimensional point in an arbitrary coordinate space."""

    x: float
    y: float


@dataclass(frozen=True)
class Rect:
    """Rectangle bounds in an arbitrary coordinate space."""

    x: float
    y: float
    width: float
    height: float

    def contains(self, point: Point2D) -> bool:
        """Return whether `point` is inside or on the rectangle boundary."""

        within_x = self.x <= point.x <= self.x + self.width
        within_y = self.y <= point.y <= self.y + self.height
        return within_x and within_y


@dataclass(frozen=True)
class FrameMetadata:
    """Frame metadata without the image payload."""

    timestamp: float
    camera_id: int | str
    width: int
    height: int
    channels: int


@dataclass(frozen=True)
class RawObservation:
    """Raw tracker observation before calibration to screen coordinates."""

    timestamp: float
    valid: bool
    confidence: float
    face_bounds: Rect | None = None
    left_iris: Point2D | None = None
    right_iris: Point2D | None = None
    feature_vector: tuple[float, ...] = ()
    reason: str | None = None

    @classmethod
    def invalid(cls, timestamp: float, reason: str) -> RawObservation:
        """Create an invalid observation with zero confidence."""

        return cls(timestamp=timestamp, valid=False, confidence=0.0, reason=reason)


@dataclass(frozen=True)
class GazeSample:
    """Calibrated gaze estimate in screen coordinates."""

    timestamp: float
    x: float
    y: float
    confidence: float
    valid: bool
    region_id: str | None = None


@dataclass(frozen=True)
class CalibrationTarget:
    """A normalized calibration target point."""

    id: str
    x: float
    y: float


@dataclass(frozen=True)
class CalibrationSample:
    """A raw observation captured for a known calibration target."""

    target: CalibrationTarget
    observation: RawObservation


@dataclass(frozen=True)
class WindowCandidate:
    """A visible application window candidate for a gaze point."""

    app_name: str
    title: str
    bounds: Rect
    score: float
