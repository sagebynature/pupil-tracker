"""Transparent gaze overlay state and widget."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Final

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from pupil_tracker.calibration import ValidationTarget
from pupil_tracker.models import GazeSample, Point2D

_DEFAULT_DOT_RADIUS: Final[float] = 6.0
_DEFAULT_MIN_HALO_RADIUS: Final[float] = 14.0
_DEFAULT_MAX_HALO_RADIUS: Final[float] = 48.0


@dataclass(frozen=True)
class CursorRenderState:
    """Computed overlay drawing state for one gaze sample."""

    x: float
    y: float
    dot_radius: float
    halo_radius: float
    opacity: float
    visible: bool


class OverlayState:
    """Pure confidence-aware overlay state with bounded debug trail."""

    def __init__(
        self,
        *,
        dot_radius: float = _DEFAULT_DOT_RADIUS,
        min_halo_radius: float = _DEFAULT_MIN_HALO_RADIUS,
        max_halo_radius: float = _DEFAULT_MAX_HALO_RADIUS,
        max_trail_length: int = 30,
    ) -> None:
        if dot_radius <= 0:
            msg = "dot_radius must be positive"
            raise ValueError(msg)
        if min_halo_radius <= 0 or max_halo_radius < min_halo_radius:
            msg = "halo radius bounds are invalid"
            raise ValueError(msg)
        if max_trail_length <= 0:
            msg = "max_trail_length must be positive"
            raise ValueError(msg)

        self.dot_radius = dot_radius
        self.min_halo_radius = min_halo_radius
        self.max_halo_radius = max_halo_radius
        self._trail: deque[Point2D] = deque(maxlen=max_trail_length)
        self.current: CursorRenderState | None = None

    @property
    def trail(self) -> tuple[Point2D, ...]:
        """Return bounded valid gaze history for debug drawing."""

        return tuple(self._trail)

    def render_state_for(self, sample: GazeSample) -> CursorRenderState:
        """Return drawing state for a gaze sample without mutating trail history."""

        if not sample.valid:
            return CursorRenderState(
                x=sample.x,
                y=sample.y,
                dot_radius=self.dot_radius,
                halo_radius=0.0,
                opacity=0.0,
                visible=False,
            )

        confidence = _clamp(sample.confidence, 0.0, 1.0)
        halo_radius = self.max_halo_radius - (
            confidence * (self.max_halo_radius - self.min_halo_radius)
        )
        return CursorRenderState(
            x=sample.x,
            y=sample.y,
            dot_radius=self.dot_radius,
            halo_radius=halo_radius,
            opacity=confidence,
            visible=True,
        )

    def update(self, sample: GazeSample) -> CursorRenderState:
        """Update current render state and append valid samples to the trail."""

        self.current = self.render_state_for(sample)
        if sample.valid:
            self._trail.append(Point2D(sample.x, sample.y))
        return self.current


@dataclass(frozen=True)
class ErrorSegment:
    """Screen-space validation error line between target and prediction."""

    start: Point2D
    end: Point2D


@dataclass(frozen=True)
class ValidationRenderState:
    """Computed validation overlay drawing state."""

    target: Point2D
    prediction: CursorRenderState | None
    error_segment: ErrorSegment | None


class ValidationOverlayState:
    """Pure validation overlay state with target, prediction, error line, and trail."""

    def __init__(
        self,
        *,
        dot_radius: float = _DEFAULT_DOT_RADIUS,
        min_halo_radius: float = _DEFAULT_MIN_HALO_RADIUS,
        max_halo_radius: float = _DEFAULT_MAX_HALO_RADIUS,
        max_trail_length: int = 30,
    ) -> None:
        self._cursor_state = OverlayState(
            dot_radius=dot_radius,
            min_halo_radius=min_halo_radius,
            max_halo_radius=max_halo_radius,
            max_trail_length=max_trail_length,
        )
        self.current: ValidationRenderState | None = None

    @property
    def trail(self) -> tuple[Point2D, ...]:
        """Return bounded valid prediction history for validation drawing."""

        return self._cursor_state.trail

    def update_validation(
        self,
        *,
        target: ValidationTarget,
        sample: GazeSample,
        screen_width: float,
        screen_height: float,
    ) -> ValidationRenderState:
        """Update validation overlay from one target and predicted gaze sample."""

        target_point = Point2D(target.x * screen_width, target.y * screen_height)
        prediction = self._cursor_state.update(sample)
        if prediction.visible:
            error_segment = ErrorSegment(
                start=target_point,
                end=Point2D(prediction.x, prediction.y),
            )
            self.current = ValidationRenderState(
                target=target_point,
                prediction=prediction,
                error_segment=error_segment,
            )
        else:
            self.current = ValidationRenderState(
                target=target_point,
                prediction=None,
                error_segment=None,
            )
        return self.current


class GazeOverlay(QWidget):
    """Transparent click-through overlay for drawing gaze cursor state."""

    def __init__(self, state: OverlayState | None = None) -> None:
        super().__init__()
        self.state = state if state is not None else OverlayState()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

    def update_sample(self, sample: GazeSample) -> None:
        """Update overlay drawing state from a gaze sample."""

        self.state.update(sample)
        self.update()

    def paintEvent(self, event: object) -> None:
        """Draw current gaze cursor and bounded debug trail."""

        del event
        current = self.state.current
        if current is None or not current.visible:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        _draw_trail(painter, self.state.trail)
        _draw_cursor(painter, current)
        painter.end()


def _draw_cursor(painter: QPainter, current: CursorRenderState) -> None:
    halo_color = QColor(80, 180, 255, int(90 * current.opacity))
    dot_color = QColor(80, 220, 255, int(220 * current.opacity))
    center = QPointF(current.x, current.y)
    painter.setPen(QPen(halo_color, 2.0))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(center, current.halo_radius, current.halo_radius)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(dot_color)
    painter.drawEllipse(center, current.dot_radius, current.dot_radius)


def _draw_trail(painter: QPainter, trail: Sequence[Point2D]) -> None:
    if len(trail) < 2:
        return
    painter.setPen(QPen(QColor(80, 180, 255, 80), 1.5))
    for start, end in pairwise(trail):
        painter.drawLine(QPointF(start.x, start.y), QPointF(end.x, end.y))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)
