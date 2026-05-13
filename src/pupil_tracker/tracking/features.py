"""Pure feature extraction helpers for gaze calibration."""

from __future__ import annotations

from pupil_tracker.models import Point2D, Rect


class FeatureExtractionError(ValueError):
    """Raised when required gaze features cannot be extracted."""


def iris_feature_vector(
    face_bounds: Rect,
    left_iris: Point2D | None,
    right_iris: Point2D | None,
) -> tuple[float, ...]:
    """Return stable normalized iris features relative to face bounds.

    The returned vector is:
    `(left_x, left_y, right_x, right_y, midpoint_x, midpoint_y)`, where all
    coordinates are normalized within `face_bounds`.
    """

    if face_bounds.width <= 0 or face_bounds.height <= 0:
        msg = "face bounds must have positive width and height"
        raise FeatureExtractionError(msg)
    if left_iris is None:
        msg = "left iris landmark is required"
        raise FeatureExtractionError(msg)
    if right_iris is None:
        msg = "right iris landmark is required"
        raise FeatureExtractionError(msg)

    left_x, left_y = _normalize_point(left_iris, face_bounds)
    right_x, right_y = _normalize_point(right_iris, face_bounds)
    return (
        left_x,
        left_y,
        right_x,
        right_y,
        (left_x + right_x) / 2.0,
        (left_y + right_y) / 2.0,
    )


def _normalize_point(point: Point2D, bounds: Rect) -> tuple[float, float]:
    return (
        (point.x - bounds.x) / bounds.width,
        (point.y - bounds.y) / bounds.height,
    )
