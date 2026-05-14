"""Pure feature extraction helpers for gaze calibration."""

from __future__ import annotations

from math import hypot

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


def eye_geometry_feature_vector(
    *,
    face_bounds: Rect,
    left_iris: Point2D | None,
    right_iris: Point2D | None,
    left_eye_bounds: Rect,
    right_eye_bounds: Rect,
) -> tuple[float, ...]:
    """Return iris features plus eye-relative vertical geometry.

    The first six values match :func:`iris_feature_vector` for backward
    readability. The remaining values add eye-box-relative iris positions,
    eye apertures normalized to face height, binocular vertical midpoint, and
    left/right vertical agreement.
    """

    _validate_bounds(left_eye_bounds, "left eye bounds")
    _validate_bounds(right_eye_bounds, "right eye bounds")
    base_features = iris_feature_vector(face_bounds, left_iris, right_iris)
    if left_iris is None or right_iris is None:
        msg = "iris landmarks are required"
        raise FeatureExtractionError(msg)

    left_eye_x, left_eye_y = _normalize_point(left_iris, left_eye_bounds)
    right_eye_x, right_eye_y = _normalize_point(right_iris, right_eye_bounds)
    left_aperture = left_eye_bounds.height / face_bounds.height
    right_aperture = right_eye_bounds.height / face_bounds.height
    mean_eye_y = (left_eye_y + right_eye_y) / 2.0
    vertical_agreement = left_eye_y - right_eye_y
    return (
        *base_features,
        left_eye_x,
        left_eye_y,
        right_eye_x,
        right_eye_y,
        left_aperture,
        right_aperture,
        mean_eye_y,
        vertical_agreement,
    )


def face_context_feature_vector(
    *,
    face_bounds: Rect,
    left_iris: Point2D | None,
    right_iris: Point2D | None,
    left_eye_bounds: Rect,
    right_eye_bounds: Rect,
    frame_width: int,
    frame_height: int,
) -> tuple[float, ...]:
    """Return eye geometry plus face position/scale context.

    The first fourteen values match :func:`eye_geometry_feature_vector`. The
    appended values are normalized face center, normalized face size, face
    aspect ratio, and inter-ocular distance normalized by frame width.
    """

    if frame_width <= 0 or frame_height <= 0:
        msg = "frame dimensions must have positive width and height"
        raise FeatureExtractionError(msg)
    eye_features = eye_geometry_feature_vector(
        face_bounds=face_bounds,
        left_iris=left_iris,
        right_iris=right_iris,
        left_eye_bounds=left_eye_bounds,
        right_eye_bounds=right_eye_bounds,
    )
    if left_iris is None or right_iris is None:
        msg = "iris landmarks are required"
        raise FeatureExtractionError(msg)
    face_center_x = (face_bounds.x + face_bounds.width / 2.0) / frame_width
    face_center_y = (face_bounds.y + face_bounds.height / 2.0) / frame_height
    face_width = face_bounds.width / frame_width
    face_height = face_bounds.height / frame_height
    face_aspect_ratio = face_bounds.width / face_bounds.height
    interocular_distance = hypot(left_iris.x - right_iris.x, left_iris.y - right_iris.y)
    normalized_interocular_distance = interocular_distance / frame_width
    return (
        *eye_features,
        face_center_x,
        face_center_y,
        face_width,
        face_height,
        face_aspect_ratio,
        normalized_interocular_distance,
    )


def head_pose_feature_vector(
    *,
    face_bounds: Rect,
    left_iris: Point2D | None,
    right_iris: Point2D | None,
    left_eye_bounds: Rect,
    right_eye_bounds: Rect,
    nose_tip: Point2D | None,
    frame_width: int,
    frame_height: int,
) -> tuple[float, ...]:
    """Return face context plus cheap roll/yaw/pitch head-pose proxies."""

    if nose_tip is None:
        msg = "nose tip landmark is required"
        raise FeatureExtractionError(msg)
    context_features = face_context_feature_vector(
        face_bounds=face_bounds,
        left_iris=left_iris,
        right_iris=right_iris,
        left_eye_bounds=left_eye_bounds,
        right_eye_bounds=right_eye_bounds,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    left_eye_center = _rect_center(left_eye_bounds)
    right_eye_center = _rect_center(right_eye_bounds)
    eye_midpoint = Point2D(
        x=(left_eye_center.x + right_eye_center.x) / 2.0,
        y=(left_eye_center.y + right_eye_center.y) / 2.0,
    )
    eye_dx = right_eye_center.x - left_eye_center.x
    if eye_dx == 0:
        msg = "eye centers must not be vertically aligned"
        raise FeatureExtractionError(msg)
    roll_proxy = (right_eye_center.y - left_eye_center.y) / eye_dx
    yaw_proxy = (nose_tip.x - eye_midpoint.x) / face_bounds.width
    pitch_proxy = (nose_tip.y - eye_midpoint.y) / face_bounds.height
    return (
        *context_features,
        roll_proxy,
        yaw_proxy,
        pitch_proxy,
    )


def _rect_center(bounds: Rect) -> Point2D:
    return Point2D(x=bounds.x + bounds.width / 2.0, y=bounds.y + bounds.height / 2.0)


def _validate_bounds(bounds: Rect, label: str) -> None:
    if bounds.width <= 0 or bounds.height <= 0:
        msg = f"{label} must have positive width and height"
        raise FeatureExtractionError(msg)


def _normalize_point(point: Point2D, bounds: Rect) -> tuple[float, float]:
    return (
        (point.x - bounds.x) / bounds.width,
        (point.y - bounds.y) / bounds.height,
    )
