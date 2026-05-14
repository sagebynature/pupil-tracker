"""Tests for pure gaze feature extraction helpers."""

import pytest

from pupil_tracker.models import Point2D, Rect
from pupil_tracker.tracking.features import (
    FeatureExtractionError,
    eye_geometry_feature_vector,
    face_context_feature_vector,
    head_pose_feature_vector,
    iris_feature_vector,
)


def test_iris_feature_vector_length_is_stable() -> None:
    features = iris_feature_vector(
        face_bounds=Rect(x=10, y=20, width=100, height=50),
        left_iris=Point2D(x=35, y=35),
        right_iris=Point2D(x=85, y=40),
    )

    assert len(features) == 6


def test_iris_centers_are_normalized_relative_to_face_bounds() -> None:
    features = iris_feature_vector(
        face_bounds=Rect(x=10, y=20, width=100, height=50),
        left_iris=Point2D(x=35, y=35),
        right_iris=Point2D(x=85, y=40),
    )

    assert features == pytest.approx((0.25, 0.3, 0.75, 0.4, 0.5, 0.35))


def test_eye_geometry_feature_vector_adds_vertical_eye_relative_features() -> None:
    features = eye_geometry_feature_vector(
        face_bounds=Rect(x=10, y=20, width=80, height=60),
        left_iris=Point2D(x=30, y=40),
        right_iris=Point2D(x=70, y=50),
        left_eye_bounds=Rect(x=20, y=30, width=20, height=20),
        right_eye_bounds=Rect(x=60, y=40, width=20, height=20),
    )

    assert len(features) == 14
    assert features == pytest.approx(
        (
            0.25,
            1 / 3,
            0.75,
            0.5,
            0.5,
            5 / 12,
            0.5,
            0.5,
            0.5,
            0.5,
            1 / 3,
            1 / 3,
            0.5,
            0.0,
        )
    )


def test_face_context_features_include_position_and_scale() -> None:
    eye_features = eye_geometry_feature_vector(
        face_bounds=Rect(x=20, y=30, width=160, height=90),
        left_iris=Point2D(x=70, y=70),
        right_iris=Point2D(x=150, y=75),
        left_eye_bounds=Rect(x=60, y=55, width=30, height=30),
        right_eye_bounds=Rect(x=135, y=60, width=30, height=30),
    )
    features = face_context_feature_vector(
        face_bounds=Rect(x=20, y=30, width=160, height=90),
        left_iris=Point2D(x=70, y=70),
        right_iris=Point2D(x=150, y=75),
        left_eye_bounds=Rect(x=60, y=55, width=30, height=30),
        right_eye_bounds=Rect(x=135, y=60, width=30, height=30),
        frame_width=400,
        frame_height=300,
    )

    assert len(features) == 20
    assert features[:14] == pytest.approx(eye_features)
    assert features[14:] == pytest.approx(
        (
            0.25,  # face center x in frame coordinates
            0.25,  # face center y in frame coordinates
            0.4,  # face width in frame coordinates
            0.3,  # face height in frame coordinates
            16 / 9,  # face aspect ratio
            (80**2 + 5**2) ** 0.5 / 400,  # inter-ocular distance in frame width units
        )
    )


def test_face_context_feature_vector_requires_valid_frame_dimensions() -> None:
    with pytest.raises(FeatureExtractionError, match="frame dimensions"):
        face_context_feature_vector(
            face_bounds=Rect(x=20, y=30, width=160, height=90),
            left_iris=Point2D(x=70, y=70),
            right_iris=Point2D(x=150, y=75),
            left_eye_bounds=Rect(x=60, y=55, width=30, height=30),
            right_eye_bounds=Rect(x=135, y=60, width=30, height=30),
            frame_width=0,
            frame_height=300,
        )


def test_head_pose_proxy_features_capture_pitch_yaw_roll() -> None:
    context_features = face_context_feature_vector(
        face_bounds=Rect(x=20, y=30, width=160, height=90),
        left_iris=Point2D(x=70, y=70),
        right_iris=Point2D(x=150, y=75),
        left_eye_bounds=Rect(x=60, y=55, width=30, height=30),
        right_eye_bounds=Rect(x=135, y=60, width=30, height=30),
        frame_width=400,
        frame_height=300,
    )
    features = head_pose_feature_vector(
        face_bounds=Rect(x=20, y=30, width=160, height=90),
        left_iris=Point2D(x=70, y=70),
        right_iris=Point2D(x=150, y=75),
        left_eye_bounds=Rect(x=60, y=55, width=30, height=30),
        right_eye_bounds=Rect(x=135, y=60, width=30, height=30),
        nose_tip=Point2D(x=120, y=90),
        frame_width=400,
        frame_height=300,
    )

    assert len(features) == 23
    assert features[:20] == pytest.approx(context_features)
    assert features[20:] == pytest.approx(
        (
            5 / 75,  # eye-line slope / roll proxy
            7.5 / 160,  # nose offset from eye midpoint / face width, yaw proxy
            17.5 / 90,  # nose offset from eye midpoint / face height, pitch proxy
        )
    )


def test_head_pose_feature_vector_requires_nose_tip() -> None:
    with pytest.raises(FeatureExtractionError, match="nose tip"):
        head_pose_feature_vector(
            face_bounds=Rect(x=20, y=30, width=160, height=90),
            left_iris=Point2D(x=70, y=70),
            right_iris=Point2D(x=150, y=75),
            left_eye_bounds=Rect(x=60, y=55, width=30, height=30),
            right_eye_bounds=Rect(x=135, y=60, width=30, height=30),
            nose_tip=None,
            frame_width=400,
            frame_height=300,
        )


def test_eye_geometry_feature_vector_requires_valid_eye_bounds() -> None:
    with pytest.raises(FeatureExtractionError, match="left eye bounds"):
        eye_geometry_feature_vector(
            face_bounds=Rect(x=10, y=20, width=80, height=60),
            left_iris=Point2D(x=30, y=40),
            right_iris=Point2D(x=70, y=50),
            left_eye_bounds=Rect(x=20, y=30, width=0, height=20),
            right_eye_bounds=Rect(x=60, y=40, width=20, height=20),
        )


def test_missing_left_iris_raises_feature_extraction_error() -> None:
    with pytest.raises(FeatureExtractionError, match="left iris"):
        iris_feature_vector(
            face_bounds=Rect(x=10, y=20, width=100, height=50),
            left_iris=None,
            right_iris=Point2D(x=85, y=40),
        )


def test_missing_right_iris_raises_feature_extraction_error() -> None:
    with pytest.raises(FeatureExtractionError, match="right iris"):
        iris_feature_vector(
            face_bounds=Rect(x=10, y=20, width=100, height=50),
            left_iris=Point2D(x=35, y=35),
            right_iris=None,
        )


def test_invalid_face_bounds_raise_feature_extraction_error() -> None:
    with pytest.raises(FeatureExtractionError, match="face bounds"):
        iris_feature_vector(
            face_bounds=Rect(x=10, y=20, width=0, height=50),
            left_iris=Point2D(x=35, y=35),
            right_iris=Point2D(x=85, y=40),
        )
