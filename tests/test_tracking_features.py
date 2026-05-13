"""Tests for pure gaze feature extraction helpers."""

import pytest

from pupil_tracker.models import Point2D, Rect
from pupil_tracker.tracking.features import FeatureExtractionError, iris_feature_vector


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
