from dataclasses import FrozenInstanceError

import pytest

from pupil_tracker.models import (
    CalibrationSample,
    CalibrationTarget,
    FrameMetadata,
    GazeSample,
    Point2D,
    RawObservation,
    Rect,
    WindowCandidate,
)


def test_rect_contains_point() -> None:
    rect = Rect(x=10, y=20, width=100, height=50)

    assert rect.contains(Point2D(10, 20))
    assert rect.contains(Point2D(110, 70))
    assert not rect.contains(Point2D(9, 20))
    assert not rect.contains(Point2D(10, 71))


def test_invalid_observation_has_reason_and_zero_confidence() -> None:
    observation = RawObservation.invalid(timestamp=1.25, reason="no face")

    assert not observation.valid
    assert observation.confidence == 0.0
    assert observation.reason == "no face"
    assert observation.feature_vector == ()


def test_raw_observation_can_store_frame_dimensions_without_image_payload() -> None:
    observation = RawObservation(
        timestamp=1.0,
        valid=True,
        confidence=0.9,
        frame_width=1280,
        frame_height=720,
    )

    assert observation.frame_width == 1280
    assert observation.frame_height == 720


def test_gaze_sample_stores_region_and_validity() -> None:
    sample = GazeSample(
        timestamp=2.0,
        x=320,
        y=240,
        confidence=0.8,
        valid=True,
        region_id="middle_center",
    )

    assert sample.region_id == "middle_center"
    assert sample.valid


def test_calibration_sample_links_target_to_observation() -> None:
    target = CalibrationTarget(id="r1c1", x=0.5, y=0.5)
    observation = RawObservation(
        timestamp=3.0,
        valid=True,
        confidence=0.9,
        feature_vector=(0.1, 0.2, 0.3),
    )

    sample = CalibrationSample(target=target, observation=observation)

    assert sample.target.id == "r1c1"
    assert sample.observation.feature_vector == (0.1, 0.2, 0.3)


def test_frame_metadata_stores_shape_without_image_payload() -> None:
    metadata = FrameMetadata(
        timestamp=4.0,
        camera_id=0,
        width=1280,
        height=720,
        channels=3,
    )

    assert metadata.width == 1280
    assert metadata.height == 720
    assert metadata.channels == 3


def test_window_candidate_uses_rect_bounds_and_score() -> None:
    candidate = WindowCandidate(
        app_name="Safari",
        title="Research",
        bounds=Rect(x=0, y=0, width=640, height=480),
        score=0.75,
    )

    assert candidate.bounds.contains(Point2D(100, 100))
    assert candidate.score == 0.75


def test_models_are_immutable() -> None:
    point = Point2D(1, 2)

    field_name = "x"
    with pytest.raises(FrozenInstanceError):
        setattr(point, field_name, 3)
