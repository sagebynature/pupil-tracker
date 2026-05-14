"""Tests for the MediaPipe tracker backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from pupil_tracker.models import FrameMetadata
from pupil_tracker.tracking import Frame
from pupil_tracker.tracking.mediapipe_backend import MediaPipeTrackerBackend


@dataclass(frozen=True)
class FakeLandmark:
    x: float
    y: float


class FakeFaceLandmarks:
    def __init__(self, landmarks: list[FakeLandmark]) -> None:
        self.landmark = landmarks


class FakeResult:
    def __init__(self, faces: list[FakeFaceLandmarks] | None) -> None:
        self.multi_face_landmarks = faces


class FakeFaceMesh:
    def __init__(self, result: FakeResult) -> None:
        self.result = result
        self.processed_images: list[Any] = []
        self.closed = False

    def process(self, image: Any) -> FakeResult:
        self.processed_images.append(image)
        return self.result

    def close(self) -> None:
        self.closed = True


def _frame() -> Frame:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    metadata = FrameMetadata(timestamp=1.25, camera_id=0, width=100, height=100, channels=3)
    return Frame(image=image, metadata=metadata)


def _landmarks() -> list[FakeLandmark]:
    landmarks = [FakeLandmark(x=0.5, y=0.5) for _ in range(478)]
    landmarks[0] = FakeLandmark(x=0.1, y=0.2)
    landmarks[1] = FakeLandmark(x=0.9, y=0.8)
    for index in MediaPipeTrackerBackend.LEFT_IRIS_INDICES:
        landmarks[index] = FakeLandmark(x=0.3, y=0.4)
    for index in MediaPipeTrackerBackend.RIGHT_IRIS_INDICES:
        landmarks[index] = FakeLandmark(x=0.7, y=0.5)
    for index, point in zip(
        MediaPipeTrackerBackend.LEFT_EYE_INDICES,
        (
            FakeLandmark(x=0.2, y=0.3),
            FakeLandmark(x=0.4, y=0.3),
            FakeLandmark(x=0.3, y=0.3),
            FakeLandmark(x=0.3, y=0.5),
        ),
        strict=True,
    ):
        landmarks[index] = point
    for index, point in zip(
        MediaPipeTrackerBackend.RIGHT_EYE_INDICES,
        (
            FakeLandmark(x=0.6, y=0.4),
            FakeLandmark(x=0.8, y=0.4),
            FakeLandmark(x=0.7, y=0.4),
            FakeLandmark(x=0.7, y=0.6),
        ),
        strict=True,
    ):
        landmarks[index] = point
    return landmarks


def test_backend_name_is_mediapipe() -> None:
    face_mesh = FakeFaceMesh(FakeResult(faces=None))
    backend = MediaPipeTrackerBackend(face_mesh=face_mesh)

    assert backend.name == "mediapipe"


def test_no_detected_face_returns_invalid_observation_with_reason() -> None:
    face_mesh = FakeFaceMesh(FakeResult(faces=None))
    backend = MediaPipeTrackerBackend(face_mesh=face_mesh)

    observation = backend.process(_frame())

    assert not observation.valid
    assert observation.timestamp == 1.25
    assert observation.confidence == 0.0
    assert observation.reason == "no face detected"


def test_mocked_landmarks_produce_valid_observation_with_stable_feature_vector() -> None:
    face_mesh = FakeFaceMesh(FakeResult(faces=[FakeFaceLandmarks(_landmarks())]))
    backend = MediaPipeTrackerBackend(face_mesh=face_mesh)

    observation = backend.process(_frame())

    assert observation.valid
    assert observation.timestamp == 1.25
    assert observation.confidence == pytest.approx(1.0)
    assert observation.face_bounds is not None
    assert observation.face_bounds.x == pytest.approx(10.0)
    assert observation.face_bounds.y == pytest.approx(20.0)
    assert observation.face_bounds.width == pytest.approx(80.0)
    assert observation.face_bounds.height == pytest.approx(60.0)
    assert observation.left_iris is not None
    assert observation.left_iris.x == pytest.approx(30.0)
    assert observation.left_iris.y == pytest.approx(40.0)
    assert observation.right_iris is not None
    assert observation.right_iris.x == pytest.approx(70.0)
    assert observation.right_iris.y == pytest.approx(50.0)
    assert observation.feature_vector == pytest.approx(
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


def test_close_releases_mediapipe_resources() -> None:
    face_mesh = FakeFaceMesh(FakeResult(faces=None))
    backend = MediaPipeTrackerBackend(face_mesh=face_mesh)

    backend.close()

    assert face_mesh.closed
