"""Tests for the desktop demo tracking runtime seam."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from pupil_tracker.models import FrameMetadata, Point2D, RawObservation, Rect
from pupil_tracker.tracking import Frame

APPS_ROOT = Path(__file__).resolve().parents[1] / "apps"
if str(APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPS_ROOT))


class FakeBackend:
    name = "fake"

    def __init__(self, observation: RawObservation) -> None:
        self.observation = observation
        self.processed_frames: list[Frame] = []
        self.close_calls = 0

    def process(self, frame: Frame) -> RawObservation:
        self.processed_frames.append(frame)
        return self.observation

    def close(self) -> None:
        self.close_calls += 1


def fake_frame() -> Frame:
    return Frame(
        image=np.zeros((4, 6, 3), dtype=np.uint8),
        metadata=FrameMetadata(timestamp=1.0, camera_id=0, width=6, height=4, channels=3),
    )


def valid_observation() -> RawObservation:
    return RawObservation(
        timestamp=1.0,
        valid=True,
        confidence=1.0,
        face_bounds=Rect(x=1.0, y=2.0, width=3.0, height=4.0),
        left_iris=Point2D(x=2.0, y=3.0),
        right_iris=Point2D(x=4.0, y=3.0),
        feature_vector=(0.1, 0.2),
    )


def test_tracking_runtime_returns_observation_status() -> None:
    from desktop_demo.tracking_runtime import TrackingRuntime

    backend = FakeBackend(observation=valid_observation())
    frame = fake_frame()
    runtime = TrackingRuntime(backend=backend)

    status = runtime.process(frame)

    assert status.valid is True
    assert status.confidence == 1.0
    assert status.message == "face tracked"
    assert status.left_iris == Point2D(x=2.0, y=3.0)
    assert status.right_iris == Point2D(x=4.0, y=3.0)
    assert status.face_bounds == Rect(x=1.0, y=2.0, width=3.0, height=4.0)
    assert backend.processed_frames == [frame]


def test_tracking_runtime_reports_invalid_observation_reason() -> None:
    from desktop_demo.tracking_runtime import TrackingRuntime

    backend = FakeBackend(observation=RawObservation.invalid(timestamp=1.0, reason="no face"))
    runtime = TrackingRuntime(backend=backend)

    status = runtime.process(fake_frame())

    assert status.valid is False
    assert status.confidence == 0.0
    assert status.message == "no face"
    assert status.left_iris is None
    assert status.right_iris is None
    assert status.face_bounds is None


def test_tracking_runtime_invalid_without_reason_has_clear_message() -> None:
    from desktop_demo.tracking_runtime import TrackingRuntime

    backend = FakeBackend(observation=RawObservation(timestamp=1.0, valid=False, confidence=0.0))
    runtime = TrackingRuntime(backend=backend)

    status = runtime.process(fake_frame())

    assert status.message == "tracker observation invalid"


def test_tracking_runtime_close_releases_backend() -> None:
    from desktop_demo.tracking_runtime import TrackingRuntime

    backend = FakeBackend(observation=valid_observation())
    runtime = TrackingRuntime(backend=backend)

    runtime.close()

    assert backend.close_calls == 1
