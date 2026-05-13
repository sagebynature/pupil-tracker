"""Tests for tracker annotations on preview frames."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from pupil_tracker.models import FrameMetadata, Point2D, RawObservation, Rect
from pupil_tracker.tracking import Frame

APPS_ROOT = Path(__file__).resolve().parents[1] / "apps"
if str(APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPS_ROOT))

from desktop_demo.tracking_runtime import TrackingStatus  # noqa: E402


class FakeCamera:
    def __init__(self, frame: Frame) -> None:
        self.frame = frame
        self.open_calls = 0
        self.close_calls = 0
        self.read_calls = 0

    def open(self) -> None:
        self.open_calls += 1

    def close(self) -> None:
        self.close_calls += 1

    def read(self) -> Frame:
        self.read_calls += 1
        return self.frame


class FakeTrackingRuntime:
    def __init__(self, observation: RawObservation) -> None:
        self.observation = observation
        self.processed_frames: list[Frame] = []
        self.close_calls = 0

    def process(self, frame: Frame) -> TrackingStatus:
        self.processed_frames.append(frame)
        return TrackingStatus(observation=self.observation, message="face tracked")

    def close(self) -> None:
        self.close_calls += 1


@pytest.fixture(scope="module")
def qt_app() -> Iterator[QApplication]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield cast(QApplication, app)


def preview_frame() -> Frame:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    return Frame(
        image=image,
        metadata=FrameMetadata(timestamp=1.0, camera_id=0, width=100, height=100, channels=3),
    )


def valid_observation() -> RawObservation:
    return RawObservation(
        timestamp=1.0,
        valid=True,
        confidence=0.9,
        face_bounds=Rect(x=10.0, y=20.0, width=30.0, height=40.0),
        left_iris=Point2D(x=20.0, y=30.0),
        right_iris=Point2D(x=70.0, y=30.0),
        feature_vector=(0.1, 0.2),
    )


def test_annotate_observation_draws_face_and_iris_points() -> None:
    from desktop_demo.ui.annotations import annotate_observation

    image = np.zeros((100, 100, 3), dtype=np.uint8)

    annotated = annotate_observation(image, valid_observation())

    assert np.any(annotated != image)
    assert annotated.shape == image.shape


def test_annotate_observation_does_not_mutate_input() -> None:
    from desktop_demo.ui.annotations import annotate_observation

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    original = image.copy()

    annotate_observation(image, valid_observation())

    np.testing.assert_array_equal(image, original)


def test_annotate_invalid_observation_is_safe_and_marks_frame() -> None:
    from desktop_demo.ui.annotations import annotate_observation

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    observation = RawObservation.invalid(timestamp=1.0, reason="no face")

    annotated = annotate_observation(image, observation)

    assert np.any(annotated != image)


def test_main_window_processes_tracker_and_renders_annotated_frame(qt_app: QApplication) -> None:
    from desktop_demo.ui.main_window import MainWindow

    frame = preview_frame()
    camera = FakeCamera(frame=frame)
    tracker = FakeTrackingRuntime(observation=valid_observation())
    window = MainWindow(camera_factory=lambda: camera, tracking_runtime=tracker)
    window.start_camera()

    window.update_preview_frame()

    assert tracker.processed_frames == [frame]
    assert "confidence 0.90" in window.debug_label.text()
    assert window.preview_label.pixmap() is not None
    window.close()
    qt_app.processEvents()


def test_main_window_closes_tracking_runtime(qt_app: QApplication) -> None:
    from desktop_demo.ui.main_window import MainWindow

    tracker = FakeTrackingRuntime(observation=valid_observation())
    window = MainWindow(
        camera_factory=lambda: FakeCamera(preview_frame()),
        tracking_runtime=tracker,
    )

    window.close()
    qt_app.processEvents()

    assert tracker.close_calls == 1
