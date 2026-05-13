"""Tests for desktop demo resource lifecycle cleanup."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from pupil_tracker.models import FrameMetadata, GazeSample, RawObservation
from pupil_tracker.tracking import Frame

APPS_ROOT = Path(__file__).resolve().parents[1] / "apps"
if str(APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPS_ROOT))

from desktop_demo.tracking_runtime import TrackingStatus  # noqa: E402


class FakeCamera:
    def __init__(self) -> None:
        self.open_calls = 0
        self.close_calls = 0

    def open(self) -> None:
        self.open_calls += 1

    def close(self) -> None:
        self.close_calls += 1

    def read(self) -> Frame:
        return Frame(
            image=np.zeros((20, 20, 3), dtype=np.uint8),
            metadata=FrameMetadata(timestamp=1.0, camera_id=0, width=20, height=20, channels=3),
        )


class FakeTrackingRuntime:
    def __init__(self) -> None:
        self.close_calls = 0

    def process(self, frame: Frame) -> TrackingStatus:
        del frame
        return TrackingStatus(
            observation=RawObservation.invalid(timestamp=1.0, reason="not used"),
            message="not used",
        )

    def close(self) -> None:
        self.close_calls += 1


@pytest.fixture(scope="module")
def qt_app() -> Iterator[QApplication]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield cast(QApplication, app)


def valid_gaze_sample() -> GazeSample:
    return GazeSample(
        timestamp=1.0,
        x=10.0,
        y=20.0,
        confidence=0.9,
        valid=True,
        region_id="top_left",
    )


def test_stop_camera_stops_timer_camera_tracker_overlay_and_logging(
    qt_app: QApplication,
    tmp_path: Path,
) -> None:
    from desktop_demo.ui.main_window import MainWindow

    camera = FakeCamera()
    tracker = FakeTrackingRuntime()
    window = MainWindow(
        camera_factory=lambda: camera,
        tracking_runtime=tracker,
        telemetry_path=tmp_path / "demo.jsonl",
        window_provider=lambda: (),
    )
    window.start_camera()
    window.start_logging()
    window.handle_gaze_sample(valid_gaze_sample())

    window.stop_camera()

    assert not window.preview_timer.isActive()
    assert camera.close_calls == 1
    assert tracker.close_calls == 1
    assert window.tracking_runtime is None
    assert window.telemetry_logger is None
    assert not window.gaze_overlay.isVisible()
    window.close()
    qt_app.processEvents()


def test_close_stops_timer_camera_tracker_and_logging(
    qt_app: QApplication,
    tmp_path: Path,
) -> None:
    from desktop_demo.ui.main_window import MainWindow

    camera = FakeCamera()
    tracker = FakeTrackingRuntime()
    window = MainWindow(
        camera_factory=lambda: camera,
        tracking_runtime=tracker,
        telemetry_path=tmp_path / "demo.jsonl",
        window_provider=lambda: (),
    )
    window.start_camera()
    window.start_logging()

    window.close()
    qt_app.processEvents()

    assert not window.preview_timer.isActive()
    assert camera.close_calls == 1
    assert tracker.close_calls == 1
    assert window.tracking_runtime is None
    assert window.telemetry_logger is None
    assert not window.gaze_overlay.isVisible()


def test_repeated_stop_and_close_are_idempotent(
    qt_app: QApplication,
    tmp_path: Path,
) -> None:
    from desktop_demo.ui.main_window import MainWindow

    camera = FakeCamera()
    tracker = FakeTrackingRuntime()
    window = MainWindow(
        camera_factory=lambda: camera,
        tracking_runtime=tracker,
        telemetry_path=tmp_path / "demo.jsonl",
        window_provider=lambda: (),
    )
    window.start_camera()
    window.start_logging()

    window.stop_camera()
    window.stop_camera()
    window.close()
    qt_app.processEvents()

    assert camera.close_calls == 1
    assert tracker.close_calls == 1
    assert window.telemetry_logger is None
