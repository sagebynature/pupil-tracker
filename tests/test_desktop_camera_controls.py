"""Tests for desktop camera start/stop wiring."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from pupil_tracker.camera import CameraError
from pupil_tracker.models import FrameMetadata
from pupil_tracker.tracking import Frame

APPS_ROOT = Path(__file__).resolve().parents[1] / "apps"
if str(APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPS_ROOT))


def fake_frame() -> Frame:
    return Frame(
        image=np.zeros((3, 2, 3), dtype=np.uint8),
        metadata=FrameMetadata(timestamp=1.0, camera_id=0, width=2, height=3, channels=3),
    )


class FakeCamera:
    def __init__(self, frames: list[Frame] | None = None) -> None:
        self.open_calls = 0
        self.close_calls = 0
        self.read_calls = 0
        self.frames = frames if frames is not None else []
        self.is_open = False

    def open(self) -> None:
        self.open_calls += 1
        self.is_open = True

    def close(self) -> None:
        self.close_calls += 1
        self.is_open = False

    def read(self) -> Frame:
        self.read_calls += 1
        if not self.frames:
            msg = "no fake frames"
            raise CameraError(msg)
        return self.frames.pop(0)


class FailingCamera(FakeCamera):
    def open(self) -> None:
        self.open_calls += 1
        msg = "failed to open camera 0"
        raise CameraError(msg)


@pytest.fixture(scope="module")
def qt_app() -> Iterator[QApplication]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield cast(QApplication, app)


def test_start_camera_opens_injected_camera_source(qt_app: QApplication) -> None:
    from desktop_demo.ui.main_window import MainWindow

    camera = FakeCamera()
    window = MainWindow(camera_factory=lambda: camera)
    qt_app.processEvents()

    window.start_camera()

    assert camera.open_calls == 1
    assert camera.is_open is True
    assert window.worker.running is True
    assert window.preview_label.text() == "Camera preview running"


def test_stop_camera_closes_open_camera_source(qt_app: QApplication) -> None:
    from desktop_demo.ui.main_window import MainWindow

    camera = FakeCamera()
    window = MainWindow(camera_factory=lambda: camera)
    window.start_camera()

    window.stop_camera()

    assert camera.close_calls == 1
    assert camera.is_open is False
    assert window.worker.running is False
    assert window.preview_label.text() == "Camera preview stopped"


def test_start_camera_failure_is_reported_without_running(qt_app: QApplication) -> None:
    from desktop_demo.ui.main_window import MainWindow

    camera = FailingCamera()
    window = MainWindow(camera_factory=lambda: camera)

    window.start_camera()

    assert camera.open_calls == 1
    assert window.worker.running is False
    assert "failed to open camera" in window.preview_label.text()


def test_camera_worker_tick_reads_frame_from_open_camera() -> None:
    from desktop_demo.ui.main_window import CameraPreviewWorker

    camera = FakeCamera(frames=[fake_frame()])
    worker = CameraPreviewWorker(lambda: camera)
    worker.start()

    frame = worker.tick()

    assert frame is not None
    assert frame.metadata.width == 2
    assert frame.metadata.height == 3
    assert camera.read_calls == 1


def test_camera_worker_tick_without_running_returns_none() -> None:
    from desktop_demo.ui.main_window import CameraPreviewWorker

    camera = FakeCamera(frames=[fake_frame()])
    worker = CameraPreviewWorker(lambda: camera)

    frame = worker.tick()

    assert frame is None
    assert camera.read_calls == 0


def test_camera_worker_tick_failure_stops_worker() -> None:
    from desktop_demo.ui.main_window import CameraPreviewWorker

    camera = FakeCamera()
    worker = CameraPreviewWorker(lambda: camera)
    worker.start()

    with pytest.raises(CameraError, match="no fake frames"):
        worker.tick()

    assert worker.running is False
    assert camera.close_calls == 1
