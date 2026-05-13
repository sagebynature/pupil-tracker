"""Tests for live preview timer and frame rendering."""

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


class FakeCamera:
    def __init__(self, frames: list[Frame] | None = None) -> None:
        self.frames = frames if frames is not None else []
        self.open_calls = 0
        self.close_calls = 0
        self.read_calls = 0
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
            msg = "preview read failed"
            raise CameraError(msg)
        return self.frames.pop(0)


@pytest.fixture(scope="module")
def qt_app() -> Iterator[QApplication]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield cast(QApplication, app)


def fake_frame() -> Frame:
    image = np.array(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=np.uint8,
    )
    return Frame(
        image=image,
        metadata=FrameMetadata(timestamp=1.0, camera_id=0, width=2, height=2, channels=3),
    )


def test_start_camera_starts_preview_timer(qt_app: QApplication) -> None:
    from desktop_demo.ui.main_window import MainWindow

    camera = FakeCamera(frames=[fake_frame()])
    window = MainWindow(camera_factory=lambda: camera, preview_interval_ms=33)
    qt_app.processEvents()

    window.start_camera()

    assert window.preview_timer.isActive()
    assert window.preview_timer.interval() == 33


def test_stop_camera_stops_preview_timer(qt_app: QApplication) -> None:
    from desktop_demo.ui.main_window import MainWindow

    camera = FakeCamera(frames=[fake_frame()])
    window = MainWindow(camera_factory=lambda: camera, preview_interval_ms=33)
    window.start_camera()

    window.stop_camera()

    assert not window.preview_timer.isActive()
    assert camera.close_calls == 1


def test_update_preview_frame_renders_camera_pixmap(qt_app: QApplication) -> None:
    from desktop_demo.ui.main_window import MainWindow

    camera = FakeCamera(frames=[fake_frame()])
    window = MainWindow(camera_factory=lambda: camera)
    window.start_camera()

    window.update_preview_frame()

    pixmap = window.preview_label.pixmap()
    assert pixmap is not None
    assert not pixmap.isNull()
    assert camera.read_calls == 1


def test_update_preview_frame_failure_stops_timer_and_reports_error(qt_app: QApplication) -> None:
    from desktop_demo.ui.main_window import MainWindow

    camera = FakeCamera()
    window = MainWindow(camera_factory=lambda: camera)
    window.start_camera()

    window.update_preview_frame()

    assert not window.preview_timer.isActive()
    assert window.worker.running is False
    assert "preview read failed" in window.preview_label.text()
