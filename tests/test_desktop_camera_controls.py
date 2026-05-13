"""Tests for desktop camera start/stop wiring."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
from PySide6.QtWidgets import QApplication

from pupil_tracker.camera import CameraError

APPS_ROOT = Path(__file__).resolve().parents[1] / "apps"
if str(APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPS_ROOT))


class FakeCamera:
    def __init__(self) -> None:
        self.open_calls = 0
        self.close_calls = 0
        self.is_open = False

    def open(self) -> None:
        self.open_calls += 1
        self.is_open = True

    def close(self) -> None:
        self.close_calls += 1
        self.is_open = False


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
