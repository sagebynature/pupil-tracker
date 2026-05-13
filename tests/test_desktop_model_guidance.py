"""Tests for MediaPipe model setup guidance in the desktop demo."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from pupil_tracker.models import FrameMetadata
from pupil_tracker.tracking import Frame

APPS_ROOT = Path(__file__).resolve().parents[1] / "apps"
if str(APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPS_ROOT))


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


@pytest.fixture(scope="module")
def qt_app() -> Iterator[QApplication]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield cast(QApplication, app)


def test_missing_model_path_shows_tracker_setup_guidance(qt_app: QApplication) -> None:
    from desktop_demo.ui.main_window import MainWindow

    window = MainWindow(model_asset_path=None)

    window.start_tracking()

    assert "PUPIL_TRACKER_MEDIAPIPE_MODEL" in window.debug_label.text()
    assert "FaceLandmarker" in window.debug_label.text()
    assert window.tracking_runtime is None
    window.close()
    qt_app.processEvents()


def test_invalid_model_path_shows_tracker_setup_guidance(
    qt_app: QApplication,
    tmp_path: Path,
) -> None:
    from desktop_demo.ui.main_window import MainWindow

    missing_path = tmp_path / "missing.task"
    window = MainWindow(model_asset_path=missing_path)

    window.start_tracking()

    assert str(missing_path) in window.debug_label.text()
    assert "PUPIL_TRACKER_MEDIAPIPE_MODEL" in window.debug_label.text()
    assert window.tracking_runtime is None
    window.close()
    qt_app.processEvents()


def test_start_calibration_requires_tracker_model_guidance(qt_app: QApplication) -> None:
    from desktop_demo.calibration_session import CalibrationSessionState
    from desktop_demo.ui.main_window import MainWindow

    window = MainWindow(model_asset_path=None)

    window.start_calibration()

    assert "PUPIL_TRACKER_MEDIAPIPE_MODEL" in window.debug_label.text()
    assert window.calibration_session.state is CalibrationSessionState.IDLE
    window.close()
    qt_app.processEvents()


def test_camera_preview_still_starts_without_model_path(qt_app: QApplication) -> None:
    from desktop_demo.ui.main_window import MainWindow

    camera = FakeCamera()
    window = MainWindow(model_asset_path=None, camera_factory=lambda: camera)

    window.start_camera()

    assert camera.open_calls == 1
    assert window.preview_timer.isActive()
    assert window.preview_label.text() == "Camera preview running"
    window.stop_camera()
    window.close()
    qt_app.processEvents()


def test_readme_documents_mediapipe_model_environment_variable() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "PUPIL_TRACKER_MEDIAPIPE_MODEL" in readme
    assert "export PUPIL_TRACKER_MEDIAPIPE_MODEL=" in readme
    assert "make run-demo" in readme
