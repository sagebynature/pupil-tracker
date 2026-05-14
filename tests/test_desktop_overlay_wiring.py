"""Tests for wiring calibrated gaze samples into the transparent overlay."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from pupil_tracker.calibration import ValidationTarget
from pupil_tracker.models import FrameMetadata, GazeSample, RawObservation
from pupil_tracker.tracking import Frame

APPS_ROOT = Path(__file__).resolve().parents[1] / "apps"
if str(APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPS_ROOT))

from desktop_demo.tracking_runtime import TrackingStatus  # noqa: E402


class FakeCamera:
    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def read(self) -> Frame:
        return Frame(
            image=np.zeros((20, 20, 3), dtype=np.uint8),
            metadata=FrameMetadata(timestamp=1.0, camera_id=0, width=20, height=20, channels=3),
        )


class FakeTrackingRuntime:
    def __init__(self, observation: RawObservation) -> None:
        self.observation = observation

    def process(self, frame: Frame) -> TrackingStatus:
        del frame
        return TrackingStatus(observation=self.observation, message="face tracked")

    def close(self) -> None:
        pass


class FakeGazeRuntime:
    def __init__(self, sample: GazeSample | None) -> None:
        self.sample = sample

    def update(
        self,
        observation: RawObservation,
        *,
        screen_width: float,
        screen_height: float,
    ) -> GazeSample | None:
        del observation, screen_width, screen_height
        return self.sample


@pytest.fixture(scope="module")
def qt_app() -> Iterator[QApplication]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield cast(QApplication, app)


def valid_observation() -> RawObservation:
    return RawObservation(
        timestamp=1.0,
        valid=True,
        confidence=0.8,
        feature_vector=(0.4, 0.6),
    )


def valid_sample() -> GazeSample:
    return GazeSample(
        timestamp=1.0,
        x=100.0,
        y=200.0,
        confidence=0.8,
        valid=True,
        region_id="middle_center",
    )


def test_tracking_sample_updates_overlay_state(qt_app: QApplication) -> None:
    from desktop_demo.ui.main_window import MainWindow

    window = MainWindow(camera_factory=FakeCamera)

    window.handle_gaze_sample(valid_sample())

    assert window.gaze_overlay.state.current is not None
    assert window.gaze_overlay.state.current.x == 100.0
    assert window.gaze_overlay.state.current.y == 200.0
    assert window.gaze_overlay.state.current.visible is True
    assert window.gaze_overlay.isVisible() is True
    window.close()
    qt_app.processEvents()


def test_invalid_tracking_sample_hides_overlay_cursor(qt_app: QApplication) -> None:
    from desktop_demo.ui.main_window import MainWindow

    window = MainWindow(camera_factory=FakeCamera)
    window.handle_gaze_sample(valid_sample())

    window.handle_gaze_sample(
        GazeSample(timestamp=2.0, x=0.0, y=0.0, confidence=0.0, valid=False)
    )

    assert window.gaze_overlay.state.current is not None
    assert window.gaze_overlay.state.current.visible is False
    assert window.gaze_overlay.isVisible() is False
    window.close()
    qt_app.processEvents()


def test_overlay_is_clickthrough_and_does_not_activate(qt_app: QApplication) -> None:
    from desktop_demo.ui.main_window import MainWindow

    window = MainWindow(camera_factory=FakeCamera)

    assert window.gaze_overlay.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert window.gaze_overlay.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    window.close()
    qt_app.processEvents()


def test_live_calibrated_gaze_updates_overlay(qt_app: QApplication) -> None:
    from desktop_demo.calibration_session import CalibrationSessionState
    from desktop_demo.ui.main_window import MainWindow

    observation = valid_observation()
    sample = valid_sample()
    window = MainWindow(
        camera_factory=FakeCamera,
        tracking_runtime=FakeTrackingRuntime(observation),
        gaze_runtime=FakeGazeRuntime(sample),
    )
    window.calibration_session.state = CalibrationSessionState.COMPLETE
    window.start_camera()

    window.update_preview_frame()

    assert window.gaze_overlay.state.current is not None
    assert window.gaze_overlay.state.current.x == sample.x
    assert window.gaze_overlay.state.current.y == sample.y
    assert window.gaze_overlay.isVisible() is True
    window.close()
    qt_app.processEvents()


def test_validation_sample_updates_overlay_state(qt_app: QApplication) -> None:
    from desktop_demo.ui.overlay import GazeOverlay

    overlay = GazeOverlay()
    target = ValidationTarget(id="v0", x=0.25, y=0.25)

    overlay.update_validation_sample(
        target=target,
        sample=valid_sample(),
        screen_width=400.0,
        screen_height=400.0,
    )

    assert overlay.validation_state.current is not None
    assert overlay.validation_state.current.target.x == 100.0
    assert overlay.validation_state.current.target.y == 100.0
    assert overlay.validation_state.current.prediction is not None
    assert overlay.validation_state.current.prediction.x == 100.0
    assert overlay.validation_state.current.prediction.y == 200.0
    assert overlay.validation_state.current.error_segment is not None
    overlay.close()
    qt_app.processEvents()


def test_validation_overlay_render_marks_target_pixel(qt_app: QApplication) -> None:
    from desktop_demo.ui.overlay import GazeOverlay

    overlay = GazeOverlay()
    overlay.resize(400, 400)
    overlay.update_validation_sample(
        target=ValidationTarget(id="v0", x=0.25, y=0.25),
        sample=valid_sample(),
        screen_width=400.0,
        screen_height=400.0,
    )
    image = QImage(400, 400, QImage.Format.Format_ARGB32)
    image.fill(QColor(0, 0, 0, 0))

    overlay.render(image)

    assert QColor(image.pixel(100, 100)).alpha() > 0
    overlay.close()
    qt_app.processEvents()
