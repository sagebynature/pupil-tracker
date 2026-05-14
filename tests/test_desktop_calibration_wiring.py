"""Tests for wiring live tracker observations into calibration."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import QApplication

from pupil_tracker.calibration import (
    CalibrationFitResult,
    LinearRidgeCalibrationModel,
    TimedCalibrationConfig,
    validation_pattern,
)
from pupil_tracker.models import FrameMetadata, GazeSample, RawObservation
from pupil_tracker.tracking import Frame

APPS_ROOT = Path(__file__).resolve().parents[1] / "apps"
if str(APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPS_ROOT))

from desktop_demo.calibration_session import CalibrationSessionState  # noqa: E402
from desktop_demo.tracking_runtime import TrackingStatus  # noqa: E402
from desktop_demo.ui.calibration_view import CalibrationFlowState  # noqa: E402


class FakeCamera:
    def __init__(self, frame: Frame) -> None:
        self.frame = frame

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def read(self) -> Frame:
        return self.frame


class FakeTrackingRuntime:
    def __init__(self, observations: list[RawObservation]) -> None:
        self.observations = observations
        self.process_calls = 0
        self.close_calls = 0

    def process(self, frame: Frame) -> TrackingStatus:
        del frame
        observation = self.observations[min(self.process_calls, len(self.observations) - 1)]
        self.process_calls += 1
        message = "face tracked" if observation.valid else observation.reason or "invalid"
        return TrackingStatus(observation=observation, message=message)

    def close(self) -> None:
        self.close_calls += 1


class FakeCalibrationModel:
    def __init__(self) -> None:
        self.fit_calls = 0

    def fit(
        self,
        samples,
        screen_width: float,
        screen_height: float,
    ) -> CalibrationFitResult:
        del screen_width, screen_height
        self.fit_calls += 1
        return CalibrationFitResult(
            sample_count=len(samples),
            mean_error_px=1.5,
            max_error_px=3.0,
        )


class FakeGazeRuntime:
    def __init__(self, sample: GazeSample | None) -> None:
        self.sample = sample
        self.update_calls = 0

    def update(
        self,
        observation: RawObservation,
        *,
        screen_width: float,
        screen_height: float,
    ) -> GazeSample | None:
        del observation, screen_width, screen_height
        self.update_calls += 1
        return self.sample


@pytest.fixture(scope="module")
def qt_app() -> Iterator[QApplication]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield cast(QApplication, app)


def fake_frame() -> Frame:
    return Frame(
        image=np.zeros((20, 20, 3), dtype=np.uint8),
        metadata=FrameMetadata(timestamp=1.0, camera_id=0, width=20, height=20, channels=3),
    )


def valid_observation(timestamp: float = 1.0) -> RawObservation:
    return RawObservation(
        timestamp=timestamp,
        valid=True,
        confidence=0.9,
        feature_vector=(timestamp, timestamp + 0.1),
    )


def gaze_sample(*, x: float = 250.0, y: float = 200.0) -> GazeSample:
    return GazeSample(
        timestamp=1.0,
        x=x,
        y=y,
        confidence=0.9,
        valid=True,
        region_id="validation",
    )


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def timed_config() -> TimedCalibrationConfig:
    return TimedCalibrationConfig(
        settle_seconds=1.0,
        capture_seconds=1.0,
        min_samples_per_target=2,
        min_confidence=0.6,
    )


def validation_config() -> TimedCalibrationConfig:
    return TimedCalibrationConfig(
        settle_seconds=1.0,
        capture_seconds=1.0,
        min_samples_per_target=1,
        min_confidence=0.0,
    )


def test_start_vertical_calibration_uses_dense_vertical_pattern(
    qt_app: QApplication,
) -> None:
    from desktop_demo.ui.main_window import MainWindow

    window = MainWindow(
        camera_factory=lambda: FakeCamera(fake_frame()),
        tracking_runtime=FakeTrackingRuntime([valid_observation()]),
    )

    window.start_vertical_calibration()

    flow = window.calibration_view.flow
    assert len(flow.targets) == 15
    assert flow.targets[0].id == "r0c0"
    assert flow.targets[-1].id == "r4c2"
    assert window.calibration_session.flow is flow
    assert window.calibration_view.flow is flow
    assert window.calibration_target_overlay.flow is flow
    assert isinstance(window.calibration_session.model, LinearRidgeCalibrationModel)
    assert window.calibration_view.title_label.text() == "15-point linear vertical calibration"
    assert window.calibration_target_overlay.isVisible()
    window.close()
    qt_app.processEvents()


def test_start_calibration_shows_full_screen_target_overlay(qt_app: QApplication) -> None:
    from desktop_demo.calibration_session import CalibrationSession
    from desktop_demo.ui.main_window import MainWindow

    flow = CalibrationFlowState(samples_per_target=1)
    session = CalibrationSession(
        flow=flow,
        model=FakeCalibrationModel(),
        screen_width=1000,
        screen_height=800,
    )
    window = MainWindow(
        camera_factory=lambda: FakeCamera(fake_frame()),
        calibration_session=session,
    )

    window.start_calibration()

    screen = QApplication.primaryScreen()
    expected_geometry = screen.geometry() if screen is not None else QRect(0, 0, 1920, 1080)
    assert window.calibration_target_overlay.geometry() == expected_geometry
    assert window.calibration_target_overlay.isVisible()
    assert window.calibration_target_overlay.testAttribute(
        Qt.WidgetAttribute.WA_TransparentForMouseEvents
    )
    assert (
        window.calibration_target_overlay.windowFlags()
        & Qt.WindowType.WindowTransparentForInput
    )
    assert window.calibration_target_overlay.flow is flow
    window.close()
    qt_app.processEvents()


def test_start_calibration_starts_session_and_refreshes_view(qt_app: QApplication) -> None:
    from desktop_demo.calibration_session import CalibrationSession
    from desktop_demo.ui.main_window import MainWindow

    flow = CalibrationFlowState(samples_per_target=1)
    session = CalibrationSession(
        flow=flow,
        model=FakeCalibrationModel(),
        screen_width=1000,
        screen_height=800,
    )
    window = MainWindow(
        camera_factory=lambda: FakeCamera(fake_frame()),
        calibration_session=session,
    )

    window.start_calibration()

    assert session.state is CalibrationSessionState.COLLECTING
    assert window.calibration_view.flow is flow
    assert "Calibration collecting" in window.debug_label.text()
    window.close()
    qt_app.processEvents()


def test_live_frame_updates_capture_observations_when_calibrating(qt_app: QApplication) -> None:
    from desktop_demo.calibration_session import CalibrationSession
    from desktop_demo.ui.main_window import MainWindow

    flow = CalibrationFlowState(samples_per_target=2)
    session = CalibrationSession(
        flow=flow,
        model=FakeCalibrationModel(),
        screen_width=1000,
        screen_height=800,
    )
    tracker = FakeTrackingRuntime(observations=[valid_observation()])
    window = MainWindow(
        camera_factory=lambda: FakeCamera(fake_frame()),
        tracking_runtime=tracker,
        calibration_session=session,
    )
    window.start_camera()
    window.start_calibration()

    window.update_preview_frame()

    assert len(window.calibration_session.flow.all_samples()) == 1
    assert "Samples: 1/2" in window.calibration_view.status_label.text()
    assert "confidence 0.90" in window.debug_label.text()
    window.close()
    qt_app.processEvents()


def test_invalid_live_observation_does_not_advance_calibration(qt_app: QApplication) -> None:
    from desktop_demo.calibration_session import CalibrationSession
    from desktop_demo.ui.main_window import MainWindow

    flow = CalibrationFlowState(samples_per_target=1)
    session = CalibrationSession(
        flow=flow,
        model=FakeCalibrationModel(),
        screen_width=1000,
        screen_height=800,
    )
    tracker = FakeTrackingRuntime(
        observations=[RawObservation.invalid(timestamp=1.0, reason="no face")]
    )
    window = MainWindow(
        camera_factory=lambda: FakeCamera(fake_frame()),
        tracking_runtime=tracker,
        calibration_session=session,
    )
    window.start_camera()
    window.start_calibration()

    window.update_preview_frame()

    assert window.calibration_session.flow.all_samples() == ()
    assert window.calibration_session.state is CalibrationSessionState.COLLECTING
    assert "Samples: 0/1" in window.calibration_view.status_label.text()
    window.close()
    qt_app.processEvents()


def test_completed_live_calibration_shows_fit_metrics(qt_app: QApplication) -> None:
    from desktop_demo.calibration_session import CalibrationSession
    from desktop_demo.ui.main_window import MainWindow

    model = FakeCalibrationModel()
    flow = CalibrationFlowState(samples_per_target=1)
    session = CalibrationSession(flow=flow, model=model, screen_width=1000, screen_height=800)
    observations = [valid_observation(timestamp=float(index)) for index in range(9)]
    tracker = FakeTrackingRuntime(observations=observations)
    window = MainWindow(
        camera_factory=lambda: FakeCamera(fake_frame()),
        tracking_runtime=tracker,
        calibration_session=session,
    )
    window.start_camera()
    window.start_calibration()

    for _ in observations:
        window.update_preview_frame()

    assert session.state is CalibrationSessionState.COMPLETE
    assert model.fit_calls == 1
    assert "Calibration complete" in window.debug_label.text()
    assert "mean error 1.50px" in window.debug_label.text()
    assert not window.calibration_target_overlay.isVisible()
    window.close()
    qt_app.processEvents()


def test_timed_calibration_start_shows_settle_guidance(qt_app: QApplication) -> None:
    from desktop_demo.calibration_session import CalibrationSession
    from desktop_demo.ui.main_window import MainWindow

    clock = FakeClock()
    flow = CalibrationFlowState(samples_per_target=2)
    session = CalibrationSession(
        flow=flow,
        model=FakeCalibrationModel(),
        screen_width=1000,
        screen_height=800,
        timing_config=timed_config(),
        clock=clock,
    )
    window = MainWindow(
        camera_factory=lambda: FakeCamera(fake_frame()),
        calibration_session=session,
    )

    window.start_calibration()

    assert "Settle: look at the dot" in window.calibration_view.status_label.text()
    assert "Accepted: 0/2 | Rejected: 0" in window.calibration_view.status_label.text()
    assert "Calibration target 1/9" in window.debug_label.text()
    window.close()
    qt_app.processEvents()


def test_timed_calibration_live_frame_shows_capture_progress(qt_app: QApplication) -> None:
    from desktop_demo.calibration_session import CalibrationSession
    from desktop_demo.ui.main_window import MainWindow

    clock = FakeClock()
    flow = CalibrationFlowState(samples_per_target=2)
    session = CalibrationSession(
        flow=flow,
        model=FakeCalibrationModel(),
        screen_width=1000,
        screen_height=800,
        timing_config=timed_config(),
        clock=clock,
    )
    tracker = FakeTrackingRuntime(observations=[valid_observation()])
    window = MainWindow(
        camera_factory=lambda: FakeCamera(fake_frame()),
        tracking_runtime=tracker,
        calibration_session=session,
    )
    window.start_camera()
    window.start_calibration()
    clock.advance(1.0)

    window.update_preview_frame()

    assert "Capturing:" in window.calibration_view.status_label.text()
    assert "Accepted: 1/2 | Rejected: 0" in window.calibration_view.status_label.text()
    assert "Calibration target 1/9" in window.debug_label.text()
    window.close()
    qt_app.processEvents()


def test_completed_calibration_enables_validation_control(qt_app: QApplication) -> None:
    from desktop_demo.calibration_session import CalibrationSession
    from desktop_demo.ui.main_window import MainWindow

    session = CalibrationSession(
        flow=CalibrationFlowState(samples_per_target=1),
        model=FakeCalibrationModel(),
        screen_width=1000,
        screen_height=800,
    )
    session.state = CalibrationSessionState.COMPLETE
    session.fit_result = CalibrationFitResult(
        sample_count=45,
        mean_error_px=1.0,
        max_error_px=2.0,
    )
    window = MainWindow(
        camera_factory=lambda: FakeCamera(fake_frame()),
        calibration_session=session,
    )

    window._update_calibration_status(None)

    assert window.calibration_view.validation_button.isEnabled()
    assert "Start validation" in window.debug_label.text()
    window.close()
    qt_app.processEvents()


def test_live_calibrated_gaze_updates_validation_session_and_overlay(
    qt_app: QApplication,
) -> None:
    from desktop_demo.calibration_session import CalibrationSession
    from desktop_demo.ui.main_window import MainWindow
    from desktop_demo.validation_session import ValidationSession, ValidationSessionState

    clock = FakeClock()
    calibration_session = CalibrationSession(
        flow=CalibrationFlowState(samples_per_target=1),
        model=FakeCalibrationModel(),
        screen_width=1000,
        screen_height=800,
    )
    calibration_session.state = CalibrationSessionState.COMPLETE
    validation_session = ValidationSession(
        targets=validation_pattern()[:1],
        screen_width=1000,
        screen_height=800,
        timing_config=validation_config(),
        clock=clock,
    )
    window = MainWindow(
        camera_factory=lambda: FakeCamera(fake_frame()),
        tracking_runtime=FakeTrackingRuntime([valid_observation()]),
        calibration_session=calibration_session,
        gaze_runtime=FakeGazeRuntime(gaze_sample()),
        validation_session=validation_session,
    )
    window.start_camera()
    window.start_validation()
    clock.advance(1.0)

    window.update_preview_frame()

    assert validation_session.state is ValidationSessionState.CAPTURING
    assert validation_session.accepted_for_current_target == 1
    assert window.gaze_overlay.validation_state.current is not None
    assert "Validation target 1/1" in window.debug_label.text()
    window.close()
    qt_app.processEvents()


def test_validation_completion_displays_metrics_and_retry_guidance(
    qt_app: QApplication,
) -> None:
    from desktop_demo.calibration_session import CalibrationSession
    from desktop_demo.ui.main_window import MainWindow
    from desktop_demo.validation_session import ValidationSession, ValidationSessionState

    clock = FakeClock()
    calibration_session = CalibrationSession(
        flow=CalibrationFlowState(samples_per_target=1),
        model=FakeCalibrationModel(),
        screen_width=1000,
        screen_height=800,
    )
    calibration_session.state = CalibrationSessionState.COMPLETE
    validation_session = ValidationSession(
        targets=validation_pattern()[:1],
        screen_width=1000,
        screen_height=800,
        timing_config=validation_config(),
        clock=clock,
    )
    window = MainWindow(
        camera_factory=lambda: FakeCamera(fake_frame()),
        tracking_runtime=FakeTrackingRuntime([valid_observation()]),
        calibration_session=calibration_session,
        gaze_runtime=FakeGazeRuntime(gaze_sample(x=650.0, y=200.0)),
        validation_session=validation_session,
    )
    window.start_camera()
    window.start_validation()
    clock.advance(1.0)
    window.update_preview_frame()
    clock.advance(1.1)

    window.update_preview_frame()

    assert validation_session.state is ValidationSessionState.COMPLETE
    assert validation_session.metrics is not None
    assert validation_session.metrics.recommendation == "retry"
    assert "Validation complete" in window.debug_label.text()
    assert "mean X error" in window.debug_label.text()
    assert "mean Y error" in window.debug_label.text()
    assert "Y bias" in window.debug_label.text()
    assert "retry calibration" in window.debug_label.text()
    window.close()
    qt_app.processEvents()
