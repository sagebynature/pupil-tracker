"""Tests for wiring live tracker observations into calibration."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from pupil_tracker.calibration import CalibrationFitResult, TimedCalibrationConfig
from pupil_tracker.models import FrameMetadata, RawObservation
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
