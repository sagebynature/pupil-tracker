"""Tests for safe live-loop telemetry emitted by the desktop demo."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from pupil_tracker.calibration import CalibrationFitResult
from pupil_tracker.models import FrameMetadata, GazeSample, RawObservation, Rect, WindowCandidate
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
    def __init__(self, sample: GazeSample) -> None:
        self.sample = sample

    def update(
        self,
        observation: RawObservation,
        *,
        screen_width: float,
        screen_height: float,
    ) -> GazeSample:
        del observation, screen_width, screen_height
        return self.sample


class FakeModel:
    def fit(
        self,
        samples: object,
        screen_width: float,
        screen_height: float,
    ) -> CalibrationFitResult:
        del samples, screen_width, screen_height
        return CalibrationFitResult(sample_count=1, mean_error_px=0.0, max_error_px=0.0)


@pytest.fixture(scope="module")
def qt_app() -> Iterator[QApplication]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield cast(QApplication, app)


def read_events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def valid_observation(timestamp: float = 1.0) -> RawObservation:
    return RawObservation(
        timestamp=timestamp,
        valid=True,
        confidence=0.8,
        face_bounds=Rect(x=1.0, y=2.0, width=10.0, height=12.0),
        feature_vector=(0.1, 0.2),
    )


def gaze_sample() -> GazeSample:
    return GazeSample(
        timestamp=1.0,
        x=10.0,
        y=20.0,
        confidence=0.9,
        valid=True,
        region_id="top_left",
    )


def window_candidate() -> WindowCandidate:
    return WindowCandidate(
        app_name="DemoApp",
        title="Main Window",
        bounds=Rect(x=0.0, y=0.0, width=100.0, height=100.0),
        score=1.0,
    )


def test_live_gaze_sample_logs_safe_payload_when_logging_enabled(
    qt_app: QApplication,
    tmp_path: Path,
) -> None:
    from desktop_demo.ui.main_window import MainWindow

    log_path = tmp_path / "demo.jsonl"
    window = MainWindow(telemetry_path=log_path, window_provider=lambda: ())
    window.start_logging()

    window.handle_gaze_sample(gaze_sample())
    window.stop_logging()

    events = read_events(log_path)
    assert [event["event_type"] for event in events] == ["gaze_sample", "window_candidate"]
    assert events[0]["payload"] == {
        "timestamp": 1.0,
        "x": 10.0,
        "y": 20.0,
        "confidence": 0.9,
        "valid": True,
        "region_id": "top_left",
    }
    assert events[1]["payload"] == {"candidate": None}
    assert "image" not in json.dumps(events)
    window.close()
    qt_app.processEvents()


def test_live_loop_logs_observation_gaze_and_window_candidate(
    qt_app: QApplication,
    tmp_path: Path,
) -> None:
    from desktop_demo.calibration_session import CalibrationSessionState
    from desktop_demo.ui.main_window import MainWindow

    log_path = tmp_path / "demo.jsonl"
    observation = valid_observation()
    window = MainWindow(
        telemetry_path=log_path,
        camera_factory=FakeCamera,
        tracking_runtime=FakeTrackingRuntime(observation),
        gaze_runtime=FakeGazeRuntime(gaze_sample()),
        window_provider=lambda: (window_candidate(),),
    )
    window.calibration_session.state = CalibrationSessionState.COMPLETE
    window.start_camera()
    window.start_logging()

    window.update_preview_frame()
    window.stop_logging()

    events = read_events(log_path)
    assert [event["event_type"] for event in events] == [
        "raw_observation",
        "gaze_sample",
        "window_candidate",
    ]
    assert events[0]["payload"] == {
        "timestamp": 1.0,
        "valid": True,
        "confidence": 0.8,
        "reason": None,
    }
    window_payload = cast(dict[str, Any], events[2]["payload"])
    assert window_payload["app_name"] == "DemoApp"
    assert "image" not in json.dumps(events)
    assert "feature_vector" not in json.dumps(events)
    window.close()
    qt_app.processEvents()


def test_live_calibration_logs_progress_without_frame_payload(
    qt_app: QApplication,
    tmp_path: Path,
) -> None:
    from desktop_demo.calibration_session import CalibrationSession
    from desktop_demo.ui.calibration_view import CalibrationFlowState
    from desktop_demo.ui.main_window import MainWindow

    log_path = tmp_path / "demo.jsonl"
    flow = CalibrationFlowState(samples_per_target=1)
    session = CalibrationSession(
        flow=flow,
        model=FakeModel(),
        screen_width=100.0,
        screen_height=100.0,
    )
    window = MainWindow(
        telemetry_path=log_path,
        camera_factory=FakeCamera,
        tracking_runtime=FakeTrackingRuntime(valid_observation()),
        calibration_session=session,
        window_provider=lambda: (),
    )
    window.start_camera()
    window.start_logging()
    window.start_calibration()

    window.update_preview_frame()
    window.stop_logging()

    events = read_events(log_path)
    event_types = [event["event_type"] for event in events]
    assert "raw_observation" in event_types
    assert "calibration_sample" in event_types
    calibration_event = next(
        event for event in events if event["event_type"] == "calibration_sample"
    )
    assert calibration_event["payload"] == {
        "target_id": "r0c0",
        "target_x": 0.1,
        "target_y": 0.1,
        "sample_count": 1,
    }
    assert "image" not in json.dumps(events)
    window.close()
    qt_app.processEvents()


def test_live_telemetry_is_noop_until_logging_enabled(
    qt_app: QApplication,
    tmp_path: Path,
) -> None:
    from desktop_demo.ui.main_window import MainWindow

    log_path = tmp_path / "demo.jsonl"
    window = MainWindow(telemetry_path=log_path, window_provider=lambda: ())

    window.handle_gaze_sample(gaze_sample())

    assert not log_path.exists()
    window.close()
    qt_app.processEvents()
