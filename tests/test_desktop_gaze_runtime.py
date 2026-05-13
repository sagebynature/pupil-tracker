"""Tests for calibrated gaze runtime in the desktop demo."""

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


class FakeModel:
    def __init__(self, *, fitted: bool = True) -> None:
        self.fitted = fitted
        self.predict_calls = 0

    def predict(
        self,
        observation: RawObservation,
        screen_width: float,
        screen_height: float,
    ) -> GazeSample:
        self.predict_calls += 1
        if not self.fitted:
            raise RuntimeError("calibration model is not fitted")
        return GazeSample(
            timestamp=observation.timestamp,
            x=screen_width / 2,
            y=screen_height / 2,
            confidence=observation.confidence,
            valid=True,
        )


class FakeSmoother:
    def __init__(self, *, valid: bool = True) -> None:
        self.valid = valid
        self.update_calls = 0

    def update(self, sample: GazeSample) -> GazeSample:
        self.update_calls += 1
        return GazeSample(
            timestamp=sample.timestamp,
            x=sample.x,
            y=sample.y,
            confidence=sample.confidence,
            valid=self.valid,
        )


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


def valid_observation(timestamp: float = 1.0) -> RawObservation:
    return RawObservation(
        timestamp=timestamp,
        valid=True,
        confidence=0.8,
        feature_vector=(0.4, 0.6),
    )


def test_gaze_runtime_predicts_smooths_and_maps_region() -> None:
    from desktop_demo.gaze_runtime import GazeRuntime

    model = FakeModel()
    smoother = FakeSmoother()
    runtime = GazeRuntime(model=model, smoother=smoother)

    sample = runtime.update(valid_observation(), screen_width=300, screen_height=300)

    assert sample is not None
    assert sample.x == 150
    assert sample.y == 150
    assert sample.region_id == "middle_center"
    assert model.predict_calls == 1
    assert smoother.update_calls == 1


def test_gaze_runtime_returns_none_for_invalid_observation() -> None:
    from desktop_demo.gaze_runtime import GazeRuntime

    model = FakeModel()
    smoother = FakeSmoother()
    runtime = GazeRuntime(model=model, smoother=smoother)

    sample = runtime.update(
        RawObservation.invalid(timestamp=2.0, reason="no face"),
        screen_width=300,
        screen_height=300,
    )

    assert sample is None
    assert model.predict_calls == 0
    assert smoother.update_calls == 0


def test_gaze_runtime_returns_none_for_unfitted_model() -> None:
    from desktop_demo.gaze_runtime import GazeRuntime

    model = FakeModel(fitted=False)
    smoother = FakeSmoother()
    runtime = GazeRuntime(model=model, smoother=smoother)

    sample = runtime.update(valid_observation(), screen_width=300, screen_height=300)

    assert sample is None
    assert model.predict_calls == 1
    assert smoother.update_calls == 0


def test_gaze_runtime_returns_none_when_smoother_outputs_invalid_sample() -> None:
    from desktop_demo.gaze_runtime import GazeRuntime

    model = FakeModel()
    smoother = FakeSmoother(valid=False)
    runtime = GazeRuntime(model=model, smoother=smoother)

    sample = runtime.update(valid_observation(), screen_width=300, screen_height=300)

    assert sample is None
    assert model.predict_calls == 1
    assert smoother.update_calls == 1


def test_gaze_runtime_rejects_invalid_screen_dimensions() -> None:
    from desktop_demo.gaze_runtime import GazeRuntime

    runtime = GazeRuntime(model=FakeModel(), smoother=FakeSmoother())

    with pytest.raises(ValueError, match="screen dimensions must be positive"):
        runtime.update(valid_observation(), screen_width=0, screen_height=300)


def test_main_window_updates_debug_from_gaze_runtime_after_calibration(
    qt_app: QApplication,
) -> None:
    from desktop_demo.calibration_session import CalibrationSessionState
    from desktop_demo.ui.main_window import MainWindow

    observation = valid_observation()
    gaze_sample = GazeSample(
        timestamp=observation.timestamp,
        x=150,
        y=150,
        confidence=0.8,
        valid=True,
        region_id="middle_center",
    )
    gaze_runtime = FakeGazeRuntime(sample=gaze_sample)
    window = MainWindow(
        camera_factory=FakeCamera,
        tracking_runtime=FakeTrackingRuntime(observation),
        gaze_runtime=gaze_runtime,
    )
    window.calibration_session.state = CalibrationSessionState.COMPLETE
    window.start_camera()

    window.update_preview_frame()

    assert gaze_runtime.update_calls == 1
    assert "gaze middle_center" in window.debug_label.text()
    assert "confidence 0.80" in window.debug_label.text()
    window.close()
    qt_app.processEvents()
