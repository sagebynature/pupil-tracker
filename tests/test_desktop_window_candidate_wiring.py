"""Tests for wiring calibrated gaze samples to window candidate debug status."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

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


class FakeWindowActivator:
    def __init__(self) -> None:
        self.activated: list[WindowCandidate] = []

    def __call__(self, candidate: WindowCandidate) -> None:
        self.activated.append(candidate)


@pytest.fixture(scope="module")
def qt_app() -> Iterator[QApplication]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield cast(QApplication, app)


def gaze_sample(*, x: float = 50.0, y: float = 50.0, valid: bool = True) -> GazeSample:
    return GazeSample(
        timestamp=1.0,
        x=x,
        y=y,
        confidence=0.9 if valid else 0.0,
        valid=valid,
        region_id="top_left",
    )


def raw_observation() -> RawObservation:
    return RawObservation(
        timestamp=1.0,
        valid=True,
        confidence=0.9,
        feature_vector=(0.2, 0.4),
    )


def demo_candidate() -> WindowCandidate:
    return WindowCandidate(
        app_name="DemoApp",
        title="Main Window",
        bounds=Rect(x=0.0, y=0.0, width=100.0, height=100.0),
        score=1.0,
    )


def test_gaze_sample_updates_window_candidate_debug_text(qt_app: QApplication) -> None:
    from desktop_demo.ui.main_window import MainWindow

    window = MainWindow(window_provider=lambda: (demo_candidate(),))

    window.handle_gaze_sample(gaze_sample())

    assert "window DemoApp" in window.debug_label.text()
    assert "Main Window" in window.debug_label.text()
    window.close()
    qt_app.processEvents()


def test_gaze_sample_without_window_candidate_reports_none(qt_app: QApplication) -> None:
    from desktop_demo.ui.main_window import MainWindow

    window = MainWindow(window_provider=lambda: ())

    window.handle_gaze_sample(gaze_sample())

    assert "window none" in window.debug_label.text()
    window.close()
    qt_app.processEvents()


def test_invalid_gaze_sample_does_not_query_window_provider(qt_app: QApplication) -> None:
    from desktop_demo.ui.main_window import MainWindow

    calls = 0

    def provider() -> tuple[WindowCandidate, ...]:
        nonlocal calls
        calls += 1
        return (demo_candidate(),)

    window = MainWindow(window_provider=provider)

    window.handle_gaze_sample(gaze_sample(valid=False))

    assert calls == 0
    assert "window" not in window.debug_label.text().lower()
    window.close()
    qt_app.processEvents()


def test_window_provider_failure_is_reported_unobtrusively(qt_app: QApplication) -> None:
    from desktop_demo.ui.main_window import MainWindow

    def provider() -> tuple[WindowCandidate, ...]:
        raise RuntimeError("window permission denied")

    window = MainWindow(window_provider=provider)

    window.handle_gaze_sample(gaze_sample())

    assert "window unavailable" in window.debug_label.text()
    assert "window permission denied" in window.debug_label.text()
    window.close()
    qt_app.processEvents()



def test_gaze_focus_disabled_does_not_activate_window_candidate(
    qt_app: QApplication,
) -> None:
    from desktop_demo.ui.main_window import MainWindow

    activator = FakeWindowActivator()
    window = MainWindow(
        window_provider=lambda: (demo_candidate(),),
        gaze_focus_enabled=False,
        window_activator=activator,
    )

    window.handle_gaze_sample(gaze_sample())

    assert activator.activated == []
    window.close()
    qt_app.processEvents()


def test_gaze_focus_enabled_activates_window_candidate_once_per_candidate(
    qt_app: QApplication,
) -> None:
    from desktop_demo.ui.main_window import MainWindow

    activator = FakeWindowActivator()
    window = MainWindow(
        window_provider=lambda: (demo_candidate(),),
        gaze_focus_enabled=True,
        window_activator=activator,
    )

    window.handle_gaze_sample(gaze_sample())
    window.handle_gaze_sample(gaze_sample())

    assert activator.activated == [demo_candidate()]
    assert "focus on" in window.gaze_focus_button.text().lower()
    window.close()
    qt_app.processEvents()


def test_gaze_focus_button_toggles_window_activation(qt_app: QApplication) -> None:
    from desktop_demo.ui.main_window import MainWindow

    activator = FakeWindowActivator()
    window = MainWindow(
        window_provider=lambda: (demo_candidate(),),
        gaze_focus_enabled=False,
        window_activator=activator,
    )

    window.gaze_focus_button.click()
    window.handle_gaze_sample(gaze_sample())
    window.gaze_focus_button.click()
    window.handle_gaze_sample(gaze_sample(x=75.0, y=75.0))

    assert activator.activated == [demo_candidate()]
    assert "focus off" in window.gaze_focus_button.text().lower()
    window.close()
    qt_app.processEvents()


def test_gaze_focus_activation_failure_is_reported_unobtrusively(
    qt_app: QApplication,
) -> None:
    from desktop_demo.ui.main_window import MainWindow

    def failing_activator(candidate: WindowCandidate) -> None:
        del candidate
        raise RuntimeError("activation permission denied")

    window = MainWindow(
        window_provider=lambda: (demo_candidate(),),
        gaze_focus_enabled=True,
        window_activator=failing_activator,
    )

    window.handle_gaze_sample(gaze_sample())

    assert "focus unavailable" in window.debug_label.text()
    assert "activation permission denied" in window.debug_label.text()
    window.close()
    qt_app.processEvents()

def test_live_calibrated_gaze_updates_window_candidate_debug_text(
    qt_app: QApplication,
) -> None:
    from desktop_demo.calibration_session import CalibrationSessionState
    from desktop_demo.ui.main_window import MainWindow

    sample = gaze_sample()
    window = MainWindow(
        camera_factory=FakeCamera,
        tracking_runtime=FakeTrackingRuntime(raw_observation()),
        gaze_runtime=FakeGazeRuntime(sample),
        window_provider=lambda: (demo_candidate(),),
    )
    window.calibration_session.state = CalibrationSessionState.COMPLETE
    window.start_camera()

    window.update_preview_frame()

    assert "window DemoApp" in window.debug_label.text()
    assert "Main Window" in window.debug_label.text()
    window.close()
    qt_app.processEvents()
