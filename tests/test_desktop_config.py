"""Tests for desktop demo runtime configuration."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from PySide6.QtWidgets import QApplication

APPS_ROOT = Path(__file__).resolve().parents[1] / "apps"
if str(APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPS_ROOT))


@pytest.fixture(scope="module")
def qt_app() -> Iterator[QApplication]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield cast(QApplication, app)


def test_demo_config_reads_model_asset_path_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from desktop_demo.config import DemoConfig

    monkeypatch.setenv("PUPIL_TRACKER_MEDIAPIPE_MODEL", "/tmp/face.task")

    config = DemoConfig.from_environment()

    assert config.model_asset_path == Path("/tmp/face.task")


def test_demo_config_defaults_keep_camera_preview_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from desktop_demo.config import DemoConfig

    monkeypatch.delenv("PUPIL_TRACKER_CAMERA_ID", raising=False)
    monkeypatch.delenv("PUPIL_TRACKER_MEDIAPIPE_MODEL", raising=False)
    monkeypatch.delenv("PUPIL_TRACKER_PREVIEW_FPS", raising=False)
    monkeypatch.delenv("PUPIL_TRACKER_VALIDATION_GRID_COLUMNS", raising=False)
    monkeypatch.delenv("PUPIL_TRACKER_VALIDATION_GRID_ROWS", raising=False)
    monkeypatch.delenv("PUPIL_TRACKER_GAZE_FOCUS_ENABLED", raising=False)

    config = DemoConfig.from_environment()

    assert config.camera_id == 0
    assert config.model_asset_path is None
    assert config.preview_fps == 30
    assert config.preview_interval_ms == 33
    assert config.validation_grid_columns == 4
    assert config.validation_grid_rows == 3
    assert config.calibration_sample_window == "all"
    assert config.gaze_focus_enabled is False


def test_demo_config_parses_calibration_sample_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from desktop_demo.config import DemoConfig

    monkeypatch.setenv("PUPIL_TRACKER_CALIBRATION_SAMPLE_WINDOW", "late")

    config = DemoConfig.from_environment()

    assert config.calibration_sample_window == "late"


def test_demo_config_parses_gaze_focus_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from desktop_demo.config import DemoConfig

    monkeypatch.setenv("PUPIL_TRACKER_GAZE_FOCUS_ENABLED", "true")

    config = DemoConfig.from_environment()

    assert config.gaze_focus_enabled is True


def test_demo_config_rejects_invalid_gaze_focus_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from desktop_demo.config import DemoConfig

    monkeypatch.setenv("PUPIL_TRACKER_GAZE_FOCUS_ENABLED", "maybe")

    with pytest.raises(ValueError, match="PUPIL_TRACKER_GAZE_FOCUS_ENABLED"):
        DemoConfig.from_environment()


def test_demo_config_rejects_invalid_calibration_sample_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from desktop_demo.config import DemoConfig

    monkeypatch.setenv("PUPIL_TRACKER_CALIBRATION_SAMPLE_WINDOW", "center")

    with pytest.raises(ValueError, match="PUPIL_TRACKER_CALIBRATION_SAMPLE_WINDOW"):
        DemoConfig.from_environment()


def test_demo_config_parses_camera_id_and_preview_fps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from desktop_demo.config import DemoConfig

    monkeypatch.setenv("PUPIL_TRACKER_CAMERA_ID", "1")
    monkeypatch.setenv("PUPIL_TRACKER_PREVIEW_FPS", "20")

    config = DemoConfig.from_environment()

    assert config.camera_id == 1
    assert config.preview_fps == 20
    assert config.preview_interval_ms == 50


def test_demo_config_preserves_non_numeric_camera_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from desktop_demo.config import DemoConfig

    monkeypatch.setenv("PUPIL_TRACKER_CAMERA_ID", "sample-video.mov")

    config = DemoConfig.from_environment()

    assert config.camera_id == "sample-video.mov"


def test_demo_config_parses_validation_grid_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from desktop_demo.config import DemoConfig

    monkeypatch.setenv("PUPIL_TRACKER_VALIDATION_GRID_COLUMNS", "5")
    monkeypatch.setenv("PUPIL_TRACKER_VALIDATION_GRID_ROWS", "4")

    config = DemoConfig.from_environment()

    assert config.validation_grid_columns == 5
    assert config.validation_grid_rows == 4


def test_demo_config_rejects_invalid_validation_grid_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from desktop_demo.config import DemoConfig

    monkeypatch.setenv("PUPIL_TRACKER_VALIDATION_GRID_COLUMNS", "0")

    with pytest.raises(ValueError, match="PUPIL_TRACKER_VALIDATION_GRID_COLUMNS"):
        DemoConfig.from_environment()


def test_demo_config_rejects_non_numeric_validation_grid_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from desktop_demo.config import DemoConfig

    monkeypatch.setenv("PUPIL_TRACKER_VALIDATION_GRID_ROWS", "wide")

    with pytest.raises(ValueError, match="PUPIL_TRACKER_VALIDATION_GRID_ROWS"):
        DemoConfig.from_environment()


def test_demo_config_rejects_invalid_preview_fps(monkeypatch: pytest.MonkeyPatch) -> None:
    from desktop_demo.config import DemoConfig

    monkeypatch.setenv("PUPIL_TRACKER_PREVIEW_FPS", "0")

    with pytest.raises(ValueError, match="PUPIL_TRACKER_PREVIEW_FPS"):
        DemoConfig.from_environment()


def test_create_main_window_applies_config_to_camera_and_timer(
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import desktop_demo.app as demo_app
    from desktop_demo.config import DemoConfig

    created_camera_ids: list[int | str] = []

    class FakeOpenCVCamera:
        def __init__(self, camera_id: int | str = 0) -> None:
            created_camera_ids.append(camera_id)
            self.open_calls = 0
            self.close_calls = 0

        def open(self) -> None:
            self.open_calls += 1

        def close(self) -> None:
            self.close_calls += 1

        def read(self) -> Any:
            msg = "not used in this test"
            raise AssertionError(msg)

    monkeypatch.setattr(demo_app, "OpenCVCamera", FakeOpenCVCamera)
    config = DemoConfig(
        camera_id="sample-video.mov",
        preview_fps=25,
        model_asset_path=None,
        validation_grid_columns=5,
        validation_grid_rows=4,
        calibration_sample_window="late",
        gaze_focus_enabled=True,
    )

    window = demo_app.create_main_window(config=config)
    window.start_camera()

    assert created_camera_ids == ["sample-video.mov"]
    assert window.preview_timer.interval() == 40
    assert window.validation_session.grid_columns == 5
    assert window.validation_session.grid_rows == 4
    assert window.calibration_session.calibration_sample_window == "late"
    assert window.gaze_focus_enabled is True
    window.stop_camera()
    qt_app.processEvents()
