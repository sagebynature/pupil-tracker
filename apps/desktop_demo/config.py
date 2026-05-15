"""Runtime configuration for the desktop demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from desktop_demo.calibration_session import CalibrationSampleWindow


def _parse_camera_id(value: str) -> int | str:
    stripped = value.strip()
    if stripped == "":
        return 0
    try:
        return int(stripped)
    except ValueError:
        return stripped


def _parse_preview_fps(value: str) -> int:
    try:
        preview_fps = int(value)
    except ValueError as error:
        msg = "PUPIL_TRACKER_PREVIEW_FPS must be a positive integer"
        raise ValueError(msg) from error
    if preview_fps <= 0:
        msg = "PUPIL_TRACKER_PREVIEW_FPS must be a positive integer"
        raise ValueError(msg)
    return preview_fps


def _parse_positive_int(value: str, *, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg) from error
    if parsed <= 0:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
    return parsed


def _parse_calibration_sample_window(value: str) -> CalibrationSampleWindow:
    normalized = value.strip().lower()
    if normalized in {"all", "early", "middle", "late"}:
        return cast(CalibrationSampleWindow, normalized)
    msg = "PUPIL_TRACKER_CALIBRATION_SAMPLE_WINDOW must be one of: all, early, middle, late"
    raise ValueError(msg)


@dataclass(frozen=True)
class DemoConfig:
    """Configuration values needed by the desktop demo shell."""

    camera_id: int | str = 0
    preview_fps: int = 30
    model_asset_path: Path | None = None
    validation_grid_columns: int = 4
    validation_grid_rows: int = 3
    calibration_sample_window: CalibrationSampleWindow = "all"

    @classmethod
    def from_environment(cls) -> DemoConfig:
        """Load demo configuration from environment variables."""

        camera_id = _parse_camera_id(os.environ.get("PUPIL_TRACKER_CAMERA_ID", "0"))
        preview_fps = _parse_preview_fps(os.environ.get("PUPIL_TRACKER_PREVIEW_FPS", "30"))
        validation_grid_columns = _parse_positive_int(
            os.environ.get("PUPIL_TRACKER_VALIDATION_GRID_COLUMNS", "4"),
            name="PUPIL_TRACKER_VALIDATION_GRID_COLUMNS",
        )
        validation_grid_rows = _parse_positive_int(
            os.environ.get("PUPIL_TRACKER_VALIDATION_GRID_ROWS", "3"),
            name="PUPIL_TRACKER_VALIDATION_GRID_ROWS",
        )
        calibration_sample_window = _parse_calibration_sample_window(
            os.environ.get("PUPIL_TRACKER_CALIBRATION_SAMPLE_WINDOW", "all")
        )
        model_asset_value = os.environ.get("PUPIL_TRACKER_MEDIAPIPE_MODEL")
        model_asset_path = Path(model_asset_value) if model_asset_value else None
        return cls(
            camera_id=camera_id,
            preview_fps=preview_fps,
            model_asset_path=model_asset_path,
            validation_grid_columns=validation_grid_columns,
            validation_grid_rows=validation_grid_rows,
            calibration_sample_window=calibration_sample_window,
        )

    @property
    def preview_interval_ms(self) -> int:
        """Return the QTimer interval for the configured preview FPS."""

        return max(1, round(1000 / self.preview_fps))
