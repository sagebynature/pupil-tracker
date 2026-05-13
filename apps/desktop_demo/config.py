"""Runtime configuration for the desktop demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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


@dataclass(frozen=True)
class DemoConfig:
    """Configuration values needed by the desktop demo shell."""

    camera_id: int | str = 0
    preview_fps: int = 30
    model_asset_path: Path | None = None

    @classmethod
    def from_environment(cls) -> DemoConfig:
        """Load demo configuration from environment variables."""

        camera_id = _parse_camera_id(os.environ.get("PUPIL_TRACKER_CAMERA_ID", "0"))
        preview_fps = _parse_preview_fps(os.environ.get("PUPIL_TRACKER_PREVIEW_FPS", "30"))
        model_asset_value = os.environ.get("PUPIL_TRACKER_MEDIAPIPE_MODEL")
        model_asset_path = Path(model_asset_value) if model_asset_value else None
        return cls(
            camera_id=camera_id,
            preview_fps=preview_fps,
            model_asset_path=model_asset_path,
        )

    @property
    def preview_interval_ms(self) -> int:
        """Return the QTimer interval for the configured preview FPS."""

        return max(1, round(1000 / self.preview_fps))
