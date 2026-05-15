"""Desktop demo application shell."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import cast

from PySide6.QtWidgets import QApplication

from desktop_demo.config import DemoConfig
from desktop_demo.ui.main_window import MainWindow
from pupil_tracker import configure_logging, get_logger
from pupil_tracker.camera import OpenCVCamera

_LOGGER = get_logger("desktop_demo")


def create_app(argv: Sequence[str] | None = None) -> QApplication:
    """Create the Qt application instance."""

    existing = QApplication.instance()
    if existing is not None:
        return cast(QApplication, existing)
    return QApplication(list(argv) if argv is not None else sys.argv)


def create_main_window(*, config: DemoConfig | None = None) -> MainWindow:
    """Create the main window from runtime configuration."""

    resolved_config = config if config is not None else DemoConfig.from_environment()
    return MainWindow(
        camera_factory=lambda: OpenCVCamera(camera_id=resolved_config.camera_id),
        preview_interval_ms=resolved_config.preview_interval_ms,
        model_asset_path=resolved_config.model_asset_path,
        validation_grid_columns=resolved_config.validation_grid_columns,
        validation_grid_rows=resolved_config.validation_grid_rows,
        calibration_sample_window=resolved_config.calibration_sample_window,
        gaze_focus_enabled=resolved_config.gaze_focus_enabled,
        posture_stability_max_delta=resolved_config.posture_stability_max_delta,
        context_stability_max_delta=resolved_config.context_stability_max_delta,
        solvepnp_style_features_enabled=resolved_config.solvepnp_style_features_enabled,
    )


def run(argv: Sequence[str] | None = None) -> int:
    """Run the desktop demo application."""

    configure_logging()
    app = create_app(argv)
    window = create_main_window(config=DemoConfig.from_environment())
    window.show()
    _LOGGER.info("started desktop demo")
    return app.exec()
