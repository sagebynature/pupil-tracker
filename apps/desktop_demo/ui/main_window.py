"""Main window for the desktop demo."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

from desktop_demo.ui.calibration_view import CalibrationView
from desktop_demo.ui.overlay import GazeOverlay
from pupil_tracker import get_logger
from pupil_tracker.telemetry import JsonlLogger

_LOGGER = get_logger("desktop_demo.ui")


class CameraPreviewWorker(QObject):
    """Placeholder camera preview worker controlled by the main window."""

    started = Signal()
    stopped = Signal()
    instances_started = 0

    def __init__(self) -> None:
        super().__init__()
        self.running = False

    def start(self) -> None:
        """Start the preview worker placeholder."""

        if self.running:
            return
        self.running = True
        type(self).instances_started += 1
        _LOGGER.info("camera preview started")
        self.started.emit()

    def stop(self) -> None:
        """Stop the preview worker placeholder."""

        if not self.running:
            return
        self.running = False
        _LOGGER.info("camera preview stopped")
        self.stopped.emit()


class MainWindow(QMainWindow):
    """Desktop demo shell with camera controls and debug placeholders."""

    def __init__(self, *, telemetry_path: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Pupil Tracker Demo")
        self.worker = CameraPreviewWorker()
        self.gaze_overlay = GazeOverlay()
        self.telemetry_path = (
            telemetry_path if telemetry_path is not None else Path("metrics/demo.jsonl")
        )
        self.telemetry_logger: JsonlLogger | None = None

        self.preview_label = QLabel("Camera preview stopped")
        self.preview_label.setMinimumSize(640, 360)
        self.preview_label.setStyleSheet("background: #111; color: #ddd; padding: 16px;")

        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        self.start_logging_button = QPushButton("Start Logging")
        self.stop_logging_button = QPushButton("Stop Logging")
        self.calibration_view = CalibrationView()
        self.debug_label = QLabel("Debug: confidence -- | region -- | fps --")

        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.start_logging_button.clicked.connect(self.start_logging)
        self.stop_logging_button.clicked.connect(self.stop_logging)

        controls = QHBoxLayout()
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.start_logging_button)
        controls.addWidget(self.stop_logging_button)
        controls.addStretch(1)

        layout = QVBoxLayout()
        layout.addWidget(self.preview_label)
        layout.addLayout(controls)
        layout.addWidget(self.calibration_view)
        layout.addWidget(self.debug_label)

        root = QWidget()
        root.setLayout(layout)
        self.setCentralWidget(root)

    def start_camera(self) -> None:
        """Start the camera preview placeholder."""

        self.worker.start()
        self.preview_label.setText("Camera preview running")

    def stop_camera(self) -> None:
        """Stop the camera preview placeholder."""

        self.worker.stop()
        self.preview_label.setText("Camera preview stopped")

    def start_logging(self) -> None:
        """Start JSONL telemetry logging only after explicit user action."""

        if self.telemetry_logger is not None:
            return
        self.telemetry_logger = JsonlLogger(self.telemetry_path)
        self.debug_label.setText(f"Telemetry logging: {self.telemetry_path}")
        _LOGGER.info("telemetry logging started at %s", self.telemetry_path)

    def stop_logging(self) -> None:
        """Stop JSONL telemetry logging and flush the file."""

        if self.telemetry_logger is None:
            return
        self.telemetry_logger.close()
        self.telemetry_logger = None
        self.debug_label.setText("Telemetry logging stopped")
        _LOGGER.info("telemetry logging stopped")

    def log_telemetry_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        """Write one telemetry event when logging is enabled."""

        if self.telemetry_logger is None:
            return
        self.telemetry_logger.write_event(event_type, payload)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Flush telemetry on close."""

        self.stop_logging()
        super().closeEvent(event)
