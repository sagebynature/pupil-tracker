"""Main window for the desktop demo."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

from pupil_tracker import get_logger

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

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Pupil Tracker Demo")
        self.worker = CameraPreviewWorker()

        self.preview_label = QLabel("Camera preview stopped")
        self.preview_label.setMinimumSize(640, 360)
        self.preview_label.setStyleSheet("background: #111; color: #ddd; padding: 16px;")

        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        self.debug_label = QLabel("Debug: confidence -- | region -- | fps --")

        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)

        controls = QHBoxLayout()
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addStretch(1)

        layout = QVBoxLayout()
        layout.addWidget(self.preview_label)
        layout.addLayout(controls)
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
