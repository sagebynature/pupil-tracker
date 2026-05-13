"""Main window for the desktop demo."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from desktop_demo.calibration_session import CalibrationSession, CalibrationSessionState
from desktop_demo.gaze_runtime import GazeRuntime
from desktop_demo.tracking_runtime import TrackingStatus
from desktop_demo.ui.annotations import annotate_observation
from desktop_demo.ui.calibration_view import CalibrationFlowState, CalibrationView
from desktop_demo.ui.frame_image import bgr_ndarray_to_qimage
from desktop_demo.ui.overlay import GazeOverlay
from pupil_tracker import get_logger
from pupil_tracker.calibration import PolynomialRidgeCalibrationModel
from pupil_tracker.camera import CameraError, OpenCVCamera
from pupil_tracker.models import GazeSample
from pupil_tracker.telemetry import JsonlLogger
from pupil_tracker.tracking import Frame

_LOGGER = get_logger("desktop_demo.ui")


class CameraSource(Protocol):
    """Minimal camera source surface needed by the demo window."""

    def open(self) -> None: ...

    def close(self) -> None: ...

    def read(self) -> Frame: ...


class TrackingRuntimeLike(Protocol):
    """Tracking runtime surface needed by the demo window."""

    def process(self, frame: Frame) -> TrackingStatus: ...

    def close(self) -> None: ...


class GazeRuntimeLike(Protocol):
    """Calibrated gaze runtime surface needed by the demo window."""

    def update(
        self,
        observation: Any,
        *,
        screen_width: float,
        screen_height: float,
    ) -> Any: ...


class CameraPreviewWorker(QObject):
    """Camera preview worker controlled by the main window."""

    started = Signal()
    stopped = Signal()
    instances_started = 0

    def __init__(self, camera_factory: Callable[[], CameraSource]) -> None:
        super().__init__()
        self._camera_factory = camera_factory
        self._camera: CameraSource | None = None
        self.running = False

    def start(self) -> None:
        """Start the preview worker placeholder."""

        if self.running:
            return
        if self._camera is None:
            self._camera = self._camera_factory()
        self._camera.open()
        self.running = True
        type(self).instances_started += 1
        _LOGGER.info("camera preview started")
        self.started.emit()

    def stop(self) -> None:
        """Stop the preview worker placeholder."""

        if not self.running and self._camera is None:
            return
        if self._camera is not None:
            self._camera.close()
            self._camera = None
        self.running = False
        _LOGGER.info("camera preview stopped")
        self.stopped.emit()

    def tick(self) -> Frame | None:
        """Read one frame when the preview worker is running."""

        if not self.running or self._camera is None:
            return None
        try:
            return self._camera.read()
        except CameraError:
            self.stop()
            raise


class MainWindow(QMainWindow):
    """Desktop demo shell with camera controls and debug placeholders."""

    def __init__(
        self,
        *,
        telemetry_path: Path | None = None,
        camera_factory: Callable[[], CameraSource] | None = None,
        preview_interval_ms: int = 33,
        tracking_runtime: TrackingRuntimeLike | None = None,
        calibration_session: CalibrationSession | None = None,
        gaze_runtime: GazeRuntimeLike | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Pupil Tracker Demo")
        self.worker = CameraPreviewWorker(
            camera_factory if camera_factory is not None else OpenCVCamera
        )
        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(preview_interval_ms)
        self.preview_timer.timeout.connect(self.update_preview_frame)
        self.tracking_runtime = tracking_runtime
        self.gaze_runtime = gaze_runtime
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
        self.calibration_view = CalibrationView(
            flow=(
                cast(CalibrationFlowState, calibration_session.flow)
                if calibration_session is not None
                else None
            )
        )
        self.calibration_session = (
            calibration_session
            if calibration_session is not None
            else self._create_default_calibration_session()
        )
        if self.gaze_runtime is None:
            self.gaze_runtime = GazeRuntime(model=cast(Any, self.calibration_session.model))
        self.debug_label = QLabel("Debug: confidence -- | region -- | fps --")

        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.start_logging_button.clicked.connect(self.start_logging)
        self.stop_logging_button.clicked.connect(self.stop_logging)
        self.calibration_view.start_button.clicked.connect(self.start_calibration)

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

    def _screen_size(self) -> tuple[float, float]:
        """Return primary screen size with a safe fallback for headless tests."""

        screen = QApplication.primaryScreen()
        if screen is None:
            return (1920.0, 1080.0)
        size = screen.geometry().size()
        return (float(size.width()), float(size.height()))

    def _create_default_calibration_session(self) -> CalibrationSession:
        """Create the default calibration session for the current screen."""

        screen_width, screen_height = self._screen_size()
        return CalibrationSession(
            flow=self.calibration_view.flow,
            model=PolynomialRidgeCalibrationModel(),
            screen_width=screen_width,
            screen_height=screen_height,
        )

    def start_camera(self) -> None:
        """Open the camera source and mark the preview as running."""

        try:
            self.worker.start()
        except CameraError as error:
            self.preview_label.setText(str(error))
            self.debug_label.setText("Debug: camera failed to start")
            _LOGGER.warning("camera preview failed to start: %s", error)
            return
        self.preview_timer.start()
        self.preview_label.setText("Camera preview running")

    def stop_camera(self) -> None:
        """Close the camera source and mark the preview as stopped."""

        self.preview_timer.stop()
        self.worker.stop()
        self.gaze_overlay.hide()
        self.preview_label.setText("Camera preview stopped")

    def start_calibration(self) -> None:
        """Start collecting calibration samples from live tracker observations."""

        self.calibration_session.start()
        self.calibration_view.refresh()
        self.debug_label.setText("Calibration collecting: look at the visible target")

    def update_preview_frame(self) -> None:
        """Read and display one preview frame from the running camera."""

        try:
            frame = self.worker.tick()
        except CameraError as error:
            self.preview_timer.stop()
            self.preview_label.setText(str(error))
            self.debug_label.setText("Debug: camera frame read failed")
            _LOGGER.warning("camera frame read failed: %s", error)
            return
        if frame is None:
            return
        image = self._preview_image_for_frame(frame)
        qimage = bgr_ndarray_to_qimage(image)
        self.preview_label.setPixmap(QPixmap.fromImage(qimage))

    def _preview_image_for_frame(self, frame: Frame) -> NDArray[np.uint8]:
        """Return the image to render for a frame, annotated when tracking is enabled."""

        if self.tracking_runtime is None:
            return frame.image
        status = self.tracking_runtime.process(frame)
        self._handle_tracking_status(status)
        return annotate_observation(frame.image, status.observation)

    def _handle_tracking_status(self, status: TrackingStatus) -> None:
        """Update UI for one tracker status and capture calibration if active."""

        if self.calibration_session.is_collecting:
            self.calibration_session.capture(status.observation)
            self.calibration_view.refresh()
            self._update_calibration_status(status)
            return
        if self.calibration_session.state is CalibrationSessionState.COMPLETE:
            self._update_gaze_status(status)
            return
        self._update_tracking_status(status)

    def _update_gaze_status(self, status: TrackingStatus) -> None:
        """Update the debug label from calibrated gaze when tracking is active."""

        if self.gaze_runtime is None:
            self._update_tracking_status(status)
            return
        screen_width, screen_height = self._screen_size()
        sample = self.gaze_runtime.update(
            status.observation,
            screen_width=screen_width,
            screen_height=screen_height,
        )
        if sample is None:
            self._update_tracking_status(status)
            return
        self.handle_gaze_sample(sample)
        self.debug_label.setText(
            f"Debug: gaze {sample.region_id} | confidence {sample.confidence:.2f}"
        )

    def handle_gaze_sample(self, sample: GazeSample) -> None:
        """Update the transparent overlay from one calibrated gaze sample."""

        self.gaze_overlay.update_sample(sample)
        if sample.valid:
            self.gaze_overlay.show()
        else:
            self.gaze_overlay.hide()

    def _update_calibration_status(self, status: TrackingStatus) -> None:
        """Update the debug label with calibration status."""

        session = self.calibration_session
        if session.state is CalibrationSessionState.COMPLETE and session.fit_result is not None:
            result = session.fit_result
            self.debug_label.setText(
                "Calibration complete: "
                f"{result.sample_count} samples | mean error {result.mean_error_px:.2f}px | "
                f"max error {result.max_error_px:.2f}px"
            )
        elif session.state is CalibrationSessionState.FAILED:
            self.debug_label.setText(f"Calibration failed: {session.error_message}")
        elif status.valid:
            self.debug_label.setText(
                f"Calibration collecting: confidence {status.confidence:.2f}"
            )
        else:
            self.debug_label.setText(f"Calibration collecting: {status.message}")

    def _update_tracking_status(self, status: TrackingStatus) -> None:
        """Update the debug label with tracker status."""

        if status.valid:
            self.debug_label.setText(
                f"Debug: confidence {status.confidence:.2f} | tracking {status.message}"
            )
        else:
            self.debug_label.setText(f"Debug: tracking {status.message}")

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
        self.stop_camera()
        if self.tracking_runtime is not None:
            self.tracking_runtime.close()
            self.tracking_runtime = None
        self.gaze_overlay.close()
        super().closeEvent(event)
