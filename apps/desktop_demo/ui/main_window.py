"""Main window for the desktop demo."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from desktop_demo.calibration_session import (
    CalibrationSampleWindow,
    CalibrationSession,
    CalibrationSessionState,
)
from desktop_demo.gaze_runtime import GazeRuntime
from desktop_demo.tracking_runtime import TrackingRuntime, TrackingStatus
from desktop_demo.ui.annotations import annotate_observation
from desktop_demo.ui.calibration_view import (
    CalibrationFlowState,
    CalibrationTargetWidget,
    CalibrationView,
)
from desktop_demo.ui.frame_image import bgr_ndarray_to_qimage
from desktop_demo.ui.overlay import GazeOverlay
from desktop_demo.validation_session import ValidationSession, ValidationSessionState
from pupil_tracker import get_logger
from pupil_tracker.calibration import (
    CalibrationQualityFilter,
    FeatureStabilityConfig,
    LinearRidgeCalibrationModel,
    PolynomialRidgeCalibrationModel,
    TimedCalibrationConfig,
    edge_dense_calibration_pattern,
    summarize_feature_diagnostics,
    top_left_focus_calibration_pattern,
    top_row_focus_calibration_pattern,
    validation_pattern,
    vertical_grid_pattern,
)
from pupil_tracker.camera import CameraError, OpenCVCamera
from pupil_tracker.models import CalibrationTarget, GazeSample, Point2D, WindowCandidate
from pupil_tracker.platform import (
    activate_window_candidate,
    candidate_at_point,
    list_visible_windows,
)
from pupil_tracker.telemetry import (
    JsonlLogger,
    calibration_config_payload,
    calibration_event_payload,
    calibration_replay_sample_payload,
    calibration_target_quality_payload,
    feature_diagnostics_payload,
    gaze_event_payload,
    raw_observation_event_payload,
    validation_metrics_payload,
    validation_replay_sample_payload,
    validation_sample_payload,
    window_candidate_payload,
)
from pupil_tracker.tracking import Frame, MediaPipeTrackerBackend

_LOGGER = get_logger("desktop_demo.ui")
_HEAD_POSE_FEATURE_INDICES = (20, 21, 22)


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


class WindowProvider(Protocol):
    """Provider for visible window candidates under the gaze point."""

    def __call__(self) -> tuple[WindowCandidate, ...]: ...


class WindowActivator(Protocol):
    """Side-effecting focus action for a gaze-selected window candidate."""

    def __call__(self, candidate: WindowCandidate) -> None: ...


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
        validation_session: ValidationSession | None = None,
        gaze_runtime: GazeRuntimeLike | None = None,
        window_provider: WindowProvider | None = None,
        window_activator: WindowActivator | None = None,
        model_asset_path: Path | None = None,
        validation_grid_columns: int = 4,
        validation_grid_rows: int = 3,
        calibration_sample_window: CalibrationSampleWindow = "all",
        gaze_focus_enabled: bool = False,
        posture_stability_max_delta: float | None = None,
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
        self.model_asset_path = model_asset_path
        self._uses_default_calibration_session = calibration_session is None
        self.gaze_runtime = gaze_runtime
        self.window_provider = (
            window_provider if window_provider is not None else list_visible_windows
        )
        self.window_activator = (
            window_activator
            if window_activator is not None
            else activate_window_candidate
        )
        self.gaze_focus_enabled = gaze_focus_enabled
        self._last_activated_window_key: tuple[Any, ...] | None = None
        self.gaze_overlay = GazeOverlay()
        self.telemetry_path = (
            telemetry_path if telemetry_path is not None else Path("metrics/demo.jsonl")
        )
        self.telemetry_logger: JsonlLogger | None = None
        if validation_grid_columns <= 0 or validation_grid_rows <= 0:
            msg = "validation grid dimensions must be positive"
            raise ValueError(msg)
        self.validation_grid_columns = validation_grid_columns
        self.validation_grid_rows = validation_grid_rows
        self.calibration_sample_window: CalibrationSampleWindow = calibration_sample_window
        self.calibration_path_name = (
            "default_9_point" if self._uses_default_calibration_session else "injected"
        )
        self.posture_stability_max_delta = posture_stability_max_delta
        self._last_preview_qimage: QImage | None = None

        self.preview_label = QLabel("Camera preview stopped")
        self.preview_label.setMinimumSize(640, 360)
        self.preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background: #111; color: #ddd; padding: 16px;")

        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        self.start_logging_button = QPushButton("Start Logging")
        self.stop_logging_button = QPushButton("Stop Logging")
        self.show_heatmap_button = QPushButton("Show Heatmap")
        self.show_heatmap_button.setCheckable(True)
        self.clear_heatmap_button = QPushButton("Clear Heatmap")
        self.gaze_focus_button = QPushButton()
        self.gaze_focus_button.setCheckable(True)
        self.gaze_focus_button.setChecked(self.gaze_focus_enabled)
        self._sync_gaze_focus_button_text()
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
        self.calibration_target_overlay = CalibrationTargetWidget(self.calibration_view.flow)
        self.calibration_target_overlay.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.Tool
        )
        self.calibration_target_overlay.setAttribute(
            Qt.WidgetAttribute.WA_ShowWithoutActivating
        )
        self.calibration_target_overlay.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self.validation_session = (
            validation_session
            if validation_session is not None
            else self._create_default_validation_session()
        )
        if self.gaze_runtime is None:
            self.gaze_runtime = GazeRuntime(model=cast(Any, self.calibration_session.model))
        self.debug_label = QLabel("Debug: confidence -- | region -- | fps --")

        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.start_logging_button.clicked.connect(self.start_logging)
        self.stop_logging_button.clicked.connect(self.stop_logging)
        self.show_heatmap_button.toggled.connect(self.set_heatmap_enabled)
        self.clear_heatmap_button.clicked.connect(self.clear_heatmap)
        self.gaze_focus_button.toggled.connect(self.set_gaze_focus_enabled)
        self.calibration_view.start_button.clicked.connect(self.start_calibration)
        self.calibration_view.vertical_calibration_button.clicked.connect(
            self.start_vertical_calibration
        )
        self.calibration_view.edge_dense_calibration_button.clicked.connect(
            self.start_edge_dense_calibration
        )
        self.calibration_view.top_left_focus_calibration_button.clicked.connect(
            self.start_top_left_focus_calibration
        )
        self.calibration_view.top_row_focus_calibration_button.clicked.connect(
            self.start_top_row_focus_calibration
        )
        self.calibration_view.validation_button.clicked.connect(self.start_validation)

        controls = QHBoxLayout()
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.start_logging_button)
        controls.addWidget(self.stop_logging_button)
        controls.addWidget(self.show_heatmap_button)
        controls.addWidget(self.clear_heatmap_button)
        controls.addWidget(self.gaze_focus_button)
        controls.addStretch(1)

        layout = QVBoxLayout()
        layout.addWidget(self.preview_label, 1)
        layout.addLayout(controls)
        layout.addWidget(self.calibration_view)
        layout.addWidget(self.debug_label)

        root = QWidget()
        root.setLayout(layout)
        self.setCentralWidget(root)

    def _screen_size(self) -> tuple[float, float]:
        """Return primary screen size with a safe fallback for headless tests."""

        size = self._primary_screen_geometry().size()
        return (float(size.width()), float(size.height()))

    def _primary_screen_geometry(self) -> QRect:
        """Return primary screen geometry with a safe fallback for headless tests."""

        screen = QApplication.primaryScreen()
        if screen is None:
            return QRect(0, 0, 1920, 1080)
        return screen.geometry()

    def _prepare_screen_overlay(self, overlay: QWidget) -> None:
        """Size a top-level overlay to the primary screen coordinate space."""

        overlay.setGeometry(self._primary_screen_geometry())

    def _create_default_calibration_session(self) -> CalibrationSession:
        """Create the default calibration session for the current screen."""

        screen_width, screen_height = self._screen_size()
        timing_config = TimedCalibrationConfig()
        return CalibrationSession(
            flow=self.calibration_view.flow,
            model=PolynomialRidgeCalibrationModel(),
            screen_width=screen_width,
            screen_height=screen_height,
            timing_config=timing_config,
            quality_filter=self._create_calibration_quality_filter(timing_config),
            calibration_sample_window=self.calibration_sample_window,
        )

    def _create_calibration_quality_filter(
        self,
        timing_config: TimedCalibrationConfig,
    ) -> CalibrationQualityFilter:
        stability_config = (
            FeatureStabilityConfig(
                feature_indices=_HEAD_POSE_FEATURE_INDICES,
                max_delta=self.posture_stability_max_delta,
            )
            if self.posture_stability_max_delta is not None
            else None
        )
        return CalibrationQualityFilter(
            min_confidence=timing_config.min_confidence,
            stability_config=stability_config,
        )

    def _create_default_validation_session(self) -> ValidationSession:
        """Create the default post-calibration validation session."""

        screen_width, screen_height = self._screen_size()
        return ValidationSession(
            targets=validation_pattern(),
            screen_width=screen_width,
            screen_height=screen_height,
            timing_config=TimedCalibrationConfig(
                settle_seconds=1.0,
                capture_seconds=1.5,
                min_samples_per_target=10,
                min_confidence=0.0,
            ),
            grid_columns=self.validation_grid_columns,
            grid_rows=self.validation_grid_rows,
        )

    def _show_model_setup_guidance(self, detail: str | None = None) -> None:
        """Show actionable MediaPipe model setup guidance."""

        message = (
            "Tracker setup required: set PUPIL_TRACKER_MEDIAPIPE_MODEL "
            "to a MediaPipe FaceLandmarker .task file"
        )
        if detail:
            message = f"{message} ({detail})"
        self.debug_label.setText(message)

    def start_tracking(self) -> bool:
        """Initialize the default MediaPipe tracker when model setup is available."""

        if self.tracking_runtime is not None:
            return True
        if self.model_asset_path is None:
            self._show_model_setup_guidance()
            return False
        if not self.model_asset_path.exists():
            self._show_model_setup_guidance(f"missing: {self.model_asset_path}")
            return False
        try:
            backend = MediaPipeTrackerBackend(model_asset_path=str(self.model_asset_path))
        except Exception as error:
            self._show_model_setup_guidance(str(error))
            _LOGGER.warning("tracker setup failed: %s", error)
            return False
        self.tracking_runtime = TrackingRuntime(backend=backend)
        self.debug_label.setText("Tracker ready")
        return True

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
        """Close live demo resources and mark the preview as stopped."""

        self.preview_timer.stop()
        self.worker.stop()
        if self.tracking_runtime is not None:
            self.tracking_runtime.close()
            self.tracking_runtime = None
        self.gaze_overlay.hide()
        self.calibration_target_overlay.hide()
        self.stop_logging()
        self.preview_label.setText("Camera preview stopped")

    def start_calibration(self) -> None:
        """Start collecting calibration samples from live tracker observations."""

        if self._uses_default_calibration_session and not self.start_tracking():
            return
        self.calibration_view.validation_button.setEnabled(False)
        self.calibration_session.start()
        self._log_calibration_config_event()
        self._prepare_screen_overlay(self.calibration_target_overlay)
        self.calibration_target_overlay.show()
        self._refresh_calibration_view_status()
        self._update_calibration_status(None)

    def start_vertical_calibration(self) -> None:
        """Start the denser vertical calibration strategy for live accuracy checks."""

        self._start_linear_calibration_with_targets(
            calibration_path="vertical_linear",
            targets=vertical_grid_pattern(),
            title="15-point linear vertical calibration",
            unavailable_message="Vertical calibration unavailable for injected sessions",
        )

    def start_edge_dense_calibration(self) -> None:
        """Start the experimental edge-dense geometry calibration strategy."""

        self._start_linear_calibration_with_targets(
            calibration_path="edge_dense",
            targets=edge_dense_calibration_pattern(),
            title="17-point edge-dense calibration",
            unavailable_message="Edge-dense calibration unavailable for injected sessions",
        )

    def start_top_left_focus_calibration(self) -> None:
        """Start the experimental top-left-focused geometry calibration strategy."""

        self._start_linear_calibration_with_targets(
            calibration_path="top_left_focus",
            targets=top_left_focus_calibration_pattern(),
            title="25-point top-left focus calibration",
            unavailable_message="Top-left focus calibration unavailable for injected sessions",
        )

    def start_top_row_focus_calibration(self) -> None:
        """Start the experimental top-row-focused geometry calibration strategy."""

        self._start_linear_calibration_with_targets(
            calibration_path="top_row_focus",
            targets=top_row_focus_calibration_pattern(),
            title="33-point top-row focus calibration",
            unavailable_message="Top-row focus calibration unavailable for injected sessions",
        )

    def _start_linear_calibration_with_targets(
        self,
        *,
        calibration_path: str,
        targets: Sequence[CalibrationTarget],
        title: str,
        unavailable_message: str,
    ) -> None:
        if not self._uses_default_calibration_session:
            self.debug_label.setText(unavailable_message)
            return
        flow = CalibrationFlowState(targets=targets)
        self.calibration_path_name = calibration_path
        self.calibration_view.set_flow(flow, title=title)
        self.calibration_target_overlay.flow = flow
        screen_width, screen_height = self._screen_size()
        timing_config = TimedCalibrationConfig()
        self.calibration_session = CalibrationSession(
            flow=flow,
            model=LinearRidgeCalibrationModel(alpha=1.0),
            screen_width=screen_width,
            screen_height=screen_height,
            timing_config=timing_config,
            quality_filter=self._create_calibration_quality_filter(timing_config),
            calibration_sample_window=self.calibration_sample_window,
        )
        self.gaze_runtime = GazeRuntime(model=cast(Any, self.calibration_session.model))
        self.start_calibration()

    def _log_calibration_config_event(self) -> None:
        """Log scalar calibration configuration at the start of a run."""

        self.log_telemetry_event(
            "calibration_config",
            calibration_config_payload(
                calibration_path=self.calibration_path_name,
                targets=self.calibration_view.flow.targets,
                model_name=type(self.calibration_session.model).__name__,
                calibration_sample_window=self.calibration_sample_window,
                screen_width=self.calibration_session.screen_width,
                screen_height=self.calibration_session.screen_height,
                posture_stability_max_delta=self.posture_stability_max_delta,
                posture_feature_indices=_HEAD_POSE_FEATURE_INDICES,
            ),
        )

    def start_validation(self) -> None:
        """Start post-calibration validation against known targets."""

        if self.calibration_session.state is not CalibrationSessionState.COMPLETE:
            self.debug_label.setText("Validation unavailable: complete calibration first")
            return
        if self.tracking_runtime is None and not self.start_tracking():
            return
        self.validation_session.start()
        self.calibration_view.validation_button.setEnabled(False)
        self._update_validation_debug()

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
        self._set_preview_image(qimage)

    def _set_preview_image(self, qimage: QImage) -> None:
        """Render a preview image scaled to the available preview panel."""

        self._last_preview_qimage = qimage
        target_size = self.preview_label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            target_size = qimage.size()
        pixmap = QPixmap.fromImage(qimage).scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(pixmap)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Rescale the last preview frame when the demo window is resized."""

        super().resizeEvent(event)
        if self._last_preview_qimage is not None:
            self._set_preview_image(self._last_preview_qimage)

    def _preview_image_for_frame(self, frame: Frame) -> NDArray[np.uint8]:
        """Return the image to render for a frame, annotated when tracking is enabled."""

        if self.tracking_runtime is None:
            return frame.image
        status = self.tracking_runtime.process(frame)
        self.log_telemetry_event(
            "raw_observation",
            raw_observation_event_payload(status.observation),
        )
        self._handle_tracking_status(status)
        return annotate_observation(frame.image, status.observation)

    def _handle_tracking_status(self, status: TrackingStatus) -> None:
        """Update UI for one tracker status and capture calibration if active."""

        if self.calibration_session.is_collecting:
            target = self.calibration_session.flow.current_target
            sample_count_before_capture = len(self.calibration_session.flow.all_samples())
            advanced = self.calibration_session.capture(status.observation)
            sample_count_after_capture = len(self.calibration_session.flow.all_samples())
            sample_accepted = sample_count_after_capture > sample_count_before_capture
            decision = self.calibration_session.last_capture_decision
            decision_reason = (
                decision.reason
                if decision is not None
                else "accepted" if sample_accepted else "not_evaluated"
            )
            if target is not None and status.valid:
                self.log_telemetry_event(
                    "calibration_sample",
                    calibration_event_payload(
                        target,
                        sample_count=sample_count_after_capture,
                    ),
                )
                self.log_telemetry_event(
                    "calibration_replay_sample",
                    calibration_replay_sample_payload(
                        target,
                        status.observation,
                        capture_phase=self.calibration_session.phase.value,
                        sample_accepted=sample_accepted,
                        decision_reason=decision_reason,
                    ),
                )
            if advanced:
                self._log_calibration_quality_events()
                if self.calibration_session.state is CalibrationSessionState.COMPLETE:
                    self._log_calibration_feature_diagnostics()
            self._refresh_calibration_view_status()
            self._update_calibration_status(status)
            return
        if self._validation_is_active():
            self._update_validation_status(status)
            return
        if self.calibration_session.state is CalibrationSessionState.COMPLETE:
            self._update_gaze_status(status)
            return
        self._update_tracking_status(status)

    def _log_calibration_quality_events(self) -> None:
        quality = self.calibration_session.target_quality
        if quality is None:
            return
        self.log_telemetry_event(
            "calibration_target_quality",
            calibration_target_quality_payload(quality),
        )
        if quality.recommendation == "retry":
            self.log_telemetry_event(
                "calibration_retry",
                calibration_target_quality_payload(quality),
            )

    def _log_calibration_feature_diagnostics(self) -> None:
        summary = summarize_feature_diagnostics(self.calibration_session.flow.all_samples())
        self.log_telemetry_event(
            "calibration_feature_diagnostics",
            feature_diagnostics_payload(summary),
        )

    def _validation_is_active(self) -> bool:
        return self.validation_session.state in {
            ValidationSessionState.SETTLING,
            ValidationSessionState.CAPTURING,
        }

    def _update_validation_status(self, status: TrackingStatus) -> None:
        """Update validation session and overlay from calibrated gaze."""

        if self.gaze_runtime is None:
            self._update_tracking_status(status)
            return
        target = self.validation_session.current_target
        screen_width = self.validation_session.screen_width
        screen_height = self.validation_session.screen_height
        sample = self.gaze_runtime.update(
            status.observation,
            screen_width=screen_width,
            screen_height=screen_height,
        )
        if sample is None:
            self._update_tracking_status(status)
            return
        if target is not None:
            self._prepare_screen_overlay(self.gaze_overlay)
            self.gaze_overlay.update_validation_sample(
                target=target,
                sample=sample,
                screen_width=screen_width,
                screen_height=screen_height,
            )
            self.log_telemetry_event(
                "validation_sample",
                validation_sample_payload(target, sample),
            )
            self.log_telemetry_event(
                "validation_replay_sample",
                validation_replay_sample_payload(target, status.observation),
            )
            self.gaze_overlay.show()
        advanced = self.validation_session.capture(sample)
        if advanced and self.validation_session.metrics is not None:
            self.log_telemetry_event(
                "validation_metrics",
                validation_metrics_payload(self.validation_session.metrics),
            )
        self._update_validation_debug()

    def _update_validation_debug(self) -> None:
        """Update debug/status labels for validation progress and metrics."""

        session = self.validation_session
        if session.state is ValidationSessionState.COMPLETE and session.metrics is not None:
            metrics = session.metrics
            recommendation = metrics.recommendation
            guidance = (
                "retry calibration"
                if recommendation == "retry"
                else f"{recommendation} calibration"
            )
            self.debug_label.setText(
                "Validation complete: "
                f"mean error {metrics.mean_error_px:.2f}px | "
                f"mean X error {metrics.mean_abs_x_error_px:.2f}px | "
                f"mean Y error {metrics.mean_abs_y_error_px:.2f}px | "
                f"Y bias {metrics.mean_signed_y_error_px:+.2f}px | "
                f"grid {metrics.grid_columns}x{metrics.grid_rows} accuracy "
                f"{metrics.grid_cell_accuracy:.0%} | "
                f"max error {metrics.max_error_px:.2f}px | {guidance}"
            )
            self.calibration_view.validation_button.setEnabled(True)
            return
        target_index = session.current_index + 1
        target_total = len(session.targets)
        self.debug_label.setText(
            f"Validation target {target_index}/{target_total} | "
            f"accepted {session.accepted_for_current_target}/"
            f"{session.timing_config.min_samples_per_target}"
        )

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

    def handle_gaze_sample(self, sample: GazeSample) -> None:
        """Update the transparent overlay from one calibrated gaze sample."""

        if sample.valid:
            self._prepare_screen_overlay(self.gaze_overlay)
        self.gaze_overlay.update_sample(sample)
        if sample.valid:
            self.log_telemetry_event("gaze_sample", gaze_event_payload(sample))
            self.gaze_overlay.show()
            self._update_window_candidate_status(sample)
        else:
            self.gaze_overlay.update_window_candidate(None)
            self.gaze_overlay.hide()

    def set_heatmap_enabled(self, enabled: bool) -> None:
        """Toggle verification heatmap accumulation and rendering."""

        screen_width, screen_height = self._screen_size()
        self._prepare_screen_overlay(self.gaze_overlay)
        self.gaze_overlay.configure_heatmap(
            screen_width=screen_width,
            screen_height=screen_height,
        )
        self.gaze_overlay.set_heatmap_enabled(enabled)
        self.show_heatmap_button.setChecked(enabled)
        self.show_heatmap_button.setText("Hide Heatmap" if enabled else "Show Heatmap")
        if enabled:
            if self.calibration_session.state is CalibrationSessionState.COMPLETE:
                message = "Heatmap enabled: waiting for calibrated gaze samples"
            else:
                message = (
                    "Heatmap enabled: waiting for calibrated gaze samples; "
                    "complete calibration first"
                )
            self.debug_label.setText(message)
        else:
            self.debug_label.setText("Heatmap disabled")

    def clear_heatmap(self) -> None:
        """Clear verification heatmap samples."""

        self.gaze_overlay.clear_heatmap()
        self.debug_label.setText("Heatmap cleared")

    def set_gaze_focus_enabled(self, enabled: bool) -> None:
        """Toggle opt-in gaze-selected window activation."""

        self.gaze_focus_enabled = enabled
        self._last_activated_window_key = None
        self.gaze_focus_button.setChecked(enabled)
        self._sync_gaze_focus_button_text()
        self.debug_label.setText(
            "Gaze focus enabled" if enabled else "Gaze focus disabled"
        )

    def _sync_gaze_focus_button_text(self) -> None:
        self.gaze_focus_button.setText(
            "Gaze Focus ON" if self.gaze_focus_enabled else "Gaze Focus OFF"
        )

    def _activate_window_candidate_if_enabled(self, candidate: WindowCandidate) -> bool:
        if not self.gaze_focus_enabled:
            return True
        activation_key = self._window_candidate_activation_key(candidate)
        if activation_key == self._last_activated_window_key:
            return True
        try:
            self.window_activator(candidate)
        except Exception as error:
            self.debug_label.setText(f"Debug: focus unavailable: {error}")
            return False
        self._last_activated_window_key = activation_key
        return True

    @staticmethod
    def _window_candidate_activation_key(candidate: WindowCandidate) -> tuple[Any, ...]:
        return (
            candidate.process_id,
            candidate.app_name,
            candidate.title,
            candidate.bounds.x,
            candidate.bounds.y,
            candidate.bounds.width,
            candidate.bounds.height,
        )

    def _update_window_candidate_status(self, sample: GazeSample) -> None:
        """Update debug output with the current visible window candidate."""

        try:
            candidate = candidate_at_point(
                Point2D(sample.x, sample.y),
                self.window_provider(),
            )
        except Exception as error:
            self.debug_label.setText(f"Debug: window unavailable: {error}")
            return
        self.log_telemetry_event("window_candidate", window_candidate_payload(candidate))
        self.gaze_overlay.update_window_candidate(candidate)
        if candidate is None:
            self._last_activated_window_key = None
            self.debug_label.setText(
                f"Debug: gaze {sample.region_id} | confidence {sample.confidence:.2f} | window none"
            )
            return
        if not self._activate_window_candidate_if_enabled(candidate):
            return
        title = f" — {candidate.title}" if candidate.title else ""
        self.debug_label.setText(
            f"Debug: gaze {sample.region_id} | confidence {sample.confidence:.2f} | "
            f"window {candidate.app_name}{title}"
        )

    def _refresh_calibration_view_status(self) -> None:
        """Refresh calibration labels, including timed quality progress when enabled."""

        self.calibration_view.refresh()
        self.calibration_target_overlay.update()
        session = self.calibration_session
        if session.timing_config is None or session.state is not CalibrationSessionState.COLLECTING:
            return
        self.calibration_view.show_quality_progress(
            phase=session.phase,
            progress=session.capture_progress,
            accepted_count=session.accepted_for_current_target,
            min_samples=session.timing_config.min_samples_per_target,
            rejected_count=session.rejected_for_current_target,
            quality=session.target_quality,
        )

    def _update_calibration_status(self, status: TrackingStatus | None) -> None:
        """Update the debug label with calibration status."""

        session = self.calibration_session
        if session.state is CalibrationSessionState.COMPLETE and session.fit_result is not None:
            result = session.fit_result
            self.calibration_target_overlay.hide()
            self.calibration_view.validation_button.setEnabled(True)
            self.debug_label.setText(
                "Calibration complete: "
                f"{result.sample_count} samples | mean error {result.mean_error_px:.2f}px | "
                f"max error {result.max_error_px:.2f}px | Start validation"
            )
        elif session.state is CalibrationSessionState.FAILED:
            self.calibration_target_overlay.hide()
            self.debug_label.setText(f"Calibration failed: {session.error_message}")
        elif session.timing_config is not None:
            phase_message = self._calibration_phase_message()
            target_index = self.calibration_view.flow.current_index + 1
            target_total = len(self.calibration_view.flow.targets)
            self.debug_label.setText(
                f"Calibration target {target_index}/{target_total} | {phase_message} | "
                f"accepted {session.accepted_for_current_target}/"
                f"{session.timing_config.min_samples_per_target} | "
                f"rejected {session.rejected_for_current_target}"
            )
        elif status is not None and status.valid:
            self.debug_label.setText(
                f"Calibration collecting: confidence {status.confidence:.2f}"
            )
        elif status is not None:
            self.debug_label.setText(f"Calibration collecting: {status.message}")
        else:
            self.debug_label.setText("Calibration collecting: look at the visible target")

    def _calibration_phase_message(self) -> str:
        """Return concise text for the current timed calibration phase."""

        phase = self.calibration_session.phase
        if phase.name == "SETTLING":
            return "Settle: look at the dot"
        if phase.name == "CAPTURING":
            return f"Capturing: {round(self.calibration_session.capture_progress * 100)}%"
        if phase.name == "REVIEWING":
            return "Reviewing target quality"
        return "Calibration complete"

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
        self.calibration_target_overlay.close()
        super().closeEvent(event)
