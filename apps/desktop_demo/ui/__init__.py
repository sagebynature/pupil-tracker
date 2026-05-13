"""Qt UI widgets for the desktop demo."""

from desktop_demo.ui.annotations import annotate_observation
from desktop_demo.ui.calibration_view import CalibrationFlowState, CalibrationView
from desktop_demo.ui.frame_image import bgr_ndarray_to_qimage
from desktop_demo.ui.main_window import CameraPreviewWorker, MainWindow
from desktop_demo.ui.overlay import CursorRenderState, GazeOverlay, OverlayState

__all__ = [
    "CalibrationFlowState",
    "CalibrationView",
    "CameraPreviewWorker",
    "CursorRenderState",
    "GazeOverlay",
    "MainWindow",
    "OverlayState",
    "annotate_observation",
    "bgr_ndarray_to_qimage",
]
