"""Tests for visible calibration target rendering."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from pupil_tracker.calibration import CalibrationPhase, TargetQualitySummary

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


def test_calibration_view_exposes_current_target_position_without_embedded_target_panel(
    qt_app: QApplication,
) -> None:
    from desktop_demo.ui.calibration_view import (
        CalibrationFlowState,
        CalibrationTargetWidget,
        CalibrationView,
    )

    flow = CalibrationFlowState(samples_per_target=1)
    view = CalibrationView(flow=flow)

    assert view.current_target_position() == (0.1, 0.1)
    assert view.findChildren(CalibrationTargetWidget) == []
    view.close()
    qt_app.processEvents()


def test_target_widget_maps_normalized_target_to_widget_pixels(qt_app: QApplication) -> None:
    from desktop_demo.ui.calibration_view import CalibrationFlowState, CalibrationTargetWidget

    target_widget = CalibrationTargetWidget(CalibrationFlowState(samples_per_target=1))
    target_widget.resize(400, 300)

    assert target_widget.current_target_position() == (0.1, 0.1)
    assert target_widget.current_target_pixel() == (40, 30)
    target_widget.close()
    qt_app.processEvents()


def test_target_widget_updates_after_flow_advances(qt_app: QApplication) -> None:
    from desktop_demo.ui.calibration_view import CalibrationFlowState, CalibrationTargetWidget
    from pupil_tracker.models import RawObservation

    flow = CalibrationFlowState(samples_per_target=1)
    target_widget = CalibrationTargetWidget(flow)
    target_widget.resize(400, 300)

    assert flow.capture_observation(
        RawObservation(timestamp=1.0, valid=True, confidence=0.9, feature_vector=(0.1, 0.2))
    )

    assert target_widget.current_target_position() == (0.5, 0.1)
    assert target_widget.current_target_pixel() == (200, 30)
    target_widget.close()
    qt_app.processEvents()


def test_target_widget_render_marks_target_pixels(qt_app: QApplication) -> None:
    from desktop_demo.ui.calibration_view import CalibrationFlowState, CalibrationTargetWidget

    target_widget = CalibrationTargetWidget(CalibrationFlowState(samples_per_target=1))
    target_widget.resize(200, 200)

    image = QImage(target_widget.size(), QImage.Format.Format_RGB32)
    image.fill(0)
    target_widget.render(image)

    target_pixel = target_widget.current_target_pixel()
    assert target_pixel is not None
    x, y = target_pixel
    target_pixel_color = image.pixelColor(x, y)
    background_pixel = image.pixelColor(199, 199)

    assert target_pixel_color != background_pixel
    target_widget.close()
    qt_app.processEvents()


def test_calibration_view_shows_settle_phase_status(qt_app: QApplication) -> None:
    from desktop_demo.ui.calibration_view import CalibrationFlowState, CalibrationView

    view = CalibrationView(flow=CalibrationFlowState(samples_per_target=20))

    view.show_quality_progress(
        phase=CalibrationPhase.SETTLING,
        progress=0.25,
        accepted_count=0,
        min_samples=20,
        rejected_count=0,
        quality=None,
    )

    assert "Settle: look at the dot" in view.status_label.text()
    assert "Accepted: 0/20 | Rejected: 0" in view.status_label.text()
    view.close()
    qt_app.processEvents()


def test_calibration_view_shows_capture_progress_status(qt_app: QApplication) -> None:
    from desktop_demo.ui.calibration_view import CalibrationFlowState, CalibrationView

    view = CalibrationView(flow=CalibrationFlowState(samples_per_target=20))

    view.show_quality_progress(
        phase=CalibrationPhase.CAPTURING,
        progress=0.42,
        accepted_count=14,
        min_samples=20,
        rejected_count=3,
        quality=None,
    )

    assert "Capturing: 42%" in view.status_label.text()
    assert "Accepted: 14/20 | Rejected: 3" in view.status_label.text()
    view.close()
    qt_app.processEvents()


def test_calibration_view_shows_retry_quality_status(qt_app: QApplication) -> None:
    from desktop_demo.ui.calibration_view import CalibrationFlowState, CalibrationView

    view = CalibrationView(flow=CalibrationFlowState(samples_per_target=20))

    view.show_quality_progress(
        phase=CalibrationPhase.SETTLING,
        progress=0.0,
        accepted_count=0,
        min_samples=20,
        rejected_count=0,
        quality=TargetQualitySummary(
            target_id="r0c0",
            accepted_count=8,
            rejected_count=12,
            mean_confidence=0.72,
            meets_min_samples=False,
            recommendation="retry",
        ),
    )

    assert "Quality: retrying target" in view.status_label.text()
    view.close()
    qt_app.processEvents()
