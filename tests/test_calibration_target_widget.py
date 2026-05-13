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


def test_calibration_view_exposes_current_target_position(qt_app: QApplication) -> None:
    from desktop_demo.ui.calibration_view import CalibrationFlowState, CalibrationView

    flow = CalibrationFlowState(samples_per_target=1)
    view = CalibrationView(flow=flow)

    assert view.current_target_position() == (0.1, 0.1)
    view.close()
    qt_app.processEvents()


def test_target_widget_maps_normalized_target_to_widget_pixels(qt_app: QApplication) -> None:
    from desktop_demo.ui.calibration_view import CalibrationFlowState, CalibrationView

    view = CalibrationView(flow=CalibrationFlowState(samples_per_target=1))
    view.target_widget.resize(400, 300)

    assert view.target_widget.current_target_position() == (0.1, 0.1)
    assert view.target_widget.current_target_pixel() == (40, 30)
    view.close()
    qt_app.processEvents()


def test_target_widget_updates_after_flow_advances(qt_app: QApplication) -> None:
    from desktop_demo.ui.calibration_view import CalibrationFlowState, CalibrationView
    from pupil_tracker.models import RawObservation

    flow = CalibrationFlowState(samples_per_target=1)
    view = CalibrationView(flow=flow)
    view.target_widget.resize(400, 300)

    assert flow.capture_observation(
        RawObservation(timestamp=1.0, valid=True, confidence=0.9, feature_vector=(0.1, 0.2))
    )
    view.refresh()

    assert view.current_target_position() == (0.5, 0.1)
    assert view.target_widget.current_target_pixel() == (200, 30)
    view.close()
    qt_app.processEvents()


def test_target_widget_render_marks_target_pixels(qt_app: QApplication) -> None:
    from desktop_demo.ui.calibration_view import CalibrationFlowState, CalibrationView

    view = CalibrationView(flow=CalibrationFlowState(samples_per_target=1))
    view.target_widget.resize(200, 200)

    image = QImage(view.target_widget.size(), QImage.Format.Format_RGB32)
    image.fill(0)
    view.target_widget.render(image)

    target_pixel = view.target_widget.current_target_pixel()
    assert target_pixel is not None
    x, y = target_pixel
    target_pixel_color = image.pixelColor(x, y)
    background_pixel = image.pixelColor(199, 199)

    assert target_pixel_color != background_pixel
    view.close()
    qt_app.processEvents()
