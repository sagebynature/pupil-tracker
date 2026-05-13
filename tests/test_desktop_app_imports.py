"""Headless smoke tests for the desktop demo shell."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
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


def test_main_module_imports_without_starting_app() -> None:
    from desktop_demo import main

    assert callable(main.main)


def test_main_window_class_can_be_imported() -> None:
    from desktop_demo.ui.main_window import MainWindow

    assert MainWindow.__name__ == "MainWindow"


def test_camera_worker_is_not_started_on_import() -> None:
    from desktop_demo.ui.main_window import CameraPreviewWorker

    assert CameraPreviewWorker.instances_started == 0


def test_main_window_contains_camera_preview_shell(qt_app: QApplication) -> None:
    from desktop_demo.ui.main_window import MainWindow

    window = MainWindow()
    qt_app.processEvents()

    assert window.windowTitle() == "Pupil Tracker Demo"
    assert window.start_button.text() == "Start Camera"
    assert window.stop_button.text() == "Stop Camera"
    assert window.preview_label.text() == "Camera preview stopped"
    assert "Debug" in window.debug_label.text()
