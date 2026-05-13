"""Desktop demo application shell."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import cast

from PySide6.QtWidgets import QApplication

from desktop_demo.ui.main_window import MainWindow
from pupil_tracker import configure_logging, get_logger

_LOGGER = get_logger("desktop_demo")


def create_app(argv: Sequence[str] | None = None) -> QApplication:
    """Create the Qt application instance."""

    existing = QApplication.instance()
    if existing is not None:
        return cast(QApplication, existing)
    return QApplication(list(argv) if argv is not None else sys.argv)


def run(argv: Sequence[str] | None = None) -> int:
    """Run the desktop demo application."""

    configure_logging()
    app = create_app(argv)
    window = MainWindow()
    window.show()
    _LOGGER.info("started desktop demo")
    return app.exec()
