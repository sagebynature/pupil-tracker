"""Logging configuration for Pupil Tracker.

Application and library code should use this module instead of `print` so logs
can be routed, filtered, and formatted consistently by callers.
"""

from __future__ import annotations

import logging
import sys

_LOGGER_NAME = "pupil_tracker"
_DEFAULT_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the package logger.

    The function is idempotent: repeated calls update the logger level without
    attaching duplicate stream handlers.
    """

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        logger.addHandler(handler)

    for handler in logger.handlers:
        handler.setLevel(level)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the package logger or a named child logger."""

    if name is None or name == "":
        return logging.getLogger(_LOGGER_NAME)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")
