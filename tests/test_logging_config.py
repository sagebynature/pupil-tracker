import logging

from pupil_tracker.logging_config import configure_logging, get_logger


def test_configure_logging_sets_named_root_level() -> None:
    logger = configure_logging(level=logging.DEBUG)

    assert logger.name == "pupil_tracker"
    assert logger.level == logging.DEBUG
    assert logger.handlers


def test_get_logger_returns_child_logger() -> None:
    logger = get_logger("tracking")

    assert logger.name == "pupil_tracker.tracking"
