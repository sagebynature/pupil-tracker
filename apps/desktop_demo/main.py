"""Desktop demo entry point placeholder.

The real demo implementation will be added from docs/plans/mvp.md.
"""

from pupil_tracker import configure_logging, get_logger


def main() -> int:
    configure_logging()
    logger = get_logger("desktop_demo")
    logger.info("Desktop demo is not implemented yet. See docs/plans/mvp.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
