# Pupil Tracker

A webcam-first pupil and gaze tracking library with a macOS desktop demo application.

The project is currently in MVP planning. The first implementation target is a Python library plus PySide6/Qt demo app that uses a webcam to calibrate gaze, estimate approximate screen position, show a confidence-aware cursor overlay, classify the current 3x3 screen region, and identify the likely visible macOS window target in debug mode.

## MVP Direction

- Platform: macOS first.
- Stack: Python, uv, Make, PySide6/Qt, OpenCV, MediaPipe, NumPy, scikit-learn, pytest, ruff, and ty.
- Tracking mode: commodity webcam first, with architecture for future IR / near-eye backends.
- Output levels:
  - raw face / eye / iris observations
  - derived eye movement features
  - calibrated screen gaze samples
- Calibration: configurable patterns with a default 9-point grid.
- Demo: simple user-facing calibration and tracking flow plus developer debug panel.
- Privacy: no camera video or frame recording by default.
- Future use case: gaze-assisted application window focus, not enabled in the MVP.

## Repository Layout

Planned layout:

```text
pupil-tracker/
  src/pupil_tracker/       # importable library package
  apps/desktop_demo/       # PySide6 desktop demo app
  tests/                   # unit and synthetic calibration tests
  docs/
    requirements.md        # interview decisions and MVP requirements
    plans/mvp.md           # task-by-task implementation plan
```

The demo app should consume the library rather than owning core tracking, calibration, or platform logic.

## Documentation

Start here:

- `docs/requirements.md` — product/research decisions, MVP scope, non-goals, and open questions.
- `docs/plans/mvp.md` — implementation plan with tasks, file paths, tests, and verification steps.

## Development Status

This repository currently contains planning documentation and project setup files. The implementation plan starts with package initialization, core data models, calibration patterns, screen-region mapping, smoothing, calibration regression, tracker backend interfaces, and then the desktop demo.

## Intended Development Setup

The project uses `uv` for dependency and lockfile management, `make` for common developer commands, `pytest` for tests, `ruff` for linting/formatting, and `ty` for type checking.

```bash
make sync
make check
make run-demo
```

Python code should use the standard-library `logging` module through `pupil_tracker.logging_config`; avoid `print`/printf-style diagnostics in source code so application callers can route logs consistently.

The desktop demo will require macOS camera permission. Future window-target and focus-related work may require macOS Accessibility permission, but the MVP must not actually focus or raise windows.

## Licensing Posture

The core project is permissive-first and uses the MIT License. GPL eye-tracking projects may be used as research references, but GPL code should not be copied into the core package. Optional GPL-compatible adapters may be considered later only with clear licensing boundaries.

## Non-Goals for MVP

- Pixel-perfect mouse replacement.
- Actual app/window focus changes.
- Windows or Linux support.
- Wayland global overlay/focus behavior.
- Video/frame recording by default.
- Product-polished UI.
