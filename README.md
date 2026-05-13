# Pupil Tracker

A webcam-first pupil and gaze tracking library with a macOS desktop demo application.

The MVP provides a Python library plus PySide6/Qt demo shell for coarse gaze tracking experiments: webcam capture, MediaPipe-based iris/face observations, 9-point calibration, polynomial/ridge gaze calibration, 3x3 screen-region mapping, a confidence-aware transparent gaze overlay, macOS visible-window candidate scoring, and opt-in JSONL telemetry.

The future product direction is gaze-assisted application window focus. The MVP deliberately does not focus, raise, click, or activate windows.

## Status

This repo contains the first macOS-focused MVP implementation slices and automated tests. Hardware/live-GUI validation is still manual because the project uses the local camera, desktop overlay, and macOS window enumeration.

Implemented library/demo areas:

- Core immutable models for observations, calibration samples, gaze samples, and window candidates.
- 9-point calibration target generation and sample collection.
- Polynomial/ridge calibration model.
- Exponential moving-average gaze smoothing.
- 3x3 screen-region classification.
- Pluggable tracker backend protocol.
- OpenCV camera source.
- MediaPipe Tasks/FaceLandmarker-backed tracker adapter with injectable fakes for tests.
- Synchronous runtime pipeline for camera/backend/calibration/smoothing/region mapping.
- PySide6 desktop demo shell with camera, calibration, overlay, and telemetry controls.
- Transparent confidence-aware gaze overlay widget.
- macOS CoreGraphics visible-window enumeration and pure candidate scoring.
- Privacy-conscious JSONL telemetry with no frame/video payloads by default.

## Requirements

- macOS for the desktop MVP.
- Python 3.11.
- `uv` installed.
- A webcam for live tracking.
- macOS camera permission for live camera usage.
- A MediaPipe FaceLandmarker model asset for real MediaPipe Tasks inference when using the default backend path.

Accessibility permission is not required for the MVP because the app does not focus, raise, activate, click, or control other windows. Future gaze-to-focus work may require Accessibility permission.

## Setup

```bash
make sync
```

This runs `uv sync --dev` and installs the locked runtime/dev environment.

## Verification

Run all automated checks:

```bash
make check
```

This runs:

- `ruff check src apps tests`
- `ty check src apps tests`
- `pytest -v`

Optional diff hygiene before committing:

```bash
git diff --check
```

## Launch the Demo

Real tracker/calibration mode requires a MediaPipe FaceLandmarker model asset. Point the demo at the `.task` file before launch:

```bash
export PUPIL_TRACKER_MEDIAPIPE_MODEL=/absolute/path/to/face_landmarker.task
make run-demo
```

If this variable is missing or points to a non-existent file, camera preview can still start, but tracker-backed calibration will show in-app setup guidance instead of failing silently.

```bash
make run-demo
```

The demo launches the PySide6 desktop shell. It does not start the camera on import; camera use should happen only after explicit user interaction.

Manual live testing should follow `docs/manual-test-checklist.md`.

## Privacy and Telemetry

The app is privacy-conscious by default:

- No camera video is recorded by default.
- No frame/image arrays are written to telemetry by default.
- JSONL telemetry is opt-in through Start Logging / Stop Logging controls.
- Default demo telemetry path is under `metrics/`, which is ignored by git.
- Telemetry serializers include scalar summaries such as timestamps, gaze coordinates, confidence, calibration target ids, sample counts, and visible-window candidate metadata.

Any future video/frame capture feature must be explicit opt-in and documented separately.

## Known MVP Limitations

- Commodity webcam gaze tracking is coarse; expect screen-region/window-level utility, not pixel-perfect cursor replacement.
- Accuracy depends heavily on lighting, camera placement, face visibility, head movement, and calibration quality.
- The demo is macOS-first and developer-oriented; Windows/Linux packaging is out of scope for the MVP.
- Multi-monitor behavior is not fully specified.
- The MediaPipe backend uses the installed MediaPipe Tasks API; real inference requires an appropriate FaceLandmarker model asset path.
- The current UI is an MVP shell; full runtime wiring from camera to calibrated overlay is still a follow-on integration/polish area.
- The app enumerates and scores visible windows for debug purposes only and does not change focus.

## Repository Layout

```text
pupil-tracker/
  src/pupil_tracker/       # importable library package
  apps/desktop_demo/       # PySide6 desktop demo app
  tests/                   # unit and headless smoke tests
  docs/
    requirements.md        # interview decisions and MVP requirements
    plans/                 # implementation plans
    manual-test-checklist.md
```

The demo app consumes the library rather than owning core tracking, calibration, or platform logic.

## Development Conventions

- Use `uv` for dependency and lockfile management.
- Use `make check` before commits.
- Use standard-library `logging` through `pupil_tracker.logging_config`; avoid `print`/printf-style diagnostics in source code.
- Keep automated tests hardware-free: use fakes for OpenCV, MediaPipe, Qt, and CoreGraphics where possible.
- Keep core library behavior independent of Qt/OpenCV/MediaPipe where practical.

## Documentation

Start here:

- `docs/requirements.md` — product/research decisions, MVP scope, non-goals, and resolved implementation choices.
- `docs/plans/mvp.md` — high-level implementation plan.
- `docs/plans/implementation-task-plan.md` — completed task-by-task TDD execution plan.
- `docs/manual-test-checklist.md` — manual live-camera/live-GUI validation steps.

## Licensing Posture

The core project uses the MIT License and is permissive-first. GPL eye-tracking projects may be used as research references, but GPL code should not be copied into the core package. Optional GPL-compatible adapters may be considered later only with clear licensing boundaries.

## Non-Goals for MVP

- Pixel-perfect mouse replacement.
- Actual app/window focus changes.
- Windows or Linux support.
- Wayland global overlay/focus behavior.
- Video/frame recording by default.
- Product-polished UI.
