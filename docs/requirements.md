# Pupil Tracker MVP Requirements

## Goal

Build a reusable pupil/gaze tracking library with a macOS-first desktop demo application. The demo uses a webcam to calibrate gaze, track eye/pupil movement, show an approximate tracking cursor on screen, classify the likely 3x3 screen region, and identify a likely visible application window target in debug mode.

Future use case: use gaze/pupil tracking to decide which application window should be placed in focus. The MVP must validate this direction without actually changing focus yet.

## Research Summary

Open-source ecosystem findings:

- MediaPipe Face Landmarker / Face Mesh / Iris is the best permissive first backend for webcam-based face, eye, and iris landmarks.
- WebGazer.js is useful as a reference for webcam gaze calibration, but GPL-3.0/browser-centric constraints make it a poor core dependency for a permissive Python library.
- Pupil Labs and EyeRecToo are strong references for serious IR/near-eye pupil tracking, but are app/hardware-oriented and commonly GPL-family.
- GazeTracking, eyeLike, OpenGazer, and ITU Gaze Tracker are useful educational or historical references, but not robust modern foundations.
- Commodity webcam gaze tracking is feasible for coarse region/window targeting and approximate cursor feedback, but not realistic for pixel-perfect mouse replacement.

## Decisions

### 1. Target Tracking Mode

Decision: Webcam first, architecture supports IR / near-eye backends later.

Implications:
- The first implementation targets an ordinary webcam.
- The core architecture must not assume MediaPipe or RGB webcam forever.
- Future tracker backends should support IR cameras, near-eye cameras, or vendor/hardware trackers.

### 2. Library Output Levels

Decision: Multi-level output.

The library should expose:

1. Raw observations:
   - face box / landmarks
   - eye landmarks
   - iris or pupil center in camera coordinates
   - blink / occlusion / confidence

2. Derived eye movement:
   - normalized eye position
   - gaze direction estimate
   - head pose-ish features if available
   - smoothed movement vector

3. Calibrated screen gaze:
   - display id
   - x/y screen coordinates
   - confidence
   - calibration quality / error
   - timestamp
   - validity flags

### 3. MVP Accuracy Target

Decision: Window-level targeting plus approximate cursor feedback.

The MVP should optimize for:
- likely window / screen-region targeting
- approximate gaze cursor visualization

The MVP should not attempt:
- pixel-perfect mouse replacement
- real focus stealing or gaze-clicking

### 4. First Platform

Decision: macOS only first.

Implications:
- The MVP can use macOS-specific APIs and packaging assumptions.
- Future platform abstractions should be retained, but Windows/Linux implementation is not required for the first milestone.
- Future window focus work will likely require macOS Accessibility permissions.

### 5. MVP Stack

Decision: Python + uv + Make + PySide6/Qt + OpenCV + MediaPipe + pytest + ruff + ty.

Rationale:
- Fast iteration for CV experiments.
- `uv` provides reproducible environment and lockfile management.
- `Makefile` provides stable developer commands for sync, lint, typecheck, test, and demo launch.
- `ty` is the project typechecker.
- Python source should use standard-library logging through project logging helpers instead of `print`/printf-style diagnostics.
- Good support for camera preview, calibration UI, debug panels, and transparent overlay windows.
- Clean path to later port performance-critical or native pieces to C++/Rust/Swift if needed.

### 6. Repository Shape

Decision: Library package plus demo app in the same repo.

Proposed shape:

```text
pupil-tracker/
  pyproject.toml
  src/pupil_tracker/
    camera/
    tracking/
    calibration/
    smoothing/
    screen/
    platform/
    models.py
  apps/desktop_demo/
    main.py
    ui/
      overlay.py
      calibration_view.py
      debug_view.py
  tests/
  docs/
```

The demo app must consume the library instead of owning core tracking logic.

### 7. Calibration UX

Decision: Default 9-point calibration with configurable calibration patterns.

Requirements:
- Default pattern: 3x3 grid.
- Calibration pattern definitions should support 5, 9, 13, 16, and custom layouts later.
- Collect multiple samples per target.
- Surface calibration quality / error.
- Allow recalibration when confidence or quality is poor.

### 8. Demo Cursor Behavior

Decision: Confidence-aware cursor plus debug overlay.

Normal mode:
- estimated gaze dot
- confidence halo / radius
- color-coded validity if useful

Debug mode:
- dot
- trail
- confidence value
- FPS / latency
- current 3x3 region
- likely window candidate once available
- blink / occlusion state if available

### 9. MVP Success Definition

Decision: A + B + C.

The MVP is successful when it can:
- calibrate from webcam
- show a stable approximate gaze cursor
- classify gaze into a 3x3 screen region
- enumerate visible macOS windows and identify the likely target window in debug mode

The MVP must not actually change app focus.

### 10. Tracking Backend

Decision: Plugin-style backend interface with MediaPipe implemented first.

Requirements:
- Define a tracker backend interface.
- Implement `MediaPipeTrackerBackend` first.
- Leave room for OpenCV heuristic, EyeGestures, Pupil Labs / IR, and hardware backends later.

### 11. License Posture

Decision: Permissive-first core, likely MIT or Apache-2.0; optional GPL adapters later only with clear boundaries.

Rules:
- Do not copy GPL code into the core.
- Avoid GPL dependencies in the default path.
- Treat GPL projects as research references unless intentionally isolated.
- MediaPipe/OpenCV/PySide-compatible dependencies are preferred for the first implementation.

### 12. Logging / Data Recording

Decision: Metrics and calibration sample export now; optional video/frame capture later.

MVP logs should include non-video telemetry:
- timestamps
- tracker backend name/version
- observation summaries
- calibrated gaze point
- confidence
- FPS / latency
- 3x3 region candidate
- visible-window candidate once implemented

Calibration exports should include:
- target point
- collected samples
- fitted model metadata
- calibration error / quality

Do not record camera video or frames by default. Any future video/frame capture must be explicit opt-in.

### 13. Calibration Model

Decision: Pluggable calibration model using continuous polynomial/ridge regression first, with region classification derived from calibrated coordinates.

Requirements:
- Collect 9-point calibration samples.
- Build feature vectors from MediaPipe landmarks / iris positions / head-pose-ish values.
- Fit a regularized polynomial/ridge regression model for screen x/y.
- Derive 3x3 region from calibrated x/y.
- Measure and surface calibration error and confidence.
- Keep affine, classifier, and more advanced models pluggable.

### 14. API Boundary

Decision: Layered API.

Core:
- synchronous, deterministic pull API
- easy to unit test
- no Qt dependency

Runtime:
- callback or async/event stream over observations and gaze samples

Demo:
- adapts runtime events into Qt signals

Future:
- local WebSocket/IPC service for other applications.

### 15. Demo Polish

Decision: Simple user-facing demo plus debug panel.

User-facing flow:
- choose camera
- run 9-point calibration
- show gaze cursor with confidence halo
- show region/window target estimate

Debug panel:
- camera preview
- face/eye/iris landmarks
- FPS / latency
- confidence
- calibration quality / error
- current 3x3 region
- likely window candidate
- logging/session controls

### 16. First Milestone

Decision: End-to-end minimal demo in thin vertical slices.

Sequence:
1. camera preview
2. MediaPipe observation stream
3. feature extraction
4. 9-point calibration collection
5. first polynomial/ridge regression calibration model
6. transparent overlay cursor with confidence halo
7. 3x3 region classification
8. macOS visible-window candidate debug readout

### 17. Testing / Validation

Decision: Unit tests plus synthetic calibration fixtures.

Automated tests should cover:
- data models
- calibration target generation
- polynomial/ridge calibration model
- smoothing/filtering
- confidence handling
- 3x3 region mapping
- tracker backend interface using mocks

Manual validation covers:
- camera permissions
- MediaPipe runtime
- overlay cursor
- macOS window enumeration

Future validation:
- opt-in recorded sample replay tests.

## Non-Goals for MVP

- Pixel-perfect cursor / mouse replacement.
- Actually focusing or raising application windows.
- Windows/Linux packaging or support.
- Wayland global overlay/focus behavior.
- Video/frame recording by default.
- GPL dependency adoption in the core package.
- Product-polished UI.

## Open Questions for Later

- MIT vs Apache-2.0 final license.
- Whether to support multiple monitors in MVP or immediately after.
- Exact macOS window enumeration approach: Accessibility API, CoreGraphics, AppKit, or combination.
- Packaging strategy: editable/developer app first vs packaged `.app` early.
- Whether MediaPipe Tasks API or legacy solutions API is the best Python dependency path at implementation time.
