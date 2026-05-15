# Pupil Tracker

A webcam-first pupil and gaze tracking library with a macOS desktop demo application.

The MVP provides a Python library plus PySide6/Qt demo shell for coarse gaze tracking experiments: webcam capture, MediaPipe-based iris/face observations, timed quality-gated 9-point calibration, post-calibration validation metrics, polynomial/ridge gaze calibration, 3x3 screen-region mapping, confidence-aware transparent gaze/validation overlays, a gaze heatmap, macOS visible-window candidate scoring, opt-in gaze-to-focus, and opt-in JSONL telemetry.

Gaze-assisted application focus is available only behind the explicit Gaze Focus toggle or `PUPIL_TRACKER_GAZE_FOCUS_ENABLED=true`. The default demo path still observes windows without focusing, raising, clicking, or activating them.

## Status

This repo contains the first macOS-focused MVP implementation slices and automated tests. Hardware/live-GUI validation is still manual because the project uses the local camera, desktop overlay, and macOS window enumeration.

Implemented library/demo areas:

- Core immutable models for observations, calibration samples, gaze samples, and window candidates.
- 9-point calibration target generation and sample collection.
- Timed calibration phases with settle/capture/review windows and quality-gated retries.
- Polynomial/ridge calibration model.
- Post-calibration validation targets, validation session controller, and mean/median/max error metrics.
- Exponential moving-average gaze smoothing.
- 3x3 screen-region classification.
- Pluggable tracker backend protocol.
- OpenCV camera source.
- MediaPipe Tasks/FaceLandmarker-backed tracker adapter with injectable fakes for tests.
- Synchronous runtime pipeline for camera/backend/calibration/smoothing/region mapping.
- PySide6 desktop demo shell with camera, calibration, validation, overlay, heatmap, opt-in gaze focus, and telemetry controls.
- Transparent confidence-aware gaze overlay and validation target/prediction/error-line overlay.
- Gaze trail and heatmap verification helpers for live accuracy checks.
- macOS CoreGraphics visible-window enumeration, pure candidate scoring, and separate opt-in AppKit application activation.
- Privacy-conscious JSONL telemetry with no frame/video payloads by default.

## Requirements

- macOS for the desktop MVP.
- Python 3.11.
- `uv` installed.
- A webcam for live tracking.
- macOS camera permission for live camera usage.
- A MediaPipe FaceLandmarker model asset for real MediaPipe Tasks inference when using the default backend path.

Accessibility permission is not required for candidate scoring. The optional Gaze Focus mode uses AppKit application activation, does not synthesize clicks, and reports an in-app `focus unavailable` status if macOS refuses activation.

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

Real tracker/calibration mode requires a MediaPipe FaceLandmarker model asset. The easiest path is to download the default model into `models/`:

```bash
make download-model
export PUPIL_TRACKER_MEDIAPIPE_MODEL=$(pwd)/models/face_landmarker.task
make run-demo
```

You can also point the demo at any compatible `.task` file directly:

```bash
PUPIL_TRACKER_MEDIAPIPE_MODEL=/absolute/path/to/face_landmarker.task make run-demo
```

If this variable is missing or points to a non-existent file, camera preview can still start, but tracker-backed calibration will show in-app setup guidance instead of failing silently.

```bash
make run-demo
```

The demo launches the PySide6 desktop shell. It does not start the camera on import; camera use happens only after explicit user interaction.

Expected manual path:

1. Click Start Camera and confirm live preview.
2. Center your face and confirm tracker annotations appear.
3. Click Start Calibration and follow the 9 fullscreen targets. For replay-backed geometry experiments, use Start Edge-Dense Calibration for the 17-point edge/corner path, Start Top-Left Focus Calibration for the 25-point top-left `v0` collapse path, or Start Top-Row Focus Calibration for the 33-point `v0`/`v1` top-row collapse path.
4. Hold gaze through each target's Settle and Capture phases.
5. Confirm calibration completes with fit metrics.
6. Click Start Validation and follow the validation targets.
7. Confirm target dot, predicted dot, error line, and validation metrics are understandable.
8. Move gaze around the screen and confirm overlay, 3x3 region, heatmap, and window-candidate debug text update plausibly.
9. Optional: enable Gaze Focus and confirm the app under the current gaze candidate comes forward; disable it before continuing accuracy diagnostics if focus changes are distracting.
10. If logging is enabled, confirm JSONL telemetry contains scalar events only and no frame/image payloads.
11. Stop Camera or close the app and confirm camera/tracker/overlay/log resources are released.

Manual live testing should follow `docs/manual-test-checklist.md`.

## Calibration, Validation, and Accuracy Checks

Calibration is accuracy-first. The demo intentionally asks for stable timed samples before trusting a fit:

- Settle: look at the target dot and hold still. Samples are ignored during this short window.
- Capture: keep looking at the target. Valid, confident observations are counted.
- Review: the session checks accepted/rejected counts. Low-quality targets are retried instead of silently advancing.

After calibration completes, run validation before judging tracking quality. Validation uses held-out target points and reports:

- Mean error: average distance between validation target and predicted gaze.
- Median error: typical distance, less sensitive to outliers.
- Max error: worst observed distance.
- Mean X error: average horizontal miss distance.
- Mean Y error: average vertical miss distance.
- Y bias: signed vertical offset; positive means predictions are lower on screen than the target, negative means predictions are higher.
- Grid accuracy: practical same-cell hit rate for a configurable validation grid. Defaults to `4x3`, configurable with `PUPIL_TRACKER_VALIDATION_GRID_COLUMNS` and `PUPIL_TRACKER_VALIDATION_GRID_ROWS`.
- Recommendation: `excellent`, `good`, `usable`, or `retry`.

Use the validation overlay to diagnose failures. The target dot is where you should look, the predicted dot is the calibrated estimate, and the line between them is the current error. If vertical tracking feels weaker than horizontal tracking, compare Mean X error, Mean Y error, and Y bias before changing model settings:

- High Y bias in the same direction across runs usually means camera angle, seating position, or calibration posture is systematically offset. Reposition the camera, reduce head pitch, improve lighting, and recalibrate.
- High Mean Y error with low signed Y bias usually means vertical estimates are noisy or compressed around the center. That points to feature extraction improvements rather than blind model tuning.
- High X and Y error together usually means the full calibration was poor; improve face visibility/head stability and retry.

If the recommendation is `retry`, improve lighting/camera position, reduce head movement, and recalibrate.

Calibration targets are shown fullscreen because the fitted model maps observations to full-monitor coordinates. The outer 9-point targets remain inset from the physical edges so they sample the usable screen area without forcing hard-to-hold edge fixations.

Start Edge-Dense Calibration is an experimental, non-default geometry check for top-row and edge/corner failures seen in replay analysis. It uses 17 targets: denser top/bottom edge rows, upper/lower quadrant points aligned with validation hot spots, and middle left/center/right anchors. Use it for a fresh logged manual validation run before changing live defaults.

Start Top-Left Focus Calibration is a second experimental, non-default geometry check for the persistent `v0` top-left collapse. It uses 25 targets: the edge-dense broad anchors plus a 3x3 local cluster around the held-out top-left validation region `(0.25, 0.25)`. Use it only for logged repeat-run comparison against the latest edge-dense runs; do not treat one improved run as a default-change signal.

Start Top-Row Focus Calibration is a third experimental, non-default geometry check for paired `v0`/`v1` top-row collapse. It uses 33 targets: broad edge anchors plus two 3x3 local clusters around the held-out top validation regions `(0.25, 0.25)` and `(0.75, 0.25)`. Use it only after top-left focus evidence shows top-row failures move laterally or affect both top validation targets; compare predicted-cell distributions for `v0`, `v1`, `v3`, and `v4` before changing defaults.

For longer live checks, enable Show Heatmap and stare at fixed points. The heatmap should cluster where you hold your gaze. Use Clear Heatmap between trials.

Gaze Focus is off by default. Turn on the Gaze Focus button, or launch with `PUPIL_TRACKER_GAZE_FOCUS_ENABLED=true`, to immediately activate the current visible-window candidate when calibrated gaze lands on it. The activation path is app-level focus by macOS process id; it does not click inside the target app and it avoids repeated activation while gaze remains on the same candidate.

## Privacy and Telemetry

The app is privacy-conscious by default:

- No camera video is recorded by default.
- No frame/image arrays are written to telemetry by default.
- JSONL telemetry is opt-in through Start Logging / Stop Logging controls.
- Default demo telemetry path is under `metrics/`, which is ignored by git.
- Telemetry serializers include scalar summaries such as timestamps, gaze coordinates, confidence, calibration target ids, sample counts, calibration quality, feature diagnostics, replayable scalar feature samples, validation samples, validation metrics, and visible-window candidate metadata.

After a logged calibration run, inspect feature separability with:

```bash
uv run python tools/analyze_feature_diagnostics.py metrics/demo.jsonl
```

Use the report to compare top/center/bottom feature deltas before adding new gaze features or tuning the calibration model.

After a fresh logged calibration and validation run, compare calibration model variants offline with:

```bash
uv run python tools/evaluate_calibration_models.py metrics/demo.jsonl --screen-width 1512 --screen-height 982 --grid-columns 4 --grid-rows 3 --objective grid --calibration-sample-window middle
```

Use the same screen dimensions as the manual run. The evaluator uses only `calibration_replay_sample` and `validation_replay_sample` scalar payloads, so it can compare candidate models without saving frames or re-running the camera session. Use `--calibration-sample-window all|early|middle|late` to test whether target-capture timing affects the model fit. The evaluator also includes replay-only target-weighted candidates for vertical edges, screen edges, and corners, replay-only asymmetric quadrant correction candidates, plus replay-only vertical-bias and per-band correction candidates; these are comparisons, not live behavior. Add `--include-target-residuals` when a run regresses to append per-target calibration and validation residual tables for the top-ranked model.

When two logged live runs disagree, compare target-specific validation behavior before changing defaults:

```bash
uv run python tools/analyze_repeat_run_diagnostics.py metrics/demo.jsonl --run START1:END1 --run START2:END2 --screen-width WIDTH --screen-height HEIGHT --grid-columns 4 --grid-rows 3
```

The repeat-run analyzer uses scalar `validation_sample`, `validation_metrics`, and `calibration_replay_sample` events, trims each run to the latest metrics sample window per target, and reports signed residual shifts, grid collapse/recovery flags, predicted grid-cell distributions, and calibration feature-drift deltas with named dominant feature changes. It does not require or emit frames, screenshots, or landmark dumps.

To validate the live late-sample policy without changing the default, launch the demo with:

```bash
PUPIL_TRACKER_CALIBRATION_SAMPLE_WINDOW=late PUPIL_TRACKER_MEDIAPIPE_MODEL=$(pwd)/models/face_landmarker.task make run-demo
```

The default live calibration sample window remains `all`; use `late` only for replay-backed manual validation until a fresh run confirms it improves practical grid/window selection.

To test the opt-in posture/head-pose stability gate during calibration capture, launch with a positive feature-drift threshold:

```bash
PUPIL_TRACKER_POSTURE_STABILITY_MAX_DELTA=0.05 PUPIL_TRACKER_MEDIAPIPE_MODEL=$(pwd)/models/face_landmarker.task make run-demo
```

The posture gate compares each target's captured samples against the first accepted sample for that target using the head-pose proxy features: roll, yaw, and pitch. Samples whose selected feature drift exceeds the threshold are rejected before calibration storage. Calibration start logs a scalar `calibration_config` event with the active path, target count, model, sample window, screen size, posture threshold, and posture feature indices so repeat-run analysis can confirm the exact test condition. Keep this gate experimental until a logged validation run shows better 4x3 grid accuracy without moving failures to other targets.

Any future video/frame capture feature must be explicit opt-in and documented separately.

## Known MVP Limitations

- Commodity webcam gaze tracking is coarse; expect screen-region/window-level utility, not pixel-perfect cursor replacement.
- Accuracy depends heavily on lighting, camera placement, face visibility, head movement, and calibration quality.
- The demo is macOS-first and developer-oriented; Windows/Linux packaging is out of scope for the MVP.
- Multi-monitor behavior is not fully specified.
- The MediaPipe backend uses the installed MediaPipe Tasks API; real inference requires an appropriate FaceLandmarker model asset path.
- Live GUI/hardware behavior still needs manual validation on each target Mac.
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
