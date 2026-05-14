# Vertical Accuracy Diagnostics Implementation Plan

> **For Hermes:** Use test-driven-development skill to implement this plan task-by-task. Keep each task small, verify RED/GREEN, run `make check`, inspect diffs, and commit before proceeding to the next task.

**Goal:** Improve weak vertical gaze tracking by first making the failure measurable, then adding vertical-sensitive eye features, then adding a denser vertical calibration option if metrics justify it.

**Architecture:** Do not tune the polynomial model blindly. Keep calibration/validation logic pure and testable, keep PySide6 as a thin adapter, and keep telemetry scalar-only. Use validation metrics to prove whether vertical error is bias, compression, or noise before changing tracker features.

**Tech Stack:** Python 3.11, uv, pytest, ruff, ty, PySide6, MediaPipe Tasks, scikit-learn ridge/polynomial calibration.

---

## Task 1: Add per-axis validation metrics

**Objective:** Compute X/Y absolute error, signed Y bias, and per-target Y bias so vertical failures are visible in tests, UI, and telemetry.

**Files:**
- Modify: `src/pupil_tracker/calibration/validation.py`
- Modify: `src/pupil_tracker/telemetry/jsonl.py`
- Modify: `apps/desktop_demo/ui/main_window.py`
- Test: `tests/test_calibration_validation.py`
- Test: `tests/test_telemetry_privacy.py` or `tests/test_jsonl_logger.py`
- Test: `tests/test_desktop_calibration_wiring.py`

**Step 1: Write failing tests**

Add validation tests that use known samples where X error and Y error differ. Assert:

- `mean_abs_x_error_px`
- `mean_abs_y_error_px`
- `mean_signed_y_error_px`
- `per_target_signed_y_error_px`

Add telemetry/UI tests asserting those fields are serialized and shown in validation completion debug text.

**Step 2: Verify RED**

Run:

```bash
uv run pytest tests/test_calibration_validation.py::test_validation_metrics_report_per_axis_error -v
```

Expected: FAIL because `ValidationMetrics` lacks the per-axis fields.

**Step 3: Implement minimally**

Compute deltas during `compute_validation_metrics()`:

- `dx = gaze_x - target_x`
- `dy = gaze_y - target_y`
- absolute means for X/Y
- signed mean for Y
- signed mean by target

Keep existing recommendation thresholds based on aggregate radial mean error for now.

**Step 4: Verify GREEN**

Run focused tests, then:

```bash
make check
git diff --check
```

**Step 5: Commit**

```bash
git add src/pupil_tracker/calibration/validation.py src/pupil_tracker/telemetry/jsonl.py apps/desktop_demo/ui/main_window.py tests/test_calibration_validation.py tests/test_telemetry_privacy.py tests/test_desktop_calibration_wiring.py
git commit -m "feat: report per-axis validation error"
```

---

## Task 2: Add a vertical diagnostic mode to manual validation docs

**Objective:** Give the live tester a repeatable protocol for determining whether vertical weakness is systematic bias or noisy jitter.

**Files:**
- Modify: `README.md`
- Modify: `docs/manual-test-checklist.md`

**Step 1: Update docs**

Document a short vertical diagnostic path:

1. Calibrate with stable head position.
2. Run validation.
3. Compare mean X error vs mean Y error and signed Y bias.
4. If signed Y bias is consistently positive/negative, recalibrate with better camera angle.
5. If mean Y error is high but signed bias is near zero, prioritize feature extraction improvements.

**Step 2: Verify docs/gates**

Run:

```bash
make check
git diff --check
```

**Step 3: Commit**

```bash
git add README.md docs/manual-test-checklist.md
git commit -m "docs: add vertical validation diagnostic workflow"
```

---

## Task 3: Add vertical-sensitive eye geometry features

**Objective:** Improve the feature vector so vertical gaze has stronger signal than whole-face-normalized iris center alone.

**Files:**
- Modify: `src/pupil_tracker/tracking/features.py`
- Modify: `src/pupil_tracker/tracking/mediapipe_backend.py`
- Test: `tests/test_tracking_features.py`
- Test: `tests/test_mediapipe_backend.py`

**Step 1: Write failing pure feature tests**

Add a new helper/API that includes:

- iris center relative to an eye bounding box or eye corner line,
- vertical eye aperture where available,
- left/right vertical agreement,
- optional face-pitch proxy.

Keep tests deterministic with synthetic `Point2D`/`Rect` inputs. Do not import MediaPipe in pure feature tests.

**Step 2: Verify RED**

Run:

```bash
uv run pytest tests/test_tracking_features.py -v
```

Expected: FAIL because the new feature helper/API is missing.

**Step 3: Implement minimally**

Add the feature helper in `features.py` and wire MediaPipe landmark extraction in `mediapipe_backend.py`. Preserve stable feature vector length and explicit errors for missing required landmarks.

**Step 4: Verify GREEN**

Run:

```bash
uv run pytest tests/test_tracking_features.py tests/test_mediapipe_backend.py -v
make check
git diff --check
```

**Step 5: Commit**

```bash
git add src/pupil_tracker/tracking/features.py src/pupil_tracker/tracking/mediapipe_backend.py tests/test_tracking_features.py tests/test_mediapipe_backend.py
git commit -m "feat: add vertical-sensitive gaze features"
```

---

## Task 4: Add a denser vertical calibration pattern option

**Objective:** Provide a deliberate 3x5 calibration mode for vertical accuracy experiments without replacing the default 9-point flow yet.

**Files:**
- Modify: `src/pupil_tracker/calibration/patterns.py`
- Modify: `apps/desktop_demo/ui/calibration_view.py`
- Modify: runtime config if needed under `apps/desktop_demo/config.py`
- Test: `tests/test_calibration_patterns.py`
- Test: `tests/test_calibration_flow.py`
- Test: desktop config/wiring tests as needed

**Step 1: Write failing tests**

Add tests for a 3-column, 5-row pattern with stable IDs and inset points. Add a config/wiring test only if exposing it through env/config in this slice.

**Step 2: Verify RED**

Run:

```bash
uv run pytest tests/test_calibration_patterns.py tests/test_calibration_flow.py -v
```

**Step 3: Implement minimally**

Add pattern generation/config plumbing without changing the default user path.

**Step 4: Verify GREEN**

Run:

```bash
make check
git diff --check
```

**Step 5: Commit**

```bash
git add src/pupil_tracker/calibration/patterns.py apps/desktop_demo/ui/calibration_view.py tests/test_calibration_patterns.py tests/test_calibration_flow.py
git commit -m "feat: add vertical calibration pattern option"
```

---

## Task 5: Manual live verification gate

**Objective:** Run the real demo and decide whether vertical features or denser calibration improved the failure mode.

**Files:**
- Modify only docs if recording manual results is needed.

**Steps:**

1. Run:

```bash
make download-model
PUPIL_TRACKER_MEDIAPIPE_MODEL=$(pwd)/models/face_landmarker.task make run-demo
```

2. Calibrate with stable head position.
3. Run validation and record mean X/Y error plus signed Y bias.
4. Test horizontal-only and vertical-only gaze sweeps.
5. Confirm fullscreen overlays remain click-through.
6. Decide whether to keep 3x5 as experimental, make it default, or add more features first.

**Commit:** Only commit docs if manual results are written down.
