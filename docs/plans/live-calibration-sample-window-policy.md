# Live Calibration Sample-Window Policy Implementation Plan

> **For Hermes:** Use test-driven-development skill to implement this plan task-by-task.

**Goal:** Move the replay-backed calibration sample-window finding into the live calibration path without changing the default behavior until a fresh manual run confirms it.

**Decision Context:** Replay evaluation showed `middle` and `late` per-target calibration samples slightly beat the latest live 4x3 grid accuracy (`40.9%` and `42.2%` vs `40.0%`). This is enough to test live, not enough to make unconditional default behavior.

**Architecture:** Add a small, typed sample-window policy seam to `CalibrationSession`. The default policy remains `all`. Runtime config can set `PUPIL_TRACKER_CALIBRATION_SAMPLE_WINDOW=late` for manual validation. The session still stores/logs all accepted samples; only the samples passed to model fitting are filtered.

---

## Task 1: Add Live Sample-Window Fit Policy

**Objective:** Fit `CalibrationSession.model` from a configured per-target sample window while preserving collection and telemetry behavior.

**Files:**
- Modify: `apps/desktop_demo/calibration_session.py`
- Modify: `tests/test_calibration_session.py`

**TDD:**
1. Add a test where each target collects six valid samples and `calibration_sample_window="late"` is configured.
2. Expected behavior:
   - flow still stores all samples,
   - model receives only the last third per target,
   - session completes successfully.
3. Verify RED:
   ```bash
   uv run pytest tests/test_calibration_session.py::test_session_fits_model_from_configured_late_sample_window -v
   ```
4. Implement:
   - `CalibrationSampleWindow = Literal["all", "early", "middle", "late"]`
   - `select_calibration_samples_by_window(samples, window=...)`
   - `CalibrationSession(..., calibration_sample_window="all")`
   - apply selection in `_fit_completed_flow()`.
5. Verify focused test and full calibration session tests.

## Task 2: Wire Runtime Config

**Objective:** Expose the policy through demo runtime config and pass it into the default main-window calibration session.

**Files:**
- Modify: `apps/desktop_demo/config.py`
- Modify: `apps/desktop_demo/app.py`
- Modify: `apps/desktop_demo/ui/main_window.py`
- Modify: `tests/test_desktop_config.py`

**TDD:**
1. Add config test for `PUPIL_TRACKER_CALIBRATION_SAMPLE_WINDOW=late`.
2. Add config rejection test for invalid values.
3. Extend `test_create_main_window_applies_config_to_camera_and_timer` to assert the created default calibration session uses the config value.
4. Verify RED:
   ```bash
   QT_QPA_PLATFORM=offscreen uv run pytest tests/test_desktop_config.py::test_demo_config_parses_calibration_sample_window -v
   ```
5. Implement smallest config/parser/wiring change.
6. Verify:
   ```bash
   QT_QPA_PLATFORM=offscreen uv run pytest tests/test_desktop_config.py -v
   uv run pytest tests/test_calibration_session.py -v
   make check
   git diff --check
   ```
7. Commit and push:
   ```bash
   git add apps/desktop_demo/calibration_session.py apps/desktop_demo/config.py apps/desktop_demo/app.py apps/desktop_demo/ui/main_window.py tests/test_calibration_session.py tests/test_desktop_config.py
   git commit -m "feat: configure live calibration sample window"
   git push origin HEAD
   ```

## Task 3: Manual Validation Docs

**Objective:** Document the exact command to run a replay-backed `late` policy validation.

**Files:**
- Modify: `README.md`
- Modify: `docs/manual-test-checklist.md`
- Modify: `docs/experiments/feature-diagnostics-comparison.md`

**Steps:**
1. Document manual command:
   ```bash
   PUPIL_TRACKER_CALIBRATION_SAMPLE_WINDOW=late PUPIL_TRACKER_MEDIAPIPE_MODEL=$(pwd)/models/face_landmarker.task make run-demo
   ```
2. Note default remains `all` until fresh manual evidence confirms the late policy.
3. Verify docs diff:
   ```bash
   git diff --check
   ```
4. Commit and push:
   ```bash
   git add README.md docs/manual-test-checklist.md docs/experiments/feature-diagnostics-comparison.md
   git commit -m "docs: prepare late-sample calibration validation"
   git push origin HEAD
   ```
