# Scalar Replay and Offline Model Evaluation Implementation Plan

> **For Hermes:** Use test-driven-development skill to implement this plan task-by-task.

**Goal:** Capture opt-in scalar calibration/validation replay samples and evaluate multiple calibration model strategies offline from one manual run.

**Architecture:** Keep live telemetry privacy-preserving by logging scalar feature arrays only after explicit Start Logging, with no frames, images, screenshots, or landmarks. Add a pure parser/evaluator tool that reconstructs calibration samples and validation observations from JSONL replay events, fits candidate models on the same captured calibration data, and compares validation metrics.

**Tech Stack:** Python 3.11, uv, pytest, ruff, ty, scikit-learn-backed existing calibration models, JSONL telemetry.

---

## Task 1: Add Scalar Replay Payload Serializers

**Objective:** Add telemetry payload helpers for replayable calibration and validation samples.

**Files:**
- Modify: `src/pupil_tracker/telemetry/jsonl.py`
- Modify: `src/pupil_tracker/telemetry/__init__.py`
- Test: `tests/test_telemetry_privacy.py`

**TDD:**
1. Add tests for `calibration_replay_sample_payload(target, observation)` and `validation_replay_sample_payload(target, observation)`.
2. Verify RED with:
   ```bash
   uv run pytest tests/test_telemetry_privacy.py::test_replay_sample_payloads_are_scalar_only -v
   ```
3. Implement payloads with keys:
   - `target_id`, `target_x`, `target_y`
   - `timestamp`, `confidence`, `valid`
   - `feature_count`, `features`
4. Preserve privacy assertions: no `image`, `frame`, raw landmarks, screenshots, or `feature_vector` key.
5. Run focused tests, then `make check`, `git diff --check`.
6. Commit:
   ```bash
   git add src/pupil_tracker/telemetry/jsonl.py src/pupil_tracker/telemetry/__init__.py tests/test_telemetry_privacy.py
   git commit -m "feat: serialize scalar replay telemetry"
   git push origin HEAD
   ```

## Task 2: Log Replay Samples from Desktop Demo

**Objective:** Emit replay sample events during calibration and validation while logging is active.

**Files:**
- Modify: `apps/desktop_demo/ui/main_window.py`
- Test: `tests/test_desktop_live_telemetry.py`

**TDD:**
1. Extend calibration telemetry test to expect `calibration_replay_sample` with scalar features.
2. Extend validation telemetry test to expect `validation_replay_sample` with scalar features from the observation used to produce the gaze sample.
3. Verify RED:
   ```bash
   QT_QPA_PLATFORM=offscreen uv run pytest tests/test_desktop_live_telemetry.py::test_live_calibration_logs_progress_without_frame_payload tests/test_desktop_live_telemetry.py::test_live_validation_logs_sample_and_metrics_without_frame_payload -v
   ```
4. Import payload helpers and log events only through existing `log_telemetry_event`, so no file is created before Start Logging.
5. Run focused tests, `make check`, `git diff --check`.
6. Commit and push:
   ```bash
   git add apps/desktop_demo/ui/main_window.py tests/test_desktop_live_telemetry.py
   git commit -m "feat: log scalar replay samples"
   git push origin HEAD
   ```

## Task 3: Build Offline Model Evaluator

**Objective:** Compare calibration models on one captured JSONL replay dataset.

**Files:**
- Create: `tools/evaluate_calibration_models.py`
- Create: `tests/test_evaluate_calibration_models.py`
- Modify: `README.md`
- Modify: `docs/manual-test-checklist.md`

**TDD:**
1. Add parser/evaluator tests using synthetic JSONL replay events.
2. Verify RED with:
   ```bash
   uv run pytest tests/test_evaluate_calibration_models.py -v
   ```
3. Implement a pure evaluator with public functions:
   - `load_replay_dataset(path)`
   - `evaluate_replay_models(dataset, screen_width, screen_height, grid_columns, grid_rows)`
   - `format_model_evaluation_report(results)`
4. Initial candidates:
   - `linear-alpha-0.1`
   - `linear-alpha-1.0`
   - `linear-alpha-10.0`
   - `poly2-alpha-0.1`
   - `poly2-alpha-1.0`
   - `poly2-alpha-10.0`
5. Use existing `compute_validation_metrics` to calculate mean error, mean X/Y, signed Y, and grid accuracy.
6. CLI:
   ```bash
   uv run python tools/evaluate_calibration_models.py metrics/demo.jsonl --screen-width 1512 --screen-height 982 --grid-columns 4 --grid-rows 3
   ```
7. Run focused tests, a CLI smoke test on synthetic data or current log if replay events exist, `make check`, `git diff --check`.
8. Commit and push:
   ```bash
   git add tools/evaluate_calibration_models.py tests/test_evaluate_calibration_models.py README.md docs/manual-test-checklist.md
   git commit -m "feat: evaluate calibration models offline"
   git push origin HEAD
   ```

## Task 4: Manual Replay Capture Gate

**Objective:** Run one fresh manual capture with replay telemetry and use the offline evaluator to choose the next modeling slice.

**Manual protocol:**
1. Launch demo:
   ```bash
   PUPIL_TRACKER_MEDIAPIPE_MODEL=$(pwd)/models/face_landmarker.task make run-demo
   ```
2. Start Camera.
3. Start Logging.
4. Start Vertical Calibration.
5. Start Validation.
6. Stop Logging.
7. Run:
   ```bash
   uv run python tools/evaluate_calibration_models.py metrics/demo.jsonl --screen-width 1512 --screen-height 982 --grid-columns 4 --grid-rows 3
   ```
8. Document best model and whether it improves 4x3 grid accuracy.

**Decision:** If an offline variant materially beats current live metrics, implement that model strategy next. If no variant improves grid accuracy, shift to calibration target weighting/geometry or real head-pose estimation rather than adding more face landmarks.
