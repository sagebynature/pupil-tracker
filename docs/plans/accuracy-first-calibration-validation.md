# Accuracy-First Calibration and Validation Implementation Plan

> **For Hermes:** Use test-driven-development skill to implement this plan task-by-task. Do not start implementation until the user approves this plan. Commit each task independently after RED/GREEN verification, `make check`, and `git diff --check`.

**Goal:** Replace fast frame-count calibration with quality-gated timed calibration, then add post-calibration validation and visual gaze verification so the user can judge whether the calibrated gaze is accurate enough to trust.

**Architecture:** Keep calibration timing, sample filtering, quality scoring, and validation metrics in pure/testable modules under `src/pupil_tracker` or non-Qt controller modules under `apps/desktop_demo`. Keep PySide6 widgets thin: they should render phase/progress/metrics and delegate decisions to injected state/session objects. Automated tests must use fake clocks, fake observations, and offscreen Qt; live webcam/MediaPipe checks remain manual.

**Tech Stack:** Python 3.11, uv, pytest, ruff, ty, PySide6, OpenCV, MediaPipe Tasks, NumPy, scikit-learn.

---

## Product Direction

The priority is calibration correctness and accuracy, not speed.

The app should not treat a quickly completed 9-dot flow as success. It should:

1. Give the user time to settle gaze on each target.
2. Capture many observations over a timed window.
3. Filter low-quality observations.
4. Retry targets with poor sample quality.
5. Fit only from accepted samples.
6. Validate calibration after fitting.
7. Show visible evidence: predicted gaze dot, target dot, error line, trail, and optional heatmap.

## Non-Goals

- Do not focus, raise, click, or activate external windows.
- Do not require live webcam/MediaPipe in automated tests.
- Do not tune final thresholds from theory alone; expose defaults and keep them easy to adjust after manual testing.
- Do not replace the model algorithm until validation metrics show model fit is the bottleneck.

## Acceptance Criteria

Accuracy-first calibration is complete when:

1. Start Calibration shows one target long enough to look at it deliberately.
2. The app has an explicit settle phase before capture.
3. The app captures over time, not merely the first few valid frames.
4. Low-confidence samples are rejected.
5. Each target has visible progress and accepted/rejected counts.
6. Poor-quality targets are retried instead of silently advancing.
7. Calibration fit reports sample count and training error metrics.

Validation/verification is complete when:

1. After calibration, the app can enter validation mode.
2. Validation uses known targets and predicted gaze samples from the fitted model.
3. The app computes mean, median, max, and per-target error in pixels.
4. The app gives a pass/warn/fail recommendation.
5. The app renders target dot, predicted gaze dot, and error line.
6. The app can show a live gaze trail and/or heatmap so the user can visually verify stare location.

---

# Phase 1: Quality-Gated Timed Calibration

## Task 1: Add a pure calibration timing state machine

**Objective:** Represent per-target phases without Qt timers so calibration behavior is deterministic and testable.

**Files:**
- Create: `src/pupil_tracker/calibration/timing.py`
- Modify: `src/pupil_tracker/calibration/__init__.py`
- Test: `tests/test_calibration_timing.py`

**Design:**

Add:

- `CalibrationPhase` enum:
  - `SETTLING`
  - `CAPTURING`
  - `REVIEWING`
  - `COMPLETE`
- `TimedCalibrationConfig` frozen dataclass:
  - `settle_seconds: float = 1.0`
  - `capture_seconds: float = 2.0`
  - `min_samples_per_target: int = 20`
  - `min_confidence: float = 0.60`
- `TimedTargetState` frozen dataclass:
  - `phase`
  - `target_started_at`
  - `capture_started_at`
  - `accepted_count`
  - `rejected_count`
  - `progress: float`

**Step 1: Write failing tests**

Tests should cover:

```python
def test_timed_state_starts_in_settling_phase() -> None: ...

def test_timed_state_moves_to_capturing_after_settle_duration() -> None: ...

def test_timed_state_reports_capture_progress() -> None: ...

def test_timed_config_rejects_invalid_thresholds() -> None: ...
```

**Step 2: Run RED**

`uv run pytest tests/test_calibration_timing.py -v`

Expected: FAIL because `pupil_tracker.calibration.timing` does not exist.

**Step 3: Implement minimal pure state machine**

Keep it independent of `CalibrationFlowState`. It should accept `now_seconds` as an argument; do not call `time.time()` inside the pure object.

**Step 4: Verify**

`uv run pytest tests/test_calibration_timing.py -v`

**Step 5: Full gate and commit**

```bash
make check
git diff --check
git add src/pupil_tracker/calibration/timing.py src/pupil_tracker/calibration/__init__.py tests/test_calibration_timing.py
git commit -m "feat: add timed calibration state"
```

---

## Task 2: Add calibration sample quality filtering

**Objective:** Decide whether an observation should be accepted for calibration based on validity, confidence, and feature-vector sanity.

**Files:**
- Create: `src/pupil_tracker/calibration/quality.py`
- Modify: `src/pupil_tracker/calibration/__init__.py`
- Test: `tests/test_calibration_quality.py`

**Design:**

Add:

- `CalibrationSampleDecision` frozen dataclass:
  - `accepted: bool`
  - `reason: str`
- `CalibrationQualityFilter`:
  - `min_confidence: float`
  - `expected_feature_count: int | None = None`
  - `decide(observation: RawObservation) -> CalibrationSampleDecision`

Initial acceptance rules:

1. Reject invalid observations.
2. Reject confidence below threshold.
3. Reject empty feature vectors.
4. If `expected_feature_count` is set, reject feature vectors with mismatched length.
5. Accept otherwise.

**Step 1: Write failing tests**

Tests should cover:

```python
def test_accepts_valid_high_confidence_observation() -> None: ...

def test_rejects_invalid_observation() -> None: ...

def test_rejects_low_confidence_observation() -> None: ...

def test_rejects_empty_feature_vector() -> None: ...

def test_rejects_feature_count_mismatch() -> None: ...
```

**Step 2: Run RED**

`uv run pytest tests/test_calibration_quality.py -v`

Expected: FAIL because module/API does not exist.

**Step 3: Implement minimal quality filter**

Do not add variance/outlier filtering yet; add that after timed capture gives us enough samples.

**Step 4: Verify and commit**

```bash
uv run pytest tests/test_calibration_quality.py -v
make check
git diff --check
git add src/pupil_tracker/calibration/quality.py src/pupil_tracker/calibration/__init__.py tests/test_calibration_quality.py
git commit -m "feat: filter calibration sample quality"
```

---

## Task 3: Add quality summary per calibration target

**Objective:** Summarize accepted/rejected samples and decide whether the target should advance or retry.

**Files:**
- Modify: `src/pupil_tracker/calibration/quality.py`
- Test: `tests/test_calibration_quality.py`

**Design:**

Add:

- `TargetQualitySummary` frozen dataclass:
  - `target_id: str`
  - `accepted_count: int`
  - `rejected_count: int`
  - `mean_confidence: float`
  - `meets_min_samples: bool`
  - `recommendation: Literal["advance", "retry"]`

Function:

```python
def summarize_target_quality(
    *,
    target_id: str,
    accepted_observations: Sequence[RawObservation],
    rejected_count: int,
    min_samples: int,
) -> TargetQualitySummary: ...
```

**Step 1: Write failing tests**

Tests:

- Enough accepted samples => `advance`.
- Too few accepted samples => `retry`.
- Mean confidence is computed from accepted samples only.
- Empty accepted samples reports `0.0` mean confidence.

**Step 2: Run RED**

`uv run pytest tests/test_calibration_quality.py -v`

**Step 3: Implement minimal summary**

**Step 4: Verify and commit**

```bash
uv run pytest tests/test_calibration_quality.py -v
make check
git diff --check
git add src/pupil_tracker/calibration/quality.py tests/test_calibration_quality.py
git commit -m "feat: summarize calibration target quality"
```

---

## Task 4: Wire timed quality capture into `CalibrationSession`

**Objective:** Make the desktop calibration controller collect samples only during capture phase and retry low-quality targets.

**Files:**
- Modify: `apps/desktop_demo/calibration_session.py`
- Test: `tests/test_calibration_session.py`

**Design:**

Extend `CalibrationSession` with injected dependencies:

- `timing_config: TimedCalibrationConfig | None = None`
- `clock: Callable[[], float] | None = None`
- `quality_filter: CalibrationQualityFilter | None = None`

Add session-visible fields:

- `phase: CalibrationPhase`
- `target_quality: TargetQualitySummary | None`
- `accepted_for_current_target: int`
- `rejected_for_current_target: int`
- `capture_progress: float`

Behavior:

1. `start()` resets flow and enters `SETTLING` for first target.
2. During `SETTLING`, `capture(observation)` ignores observations.
3. After settle duration, session enters `CAPTURING`.
4. During `CAPTURING`, quality filter decides whether to add observation to flow.
5. When capture duration completes:
   - if target quality passes, advance target.
   - if target quality fails, reset only that target's samples and retry same target.
6. Fit only after all targets pass.

**Important implementation note:**
Current `CalibrationFlowState.capture_observation()` advances immediately when sample count reaches `samples_per_target`. For timed quality capture, either:

- raise `samples_per_target` to `min_samples_per_target` and only call capture during timed capture, or
- add a method to add samples without auto-advancing and let `CalibrationSession` own advancement.

Recommendation: add minimal methods to `CalibrationFlowState` in Task 5 before finalizing this task if needed. Do not force timing logic into the widget.

**Step 1: Write failing tests**

Tests should use a fake clock:

```python
def test_settle_phase_ignores_valid_observations() -> None: ...

def test_capture_phase_accepts_high_quality_observations() -> None: ...

def test_low_quality_target_retries_instead_of_advancing() -> None: ...

def test_high_quality_target_advances_after_capture_duration() -> None: ...
```

**Step 2: Run RED**

`uv run pytest tests/test_calibration_session.py -v`

Expected: FAIL because session has no timed/quality fields.

**Step 3: Implement minimally**

Keep old default behavior available only if necessary for existing tests, but prefer updating existing tests to the new accuracy-first behavior where appropriate.

**Step 4: Verify and commit**

```bash
uv run pytest tests/test_calibration_session.py tests/test_calibration_flow.py -v
make check
git diff --check
git add apps/desktop_demo/calibration_session.py tests/test_calibration_session.py tests/test_calibration_flow.py
git commit -m "feat: quality gate live calibration capture"
```

---

## Task 5: Add calibration flow support for retry/current-target sample management

**Objective:** Give the session precise control over target retry without resetting the entire calibration flow.

**Files:**
- Modify: `apps/desktop_demo/ui/calibration_view.py`
- Test: `tests/test_calibration_flow.py`

**Design:**

Add methods to `CalibrationFlowState`:

```python
def clear_current_target_samples(self) -> None: ...

def advance_target(self) -> bool: ...

def add_current_target_sample(self, observation: RawObservation) -> bool: ...
```

`add_current_target_sample()` should store valid samples but not advance by itself. This separates collection from advancement.

**Step 1: Write failing tests**

Tests:

- clear current target samples does not clear previous target samples.
- advance target moves from `r0c0` to `r0c1`.
- add current target sample stores without advancing.

**Step 2: Run RED**

`uv run pytest tests/test_calibration_flow.py -v`

**Step 3: Implement with backwards compatibility**

Keep `capture_observation()` if existing tests still use it, but route new session code through the new explicit methods.

**Step 4: Verify and commit**

```bash
uv run pytest tests/test_calibration_flow.py tests/test_calibration_target_widget.py -v
make check
git diff --check
git add apps/desktop_demo/ui/calibration_view.py tests/test_calibration_flow.py
git commit -m "feat: control calibration target retries"
```

---

## Task 6: Render calibration phase and quality progress in the UI

**Objective:** Make calibration understandable: show settle/capture/retry status, target number, progress, and sample counts.

**Files:**
- Modify: `apps/desktop_demo/ui/calibration_view.py`
- Modify: `apps/desktop_demo/ui/main_window.py`
- Test: `tests/test_calibration_target_widget.py`
- Test: `tests/test_desktop_calibration_wiring.py`

**Design:**

Display:

- `Target 3/9: r0c2`
- `Settle: look at the dot`
- `Capturing: 42%`
- `Accepted: 14/20 | Rejected: 3`
- `Quality: retrying target` or `Quality: good`

**Step 1: Write failing tests**

Tests should assert labels update for:

- settling phase
- capture progress
- retry message
- completion metrics

**Step 2: Run RED**

`uv run pytest tests/test_calibration_target_widget.py tests/test_desktop_calibration_wiring.py -v`

**Step 3: Implement UI label updates**

Do not put timing decisions in UI. UI reads state from `CalibrationSession` or a small view model.

**Step 4: Verify and commit**

```bash
uv run pytest tests/test_calibration_target_widget.py tests/test_desktop_calibration_wiring.py -v
make check
git diff --check
git add apps/desktop_demo/ui/calibration_view.py apps/desktop_demo/ui/main_window.py tests/test_calibration_target_widget.py tests/test_desktop_calibration_wiring.py
git commit -m "feat: show calibration quality progress"
```

---

# Phase 2: Post-Calibration Validation Metrics

## Task 7: Add validation target pattern and metric model

**Objective:** Define validation targets and metric types independent of Qt.

**Files:**
- Create: `src/pupil_tracker/calibration/validation.py`
- Modify: `src/pupil_tracker/calibration/__init__.py`
- Test: `tests/test_calibration_validation.py`

**Design:**

Add:

- `ValidationTarget`
  - id, x, y
- `ValidationSample`
  - target, gaze_sample
- `ValidationMetrics`
  - sample_count
  - mean_error_px
  - median_error_px
  - max_error_px
  - per_target_error_px: Mapping[str, float]
  - recommendation: `excellent | good | usable | retry`

Function:

```python
def validation_pattern() -> tuple[ValidationTarget, ...]: ...

def compute_validation_metrics(
    samples: Sequence[ValidationSample],
    *,
    screen_width: float,
    screen_height: float,
) -> ValidationMetrics: ...
```

Initial validation pattern recommendation:

- Use 5 targets not identical to the 9 training targets:
  - center-ish and intermediate positions, e.g. `(0.25, 0.25)`, `(0.75, 0.25)`, `(0.50, 0.50)`, `(0.25, 0.75)`, `(0.75, 0.75)`.

**Step 1: Write failing tests**

Tests:

- pattern returns stable targets.
- exact gaze at target returns zero error.
- mean/median/max computed correctly.
- recommendation thresholds classify output.

**Step 2: Run RED**

`uv run pytest tests/test_calibration_validation.py -v`

**Step 3: Implement pure metrics**

Start thresholds:

- excellent: mean < 75 px
- good: mean < 125 px
- usable: mean < 200 px
- retry: otherwise

**Step 4: Verify and commit**

```bash
uv run pytest tests/test_calibration_validation.py -v
make check
git diff --check
git add src/pupil_tracker/calibration/validation.py src/pupil_tracker/calibration/__init__.py tests/test_calibration_validation.py
git commit -m "feat: compute calibration validation metrics"
```

---

## Task 8: Add validation session controller

**Objective:** Collect predicted gaze samples against known validation targets after calibration completes.

**Files:**
- Create: `apps/desktop_demo/validation_session.py`
- Test: `tests/test_validation_session.py`

**Design:**

Add `ValidationSession` with:

- targets
- current target
- state: idle / settling / capturing / complete
- settle/capture durations, same shape as calibration timing
- accepted gaze samples per target
- metrics after completion

Inputs:

- `GazeSample` from `GazeRuntime`
- fake clock for tests

Behavior:

- ignore samples during settle
- capture valid gaze samples during capture window
- compute metrics after all validation targets complete

**Step 1: Write failing tests**

Tests:

```python
def test_validation_ignores_samples_during_settle() -> None: ...

def test_validation_collects_valid_gaze_during_capture() -> None: ...

def test_validation_computes_metrics_after_all_targets() -> None: ...
```

**Step 2: Run RED**

`uv run pytest tests/test_validation_session.py -v`

**Step 3: Implement minimal controller**

Keep it non-Qt. It should not know about widgets.

**Step 4: Verify and commit**

```bash
uv run pytest tests/test_validation_session.py tests/test_calibration_validation.py -v
make check
git diff --check
git add apps/desktop_demo/validation_session.py tests/test_validation_session.py
git commit -m "feat: collect post-calibration validation samples"
```

---

## Task 9: Add validation mode to desktop demo state machine

**Objective:** Make validation an explicit mode after calibration, before trusting window focus experiments.

**Files:**
- Modify: `apps/desktop_demo/state.py`
- Test: `tests/test_desktop_state.py`

**Design:**

Add states or transitions:

- `CALIBRATING`
- `VALIDATING`
- `TRACKING`

Flow:

- calibration completed -> validation available or auto-start validation
- validation passed -> tracking
- validation failed -> retry calibration or remain validation failed

**Step 1: Write failing tests**

Tests:

- calibration completion enters validation state.
- validation pass enters tracking.
- validation fail can retry calibration.

**Step 2: Run RED**

`uv run pytest tests/test_desktop_state.py -v`

**Step 3: Implement state transitions**

**Step 4: Verify and commit**

```bash
uv run pytest tests/test_desktop_state.py -v
make check
git diff --check
git add apps/desktop_demo/state.py tests/test_desktop_state.py
git commit -m "feat: add validation demo state"
```

---

# Phase 3: Visual Verification Overlay

## Task 10: Add validation overlay state for target/prediction/error line

**Objective:** Represent validation visual evidence without Qt painting first.

**Files:**
- Modify: `apps/desktop_demo/ui/overlay.py`
- Test: `tests/test_overlay_state.py`

**Design:**

Extend overlay state to support:

- predicted gaze dot
- validation target dot
- error line endpoints
- trail of recent valid gaze samples
- confidence halo

If this makes `OverlayState` too broad, create a new pure `ValidationOverlayState` in `apps/desktop_demo/ui/overlay.py` or `apps/desktop_demo/ui/validation_overlay.py`.

**Step 1: Write failing tests**

Tests:

- target + prediction creates an error segment.
- invalid gaze hides prediction but keeps target visible.
- trail remains bounded.

**Step 2: Run RED**

`uv run pytest tests/test_overlay_state.py -v`

**Step 3: Implement pure state**

**Step 4: Verify and commit**

```bash
uv run pytest tests/test_overlay_state.py -v
make check
git diff --check
git add apps/desktop_demo/ui/overlay.py tests/test_overlay_state.py
git commit -m "feat: model validation overlay state"
```

---

## Task 11: Paint validation target, prediction, and error line

**Objective:** Make the overlay visually show where the user should look and where the model thinks they are looking.

**Files:**
- Modify: `apps/desktop_demo/ui/overlay.py`
- Test: `tests/test_desktop_overlay_wiring.py`

**Design:**

Paint:

- target dot: blue/white ring
- predicted gaze dot: green/yellow based on confidence
- error line: red/orange line between target and prediction
- optional short trail: translucent points

**Step 1: Write failing tests**

Use offscreen Qt render tests where practical:

- update overlay with validation target and sample.
- assert state stores both points.
- render smoke can check non-empty image or state only if pixel tests are flaky.

**Step 2: Run RED**

`QT_QPA_PLATFORM=offscreen uv run pytest tests/test_desktop_overlay_wiring.py -v`

**Step 3: Implement painting**

Keep rendering thin over pure state.

**Step 4: Verify and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/test_desktop_overlay_wiring.py -v
make check
git diff --check
git add apps/desktop_demo/ui/overlay.py tests/test_desktop_overlay_wiring.py
git commit -m "feat: render calibration validation overlay"
```

---

## Task 12: Wire validation session into `MainWindow`

**Objective:** After calibration completes, let the user validate and see metrics/overlay evidence.

**Files:**
- Modify: `apps/desktop_demo/ui/main_window.py`
- Modify: `apps/desktop_demo/ui/calibration_view.py`
- Test: `tests/test_desktop_calibration_wiring.py`
- Test: `tests/test_desktop_gaze_runtime.py`

**Design:**

Add UI controls/status:

- `Start Validation` button, or auto-start after calibration with clear status.
- validation target display in the calibration/validation panel.
- debug label includes metrics after completion.

Loop behavior:

1. Calibration session completes and model is fitted.
2. User starts validation.
3. Each live tracker status produces a calibrated gaze sample via `GazeRuntime`.
4. Validation session captures gaze sample for current validation target.
5. Overlay shows target, prediction, and error line.
6. After validation completes, display metrics and recommendation.

**Step 1: Write failing tests**

Tests:

- completed calibration enables validation state/control.
- live calibrated gaze updates validation session.
- validation completion updates debug/status with mean/max error.
- failed recommendation is surfaced as retry guidance.

**Step 2: Run RED**

`QT_QPA_PLATFORM=offscreen uv run pytest tests/test_desktop_calibration_wiring.py tests/test_desktop_gaze_runtime.py -v`

**Step 3: Implement minimal wiring**

Keep existing live gaze overlay behavior for tracking mode; validation overlay should take precedence during validation.

**Step 4: Verify and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/test_desktop_calibration_wiring.py tests/test_desktop_gaze_runtime.py tests/test_desktop_overlay_wiring.py -v
make check
git diff --check
git add apps/desktop_demo/ui/main_window.py apps/desktop_demo/ui/calibration_view.py tests/test_desktop_calibration_wiring.py tests/test_desktop_gaze_runtime.py tests/test_desktop_overlay_wiring.py
git commit -m "feat: wire post-calibration validation mode"
```

---

# Phase 4: Heatmap / Trail Verification

## Task 13: Add pure gaze heatmap accumulator

**Objective:** Accumulate recent gaze samples into a lightweight heatmap that can be rendered without storing raw frames.

**Files:**
- Create: `src/pupil_tracker/screen/heatmap.py`
- Modify: `src/pupil_tracker/screen/__init__.py`
- Test: `tests/test_heatmap.py`

**Design:**

Add:

- `HeatmapConfig`
  - screen_width
  - screen_height
  - cols: default 64
  - rows: default 36
  - decay: default 0.95
- `GazeHeatmap`
  - `add(sample: GazeSample) -> None`
  - `decay() -> None`
  - `normalized_cells() -> tuple[tuple[float, ...], ...]`

Rules:

- ignore invalid samples.
- clamp out-of-bounds samples.
- no image/frame payloads.

**Step 1: Write failing tests**

Tests:

- valid gaze increments expected cell.
- invalid gaze ignored.
- decay reduces intensity.
- normalized output max is <= 1.

**Step 2: Run RED**

`uv run pytest tests/test_heatmap.py -v`

**Step 3: Implement pure accumulator**

**Step 4: Verify and commit**

```bash
uv run pytest tests/test_heatmap.py -v
make check
git diff --check
git add src/pupil_tracker/screen/heatmap.py src/pupil_tracker/screen/__init__.py tests/test_heatmap.py
git commit -m "feat: accumulate gaze heatmap"
```

---

## Task 14: Render heatmap/trail overlay toggle

**Objective:** Let the user visually inspect where gaze samples cluster on screen.

**Files:**
- Modify: `apps/desktop_demo/ui/overlay.py`
- Modify: `apps/desktop_demo/ui/main_window.py`
- Test: `tests/test_desktop_overlay_wiring.py`

**Design:**

Add a toggle, default off or validation-only:

- `Show Heatmap`
- `Clear Heatmap`

Render:

- translucent heatmap cells
- recent gaze trail

**Step 1: Write failing tests**

Tests:

- valid gaze samples update heatmap when enabled.
- invalid samples do not update heatmap.
- clear heatmap resets cells.

**Step 2: Run RED**

`QT_QPA_PLATFORM=offscreen uv run pytest tests/test_desktop_overlay_wiring.py -v`

**Step 3: Implement minimal UI/state wiring**

**Step 4: Verify and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/test_desktop_overlay_wiring.py -v
make check
git diff --check
git add apps/desktop_demo/ui/overlay.py apps/desktop_demo/ui/main_window.py tests/test_desktop_overlay_wiring.py
git commit -m "feat: show gaze verification heatmap"
```

---

# Phase 5: Telemetry, Docs, and Manual Validation

## Task 15: Add calibration/validation telemetry events

**Objective:** Log scalar-only calibration and validation quality events for offline inspection.

**Files:**
- Modify: `src/pupil_tracker/telemetry/jsonl.py`
- Modify: `src/pupil_tracker/telemetry/__init__.py`
- Modify: `apps/desktop_demo/ui/main_window.py`
- Test: `tests/test_telemetry_privacy.py`
- Test: `tests/test_desktop_live_telemetry.py`

**Events:**

- `calibration_target_quality`
- `calibration_retry`
- `validation_sample`
- `validation_metrics`

Payloads must include only scalar/list/dict values. No frames, images, raw feature vectors, or video.

**Step 1: Write failing tests**

Tests:

- quality payload serializes to JSON.
- validation metrics payload serializes to JSON.
- payloads do not include image/frame/feature_vector.

**Step 2: Run RED**

`uv run pytest tests/test_telemetry_privacy.py tests/test_desktop_live_telemetry.py -v`

**Step 3: Implement payload helpers and live logging**

**Step 4: Verify and commit**

```bash
uv run pytest tests/test_telemetry_privacy.py tests/test_desktop_live_telemetry.py -v
make check
git diff --check
git add src/pupil_tracker/telemetry/jsonl.py src/pupil_tracker/telemetry/__init__.py apps/desktop_demo/ui/main_window.py tests/test_telemetry_privacy.py tests/test_desktop_live_telemetry.py
git commit -m "feat: log calibration validation metrics"
```

---

## Task 16: Update README and manual checklist

**Objective:** Document accuracy-first calibration and validation workflow for real hardware testing.

**Files:**
- Modify: `README.md`
- Modify: `docs/manual-test-checklist.md`

**Content:**

Add:

- how to download model
- how to start demo
- what settle/capture phases mean
- how to judge validation metrics
- what pass/warn/fail means
- how to use dot/trail/heatmap verification
- what to record if calibration feels wrong

**Verification:**

```bash
make check
git diff --check
git diff -- README.md docs/manual-test-checklist.md
```

**Commit:**

```bash
git add README.md docs/manual-test-checklist.md
git commit -m "docs: document calibration validation workflow"
```

---

# Manual Validation Gate

After all tasks are implemented and committed, run this manually on the target Mac:

```bash
make download-model
PUPIL_TRACKER_MEDIAPIPE_MODEL=$(pwd)/models/face_landmarker.task make run-demo
```

Checklist:

1. Start Camera.
2. Confirm preview scales to panel.
3. Start Calibration.
4. Confirm each target has visible settle/capture time.
5. Confirm low-confidence/lost-face samples do not advance target.
6. Complete calibration.
7. Start validation.
8. Look at each validation target.
9. Confirm predicted dot and target dot are visible.
10. Confirm error line is understandable.
11. Confirm validation metrics are displayed.
12. Enable heatmap/trail and stare at fixed points.
13. Confirm heatmap clusters where you stare.
14. Stop camera and close app.
15. Confirm no camera/tracker resources remain open.

# Execution Order Recommendation

Execute in this order:

1. Task 1
2. Task 2
3. Task 3
4. Task 5
5. Task 4
6. Task 6
7. Task 7
8. Task 8
9. Task 9
10. Task 10
11. Task 11
12. Task 12
13. Task 13
14. Task 14
15. Task 15
16. Task 16

Rationale: flow retry primitives in Task 5 may be needed by the session wiring in Task 4. If Task 4 reveals missing flow control, pause and do Task 5 first.

# First Execution Slice

When approved, start with Task 1 only:

`feat: add timed calibration state`

Do not implement validation, heatmaps, or UI changes until the pure timing state is tested and committed.
