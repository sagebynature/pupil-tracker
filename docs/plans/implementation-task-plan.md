# Pupil Tracker Implementation Task Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Execute the pupil-tracker MVP from the current uv/Make/ty/logging baseline into a macOS-first webcam gaze demo with a clean reusable library boundary.

**Architecture:** Keep all reusable behavior in `src/pupil_tracker` and make `apps/desktop_demo` a consumer. Implement pure/testable data, calibration, smoothing, and runtime layers first; then add camera, MediaPipe, Qt UI, overlay, and macOS window-candidate integration as thin vertical slices.

**Tech Stack:** Python 3.11, uv, Make, pytest, ruff, ty, standard-library logging, NumPy, scikit-learn, OpenCV, MediaPipe, PySide6/Qt, macOS CoreGraphics/PyObjC-compatible APIs.

---

## Current Baseline

Already present:

- `pyproject.toml` with uv dependency groups, pytest, ruff, and ty config.
- `uv.lock`.
- `Makefile` with `sync`, `test`, `typecheck`, `lint`, `format`, `check`, `run-demo`, and `clean`.
- `src/pupil_tracker/logging_config.py`.
- `apps/desktop_demo/main.py` placeholder using logging.
- Tests enforcing logging helper behavior and no direct `print(...)` in `src/` or `apps/`.

Baseline verification command:

```bash
make check
```

Expected now: ruff passes, ty passes, pytest passes.

## Execution Rules

For every code-producing task:

1. Write or update a focused failing test first.
2. Run the focused test and verify it fails for the expected reason.
3. Implement the smallest code that passes.
4. Run the focused test again and verify it passes.
5. Run `make check` before committing.
6. Commit each task independently with a Conventional Commit message.
7. Do not use `print(...)` or printf-style diagnostics in Python source; use `get_logger(...)`.
8. Do not add automatic window focusing in the MVP.
9. Do not write camera frames/video by default.

For manual hardware/UI tasks, include the best possible automated tests around pure logic, then run the manual verification listed in the task.

---

## Phase 1: Core Library Foundation

### Task 1: Add core data models

**Objective:** Define immutable typed models that all later layers can share.

**Files:**
- Create: `src/pupil_tracker/models.py`
- Modify: `src/pupil_tracker/__init__.py`
- Create: `tests/test_models.py`

**Step 1: Write failing tests**

Add tests for:

```python
from pupil_tracker.models import (
    CalibrationSample,
    CalibrationTarget,
    FrameMetadata,
    GazeSample,
    Point2D,
    RawObservation,
    Rect,
    WindowCandidate,
)


def test_rect_contains_point() -> None:
    rect = Rect(x=10, y=20, width=100, height=50)

    assert rect.contains(Point2D(10, 20))
    assert rect.contains(Point2D(110, 70))
    assert not rect.contains(Point2D(9, 20))


def test_invalid_observation_has_reason_and_zero_confidence() -> None:
    observation = RawObservation.invalid(timestamp=1.25, reason="no face")

    assert not observation.valid
    assert observation.confidence == 0.0
    assert observation.reason == "no face"


def test_gaze_sample_stores_region_and_validity() -> None:
    sample = GazeSample(
        timestamp=2.0,
        x=320,
        y=240,
        confidence=0.8,
        valid=True,
        region_id="middle_center",
    )

    assert sample.region_id == "middle_center"
    assert sample.valid
```

**Step 2: Verify RED**

Run:

```bash
uv run pytest tests/test_models.py -v
```

Expected: fails because `pupil_tracker.models` does not exist.

**Step 3: Implement models**

Implement frozen dataclasses:

- `Point2D`
- `Rect`
- `FrameMetadata`
- `RawObservation`
- `GazeSample`
- `CalibrationTarget`
- `CalibrationSample`
- `WindowCandidate`

Keep models independent of OpenCV, MediaPipe, Qt, and macOS APIs. `FrameMetadata` should hold shape/timestamp/camera id only; image arrays belong in camera code later.

**Step 4: Verify GREEN**

Run:

```bash
uv run pytest tests/test_models.py -v
make check
```

Expected: all pass.

**Step 5: Commit**

```bash
git add src/pupil_tracker/models.py src/pupil_tracker/__init__.py tests/test_models.py
git commit -m "feat: add core tracking data models"
```

### Task 2: Add calibration target patterns

**Objective:** Generate normalized calibration target layouts, starting with a 9-point grid.

**Files:**
- Create: `src/pupil_tracker/calibration/__init__.py`
- Create: `src/pupil_tracker/calibration/patterns.py`
- Create: `tests/test_calibration_patterns.py`

**Step 1: Write failing tests**

Test:

- `grid_pattern(rows=3, cols=3, margin=0.1)` returns 9 targets.
- All targets are normalized in `[0, 1]`.
- IDs are stable: `r0c0`, `r1c1`, `r2c2`.
- Center is `(0.5, 0.5)`.
- invalid rows/cols/margin raise `ValueError`.

**Step 2: Verify RED**

```bash
uv run pytest tests/test_calibration_patterns.py -v
```

Expected: fails because module/function is missing.

**Step 3: Implement**

Implement:

```python
def grid_pattern(rows: int, cols: int, margin: float = 0.1) -> list[CalibrationTarget]:
    ...
```

Use `CalibrationTarget` from `models.py`.

**Step 4: Verify GREEN**

```bash
uv run pytest tests/test_calibration_patterns.py -v
make check
```

**Step 5: Commit**

```bash
git add src/pupil_tracker/calibration tests/test_calibration_patterns.py
git commit -m "feat: add calibration target patterns"
```

### Task 3: Add screen geometry and 3x3 region mapping

**Objective:** Convert gaze points to named screen regions.

**Files:**
- Create: `src/pupil_tracker/screen/__init__.py`
- Create: `src/pupil_tracker/screen/regions.py`
- Create: `tests/test_regions.py`

**Step 1: Write failing tests**

Cover:

- `(0, 0)` maps to `top_left`.
- center maps to `middle_center`.
- bottom-right maps to `bottom_right`.
- out-of-bounds values are clamped before mapping.
- invalid screen dimensions raise `ValueError`.

**Step 2: Verify RED**

```bash
uv run pytest tests/test_regions.py -v
```

**Step 3: Implement**

Implement:

```python
def region_3x3(x: float, y: float, width: float, height: float) -> str:
    ...
```

Use row names `top`, `middle`, `bottom` and column names `left`, `center`, `right`.

**Step 4: Verify GREEN**

```bash
uv run pytest tests/test_regions.py -v
make check
```

**Step 5: Commit**

```bash
git add src/pupil_tracker/screen tests/test_regions.py
git commit -m "feat: map gaze points to screen regions"
```

### Task 4: Add gaze smoothing

**Objective:** Smooth noisy gaze samples without hiding invalid observations.

**Files:**
- Create: `src/pupil_tracker/smoothing/__init__.py`
- Create: `src/pupil_tracker/smoothing/filters.py`
- Create: `tests/test_smoothing.py`

**Step 1: Write failing tests**

Cover:

- First valid sample initializes the smoother.
- Second valid sample is blended using alpha.
- Invalid samples preserve invalid status and do not reset the last good point by default.
- `reset()` clears smoother state.

**Step 2: Verify RED**

```bash
uv run pytest tests/test_smoothing.py -v
```

**Step 3: Implement**

Implement:

```python
class EmaGazeSmoother:
    def __init__(self, alpha: float = 0.35) -> None: ...
    def update(self, sample: GazeSample) -> GazeSample: ...
    def reset(self) -> None: ...
```

Validate `0 < alpha <= 1`.

**Step 4: Verify GREEN**

```bash
uv run pytest tests/test_smoothing.py -v
make check
```

**Step 5: Commit**

```bash
git add src/pupil_tracker/smoothing tests/test_smoothing.py
git commit -m "feat: add gaze smoothing filter"
```

---

## Phase 2: Calibration Model and Runtime Contracts

### Task 5: Add calibration sample collector

**Objective:** Store calibration samples by target and enforce valid observation collection rules.

**Files:**
- Create: `src/pupil_tracker/calibration/samples.py`
- Create: `tests/test_calibration_samples.py`

**Step 1: Write failing tests**

Cover:

- collector starts empty.
- adding a valid sample increments count for target id.
- invalid observations are rejected or skipped according to explicit API.
- `samples_for(target_id)` returns stable collected samples.
- `all_samples()` returns samples in insertion order.

**Step 2: Verify RED**

```bash
uv run pytest tests/test_calibration_samples.py -v
```

**Step 3: Implement**

Implement `CalibrationSampleCollector` with explicit methods:

```python
class CalibrationSampleCollector:
    def add(self, sample: CalibrationSample) -> bool: ...
    def samples_for(self, target_id: str) -> tuple[CalibrationSample, ...]: ...
    def all_samples(self) -> tuple[CalibrationSample, ...]: ...
    def clear(self) -> None: ...
```

**Step 4: Verify GREEN**

```bash
uv run pytest tests/test_calibration_samples.py -v
make check
```

**Step 5: Commit**

```bash
git add src/pupil_tracker/calibration/samples.py tests/test_calibration_samples.py
git commit -m "feat: collect calibration samples"
```

### Task 6: Add polynomial ridge calibration model

**Objective:** Fit screen x/y predictions from synthetic calibration feature vectors.

**Files:**
- Create: `src/pupil_tracker/calibration/model.py`
- Create: `tests/test_calibration_model.py`

**Step 1: Write failing tests**

Use synthetic deterministic data. Cover:

- fitting enough samples returns a fit result with sample count and mean error.
- prediction maps a known synthetic observation near expected screen coordinates.
- predicting before fit raises a clear error.
- fitting with too few valid samples raises `ValueError`.

**Step 2: Verify RED**

```bash
uv run pytest tests/test_calibration_model.py -v
```

**Step 3: Implement**

Implement:

```python
@dataclass(frozen=True)
class CalibrationFitResult:
    sample_count: int
    mean_error_px: float
    max_error_px: float

class PolynomialRidgeCalibrationModel:
    def __init__(self, degree: int = 2, alpha: float = 1.0) -> None: ...
    def fit(self, samples: Sequence[CalibrationSample], screen_width: float, screen_height: float) -> CalibrationFitResult: ...
    def predict(self, observation: RawObservation, screen_width: float, screen_height: float) -> GazeSample: ...
```

Use scikit-learn `PolynomialFeatures` + `Ridge`. Keep scikit-learn imports inside this module.

**Step 4: Verify GREEN**

```bash
uv run pytest tests/test_calibration_model.py -v
make check
```

**Step 5: Commit**

```bash
git add src/pupil_tracker/calibration/model.py tests/test_calibration_model.py
git commit -m "feat: add polynomial gaze calibration model"
```

### Task 7: Define tracker backend interface

**Objective:** Establish a pluggable tracker contract before adding MediaPipe.

**Files:**
- Create: `src/pupil_tracker/tracking/__init__.py`
- Create: `src/pupil_tracker/tracking/backend.py`
- Create: `tests/test_tracking_backend.py`

**Step 1: Write failing tests**

Use a fake backend to prove the protocol shape:

- backend has `name`.
- `process(frame)` returns `RawObservation`.
- `close()` can be called safely.

**Step 2: Verify RED**

```bash
uv run pytest tests/test_tracking_backend.py -v
```

**Step 3: Implement**

Implement:

```python
@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    metadata: FrameMetadata

class TrackerBackend(Protocol):
    name: str
    def process(self, frame: Frame) -> RawObservation: ...
    def close(self) -> None: ...
```

If `Frame` fits better in `models.py`, add it there and keep image-array coupling contained.

**Step 4: Verify GREEN**

```bash
uv run pytest tests/test_tracking_backend.py -v
make check
```

**Step 5: Commit**

```bash
git add src/pupil_tracker/tracking tests/test_tracking_backend.py src/pupil_tracker/models.py
git commit -m "feat: define tracker backend interface"
```

### Task 8: Add runtime pipeline with fake components

**Objective:** Compose camera source, tracker backend, calibration model, smoother, and region mapper without real hardware.

**Files:**
- Create: `src/pupil_tracker/runtime/__init__.py`
- Create: `src/pupil_tracker/runtime/pipeline.py`
- Create: `tests/test_runtime_pipeline.py`

**Step 1: Write failing tests**

Use fake camera/backend/model. Cover:

- one `step()` returns observation and optional gaze sample.
- invalid observation returns no valid gaze sample.
- valid observation runs calibration, smoothing, and 3x3 region mapping.

**Step 2: Verify RED**

```bash
uv run pytest tests/test_runtime_pipeline.py -v
```

**Step 3: Implement**

Implement a small synchronous pull pipeline first. Avoid async until needed by Qt integration.

**Step 4: Verify GREEN**

```bash
uv run pytest tests/test_runtime_pipeline.py -v
make check
```

**Step 5: Commit**

```bash
git add src/pupil_tracker/runtime tests/test_runtime_pipeline.py
git commit -m "feat: add synchronous gaze runtime pipeline"
```

---

## Phase 3: Logging and Persistence

### Task 9: Add JSONL metrics logger

**Objective:** Persist non-video telemetry for calibration and runtime debugging.

**Files:**
- Create: `src/pupil_tracker/telemetry/__init__.py`
- Create: `src/pupil_tracker/telemetry/jsonl.py`
- Create: `tests/test_jsonl_logger.py`

**Step 1: Write failing tests**

Cover:

- writing two events produces two JSON lines.
- payloads are JSON serializable.
- event contains `event_type` and `timestamp`.
- NumPy arrays / image frames are rejected or not serializable by default.

**Step 2: Verify RED**

```bash
uv run pytest tests/test_jsonl_logger.py -v
```

**Step 3: Implement**

Implement:

```python
class JsonlLogger:
    def __init__(self, path: Path) -> None: ...
    def write_event(self, event_type: str, payload: Mapping[str, Any]) -> None: ...
    def close(self) -> None: ...
```

Use `get_logger("telemetry")` for logger diagnostics. Do not use `print`.

**Step 4: Verify GREEN**

```bash
uv run pytest tests/test_jsonl_logger.py -v
make check
```

**Step 5: Commit**

```bash
git add src/pupil_tracker/telemetry tests/test_jsonl_logger.py
git commit -m "feat: add JSONL telemetry logger"
```

---

## Phase 4: Camera and MediaPipe Integration

### Task 10: Add OpenCV camera source

**Objective:** Read webcam frames through OpenCV behind a small camera abstraction.

**Files:**
- Create: `src/pupil_tracker/camera/__init__.py`
- Create: `src/pupil_tracker/camera/opencv_camera.py`
- Create: `tests/test_opencv_camera.py`

**Step 1: Write failing tests for pure behavior**

Avoid requiring real camera in automated tests. Cover:

- constructor stores camera id and optional dimensions.
- read before open raises a clear camera error.
- close is idempotent.

Use monkeypatch/fakes for `cv2.VideoCapture` if testing open/read behavior.

**Step 2: Verify RED**

```bash
uv run pytest tests/test_opencv_camera.py -v
```

**Step 3: Implement**

Implement `OpenCVCamera` and `CameraError`. Timestamp frames with `time.monotonic()` and log camera lifecycle through `get_logger("camera")`.

**Step 4: Verify GREEN**

```bash
uv run pytest tests/test_opencv_camera.py -v
make check
```

**Step 5: Manual verification**

Run a temporary one-liner, not committed:

```bash
uv run python - <<'PY'
from pupil_tracker.camera.opencv_camera import OpenCVCamera
cam = OpenCVCamera(0)
cam.open()
frame = cam.read()
print(frame.metadata)
cam.close()
PY
```

This manual snippet may print in the shell; do not commit print usage into `src/` or `apps/`.

**Step 6: Commit**

```bash
git add src/pupil_tracker/camera tests/test_opencv_camera.py
git commit -m "feat: add OpenCV camera source"
```

### Task 11: Add MediaPipe feature extraction helpers

**Objective:** Convert landmark-like inputs into stable feature vectors without depending on live MediaPipe in tests.

**Files:**
- Create: `src/pupil_tracker/tracking/features.py`
- Create: `tests/test_tracking_features.py`

**Step 1: Write failing tests**

Use synthetic landmark points. Cover:

- feature vector length is stable.
- iris centers are normalized relative to face bounds.
- missing landmarks produce invalid/empty features through explicit errors or return values.

**Step 2: Verify RED**

```bash
uv run pytest tests/test_tracking_features.py -v
```

**Step 3: Implement**

Keep pure geometry in this module. Do not import MediaPipe here if possible.

**Step 4: Verify GREEN**

```bash
uv run pytest tests/test_tracking_features.py -v
make check
```

**Step 5: Commit**

```bash
git add src/pupil_tracker/tracking/features.py tests/test_tracking_features.py
git commit -m "feat: extract gaze calibration features"
```

### Task 12: Add MediaPipe tracker backend

**Objective:** Produce `RawObservation` from real webcam frames using MediaPipe.

**Files:**
- Create: `src/pupil_tracker/tracking/mediapipe_backend.py`
- Create: `tests/test_mediapipe_backend.py`

**Step 1: Write failing tests with mocks**

Do not require MediaPipe model execution in unit tests. Cover:

- backend name is `mediapipe`.
- no detected face returns invalid observation with reason.
- mocked landmarks produce valid observation with stable feature vector.
- close releases MediaPipe resources.

**Step 2: Verify RED**

```bash
uv run pytest tests/test_mediapipe_backend.py -v
```

**Step 3: Implement**

Use current MediaPipe Python API available from installed package. Keep API-specific objects isolated to this file.

**Step 4: Verify GREEN**

```bash
uv run pytest tests/test_mediapipe_backend.py -v
make check
```

**Step 5: Manual verification**

Run a non-committed script that opens the camera, processes frames for a few seconds, and logs valid/invalid observation counts. Use logging in any temporary file if committed; shell one-liners can print for local inspection.

**Step 6: Commit**

```bash
git add src/pupil_tracker/tracking/mediapipe_backend.py tests/test_mediapipe_backend.py
git commit -m "feat: add MediaPipe tracker backend"
```

---

## Phase 5: Desktop Demo Vertical Slice

### Task 13: Add Qt app shell and camera preview

**Objective:** Replace placeholder demo with a window that can start/stop camera preview.

**Files:**
- Create: `apps/desktop_demo/app.py`
- Create: `apps/desktop_demo/ui/__init__.py`
- Create: `apps/desktop_demo/ui/main_window.py`
- Modify: `apps/desktop_demo/main.py`
- Create: `tests/test_desktop_app_imports.py`

**Step 1: Write failing import/smoke tests**

Keep automated tests headless:

- `apps.desktop_demo.main` imports without side effects.
- main window class can be imported.
- camera worker is not started on import.

**Step 2: Verify RED**

```bash
uv run pytest tests/test_desktop_app_imports.py -v
```

**Step 3: Implement**

Create basic PySide6 app shell:

- Start Camera button.
- Stop Camera button.
- preview label/widget.
- debug labels placeholders.

Use logging for app lifecycle.

**Step 4: Verify GREEN**

```bash
uv run pytest tests/test_desktop_app_imports.py -v
make check
```

**Step 5: Manual verification**

```bash
make run-demo
```

Expected: window opens and remains responsive; camera preview can be started.

**Step 6: Commit**

```bash
git add apps/desktop_demo tests/test_desktop_app_imports.py
git commit -m "feat: add desktop camera preview shell"
```

### Task 14: Add calibration view flow

**Objective:** Present 9 calibration targets and collect observation samples.

**Files:**
- Create: `apps/desktop_demo/ui/calibration_view.py`
- Modify: `apps/desktop_demo/ui/main_window.py`
- Create: `tests/test_calibration_flow.py`

**Step 1: Write failing pure flow tests**

Extract testable flow state from Qt if useful. Cover:

- initial target is first 9-point target.
- advancing after N samples moves to the next target.
- completion exposes all collected samples.
- insufficient valid samples keeps/retries current target.

**Step 2: Verify RED**

```bash
uv run pytest tests/test_calibration_flow.py -v
```

**Step 3: Implement**

Implement calibration UI on top of pure flow state. Fit `PolynomialRidgeCalibrationModel` on completion.

**Step 4: Verify GREEN**

```bash
uv run pytest tests/test_calibration_flow.py -v
make check
```

**Step 5: Manual verification**

```bash
make run-demo
```

Expected: user can run through 9 visible targets and see calibration quality/error.

**Step 6: Commit**

```bash
git add apps/desktop_demo tests/test_calibration_flow.py
git commit -m "feat: add 9-point calibration flow"
```

### Task 15: Add transparent gaze overlay

**Objective:** Draw a confidence-aware dot and halo above the desktop.

**Files:**
- Create: `apps/desktop_demo/ui/overlay.py`
- Modify: `apps/desktop_demo/ui/main_window.py`
- Create: `tests/test_overlay_state.py`

**Step 1: Write failing pure overlay-state tests**

Test a pure helper if possible:

- high confidence produces smaller halo.
- low confidence produces larger halo.
- invalid sample hides or dims cursor.
- debug trail keeps bounded history.

**Step 2: Verify RED**

```bash
uv run pytest tests/test_overlay_state.py -v
```

**Step 3: Implement**

Use Qt flags for frameless, always-on-top, translucent, click-through overlay. Keep drawing state testable outside `paintEvent` where possible.

**Step 4: Verify GREEN**

```bash
uv run pytest tests/test_overlay_state.py -v
make check
```

**Step 5: Manual verification**

```bash
make run-demo
```

Expected: overlay appears above other windows, does not intercept mouse input, and updates with gaze samples after calibration.

**Step 6: Commit**

```bash
git add apps/desktop_demo/ui/overlay.py tests/test_overlay_state.py
git commit -m "feat: add confidence-aware gaze overlay"
```

---

## Phase 6: macOS Window Candidate and MVP Finish

### Task 16: Add macOS visible-window enumeration

**Objective:** List visible windows and score the likely gaze target without focusing anything.

**Files:**
- Create: `src/pupil_tracker/platform/__init__.py`
- Create: `src/pupil_tracker/platform/macos_windows.py`
- Create: `tests/test_macos_windows.py`

**Step 1: Write failing tests around pure scoring**

Do not require live macOS windows in unit tests. Cover:

- point inside one window returns that candidate.
- overlapping windows choose the higher score/frontmost order if available.
- off-window point returns `None`.
- hidden/minimized-like records are filtered by parser rules.

**Step 2: Verify RED**

```bash
uv run pytest tests/test_macos_windows.py -v
```

**Step 3: Implement**

Use CoreGraphics `CGWindowListCopyWindowInfo` through available bindings. Keep raw OS parsing separate from pure candidate scoring.

Do not activate, raise, focus, or click any window.

**Step 4: Verify GREEN**

```bash
uv run pytest tests/test_macos_windows.py -v
make check
```

**Step 5: Manual verification**

```bash
make run-demo
```

Expected: debug panel displays likely app/window title under current estimated gaze point.

**Step 6: Commit**

```bash
git add src/pupil_tracker/platform tests/test_macos_windows.py apps/desktop_demo
git commit -m "feat: identify likely macOS window target"
```

### Task 17: Add demo telemetry controls

**Objective:** Let the demo explicitly start/stop JSONL telemetry without recording frames/video.

**Files:**
- Modify: `apps/desktop_demo/ui/main_window.py`
- Modify: `apps/desktop_demo/ui/debug_view.py` if created
- Modify: `src/pupil_tracker/telemetry/jsonl.py` if needed
- Create: `tests/test_telemetry_privacy.py`

**Step 1: Write failing tests**

Cover:

- telemetry event serializers omit frame image arrays.
- calibration/gaze/window events serialize as JSON.
- stop/close flushes file.

**Step 2: Verify RED**

```bash
uv run pytest tests/test_telemetry_privacy.py -v
```

**Step 3: Implement**

Add Start Logging / Stop Logging controls. Default path should be under ignored local runtime output, e.g. `metrics/`.

**Step 4: Verify GREEN**

```bash
uv run pytest tests/test_telemetry_privacy.py -v
make check
```

**Step 5: Manual verification**

Run demo, start logging, calibrate briefly, stop logging, inspect JSONL manually. Confirm no image/frame data is present.

**Step 6: Commit**

```bash
git add apps/desktop_demo src/pupil_tracker/telemetry tests/test_telemetry_privacy.py
git commit -m "feat: add demo telemetry controls"
```

### Task 18: Final MVP documentation and verification

**Objective:** Make the repo runnable by a future developer and document known limits.

**Files:**
- Modify: `README.md`
- Modify: `docs/requirements.md` if any decisions changed
- Modify: `docs/plans/mvp.md` if implementation diverged
- Create: `docs/manual-test-checklist.md`

**Step 1: Update docs**

Document:

- `make sync`
- `make check`
- `make run-demo`
- macOS camera permission
- optional Accessibility permission is not needed until focus feature
- privacy/logging behavior
- known webcam accuracy limitations
- manual MVP checklist

**Step 2: Verify docs and checks**

Run:

```bash
make check
git diff --check
make run-demo
```

Expected:

- automated checks pass
- demo launches
- manual checklist can be followed

**Step 3: Commit**

```bash
git add README.md docs
git commit -m "docs: document MVP demo workflow"
```

---

## Suggested Execution Order Summary

1. `feat: add core tracking data models`
2. `feat: add calibration target patterns`
3. `feat: map gaze points to screen regions`
4. `feat: add gaze smoothing filter`
5. `feat: collect calibration samples`
6. `feat: add polynomial gaze calibration model`
7. `feat: define tracker backend interface`
8. `feat: add synchronous gaze runtime pipeline`
9. `feat: add JSONL telemetry logger`
10. `feat: add OpenCV camera source`
11. `feat: extract gaze calibration features`
12. `feat: add MediaPipe tracker backend`
13. `feat: add desktop camera preview shell`
14. `feat: add 9-point calibration flow`
15. `feat: add confidence-aware gaze overlay`
16. `feat: identify likely macOS window target`
17. `feat: add demo telemetry controls`
18. `docs: document MVP demo workflow`

## Overall MVP Verification Gate

Before declaring MVP complete:

```bash
make sync
make check
git diff --check
make run-demo
```

Manual checklist:

- Camera preview starts on macOS.
- MediaPipe detects face/eyes under normal lighting.
- 9-point calibration completes.
- Gaze dot appears with confidence halo.
- 3x3 region updates plausibly.
- Likely macOS window candidate appears in debug mode.
- Telemetry JSONL is written only after explicit start.
- Telemetry contains no image/video frame payloads.
- No automatic focus/raise/click behavior occurs.
