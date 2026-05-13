# Pupil Tracker MVP Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build a macOS-first Python library and desktop demo that uses a webcam to calibrate gaze, estimate approximate screen position, show a confidence-aware cursor overlay, classify the current 3x3 screen region, and identify a likely visible macOS window target in debug mode.

**Architecture:** The project is an importable Python package plus a PySide6 desktop demo app. Core tracking, calibration, smoothing, screen mapping, and platform logic live in `src/pupil_tracker`; the desktop demo in `apps/desktop_demo` consumes the library. The first tracker backend uses MediaPipe face/eye/iris landmarks, but the backend interface remains pluggable for future IR/near-eye trackers.

**Tech Stack:** Python, uv, Make, PySide6/Qt, OpenCV, MediaPipe, NumPy, scikit-learn or NumPy-based ridge regression, pytest, ruff, ty, macOS CoreGraphics/AppKit/Accessibility-facing adapters where needed.

---

## Milestone Acceptance Criteria

The MVP is complete when:

- A user can launch the demo app on macOS.
- The app can open a webcam and show a camera preview.
- The app can run a 9-point calibration flow.
- The app can estimate a calibrated gaze point with confidence.
- A transparent overlay shows an approximate gaze dot with confidence halo.
- The app classifies the current gaze point into a 3x3 screen region.
- The debug panel shows current FPS/latency, confidence, region, and likely visible window candidate.
- The app writes metrics/calibration logs without recording video frames by default.
- Pure library behavior is covered by unit tests and synthetic calibration fixtures.

## Repo Shape

Create this structure:

```text
pupil-tracker/
  pyproject.toml
  README.md
  src/pupil_tracker/
    __init__.py
    models.py
    camera/
      __init__.py
      opencv_camera.py
    tracking/
      __init__.py
      backend.py
      mediapipe_backend.py
      features.py
    calibration/
      __init__.py
      patterns.py
      model.py
      samples.py
    smoothing/
      __init__.py
      filters.py
    screen/
      __init__.py
      geometry.py
      regions.py
    platform/
      __init__.py
      macos_windows.py
    runtime/
      __init__.py
      stream.py
    logging/
      __init__.py
      jsonl.py
  apps/desktop_demo/
    __init__.py
    main.py
    app.py
    ui/
      __init__.py
      main_window.py
      calibration_view.py
      debug_view.py
      overlay.py
  tests/
    test_models.py
    test_calibration_patterns.py
    test_calibration_model.py
    test_regions.py
    test_smoothing.py
    test_runtime_mock.py
```

## Implementation Tasks

### Task 1: Initialize Python package metadata

**Objective:** Create the project packaging skeleton and development dependencies.

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/pupil_tracker/__init__.py`
- Create: package `__init__.py` files under planned subdirectories

**Step 1: Create `pyproject.toml`**

Use dependencies appropriate for the MVP:

```toml
[project]
name = "pupil-tracker"
version = "0.1.0"
description = "Webcam-first pupil and gaze tracking library with a macOS desktop demo."
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [{ name = "Sage" }]
dependencies = [
  "numpy>=1.26",
  "opencv-python>=4.9",
  "mediapipe>=0.10",
  "PySide6>=6.6",
  "scikit-learn>=1.4",
]

[dependency-groups]
dev = [
  "pytest>=8.0",
  "pytest-cov>=4.1",
  "ruff>=0.8",
  "ty>=0.0.1a8",
]

[build-system]
requires = ["hatchling>=1.26"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
line-length = 100
src = ["src", "apps", "tests"]

[tool.ty.environment]
python-version = "3.11"
python-platform = "macos"
root = ["./src", "./apps", "./tests"]
```

**Step 2: Create minimal README**

Include:
- project goal
- MVP scope
- macOS-first note
- no video recording by default
- development install command

**Step 3: Verify packaging metadata**

Run:

```bash
uv sync --dev
make check
```

Expected:
- uv sync succeeds and writes/updates `uv.lock`
- lint, ty typechecking, and pytest pass

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock Makefile README.md src apps tests
git commit -m "chore: initialize uv project tooling"
```

### Task 2: Define core data models

**Objective:** Add typed dataclasses for frames, observations, gaze samples, calibration samples, regions, and window candidates.

**Files:**
- Create: `src/pupil_tracker/models.py`
- Create: `tests/test_models.py`

**Step 1: Write failing tests**

Test:
- `Frame` stores image shape metadata and timestamp.
- `RawObservation` can be invalid with a reason.
- `GazeSample` stores screen x/y, confidence, region id, and validity.
- `WindowCandidate` stores title, app name, bounds, and score.

**Step 2: Implement dataclasses**

Recommended classes:

```python
@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    timestamp: float
    camera_id: int | str

@dataclass(frozen=True)
class RawObservation:
    timestamp: float
    valid: bool
    confidence: float
    face_bounds: Rect | None = None
    left_iris: Point2D | None = None
    right_iris: Point2D | None = None
    feature_vector: tuple[float, ...] = ()
    reason: str | None = None

@dataclass(frozen=True)
class GazeSample:
    timestamp: float
    x: float
    y: float
    confidence: float
    valid: bool
    region_id: str | None = None

@dataclass(frozen=True)
class CalibrationTarget:
    id: str
    x: float
    y: float

@dataclass(frozen=True)
class CalibrationSample:
    target: CalibrationTarget
    observation: RawObservation

@dataclass(frozen=True)
class WindowCandidate:
    app_name: str
    title: str
    bounds: Rect
    score: float
```

**Step 3: Run tests**

```bash
pytest tests/test_models.py -v
```

Expected: pass.

**Step 4: Commit**

```bash
git add src/pupil_tracker/models.py tests/test_models.py
git commit -m "feat: define core tracking data models"
```

### Task 3: Add calibration target patterns

**Objective:** Generate configurable calibration target layouts, starting with 9-point grid.

**Files:**
- Create: `src/pupil_tracker/calibration/patterns.py`
- Create: `tests/test_calibration_patterns.py`

**Step 1: Write failing tests**

Test:
- `grid_pattern(rows=3, cols=3, margin=0.1)` returns 9 targets.
- coordinates are normalized in `[0, 1]`.
- target ids are stable, e.g. `r0c0`, `r1c1`, `r2c2`.
- center target is approximately `(0.5, 0.5)`.

**Step 2: Implement pattern generation**

Implement:

```python
def grid_pattern(rows: int, cols: int, margin: float = 0.1) -> list[CalibrationTarget]:
    ...
```

Rules:
- `rows` and `cols` must be positive.
- `margin` must satisfy `0 <= margin < 0.5`.
- interpolate x/y between `margin` and `1 - margin`.

**Step 3: Run tests**

```bash
pytest tests/test_calibration_patterns.py -v
```

Expected: pass.

**Step 4: Commit**

```bash
git add src/pupil_tracker/calibration/patterns.py tests/test_calibration_patterns.py
git commit -m "feat: add calibration target patterns"
```

### Task 4: Add screen geometry and 3x3 region mapping

**Objective:** Map normalized or absolute gaze points to screen regions.

**Files:**
- Create: `src/pupil_tracker/screen/geometry.py`
- Create: `src/pupil_tracker/screen/regions.py`
- Create: `tests/test_regions.py`

**Step 1: Write failing tests**

Test:
- a point near top-left maps to `top_left`
- center maps to `center`
- bottom-right maps to `bottom_right`
- out-of-bounds coordinates clamp or return invalid according to chosen behavior

**Step 2: Implement region mapping**

Recommended API:

```python
def region_3x3(x: float, y: float, width: float, height: float) -> str:
    ...
```

Use row names `top`, `middle`, `bottom` and column names `left`, `center`, `right`.

**Step 3: Run tests**

```bash
pytest tests/test_regions.py -v
```

Expected: pass.

**Step 4: Commit**

```bash
git add src/pupil_tracker/screen tests/test_regions.py
git commit -m "feat: map gaze points to screen regions"
```

### Task 5: Add smoothing filters

**Objective:** Smooth gaze points and confidence for a less jumpy cursor.

**Files:**
- Create: `src/pupil_tracker/smoothing/filters.py`
- Create: `tests/test_smoothing.py`

**Step 1: Write failing tests**

Test:
- exponential moving average initializes from first valid sample.
- subsequent samples move toward the new point according to alpha.
- invalid samples do not produce wild jumps.

**Step 2: Implement EMA smoother**

Recommended API:

```python
class EmaGazeSmoother:
    def __init__(self, alpha: float = 0.35) -> None: ...
    def update(self, sample: GazeSample) -> GazeSample: ...
    def reset(self) -> None: ...
```

**Step 3: Run tests**

```bash
pytest tests/test_smoothing.py -v
```

Expected: pass.

**Step 4: Commit**

```bash
git add src/pupil_tracker/smoothing/filters.py tests/test_smoothing.py
git commit -m "feat: add gaze smoothing filter"
```

### Task 6: Add calibration sample collection and regression model

**Objective:** Fit a continuous screen-coordinate model from calibration samples.

**Files:**
- Create: `src/pupil_tracker/calibration/samples.py`
- Create: `src/pupil_tracker/calibration/model.py`
- Create: `tests/test_calibration_model.py`

**Step 1: Write failing synthetic fixture tests**

Create synthetic feature vectors with known mapping to screen coordinates.

Test:
- fitting with enough samples produces low error.
- prediction returns a `GazeSample` with valid coordinates.
- invalid observations are ignored or rejected.
- model reports calibration quality metadata.

**Step 2: Implement sample collection**

Implement a collector that stores `CalibrationSample` instances and groups them by target id.

**Step 3: Implement polynomial/ridge model**

Recommended API:

```python
class CalibrationModel:
    def fit(self, samples: Sequence[CalibrationSample]) -> CalibrationFitResult: ...
    def predict(self, observation: RawObservation, screen_width: float, screen_height: float) -> GazeSample: ...
```

Use `sklearn.pipeline.make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=1.0))` or an equivalent explicit implementation.

**Step 4: Run tests**

```bash
pytest tests/test_calibration_model.py -v
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/pupil_tracker/calibration tests/test_calibration_model.py
git commit -m "feat: add gaze calibration regression model"
```

### Task 7: Define tracker backend interface and mock runtime stream

**Objective:** Provide a testable backend contract and runtime stream independent of MediaPipe.

**Files:**
- Create: `src/pupil_tracker/tracking/backend.py`
- Create: `src/pupil_tracker/runtime/stream.py`
- Create: `tests/test_runtime_mock.py`

**Step 1: Write failing tests with a fake backend**

Test:
- fake backend receives frames and emits observations.
- runtime stream can process observations into gaze samples when a calibration model is available.
- runtime handles invalid observations gracefully.

**Step 2: Implement backend protocol**

Recommended API:

```python
class TrackerBackend(Protocol):
    name: str
    def process(self, frame: Frame) -> RawObservation: ...
    def close(self) -> None: ...
```

**Step 3: Implement runtime stream skeleton**

Keep it simple for MVP:
- pull frame from camera source
- process frame through tracker backend
- optionally calibrate to `GazeSample`
- smooth and classify region

**Step 4: Run tests**

```bash
pytest tests/test_runtime_mock.py -v
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/pupil_tracker/tracking/backend.py src/pupil_tracker/runtime/stream.py tests/test_runtime_mock.py
git commit -m "feat: add tracker backend and runtime stream interfaces"
```

### Task 8: Implement OpenCV camera source

**Objective:** Capture webcam frames through OpenCV.

**Files:**
- Create: `src/pupil_tracker/camera/opencv_camera.py`

**Step 1: Implement camera source**

Recommended API:

```python
class OpenCVCamera:
    def __init__(self, camera_id: int = 0, width: int | None = None, height: int | None = None) -> None: ...
    def open(self) -> None: ...
    def read(self) -> Frame: ...
    def close(self) -> None: ...
```

**Step 2: Add defensive behavior**

- Raise a clear exception if camera cannot open.
- Timestamp frames with `time.monotonic()`.
- Convert or document BGR/RGB expectations.

**Step 3: Manual verification**

Run a short script that opens camera 0, reads one frame, prints shape, and closes camera.

Expected:
- macOS prompts for camera permission if needed.
- frame shape is printed.

**Step 4: Commit**

```bash
git add src/pupil_tracker/camera/opencv_camera.py
git commit -m "feat: add OpenCV camera source"
```

### Task 9: Implement MediaPipe tracker backend

**Objective:** Convert webcam frames into raw eye/iris observations and feature vectors.

**Files:**
- Create: `src/pupil_tracker/tracking/mediapipe_backend.py`
- Create: `src/pupil_tracker/tracking/features.py`

**Step 1: Implement feature extraction helpers**

Extract stable features such as:
- normalized left iris center within face/eye bounds
- normalized right iris center within face/eye bounds
- eye aspect or blink-ish measurements
- face box center/size
- selected landmark deltas useful for calibration

**Step 2: Implement `MediaPipeTrackerBackend`**

Use MediaPipe face landmark APIs to produce `RawObservation`.

Requirements:
- return invalid observation when no face is detected
- include confidence where available or derived
- include feature vector suitable for calibration
- keep MediaPipe-specific details behind this backend

**Step 3: Manual verification**

Run a short script:
- open camera
- process frames for 10 seconds
- print observation validity/confidence/features length

Expected:
- observations become valid when face is visible
- feature vector length is stable

**Step 4: Commit**

```bash
git add src/pupil_tracker/tracking/mediapipe_backend.py src/pupil_tracker/tracking/features.py
git commit -m "feat: add MediaPipe tracker backend"
```

### Task 10: Add metrics and calibration JSONL logging

**Objective:** Persist non-video telemetry and calibration samples for tuning.

**Files:**
- Create: `src/pupil_tracker/logging/jsonl.py`
- Add tests if serialization stays pure/simple

**Step 1: Implement JSONL writer**

Recommended API:

```python
class JsonlLogger:
    def __init__(self, path: Path) -> None: ...
    def write_event(self, event_type: str, payload: Mapping[str, Any]) -> None: ...
    def close(self) -> None: ...
```

**Step 2: Ensure no video frames are serialized**

Do not serialize `Frame.image` arrays.
Only write summaries, metrics, observations, gaze samples, and calibration metadata.

**Step 3: Manual verification**

Write three events and verify each line is valid JSON.

**Step 4: Commit**

```bash
git add src/pupil_tracker/logging/jsonl.py
git commit -m "feat: add JSONL metrics logging"
```

### Task 11: Build basic desktop app shell

**Objective:** Launch a PySide6 app with camera preview and debug panel placeholder.

**Files:**
- Create: `apps/desktop_demo/main.py`
- Create: `apps/desktop_demo/app.py`
- Create: `apps/desktop_demo/ui/main_window.py`
- Create: `apps/desktop_demo/ui/debug_view.py`

**Step 1: Implement app entrypoint**

`main.py` should create a `QApplication`, instantiate `MainWindow`, and run the event loop.

**Step 2: Implement main window**

Include:
- camera preview area
- buttons: Start Camera, Start Calibration, Toggle Overlay, Toggle Debug
- debug labels for FPS, confidence, region, window candidate

**Step 3: Wire camera preview**

Use a `QTimer` or worker thread to pull frames and display them.
Keep blocking camera work out of the UI thread if preview stutters.

**Step 4: Manual verification**

Run:

```bash
python apps/desktop_demo/main.py
```

Expected:
- app opens
- camera preview can start
- window remains responsive

**Step 5: Commit**

```bash
git add apps/desktop_demo
git commit -m "feat: add desktop demo app shell"
```

### Task 12: Add calibration UI flow

**Objective:** Present 9-point targets and collect samples per target.

**Files:**
- Create: `apps/desktop_demo/ui/calibration_view.py`
- Modify: `apps/desktop_demo/ui/main_window.py`
- Modify: runtime/app wiring as needed

**Step 1: Implement full-screen or large calibration view**

Show one target at a time using `grid_pattern(3, 3)`.

**Step 2: Collect samples**

For each target:
- wait short settling interval
- collect N valid observations
- display progress
- skip/retry if insufficient confidence

**Step 3: Fit calibration model**

After all targets:
- fit calibration model
- show calibration error/quality
- allow retry if quality is poor

**Step 4: Manual verification**

Run calibration and verify model can produce gaze samples after completion.

**Step 5: Commit**

```bash
git add apps/desktop_demo src/pupil_tracker/calibration
git commit -m "feat: add 9-point calibration flow"
```

### Task 13: Add transparent overlay cursor

**Objective:** Show gaze dot and confidence halo on top of the screen.

**Files:**
- Create: `apps/desktop_demo/ui/overlay.py`
- Modify: app wiring as needed

**Step 1: Implement overlay window**

Use PySide6/Qt window flags:
- frameless
- always on top
- translucent background
- mouse-transparent / click-through where possible

**Step 2: Draw cursor**

Draw:
- gaze dot
- confidence halo
- optional trail in debug mode

**Step 3: Wire runtime gaze samples**

Update overlay from smoothed `GazeSample` values.

**Step 4: Manual verification**

Expected:
- overlay appears above other windows
- overlay does not intercept mouse input
- dot follows estimated gaze after calibration

**Step 5: Commit**

```bash
git add apps/desktop_demo/ui/overlay.py
git commit -m "feat: add confidence-aware gaze overlay"
```

### Task 14: Add macOS visible-window enumeration

**Objective:** Identify candidate visible windows under or near the estimated gaze point.

**Files:**
- Create: `src/pupil_tracker/platform/macos_windows.py`
- Modify: `apps/desktop_demo/ui/debug_view.py`
- Modify: app wiring as needed

**Step 1: Implement visible window list**

Use macOS APIs available from Python. Likely options:
- CoreGraphics `CGWindowListCopyWindowInfo`
- PyObjC bindings if needed

Collect:
- app/process name
- window title
- bounds
- layer/visibility flags

**Step 2: Implement candidate scoring**

Given a gaze point:
- find windows whose bounds contain the point
- prefer frontmost / visible / larger score as appropriate
- return `WindowCandidate | None`

**Step 3: Add debug display**

Show current candidate app/title/score in the debug panel.

**Step 4: Manual verification**

Open a few app windows, move gaze/cursor estimate over them, and verify debug candidate changes.

Do not focus or raise windows.

**Step 5: Commit**

```bash
git add src/pupil_tracker/platform/macos_windows.py apps/desktop_demo
git commit -m "feat: show likely macOS window target"
```

### Task 15: Add end-to-end logging controls

**Objective:** Let the demo write metrics/calibration logs explicitly.

**Files:**
- Modify: `apps/desktop_demo/ui/debug_view.py`
- Modify: `apps/desktop_demo/ui/main_window.py`
- Modify: `src/pupil_tracker/logging/jsonl.py` if needed

**Step 1: Add logging controls**

Controls:
- Start logging
- Stop logging
- Show log path

**Step 2: Write runtime events**

Log:
- calibration start/end/quality
- observation summaries
- gaze samples
- region candidates
- window candidates
- FPS/latency metrics

**Step 3: Verify privacy behavior**

Confirm no image arrays or video frames are written.

**Step 4: Commit**

```bash
git add apps/desktop_demo src/pupil_tracker/logging
git commit -m "feat: add demo metrics logging controls"
```

### Task 16: Final MVP verification pass

**Objective:** Confirm tests pass and the manual demo flow works.

**Files:**
- Modify docs/README if needed based on actual run commands

**Step 1: Run automated tests**

```bash
pytest -v
```

Expected: all tests pass.

**Step 2: Run lint/type checks if configured**

```bash
ruff check src apps tests
ty check src apps tests
```

Expected: pass or document intentional skips.

**Step 3: Manual demo checklist**

Run:

```bash
python apps/desktop_demo/main.py
```

Verify:
- camera preview starts
- MediaPipe detects face/eyes
- 9-point calibration completes
- gaze dot appears with confidence halo
- 3x3 region updates
- visible window candidate appears in debug mode
- metrics logs write JSONL without frames/video
- no app focus change occurs

**Step 4: Update README**

Add:
- install command
- run command
- macOS permissions note
- MVP limitations
- privacy/logging note

**Step 5: Commit**

```bash
git add README.md docs src apps tests
git commit -m "docs: document MVP demo workflow"
```

## Implementation Notes

- Keep the core library free of Qt imports.
- Keep MediaPipe details isolated to `tracking/mediapipe_backend.py`.
- Prefer explicit confidence/validity flags over throwing exceptions during normal tracking failures.
- Avoid global mutable state in calibration/runtime code.
- Do not implement automatic app focusing in the MVP.
- Do not write camera frames/video unless a future explicit opt-in feature is added.
- Treat GPL projects as references only.

## Future Work After MVP

- Multi-monitor support.
- Drift correction.
- Passive calibration from mouse/click behavior.
- Optional recorded sample replay tests.
- IR / near-eye backend.
- Local WebSocket/IPC service.
- macOS Accessibility integration to focus windows with dwell/confirmation safeguards.
- Windows and Linux adapter planning.
