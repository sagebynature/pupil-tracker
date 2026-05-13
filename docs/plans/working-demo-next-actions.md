# Working Pupil Tracker Demo Next Actions Implementation Plan

> **For Hermes:** Use test-driven-development skill to implement this plan task-by-task. Keep each task small, verify RED/GREEN, run `make check`, inspect diffs, and commit before proceeding.

**Goal:** Turn the current camera-activating demo shell into a working end-to-end pupil/gaze tracking demo that shows live preview, tracker annotations, calibration targets, calibrated gaze overlay, region/window debug output, and safe telemetry.

**Architecture:** Keep core logic in `src/pupil_tracker` and keep PySide6 as a thin adapter in `apps/desktop_demo`. Add testable, pure or fake-driven seams for frame conversion, preview rendering, tracker status, calibration sessions, runtime state, and settings. Avoid real webcam/MediaPipe/window requirements in automated tests; reserve live hardware/UI validation for explicit manual checks.

**Tech Stack:** Python 3.11, uv, Make, PySide6/Qt, OpenCV, MediaPipe Tasks, NumPy, scikit-learn, pytest, ruff, ty, macOS CoreGraphics.

---

## Current State

The repo already has:

- `OpenCVCamera` that can open/read/close frames.
- `MediaPipeTrackerBackend` with injectable fake seam and real Tasks adapter path.
- `CalibrationFlowState` for 9-point sample collection.
- `PolynomialRidgeCalibrationModel` for fitting/predicting screen gaze.
- `RuntimePipeline` for camera/backend/calibration/smoothing/region mapping.
- `GazeOverlay` and `OverlayState`.
- `MainWindow` with camera/calibration/telemetry controls.
- macOS visible-window candidate scoring.

The missing demo-critical pieces are:

- live preview frame pump
- frame-to-QPixmap conversion
- tracker annotation rendering
- explicit MediaPipe model asset configuration
- calibration target visual presentation
- calibration capture from live observations
- fit/calibrated tracking state transition
- overlay/runtime/window candidate wiring
- useful telemetry from the live loop

## Milestone Acceptance Criteria

The next milestone is complete when a developer can:

1. Run `make run-demo`.
2. Click Start Camera and see live webcam frames.
3. See face/iris annotation/debug feedback when tracking is available.
4. Configure/load a MediaPipe FaceLandmarker model asset.
5. Start 9-point calibration, follow visible targets, and collect valid observations.
6. Fit calibration after target collection.
7. See calibrated gaze dot/halo and 3x3 region update during tracking.
8. See likely macOS window candidate update in debug text.
9. Start/stop telemetry and inspect JSONL with no frame/image payloads.
10. Stop/close the demo without leaving camera/tracker resources open.

---

## Phase 1: Make the Camera Preview Real

### Task 1: Add frame-to-Qt image conversion helper

**Objective:** Convert OpenCV BGR frames into Qt images/pixmaps through a small testable adapter.

**Files:**
- Create: `apps/desktop_demo/ui/frame_image.py`
- Test: `tests/test_frame_image.py`

**RED test:**

```python
def test_bgr_frame_converts_to_rgb_qimage() -> None:
    image = np.array([[[10, 20, 30]]], dtype=np.uint8)  # BGR
    qimage = bgr_ndarray_to_qimage(image)

    assert qimage.width() == 1
    assert qimage.height() == 1
    assert qimage.pixelColor(0, 0).red() == 30
    assert qimage.pixelColor(0, 0).green() == 20
    assert qimage.pixelColor(0, 0).blue() == 10
```

**Run RED:**

```bash
uv run pytest tests/test_frame_image.py -v
```

Expected: FAIL because `desktop_demo.ui.frame_image` does not exist.

**Implementation notes:**

- Use `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`.
- Ensure the returned `QImage` owns its memory, e.g. call `.copy()` before returning.
- Support 3-channel BGR only initially; raise `ValueError` for unsupported shapes.

**Verify:**

```bash
uv run pytest tests/test_frame_image.py -v
make check
git diff --check
```

**Commit:**

```bash
git add apps/desktop_demo/ui/frame_image.py tests/test_frame_image.py
git commit -m "feat: convert camera frames for Qt preview"
```

### Task 2: Add synchronous preview tick worker

**Objective:** Make `CameraPreviewWorker` expose a testable `tick()` method that reads one frame and emits/stores it without needing a real Qt event loop.

**Files:**
- Modify: `apps/desktop_demo/ui/main_window.py`
- Test: `tests/test_desktop_camera_controls.py`

**RED test:**

```python
def test_camera_worker_tick_reads_frame_from_open_camera() -> None:
    camera = FakeCamera(frames=[fake_frame()])
    worker = CameraPreviewWorker(lambda: camera)
    worker.start()

    frame = worker.tick()

    assert frame.metadata.width == 2
    assert camera.read_calls == 1
```

**Run RED:**

```bash
uv run pytest tests/test_desktop_camera_controls.py -v
```

Expected: FAIL because `tick()` does not exist and the camera protocol lacks `read()`.

**Implementation notes:**

- Extend the local `CameraSource` protocol to include `read() -> Frame`.
- Add `CameraPreviewWorker.tick() -> Frame | None`.
- If not running, return `None`.
- If read fails with `CameraError`, stop worker and re-raise or expose the error for `MainWindow` to display.
- Do not add a real `QTimer` yet; keep this slice pure/testable.

**Verify:**

```bash
uv run pytest tests/test_desktop_camera_controls.py -v
make check
git diff --check
```

**Commit:**

```bash
git add apps/desktop_demo/ui/main_window.py tests/test_desktop_camera_controls.py
git commit -m "feat: read preview frames from camera worker"
```

### Task 3: Wire QTimer live preview rendering

**Objective:** Start a Qt timer on Start Camera, read frames on each timeout, and render them into the preview label.

**Files:**
- Modify: `apps/desktop_demo/ui/main_window.py`
- Test: `tests/test_desktop_live_preview.py`

**RED test:**

```python
def test_start_camera_starts_preview_timer(qt_app: QApplication) -> None:
    camera = FakeCamera(frames=[fake_frame()])
    window = MainWindow(camera_factory=lambda: camera, preview_interval_ms=33)

    window.start_camera()

    assert window.preview_timer.isActive()
    assert window.preview_timer.interval() == 33
```

**Run RED:**

```bash
uv run pytest tests/test_desktop_live_preview.py -v
```

Expected: FAIL because `preview_timer` / `preview_interval_ms` do not exist.

**Implementation notes:**

- Add `QTimer` to `MainWindow`.
- Connect timeout to `update_preview_frame()`.
- `update_preview_frame()` calls `worker.tick()`, converts frame image via `bgr_ndarray_to_qimage`, sets `preview_label.setPixmap(...)`.
- Stop timer on Stop Camera and close.
- On read failure, stop camera and show error.
- Keep tests offscreen and fake-driven.

**Verify:**

```bash
uv run pytest tests/test_desktop_live_preview.py -v
make check
git diff --check
```

**Manual check:**

```bash
make run-demo
```

Click Start Camera. Expected: live preview appears. Click Stop Camera. Expected: camera stops and timer stops.

**Commit:**

```bash
git add apps/desktop_demo/ui/main_window.py apps/desktop_demo/ui/frame_image.py tests/test_desktop_live_preview.py
git commit -m "feat: show live camera preview"
```

---

## Phase 2: Tracker Configuration and Preview Annotations

### Task 4: Add demo runtime configuration

**Objective:** Provide a typed config object for camera id, preview FPS, and MediaPipe model asset path.

**Files:**
- Create: `apps/desktop_demo/config.py`
- Modify: `apps/desktop_demo/app.py`
- Test: `tests/test_desktop_config.py`

**RED test:**

```python
def test_demo_config_reads_model_asset_path_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPIL_TRACKER_MEDIAPIPE_MODEL", "/tmp/face.task")

    config = DemoConfig.from_environment()

    assert config.model_asset_path == Path("/tmp/face.task")
```

**Implementation notes:**

- Use env vars first:
  - `PUPIL_TRACKER_CAMERA_ID`, default `0`.
  - `PUPIL_TRACKER_MEDIAPIPE_MODEL`, optional path.
  - `PUPIL_TRACKER_PREVIEW_FPS`, default `30`.
- Do not fail app startup when model path is missing; camera preview should still work.
- Fail tracker startup with a clear label/debug message if tracking is requested without model path.

**Verify/commit:**

```bash
uv run pytest tests/test_desktop_config.py -v
make check
git diff --check
git add apps/desktop_demo/config.py apps/desktop_demo/app.py tests/test_desktop_config.py
git commit -m "feat: add desktop demo runtime config"
```

### Task 5: Add tracker worker seam and status model

**Objective:** Process preview frames into tracker observations through a testable tracker worker.

**Files:**
- Create: `apps/desktop_demo/tracking_runtime.py`
- Test: `tests/test_desktop_tracking_runtime.py`

**RED test:**

```python
def test_tracking_runtime_returns_observation_status() -> None:
    backend = FakeBackend(observation=valid_observation())
    runtime = TrackingRuntime(backend=backend)

    status = runtime.process(fake_frame())

    assert status.valid is True
    assert status.confidence == 1.0
    assert status.left_iris is not None
```

**Implementation notes:**

- Add `TrackingStatus` dataclass:
  - `observation: RawObservation`
  - `valid: bool`
  - `message: str`
  - maybe `face_bounds`, `left_iris`, `right_iris` convenience properties.
- Add `TrackingRuntime.process(frame: Frame) -> TrackingStatus`.
- Add `close()` that closes backend.
- Keep MediaPipe creation outside tests; inject fake backend.

**Verify/commit:**

```bash
uv run pytest tests/test_desktop_tracking_runtime.py -v
make check
git diff --check
git add apps/desktop_demo/tracking_runtime.py tests/test_desktop_tracking_runtime.py
git commit -m "feat: add desktop tracker runtime seam"
```

### Task 6: Draw tracker annotations on preview frames

**Objective:** Render face bounds and iris centers onto the preview frame before showing it.

**Files:**
- Create: `apps/desktop_demo/ui/annotations.py`
- Modify: `apps/desktop_demo/ui/main_window.py`
- Test: `tests/test_preview_annotations.py`

**RED test:**

```python
def test_annotate_frame_draws_iris_points() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    observation = RawObservation(... left_iris=Point2D(20, 30), right_iris=Point2D(70, 30), ...)

    annotated = annotate_observation(image, observation)

    assert np.any(annotated != image)
```

**Implementation notes:**

- Use `cv2.rectangle` for face bounds and `cv2.circle` for iris centers.
- Copy the input image; do not mutate the original frame image.
- If observation invalid, optionally overlay reason text but do not crash.
- Wire into `update_preview_frame()` after Task 5: read frame, process tracker if enabled, annotate, render.

**Verify/commit:**

```bash
uv run pytest tests/test_preview_annotations.py -v
make check
git diff --check
git add apps/desktop_demo/ui/annotations.py apps/desktop_demo/ui/main_window.py tests/test_preview_annotations.py
git commit -m "feat: annotate preview with tracker observations"
```

---

## Phase 3: Calibration That Actually Uses Live Observations

### Task 7: Add visual calibration target widget

**Objective:** Show a visible target dot at normalized screen/widget coordinates instead of text-only target labels.

**Files:**
- Modify: `apps/desktop_demo/ui/calibration_view.py`
- Test: `tests/test_calibration_target_widget.py`

**RED test:**

```python
def test_calibration_view_exposes_current_target_position(qt_app: QApplication) -> None:
    flow = CalibrationFlowState(samples_per_target=1)
    view = CalibrationView(flow=flow)

    target = view.current_target_position()

    assert target == (0.1, 0.1)
```

**Implementation notes:**

- Add a drawing area widget or enhance `CalibrationView.paintEvent`.
- Draw target dot/crosshair for `flow.current_target`.
- Start with embedded widget coordinates; full-screen calibration can be a later task.
- Keep a method returning normalized target position for tests.

**Verify/commit:**

```bash
uv run pytest tests/test_calibration_target_widget.py -v
make check
git diff --check
git add apps/desktop_demo/ui/calibration_view.py tests/test_calibration_target_widget.py
git commit -m "feat: draw calibration target widget"
```

### Task 8: Add calibration session controller

**Objective:** Capture valid live `RawObservation` samples into `CalibrationFlowState` and fit the calibration model on completion.

**Files:**
- Create: `apps/desktop_demo/calibration_session.py`
- Test: `tests/test_calibration_session.py`

**RED test:**

```python
def test_session_fits_model_after_all_targets_have_samples() -> None:
    model = FakeCalibrationModel()
    flow = CalibrationFlowState(samples_per_target=1)
    session = CalibrationSession(flow=flow, model=model, screen_width=1000, screen_height=800)

    for _ in flow.targets:
        session.capture(valid_observation())

    assert session.is_complete is True
    assert model.fit_calls == 1
```

**Implementation notes:**

- Add states: `idle`, `collecting`, `complete`, `failed`.
- `start()` resets/activates collection.
- `capture(observation)` no-ops unless collecting.
- Invalid observations should not advance.
- On completion, call `model.fit(flow.all_samples(), screen_width, screen_height)`.
- Store fit metrics for debug label.
- Keep screen dimensions injectable for tests.

**Verify/commit:**

```bash
uv run pytest tests/test_calibration_session.py -v
make check
git diff --check
git add apps/desktop_demo/calibration_session.py tests/test_calibration_session.py
git commit -m "feat: add live calibration session controller"
```

### Task 9: Wire calibration controls to live tracker observations

**Objective:** Make Start Calibration start collection and capture observations from the frame loop.

**Files:**
- Modify: `apps/desktop_demo/ui/main_window.py`
- Modify: `apps/desktop_demo/ui/calibration_view.py`
- Test: `tests/test_desktop_calibration_wiring.py`

**RED test:**

```python
def test_live_frame_updates_capture_observations_when_calibrating(qt_app: QApplication) -> None:
    window = MainWindow(... fakes with one valid observation ...)
    window.start_camera()
    window.start_calibration()

    window.update_preview_frame()

    assert len(window.calibration_session.flow.all_samples()) == 1
```

**Implementation notes:**

- Add `MainWindow.start_calibration()` connected to `calibration_view.start_button`.
- Build `CalibrationSession` with `PolynomialRidgeCalibrationModel`.
- During `update_preview_frame()`, after tracker observation exists, call `session.capture(observation)` if collecting.
- Refresh calibration view after each capture.
- Show status/fit metrics in debug label.

**Verify/commit:**

```bash
uv run pytest tests/test_desktop_calibration_wiring.py -v
make check
git diff --check
git add apps/desktop_demo/ui/main_window.py apps/desktop_demo/ui/calibration_view.py tests/test_desktop_calibration_wiring.py
git commit -m "feat: wire calibration to live observations"
```

---

## Phase 4: Calibrated Tracking and Overlay

### Task 10: Add demo tracking state machine

**Objective:** Represent demo mode transitions clearly: stopped, previewing, calibrating, calibrated_tracking, error.

**Files:**
- Create: `apps/desktop_demo/state.py`
- Test: `tests/test_desktop_state.py`

**RED test:**

```python
def test_state_transitions_from_calibrating_to_tracking_after_fit() -> None:
    state = DemoStateMachine()
    state.camera_started()
    state.calibration_started()
    state.calibration_completed()

    assert state.mode is DemoMode.TRACKING
```

**Implementation notes:**

- Use enum `DemoMode`.
- Keep transitions explicit and pure.
- MainWindow uses this for button enablement/debug text in later tasks.

**Verify/commit:**

```bash
uv run pytest tests/test_desktop_state.py -v
make check
git diff --check
git add apps/desktop_demo/state.py tests/test_desktop_state.py
git commit -m "feat: add desktop demo state machine"
```

### Task 11: Compute calibrated gaze during live loop

**Objective:** After calibration completes, convert each valid observation into smoothed `GazeSample` with region id.

**Files:**
- Create: `apps/desktop_demo/gaze_runtime.py`
- Modify: `apps/desktop_demo/ui/main_window.py`
- Test: `tests/test_desktop_gaze_runtime.py`

**RED test:**

```python
def test_gaze_runtime_predicts_smooths_and_maps_region() -> None:
    runtime = GazeRuntime(model=FakeModel(...), smoother=ExponentialMovingAverageSmoother(...))

    sample = runtime.update(valid_observation(), screen_width=300, screen_height=300)

    assert sample is not None
    assert sample.region_id == "middle_center"
```

**Implementation notes:**

- This can wrap existing `PolynomialRidgeCalibrationModel`, `ExponentialMovingAverageSmoother`, and `region_3x3`.
- Return `None` for invalid observations or unfitted model.
- MainWindow updates debug label and overlay only after a sample exists.

**Verify/commit:**

```bash
uv run pytest tests/test_desktop_gaze_runtime.py -v
make check
git diff --check
git add apps/desktop_demo/gaze_runtime.py apps/desktop_demo/ui/main_window.py tests/test_desktop_gaze_runtime.py
git commit -m "feat: compute calibrated gaze in demo runtime"
```

### Task 12: Wire transparent overlay to calibrated gaze

**Objective:** Move/update `GazeOverlay` from calibrated samples during tracking.

**Files:**
- Modify: `apps/desktop_demo/ui/main_window.py`
- Modify: `apps/desktop_demo/ui/overlay.py` if needed
- Test: `tests/test_desktop_overlay_wiring.py`

**RED test:**

```python
def test_tracking_sample_updates_overlay_state(qt_app: QApplication) -> None:
    window = MainWindow(... fake calibrated sample ...)

    window.handle_gaze_sample(GazeSample(timestamp=1, x=100, y=200, confidence=0.8, valid=True))

    assert window.gaze_overlay.state.current.x == 100
```

**Implementation notes:**

- Add `MainWindow.handle_gaze_sample(sample)`.
- Call `gaze_overlay.update_sample(sample)`.
- Show overlay on tracking start; hide on stop/invalid if desired.
- Avoid stealing focus or mouse events.

**Verify/commit:**

```bash
uv run pytest tests/test_desktop_overlay_wiring.py -v
make check
git diff --check
git add apps/desktop_demo/ui/main_window.py apps/desktop_demo/ui/overlay.py tests/test_desktop_overlay_wiring.py
git commit -m "feat: update overlay from calibrated gaze"
```

---

## Phase 5: Window Candidate and Telemetry in the Live Loop

### Task 13: Wire macOS window candidate debug update

**Objective:** Use calibrated gaze to identify a likely visible macOS window candidate in debug output without changing focus.

**Files:**
- Modify: `apps/desktop_demo/ui/main_window.py`
- Test: `tests/test_desktop_window_candidate_wiring.py`

**RED test:**

```python
def test_gaze_sample_updates_window_candidate_debug_text(qt_app: QApplication) -> None:
    window = MainWindow(window_provider=lambda: [candidate])

    window.handle_gaze_sample(GazeSample(timestamp=1, x=50, y=50, confidence=1.0, valid=True))

    assert "DemoApp" in window.debug_label.text()
```

**Implementation notes:**

- Inject `window_provider` for tests; default `list_visible_windows`.
- Use existing `candidate_at_point` pure scorer.
- Never focus/click/raise windows.
- Catch platform errors and show unobtrusive debug text.

**Verify/commit:**

```bash
uv run pytest tests/test_desktop_window_candidate_wiring.py -v
make check
git diff --check
git add apps/desktop_demo/ui/main_window.py tests/test_desktop_window_candidate_wiring.py
git commit -m "feat: show live window candidate debug status"
```

### Task 14: Emit live-loop telemetry events

**Objective:** When logging is enabled, write safe scalar telemetry for observations, calibration progress, gaze samples, and window candidates.

**Files:**
- Modify: `apps/desktop_demo/ui/main_window.py`
- Modify: `src/pupil_tracker/telemetry/jsonl.py` if helper gaps appear
- Test: `tests/test_desktop_live_telemetry.py`

**RED test:**

```python
def test_live_gaze_sample_logs_safe_payload_when_logging_enabled(tmp_path: Path) -> None:
    window = MainWindow(telemetry_path=tmp_path / "demo.jsonl")
    window.start_logging()

    window.handle_gaze_sample(GazeSample(timestamp=1, x=10, y=20, confidence=0.9, valid=True))
    window.stop_logging()

    assert "gaze_sample" in (tmp_path / "demo.jsonl").read_text(encoding="utf-8")
```

**Implementation notes:**

- Use existing `gaze_event_payload`, `calibration_event_payload`, `window_candidate_payload`.
- Do not serialize frames/images.
- Keep logging no-op unless explicitly enabled.

**Verify/commit:**

```bash
uv run pytest tests/test_desktop_live_telemetry.py -v
make check
git diff --check
git add apps/desktop_demo/ui/main_window.py src/pupil_tracker/telemetry/jsonl.py tests/test_desktop_live_telemetry.py
git commit -m "feat: log safe live demo telemetry"
```

---

## Phase 6: Robustness and Developer UX

### Task 15: Add in-app model asset error guidance

**Objective:** Make missing/invalid MediaPipe model setup obvious in the UI and README.

**Files:**
- Modify: `apps/desktop_demo/ui/main_window.py`
- Modify: `README.md`
- Test: `tests/test_desktop_model_guidance.py`

**RED test:**

```python
def test_missing_model_path_shows_tracker_setup_guidance(qt_app: QApplication) -> None:
    window = MainWindow(model_asset_path=None)

    window.start_tracking()

    assert "PUPIL_TRACKER_MEDIAPIPE_MODEL" in window.debug_label.text()
```

**Implementation notes:**

- Camera preview must work without model path.
- Tracking/calibration requiring tracker should show a clear message.
- README should include an example env var command once verified.

**Verify/commit:**

```bash
uv run pytest tests/test_desktop_model_guidance.py -v
make check
git diff --check
git add apps/desktop_demo/ui/main_window.py README.md tests/test_desktop_model_guidance.py
git commit -m "docs: clarify MediaPipe model setup"
```

### Task 16: Add resource lifecycle tests

**Objective:** Ensure camera, timer, tracker, overlay, and telemetry clean up on Stop Camera and window close.

**Files:**
- Modify: `tests/test_desktop_resource_lifecycle.py`
- Modify: app files as needed

**RED test:**

```python
def test_close_stops_timer_camera_tracker_and_logging(qt_app: QApplication, tmp_path: Path) -> None:
    window = MainWindow(...)
    window.start_camera()
    window.start_logging()

    window.close()

    assert not window.preview_timer.isActive()
    assert fake_camera.close_calls == 1
    assert fake_tracker.close_calls == 1
    assert window.telemetry_logger is None
```

**Implementation notes:**

- This is likely a hardening task; implement only fixes surfaced by tests.
- Avoid multiple close calls causing duplicate errors.

**Verify/commit:**

```bash
uv run pytest tests/test_desktop_resource_lifecycle.py -v
make check
git diff --check
git add apps tests/test_desktop_resource_lifecycle.py
git commit -m "fix: clean up demo resources on close"
```

### Task 17: Update manual checklist for real end-to-end flow

**Objective:** Update manual docs after the demo is truly wired.

**Files:**
- Modify: `docs/manual-test-checklist.md`
- Modify: `README.md`

**Docs-only verification:**

```bash
make check
git diff --check
```

**Manual verification:**

```bash
PUPIL_TRACKER_MEDIAPIPE_MODEL=/path/to/face_landmarker.task make run-demo
```

Expected manual path:

1. Start Camera shows live preview.
2. Tracker annotations appear when face visible.
3. Start Calibration shows visible targets.
4. Calibration completes and reports fit metrics.
5. Gaze overlay/region/window debug update.
6. Telemetry JSONL contains no frame payloads.

**Commit:**

```bash
git add README.md docs/manual-test-checklist.md
git commit -m "docs: update end-to-end demo checklist"
```

---

## Recommended Execution Order Summary

1. `feat: convert camera frames for Qt preview`
2. `feat: read preview frames from camera worker`
3. `feat: show live camera preview`
4. `feat: add desktop demo runtime config`
5. `feat: add desktop tracker runtime seam`
6. `feat: annotate preview with tracker observations`
7. `feat: draw calibration target widget`
8. `feat: add live calibration session controller`
9. `feat: wire calibration to live observations`
10. `feat: add desktop demo state machine`
11. `feat: compute calibrated gaze in demo runtime`
12. `feat: update overlay from calibrated gaze`
13. `feat: show live window candidate debug status`
14. `feat: log safe live demo telemetry`
15. `docs: clarify MediaPipe model setup`
16. `fix: clean up demo resources on close`
17. `docs: update end-to-end demo checklist`

## Global Verification Gate

Run before declaring this next milestone complete:

```bash
make sync
make check
git diff --check
```

Then run explicit manual hardware/GUI validation:

```bash
PUPIL_TRACKER_MEDIAPIPE_MODEL=/path/to/face_landmarker.task make run-demo
```

Manual pass criteria:

- Live camera preview appears and stops cleanly.
- Tracker annotation appears when face/iris landmarks are detected.
- Calibration target advances only with valid observations.
- Calibration fit completes and reports metrics.
- Gaze overlay appears and tracks approximately.
- 3x3 region and likely window candidate update plausibly.
- Telemetry is opt-in and contains no images/video frames.
- Closing the window releases camera/tracker/timer/log resources.

## Risks / Pitfalls

- MediaPipe Tasks model asset management is a product/UX issue, not just code. Keep camera preview usable when the model path is absent.
- Do not run live camera in automated tests. Use fake cameras and fake trackers.
- Do not record frames in telemetry while adding live preview.
- Do not block the Qt UI thread with long tracker processing. Start synchronous/timer-based for MVP, but isolate runtime so a future worker thread can replace it.
- Transparent overlay must stay click-through and must not steal focus.
- Window candidate scoring must remain read-only; never focus, click, raise, or activate windows in this milestone.
