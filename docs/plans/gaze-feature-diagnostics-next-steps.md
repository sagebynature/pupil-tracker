# Gaze Feature Diagnostics Next Steps Implementation Plan

> **For Hermes:** Use test-driven-development skill to implement this plan task-by-task. Keep each task small, verify RED/GREEN, run `make check`, inspect diffs, commit each task, and pause for manual validation gates when requested.

**Goal:** Determine whether the current gaze features separate top/middle/bottom calibration targets, then add the lowest-risk face geometry and pose features that improve practical desktop/window selection.

**Architecture:** Do not tune grid size or calibration model parameters blindly. First add scalar-only feature diagnostics that summarize existing calibration samples by target. Then add cheap face geometry features, then head-pose proxies, and only then revisit model form or temporal window-selection smoothing. Keep the PySide6 UI as a thin adapter and keep telemetry privacy-preserving: no frames, screenshots, full landmark dumps, or raw image payloads.

**Tech Stack:** Python 3.11, uv, pytest, ruff, ty, PySide6, MediaPipe Tasks, scikit-learn ridge/polynomial calibration, JSONL telemetry.

---

## Current Evidence

Recent manual runs show:

- Closer camera placement improved pixel accuracy:
  - mean error: `338.76 px` → `262.62 px`
  - mean Y error: `290.06 px` → `197.29 px`
  - max error: `750.42 px` → `495.67 px`
- Practical `4x3` grid accuracy remained poor:
  - `20.0%` → `10.5%`
- Calibration capture was clean:
  - 15 targets
  - about 50–51 accepted samples per target
  - 0 rejected samples
  - confidence 1.0
- Per-target signed Y still shows vertical compression:
  - top targets predict too low
  - bottom targets predict too high

Decision: **add feature-separability diagnostics before adding more landmarks or changing models.**

---

## Success Criteria

A future run is considered directionally better only if it improves practical and diagnostic metrics together:

1. Mean Y error decreases without a large X regression.
2. Signed Y bias remains near zero.
3. `4x3` grid accuracy improves materially.
4. Per-target top/bottom errors shrink, not only center error.
5. Feature summaries show top/middle/bottom separation for vertical-sensitive features.

---

## Task 1: Add Pure Feature Summary Metrics

**Objective:** Create a pure, testable helper that summarizes feature vectors by calibration target.

**Files:**
- Create: `src/pupil_tracker/calibration/feature_diagnostics.py`
- Modify: `src/pupil_tracker/calibration/__init__.py`
- Test: `tests/test_calibration_feature_diagnostics.py`

**Step 1: Write failing tests**

Add tests for a public API similar to:

```python
from pupil_tracker.calibration import summarize_feature_diagnostics
from pupil_tracker.models import CalibrationSample, CalibrationTarget, RawObservation


def test_feature_diagnostics_report_mean_and_std_per_target() -> None:
    target = CalibrationTarget(id="top", x=0.5, y=0.2)
    samples = (
        CalibrationSample(
            target=target,
            observation=RawObservation(
                timestamp=1.0,
                valid=True,
                confidence=1.0,
                feature_vector=(1.0, 2.0),
            ),
        ),
        CalibrationSample(
            target=target,
            observation=RawObservation(
                timestamp=2.0,
                valid=True,
                confidence=1.0,
                feature_vector=(3.0, 4.0),
            ),
        ),
    )

    summary = summarize_feature_diagnostics(samples)

    assert summary.target_summaries["top"].feature_mean == (2.0, 3.0)
    assert summary.target_summaries["top"].feature_std == (1.0, 1.0)
    assert summary.target_summaries["top"].accepted_count == 2
```

Also test:

- invalid observations are skipped
- empty feature vectors are skipped
- inconsistent feature vector lengths raise `ValueError`
- output is JSON-serializable without NumPy arrays

**Step 2: Verify RED**

Run:

```bash
uv run pytest tests/test_calibration_feature_diagnostics.py -v
```

Expected: FAIL because `pupil_tracker.calibration.feature_diagnostics` or `summarize_feature_diagnostics` does not exist.

**Step 3: Implement minimally**

Create frozen dataclasses:

```python
@dataclass(frozen=True)
class TargetFeatureSummary:
    target_id: str
    target_x: float
    target_y: float
    accepted_count: int
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]


@dataclass(frozen=True)
class FeatureDiagnosticsSummary:
    feature_count: int
    target_summaries: Mapping[str, TargetFeatureSummary]
```

Use only Python stdlib. Keep calculations deterministic and scalar-only.

**Step 4: Verify GREEN**

Run:

```bash
uv run pytest tests/test_calibration_feature_diagnostics.py -v
make check
git diff --check
```

**Step 5: Commit**

```bash
git add src/pupil_tracker/calibration/feature_diagnostics.py src/pupil_tracker/calibration/__init__.py tests/test_calibration_feature_diagnostics.py
git commit -m "feat: summarize calibration feature diagnostics"
```

---

## Task 2: Serialize Feature Diagnostics Telemetry

**Objective:** Add a JSONL payload function for feature diagnostics without changing live UI flow yet.

**Files:**
- Modify: `src/pupil_tracker/telemetry/jsonl.py`
- Test: `tests/test_telemetry_privacy.py`

**Step 1: Write failing tests**

Add a test that builds a `FeatureDiagnosticsSummary` and asserts the payload contains only JSON-safe scalar/list/dict data:

```python
def test_feature_diagnostics_payload_is_scalar_only() -> None:
    payload = feature_diagnostics_payload(summary)

    assert payload == {
        "feature_count": 2,
        "targets": {
            "top": {
                "target_x": 0.5,
                "target_y": 0.2,
                "accepted_count": 2,
                "feature_mean": [2.0, 3.0],
                "feature_std": [1.0, 1.0],
            }
        },
    }
```

Also assert the payload rejects or cannot contain frames/images/NumPy arrays, following existing telemetry privacy style.

**Step 2: Verify RED**

Run:

```bash
uv run pytest tests/test_telemetry_privacy.py::test_feature_diagnostics_payload_is_scalar_only -v
```

Expected: FAIL because `feature_diagnostics_payload` is missing.

**Step 3: Implement minimally**

Add `feature_diagnostics_payload(summary: FeatureDiagnosticsSummary) -> dict[str, object]`.

Do not log automatically in this task.

**Step 4: Verify GREEN**

Run:

```bash
uv run pytest tests/test_telemetry_privacy.py -v
make check
git diff --check
```

**Step 5: Commit**

```bash
git add src/pupil_tracker/telemetry/jsonl.py tests/test_telemetry_privacy.py
git commit -m "feat: serialize feature diagnostic telemetry"
```

---

## Task 3: Log Feature Diagnostics After Calibration Completes

**Objective:** Emit one `calibration_feature_diagnostics` telemetry event when a calibration session completes and logging is active.

**Files:**
- Modify: `apps/desktop_demo/calibration_session.py` only if a public accessor is needed
- Modify: `apps/desktop_demo/ui/main_window.py`
- Test: `tests/test_desktop_calibration_wiring.py`
- Test: `tests/test_desktop_live_telemetry.py` or `tests/test_telemetry_privacy.py`

**Step 1: Write failing tests**

Add a desktop wiring test with a fake completed calibration session and fake telemetry logger. Assert:

- the event type is `calibration_feature_diagnostics`
- the event is emitted only when telemetry logging is active
- the event uses the completed calibration samples, not validation samples
- no frame/image data appears in the payload

**Step 2: Verify RED**

Run the focused test:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/test_desktop_calibration_wiring.py::test_completed_calibration_logs_feature_diagnostics -v
```

Expected: FAIL because no feature diagnostics event is logged.

**Step 3: Implement minimally**

When calibration transitions to complete and fit succeeds:

1. read `calibration_session.flow.all_samples()`
2. compute `summarize_feature_diagnostics(...)`
3. serialize with `feature_diagnostics_payload(...)`
4. log event type `calibration_feature_diagnostics`

Do not log while telemetry is stopped.

**Step 4: Verify GREEN**

Run:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/test_desktop_calibration_wiring.py tests/test_desktop_live_telemetry.py tests/test_telemetry_privacy.py -v
make check
git diff --check
```

**Step 5: Commit**

```bash
git add apps/desktop_demo/ui/main_window.py apps/desktop_demo/calibration_session.py src/pupil_tracker/telemetry/jsonl.py tests/test_desktop_calibration_wiring.py tests/test_desktop_live_telemetry.py tests/test_telemetry_privacy.py
git commit -m "feat: log calibration feature diagnostics"
```

---

## Task 4: Add a Telemetry Parser for Feature Separability

**Objective:** Make manual runs actionable by adding a small script that reads `metrics/demo.jsonl` and reports top/middle/bottom feature separation.

**Files:**
- Create: `tools/analyze_feature_diagnostics.py`
- Test: `tests/test_feature_diagnostics_analysis.py`
- Modify: `README.md` or `docs/manual-test-checklist.md`

**Step 1: Write failing tests**

Create a test with a tiny JSONL fixture containing a `calibration_feature_diagnostics` event. Assert the parser reports:

- latest diagnostics line
- feature count
- per-target means/stds
- top-vs-center deltas for each feature
- bottom-vs-center deltas for each feature

**Step 2: Verify RED**

Run:

```bash
uv run pytest tests/test_feature_diagnostics_analysis.py -v
```

Expected: FAIL because the script/module is missing.

**Step 3: Implement minimally**

Implement pure functions first, then a CLI wrapper:

```bash
uv run python tools/analyze_feature_diagnostics.py metrics/demo.jsonl
```

Keep output text compact and copy/pasteable into experiment docs.

**Step 4: Verify GREEN**

Run:

```bash
uv run pytest tests/test_feature_diagnostics_analysis.py -v
make check
git diff --check
```

**Step 5: Commit**

```bash
git add tools/analyze_feature_diagnostics.py tests/test_feature_diagnostics_analysis.py README.md docs/manual-test-checklist.md
git commit -m "feat: analyze feature separability telemetry"
```

---

## Task 5: Manual Close-Camera Diagnostics Run

**Objective:** Run the real demo with the new diagnostics and decide whether current features separate vertical targets.

**Files:**
- Modify only docs if recording results:
  - `docs/experiments/manual-validation-grid-and-camera-distance.md`
  - or a new dated experiment note under `docs/experiments/`

**Steps:**

1. Launch:

```bash
PUPIL_TRACKER_MEDIAPIPE_MODEL=$(pwd)/models/face_landmarker.task make run-demo
```

2. Use the better manual setup from prior runs:
   - sit closer to camera
   - keep head centered
   - keep lighting stable

3. Flow:
   - **Start Camera**
   - **Start Logging**
   - **Start Vertical Calibration**
   - **Start Validation**
   - **Stop Logging**

4. Parse:

```bash
uv run python tools/analyze_feature_diagnostics.py metrics/demo.jsonl
```

5. Compare against `validation_metrics`:
   - mean error
   - mean X/Y error
   - signed Y bias
   - `4x3` grid accuracy
   - per-target signed Y

**Decision Gate:**

- If existing vertical-sensitive features barely change between top/middle/bottom, add more features next.
- If they separate cleanly but validation still compresses, revisit model form, target weighting, or calibration/validation sample windows.

**Commit:** Only commit docs if manual results are written down.

---

## Task 6: Add Face Position and Scale Features

**Objective:** Add low-lift camera geometry features supported by the closer-camera experiment.

**Files:**
- Modify: `src/pupil_tracker/models.py`
- Modify: `src/pupil_tracker/tracking/features.py`
- Modify: `src/pupil_tracker/tracking/mediapipe_backend.py`
- Test: `tests/test_models.py`
- Test: `tests/test_tracking_features.py`
- Test: `tests/test_mediapipe_backend.py`

**Step 1: Write failing tests**

Add or extend model data so feature extraction knows frame dimensions without logging frame payloads. Preferred small addition:

```python
@dataclass(frozen=True)
class RawObservation:
    ...
    frame_width: int | None = None
    frame_height: int | None = None
```

Feature tests should assert appended scalar features:

- normalized face center x/y
- normalized face width/height
- face aspect ratio
- normalized inter-ocular distance

Preserve the legacy first 14 feature positions from `eye_geometry_feature_vector`.

**Step 2: Verify RED**

Run:

```bash
uv run pytest tests/test_tracking_features.py::test_face_context_features_include_position_and_scale -v
```

Expected: FAIL because the helper/API does not expose face context features.

**Step 3: Implement minimally**

Add a new helper, not a breaking change to the old one:

```python
face_context_feature_vector(...)
```

It should return:

```text
existing 14 eye geometry features + face context scalars
```

Wire MediaPipe backend to use the new helper and set frame metadata fields on `RawObservation` if needed.

**Step 4: Verify GREEN**

Run:

```bash
uv run pytest tests/test_tracking_features.py tests/test_mediapipe_backend.py tests/test_models.py -v
make check
git diff --check
```

**Step 5: Commit**

```bash
git add src/pupil_tracker/models.py src/pupil_tracker/tracking/features.py src/pupil_tracker/tracking/mediapipe_backend.py tests/test_models.py tests/test_tracking_features.py tests/test_mediapipe_backend.py
git commit -m "feat: add face context gaze features"
```

---

## Task 7: Add Head-Pose Proxy Features

**Objective:** Add cheap head-pose proxies before implementing heavier solvePnP or 3D pose estimation.

**Files:**
- Modify: `src/pupil_tracker/tracking/features.py`
- Modify: `src/pupil_tracker/tracking/mediapipe_backend.py`
- Test: `tests/test_tracking_features.py`
- Test: `tests/test_mediapipe_backend.py`

**Step 1: Write failing tests**

Use deterministic synthetic landmarks to assert proxy outputs:

- eye-line slope / roll proxy
- nose-to-eye-midpoint x offset / yaw proxy
- nose-to-eye-midpoint y offset / pitch proxy
- mouth/chin-to-eye geometry if stable landmarks are already available

Do not add eyebrow movement as a standalone feature in this task.

**Step 2: Verify RED**

Run:

```bash
uv run pytest tests/test_tracking_features.py::test_head_pose_proxy_features_capture_pitch_yaw_roll -v
```

Expected: FAIL because the helper/API does not expose pose proxies.

**Step 3: Implement minimally**

Add a helper that consumes explicit points or normalized scalars. Keep the backend mapping isolated with named landmark constants.

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
git commit -m "feat: add head pose proxy gaze features"
```

---

## Task 8: Manual Feature Comparison Gate

**Objective:** Compare baseline diagnostics, face-context features, and pose-proxy features under the same manual protocol.

**Files:**
- Create: `docs/experiments/feature-diagnostics-comparison.md`

**Steps:**

For each feature set:

1. Run close-camera vertical calibration and validation.
2. Parse latest feature diagnostics.
3. Parse latest validation metrics.
4. Record:
   - mean error
   - mean X/Y error
   - signed Y
   - `4x3` grid accuracy
   - top/middle/bottom feature deltas
   - per-target signed Y

**Decision Gate:**

- Keep features that improve vertical separability and validation metrics.
- Revert or disable features that add noise or regress grid accuracy.
- Only after this gate consider temporal smoothing for the red window border.

**Commit:**

```bash
git add docs/experiments/feature-diagnostics-comparison.md
git commit -m "docs: compare gaze feature diagnostics"
```

---

## Deferred Work

Do not start these until diagnostics justify them:

- full solvePnP head pose using 2D/3D canonical face landmarks
- raw eye crops or CNN-based appearance model
- eyebrow/chin expression features as standalone model inputs
- dwell-based window selection side effects
- changing recommendation thresholds based on grid accuracy
- changing default calibration model again

---

## Manual Run Command Reference

```bash
PUPIL_TRACKER_MEDIAPIPE_MODEL=$(pwd)/models/face_landmarker.task make run-demo
```

Optional validation grid overrides:

```bash
PUPIL_TRACKER_VALIDATION_GRID_COLUMNS=4 \
PUPIL_TRACKER_VALIDATION_GRID_ROWS=3 \
PUPIL_TRACKER_MEDIAPIPE_MODEL=$(pwd)/models/face_landmarker.task \
make run-demo
```

Quality gate before every code commit:

```bash
make check
git diff --check
git status --short
```
