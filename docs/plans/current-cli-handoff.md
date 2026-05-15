# Current CLI Handoff — Pupil Tracker

Date: 2026-05-15

## Repository State

- Repo: `/Users/sage/workspace/sagebynature/pupil-tracker`
- Branch: `main`
- Remote state at handoff: clean and synced with `origin/main`
- Latest commit at handoff: `5d91321 docs: record restored feature validation`

Recent relevant commits:

```text
5d91321 docs: record restored feature validation
d707a8f fix: gate solvepnp features behind opt in
956a8ff docs: record solvepnp validation regression
c3f2b49 refactor: remove legacy calibration target panel
4967f79 feat: add solvepnp-style pose features
288c610 feat: add top-left replay correction candidate
```

## Current Decision

Do **not** add more calibration targets and do **not** promote asymmetric/residual wrappers.

The next slice should be scalar diagnostics around repeat-run posture/feature drift:

```text
per-target calibration accepted samples
vs.
per-target validation windows
```

The goal is to explain why `v0`, `v3`, and `v4` remain unstable across runs even after the live feature vector was restored to the stable 23-feature default.

## Why We Are Here

1. Top-row focus calibration produced the best live result so far:
   - run range: `115925:126151`
   - grid accuracy: `52.1%`
   - mean error: `168.03 px`
   - still failed `v0` at `0.0%`

2. SolvePnP-style scalar geometry was added, but a live 29-feature run regressed:
   - run range: `126383:135157`
   - grid accuracy: `27.4%`
   - `v0`, `v3`, and `v4` at `0.0%`

3. The solvePnP-style suffix is now opt-in only:

```bash
PUPIL_TRACKER_SOLVEPNP_STYLE_FEATURES=true
```

Default live MediaPipe path is restored to 23 features.

4. A fresh default 23-feature top-row focus run confirmed the feature gate worked technically, but did not recover the earlier best grid result:
   - run range: `136697:145435`
   - replay feature count: `23` for all `2844` replay samples
   - grid accuracy: `31.1%`
   - mean error: `169.05 px`
   - `v0`, `v3`, `v4` at `0.0%`

## Latest Target Breakdown

Latest restored 23-feature run, matched validation window:

| Target | Grid Accuracy | Predicted Cells |
|---|---:|---|
| `v0` | `0.0%` | `r0c0=34`, `r1c0=4` |
| `v1` | `55.3%` | `r0c3=21`, `r0c2=17` |
| `v2` | `100.0%` | `r1c2=38` |
| `v3` | `0.0%` | `r1c1=38` |
| `v4` | `0.0%` | `r1c2=38` |

Top-left remains scalar-separable:

```text
validation_grid_accuracy: 0.0%
validation_predicted_cells: r0c0=34, r1c0=4
assessment: separable
```

Dominant separability features from latest run:

| Feature | Normalized Δ |
|---|---:|
| `22 pitch proxy` | `+9.67` |
| `10 left eye aperture` | `+2.07` |
| `21 yaw proxy` | `-1.94` |
| `11 right eye aperture` | `+1.76` |
| `17 face height` | `-1.70` |
| `15 face center Y` | `+1.64` |

## Useful Existing Commands

Top-left separability for latest restored run:

```bash
uv run python tools/analyze_top_left_separability.py metrics/demo.jsonl \
  --run 136697:145435 \
  --screen-width 5120 \
  --screen-height 1440 \
  --grid-columns 4 \
  --grid-rows 3
```

Replay evaluator for latest restored run:

```bash
python - <<'PY'
from pathlib import Path
import json
src=Path('metrics/demo.jsonl')
out=Path('/tmp/pupil_tracker_latest_23feature_top_row_136697_145435.jsonl')
start,end=136697,145435
with src.open() as f, out.open('w') as g:
    for i,line in enumerate(f,1):
        if start <= i <= end:
            g.write(line)
print(out)
PY

uv run python tools/evaluate_calibration_models.py \
  /tmp/pupil_tracker_latest_23feature_top_row_136697_145435.jsonl \
  --screen-width 5120 \
  --screen-height 1440 \
  --grid-columns 4 \
  --grid-rows 3 \
  --objective grid \
  --calibration-sample-window all \
  --include-target-residuals
```

Compare best 23-feature run against latest restored 23-feature run:

```bash
uv run python tools/analyze_repeat_run_diagnostics.py metrics/demo.jsonl \
  --run 115925:126151 \
  --run 136697:145435 \
  --screen-width 5120 \
  --screen-height 1440 \
  --grid-columns 4 \
  --grid-rows 3
```

## Recommended Next TDD Slice

Create a scalar-only tool, likely under `tools/`, with tests first:

```text
tools/analyze_posture_validation_drift.py
tests/test_posture_validation_drift_analysis.py
```

Minimum report:

1. For each validation target, identify the relevant nearby calibration target/cluster.
2. Compare calibration accepted sample feature distributions vs validation replay sample feature distributions.
3. Report mean/std/range, signed delta, normalized delta, and dominant features.
4. Highlight posture/head-pose indices first: `20 roll`, `21 yaw`, `22 pitch`, plus face center/size/aspect indices `14-18`.
5. Include predicted-cell distributions from validation samples.
6. Keep output scalar-only markdown/text. No frames, screenshots, raw landmarks, or binary blobs.

Acceptance criteria:

- Tests cover feature-length mismatches and missing targets.
- Tests prove latest validation metrics window is used, not stale validation samples.
- Report explicitly flags targets where posture/feature drift aligns with grid collapse.
- `make check` passes before commit.

## Guardrails

- Do not promote `PUPIL_TRACKER_SOLVEPNP_STYLE_FEATURES=true`; it is opt-in/evaluator-only.
- Do not make `top_row_focus` the default while `v0` remains `0.0%`.
- Do not add another calibration geometry path until repeat-run drift is explained.
- Prioritize grid-cell accuracy over pixel error.
- Keep telemetry scalar-only.
