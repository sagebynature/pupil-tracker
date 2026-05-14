# Feature Diagnostics Comparison

Date: 2026-05-14

## Status

Pending manual live-camera validation.

The current `metrics/demo.jsonl` was produced before `calibration_feature_diagnostics` existed, so it has validation metrics but no feature-diagnostics event to compare. Do not infer feature separability from the older log.

## Implemented Feature Sets

| Feature set | Commit | Feature vector length | Notes |
|---|---|---:|---|
| Eye geometry baseline | `0d5aa2e` lineage | 14 | Iris + eye-relative vertical geometry. |
| Face context | `f615c02` | 20 | Adds normalized face center, face size, face aspect ratio, and inter-ocular distance. |
| Head-pose proxies | `6e652f4` | 23 | Adds eye-line roll, nose x-offset/yaw proxy, and nose y-offset/pitch proxy. |

## Manual Run Protocol

Run from the repo root:

```bash
PUPIL_TRACKER_MEDIAPIPE_MODEL=$(pwd)/models/face_landmarker.task make run-demo
```

Use the better setup from prior runs:

1. Sit closer to the camera.
2. Keep head centered and stable.
3. Keep lighting stable.
4. Click **Start Camera**.
5. Click **Start Logging**.
6. Click **Start Vertical Calibration**.
7. Complete calibration carefully.
8. Click **Start Validation**.
9. Complete validation.
10. Click **Stop Logging**.

Parse the latest feature diagnostics:

```bash
uv run python tools/analyze_feature_diagnostics.py metrics/demo.jsonl
```

Also inspect the latest `validation_metrics` event for:

- mean error
- mean X error
- mean Y error
- signed Y bias
- configured grid accuracy, default `4x3`
- per-target signed Y, if needed

## Results Template

| Run | Feature set | Mean Error | Mean X | Mean Y | Signed Y | 4x3 Grid Accuracy | Top-vs-Center Signal | Bottom-vs-Center Signal | Decision |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| Pending | Head-pose proxies current build | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Decision Gate

Keep the added features only if manual evidence improves at least one of these without a clear regression:

1. vertical feature separability for top/middle/bottom targets,
2. mean Y error,
3. signed Y bias stability,
4. practical `4x3` grid accuracy,
5. red window-border usefulness.

If feature separability improves but validation remains compressed, investigate model form, target weighting, or calibration/validation sampling windows before adding heavier features.

If feature separability does not improve, do not add eyebrow/chin expression features as standalone inputs. Next heavier candidate would be a real head-pose estimate such as solvePnP with canonical 3D face landmarks.
