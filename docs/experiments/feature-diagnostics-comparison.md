# Feature Diagnostics Comparison

Date: 2026-05-14

## Status

Manual live-camera validation captured on the current head-pose proxy build.

The latest `metrics/demo.jsonl` now contains both `calibration_feature_diagnostics` and `validation_metrics` events. The run provides enough evidence to keep the scalar feature expansion for another iteration, but it does not yet prove practical 4x3 window selection is solved.

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

## Results

Latest event locations:

- `calibration_feature_diagnostics`: line `40664`
- `validation_metrics`: line `43330`

| Run | Feature set | Mean Error | Mean X | Mean Y | Signed Y | 4x3 Grid Accuracy | Recommendation | Decision |
|---|---|---:|---:|---:|---:|---:|---|---|
| Current manual run | Head-pose proxies | 199.56 px | 89.63 px | 171.16 px | -31.66 px | 12.6% | usable | Keep for another measured iteration; not sufficient for 4x3 selection yet. |

Per validation target:

| Target | Error | Signed Y | 4x3 Grid Accuracy |
|---|---:|---:|---:|
| `v0` | 303.93 px | +234.84 px | 0.0% |
| `v1` | 91.26 px | +68.35 px | 39.5% |
| `v2` | 61.23 px | +44.87 px | 21.1% |
| `v3` | 357.12 px | -328.92 px | 0.0% |
| `v4` | 184.25 px | -177.42 px | 2.6% |

Compared with the prior closer-camera run from `manual-validation-grid-and-camera-distance.md`:

| Metric | Prior closer-camera run | Current head-pose run | Delta |
|---|---:|---:|---:|
| Mean error | 262.62 px | 199.56 px | -63.06 px |
| Max error | 495.67 px | 416.43 px | -79.24 px |
| Mean X error | 124.92 px | 89.63 px | -35.29 px |
| Mean Y error | 197.29 px | 171.16 px | -26.13 px |
| Signed Y bias | -17.23 px | -31.66 px | -14.43 px |
| 4x3 grid accuracy | 10.5% | 12.6% | +2.1 percentage points |

## Feature Separability

The latest diagnostics event reported:

- feature count: `23`
- calibration targets: `15`
- center-column accepted samples: top `51`, center `51`, bottom `50`

Largest top/bottom-vs-center vertical-separation signals were:

| Feature index | Feature | Top - Center | Bottom - Center | Combined Absolute Delta |
|---:|---|---:|---:|---:|
| 9 | right iris eye-relative Y | +0.0225 | -0.0912 | 0.1137 |
| 12 | eye midpoint Y | +0.0113 | -0.0676 | 0.0789 |
| 13 | eye-line slope | -0.0224 | +0.0471 | 0.0695 |
| 7 | left iris eye-relative Y | +0.0001 | -0.0441 | 0.0442 |
| 18 | face aspect ratio | -0.0107 | +0.0318 | 0.0425 |
| 20 | roll proxy | -0.0226 | +0.0176 | 0.0402 |

Interpretation:

1. The current feature vector does contain measurable top/middle/bottom signal.
2. The strongest separation still comes from eye-relative vertical geometry, not the new nose-based pitch/yaw proxy.
3. The scalar feature expansion improved pixel error and moved the recommendation from `retry` to `usable`.
4. Practical 4x3 grid accuracy is still too low for dependable window selection.

## Decision Gate

Keep the added features only if manual evidence improves at least one of these without a clear regression:

1. vertical feature separability for top/middle/bottom targets,
2. mean Y error,
3. signed Y bias stability,
4. practical `4x3` grid accuracy,
5. red window-border usefulness.

This run clears the first three gates partially and improves the fourth only slightly. Keep the features for now because pixel error improved and the diagnostic signal is measurable. Do not treat the current build as ready for dependable window selection because grid accuracy remains only `12.6%`.

If feature separability improves but validation remains compressed, investigate model form, target weighting, or calibration/validation sampling windows before adding heavier features.

If feature separability does not improve, do not add eyebrow/chin expression features as standalone inputs. Next heavier candidate would be a real head-pose estimate such as solvePnP with canonical 3D face landmarks.
