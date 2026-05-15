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

## Replay Model Evaluation

Date: 2026-05-15

After scalar replay telemetry was added, a fresh manual run produced replayable calibration and validation samples:

- `calibration_feature_diagnostics`: line `47063`
- `validation_metrics`: line `48429`
- `calibration_replay_sample` events in latest run: `1148`
- `validation_replay_sample` events in latest run: `320`

Latest live validation metrics:

| Metric | Value |
|---|---:|
| Sample count | `190` |
| Mean error | `256.20 px` |
| Median error | `260.21 px` |
| Max error | `474.61 px` |
| Mean X error | `127.41 px` |
| Mean Y error | `180.68 px` |
| Signed Y bias | `-70.26 px` |
| 4x3 grid accuracy | `40.0%` |
| Recommendation | `retry` |

Offline replay model comparison on the same scalar samples, sorted by the product objective (`--objective grid`):

| Model | Mean Error | Mean X | Mean Y | Signed Y | Grid Accuracy | Recommendation |
|---|---:|---:|---:|---:|---:|---|
| `linear-alpha-1.0-affine-corrected` | 201.60 px | 116.07 px | 131.88 px | -17.53 px | 32.2% | retry |
| `poly2-alpha-10.0` | 205.10 px | 115.63 px | 135.89 px | -11.06 px | 26.6% | retry |
| `linear-alpha-1.0` | 201.91 px | 114.58 px | 133.37 px | -13.42 px | 19.7% | retry |
| `linear-alpha-1.0-bias-corrected` | 201.91 px | 114.58 px | 133.37 px | -13.42 px | 19.7% | retry |
| `linear-alpha-10.0` | 235.69 px | 126.19 px | 170.44 px | -19.63 px | 15.3% | retry |
| `poly2-alpha-1.0` | 182.55 px | 123.71 px | 101.67 px | +37.17 px | 13.8% | usable |
| `poly2-alpha-1.0-bias-corrected` | 182.55 px | 123.71 px | 101.67 px | +37.17 px | 13.8% | usable |
| `poly2-alpha-1.0-affine-corrected` | 178.09 px | 124.48 px | 95.74 px | +40.99 px | 13.4% | usable |
| `linear-alpha-0.1` | 187.63 px | 130.74 px | 103.89 px | +30.15 px | 8.8% | usable |
| `poly2-alpha-0.1` | 248.63 px | 211.84 px | 104.91 px | +0.19 px | 3.1% | retry |

Interpretation:

1. The offline evaluator is now useful: one manual run compared ten model variants without another camera session.
2. `poly2-alpha-1.0-affine-corrected` gives the lowest pixel error (`178.09 px`) and mean Y error (`95.74 px`), but its grid accuracy is only `13.4%`.
3. `linear-alpha-1.0-affine-corrected` is the best offline grid candidate at `32.2%`, improving over the uncorrected `linear-alpha-1.0` result (`19.7%`) but still under the latest live validation result (`40.0%`).
4. Do not promote any corrected offline candidate to the live model yet. Post-fit affine correction is promising enough to keep in the evaluator, but current evidence points to calibration geometry / target sampling as the next higher-leverage slice.

## Calibration Sample-Window Evaluation

Date: 2026-05-15

The replay evaluator was extended to fit models with `all`, `early`, `middle`, or `late` calibration samples per target. This tests whether capture timing within each target affects grid-cell performance.

Top grid candidate per sample window:

| Calibration Window | Top Grid Model | Mean Error | Mean X | Mean Y | Signed Y | Grid Accuracy | Recommendation |
|---|---|---:|---:|---:|---:|---:|---|
| `all` | `linear-alpha-1.0-affine-corrected` | 201.60 px | 116.07 px | 131.88 px | -17.53 px | 32.2% | retry |
| `early` | `poly2-alpha-1.0-affine-corrected` | 217.60 px | 164.60 px | 109.98 px | +44.21 px | 16.6% | retry |
| `middle` | `poly2-alpha-1.0-affine-corrected` | 194.96 px | 130.51 px | 108.50 px | +9.16 px | 40.9% | usable |
| `late` | `poly2-alpha-1.0-affine-corrected` | 201.50 px | 130.25 px | 117.02 px | -10.96 px | 42.2% | retry |

Best pixel-error candidate per sample window:

| Calibration Window | Best Error Model | Mean Error | Grid Accuracy |
|---|---|---:|---:|
| `all` | `poly2-alpha-1.0-affine-corrected` | 178.09 px | 13.4% |
| `early` | `linear-alpha-1.0-affine-corrected` | 217.14 px | 7.8% |
| `middle` | `poly2-alpha-0.1` | 175.88 px | 23.1% |
| `late` | `poly2-alpha-0.1` | 172.89 px | 34.4% |

Interpretation:

1. Sample timing matters. `early` calibration samples are clearly worse for grid accuracy.
2. `middle` and `late` windows beat the latest live validation grid result (`40.0%`) by a small margin: `40.9%` and `42.2%` respectively.
3. The winning grid model in both cases is `poly2-alpha-1.0-affine-corrected`, but `late` is marked `retry` because mean error remains above the recommendation threshold.
4. The gain is real enough to justify a live-calibration sampling slice, but not strong enough to treat the current model as dependable.
5. Next implementation candidate: fit the live model from the late third of accepted samples per target, or expose a replay-backed calibration sample policy behind a small config seam before changing the default.

## Decision Gate

Keep the added features only if manual evidence improves at least one of these without a clear regression:

1. vertical feature separability for top/middle/bottom targets,
2. mean Y error,
3. signed Y bias stability,
4. practical `4x3` grid accuracy,
5. red window-border usefulness.

The first head-pose run cleared the first three gates partially and improved grid accuracy only slightly. The replay-enabled run improved live 4x3 grid accuracy to `40.0%`, but pixel error regressed to `256.20 px`. Grid-first offline replay now shows corrected candidates can improve over their base models, and sample-window replay shows middle/late calibration samples can slightly beat the latest live grid result. Keep the features and evaluator corrections for now; the next live slice should test a replay-backed calibration sample policy before treating the build as dependable for window selection.

If feature separability improves but validation remains compressed, investigate model form, target weighting, or calibration/validation sampling windows before adding heavier features.

If feature separability does not improve, do not add eyebrow/chin expression features as standalone inputs. Next heavier candidate would be a real head-pose estimate such as solvePnP with canonical 3D face landmarks.
