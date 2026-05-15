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
5. Next implementation candidate: run a fresh live manual validation with `PUPIL_TRACKER_CALIBRATION_SAMPLE_WINDOW=late`. The implementation now supports the policy behind config, while the default remains `all` until live evidence confirms it.

## Late-Window Live Validation

Date: 2026-05-15

A fresh manual run was launched with `PUPIL_TRACKER_CALIBRATION_SAMPLE_WINDOW=late`.

Latest run evidence:

- `calibration_feature_diagnostics`: line `52159`
- `validation_metrics`: line `53453`
- `calibration_replay_sample` events in latest run: `1150`
- `validation_replay_sample` events in latest run: `319`

Live validation metrics from the late-window run:

| Metric | Value |
|---|---:|
| Sample count | `190` |
| Mean error | `296.41 px` |
| Median error | `317.47 px` |
| Max error | `659.10 px` |
| Mean X error | `153.18 px` |
| Mean Y error | `217.05 px` |
| Signed Y bias | `+70.05 px` |
| 4x3 grid accuracy | `30.0%` |
| Recommendation | `retry` |

Per-target grid accuracy shows the failure is concentrated, not uniform:

| Target | Grid Accuracy | Signed Y Error |
|---|---:|---:|
| `v0` | `0.0%` | `+446.58 px` |
| `v1` | `47.4%` | `+145.12 px` |
| `v2` | `97.4%` | `+119.88 px` |
| `v3` | `0.0%` | `-72.49 px` |
| `v4` | `5.3%` | `-288.86 px` |

Offline replay on only the latest run confirms the live late-window policy was not a win for the current default polynomial model:

| Calibration Window | Top Grid Model | Mean Error | Mean Y | Signed Y | Grid Accuracy | Recommendation |
|---|---|---:|---:|---:|---:|---|
| `all` | `linear-alpha-0.1` | 253.15 px | 169.91 px | +147.64 px | 40.1% | retry |
| `early` | `linear-alpha-0.1` | 237.80 px | 164.46 px | +132.90 px | 43.3% | retry |
| `middle` | `poly2-alpha-1.0-affine-corrected` | 256.27 px | 179.34 px | +103.23 px | 29.8% | retry |
| `late` | `poly2-alpha-1.0` | 237.83 px | 159.95 px | +66.86 px | 28.5% | retry |

Interpretation:

1. The late-window live policy regressed practical grid accuracy from the previous live baseline (`40.0%`) to `30.0%`.
2. Do not change the default calibration sample window from `all` to `late`.
3. The latest run has severe target-specific vertical failures (`v0`, `v3`, `v4`), which points to run stability / target geometry / validation sampling rather than a simple late-sample improvement.
4. The latest replay still shows some model-form sensitivity (`linear-alpha-0.1` with early/all windows reaches `40.1%`-`43.3%` offline), but every candidate remains `retry`; do not promote a model solely from this noisy run.
5. Next implementation should compare target-specific residuals and calibration/validation target geometry before another default live policy change.

## Target Residual Replay Analysis

Date: 2026-05-15

The replay evaluator now supports `--include-target-residuals`, which appends per-target calibration and validation residual tables for the top-ranked model. The latest run was isolated to lines `48430`-`53453` before analysis.

Top grid model per calibration sample window on the latest run:

| Calibration Window | Top Grid Model | Mean Error | Mean Y | Signed Y | Grid Accuracy | Recommendation |
|---|---|---:|---:|---:|---:|---|
| `all` | `linear-alpha-0.1` | 253.15 px | 169.91 px | +147.64 px | 40.1% | retry |
| `early` | `linear-alpha-0.1` | 237.80 px | 164.46 px | +132.90 px | 43.3% | retry |
| `middle` | `poly2-alpha-1.0-affine-corrected` | 256.27 px | 179.34 px | +103.23 px | 29.8% | retry |
| `late` | `poly2-alpha-1.0` | 237.83 px | 159.95 px | +66.86 px | 28.5% | retry |

The live late-window path uses the default polynomial model, which matches the `poly2-alpha-1.0` replay candidate. Its worst calibration residuals were strongly vertical and concentrated at screen edges:

| Target | Target X | Target Y | Samples | Mean Error | Mean X | Mean Y | Signed X | Signed Y |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `r0c1` | 0.50 | 0.10 | 25 | 199.88 px | 25.18 px | 197.86 px | +25.18 px | +197.86 px |
| `r0c2` | 0.90 | 0.10 | 25 | 131.05 px | 23.39 px | 128.29 px | +22.64 px | +128.29 px |
| `r4c1` | 0.50 | 0.90 | 25 | 124.78 px | 22.35 px | 120.70 px | +21.77 px | -117.50 px |
| `r4c2` | 0.90 | 0.90 | 25 | 116.34 px | 58.89 px | 94.38 px | -58.89 px | -94.38 px |
| `r1c0` | 0.10 | 0.30 | 25 | 97.22 px | 77.86 px | 56.38 px | -77.86 px | +56.38 px |

Worst validation residuals for the same live-equivalent `late`/`poly2-alpha-1.0` path:

| Target | Target X | Target Y | Samples | Mean Error | Mean X | Mean Y | Signed X | Signed Y | Grid Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v0` | 0.25 | 0.25 | 64 | 383.70 px | 72.95 px | 368.87 px | +15.19 px | +368.87 px | 0.0% |
| `v4` | 0.75 | 0.75 | 64 | 301.57 px | 172.43 px | 184.47 px | -64.21 px | -172.33 px | 12.5% |
| `v1` | 0.75 | 0.25 | 63 | 261.44 px | 207.81 px | 118.26 px | -12.09 px | +91.43 px | 50.8% |
| `v2` | 0.50 | 0.50 | 64 | 144.52 px | 80.92 px | 81.70 px | +80.01 px | +70.23 px | 71.9% |
| `v3` | 0.25 | 0.75 | 64 | 98.28 px | 79.62 px | 45.79 px | +32.78 px | -23.54 px | 7.8% |

Interpretation:

1. The regression is dominated by vertical error, especially top-left validation target `v0` (`+368.87 px` signed Y, `0.0%` grid accuracy) and bottom-right `v4` (`-172.33 px` signed Y, `12.5%` grid accuracy).
2. Calibration residuals show the same vertical compression pattern: top calibration targets are predicted too low, while bottom targets are predicted too high.
3. Edge and corner targets are worse than the center. This points to geometry/coverage and mapping compression, not simply noisy late samples.
4. The best offline grid model for this run (`early` + `linear-alpha-0.1`, `43.3%`) still has retry-level error and severe top-row vertical bias, so it should not be promoted as-is.
5. Next implementation should test target weighting or geometry changes that penalize top/bottom residuals, while keeping the default live sample window as `all`.

## Replay Target Weighting Analysis

Date: 2026-05-15

The replay evaluator now includes evaluator-only weighted candidates. Weighting duplicates selected calibration samples before model fit; telemetry payloads and live calibration behavior are unchanged. Policies tested:

- `vertical_edges`: top/bottom targets weighted 3x.
- `screen_edges`: top/bottom/left/right targets weighted 3x.
- `corners`: corner targets weighted 3x.

Latest-run results, isolated to lines `48430`-`53453`:

| Calibration Window | Best Overall Model | Overall Grid | Best Weighted Model | Weighted Grid | Weighted Mean Error | Weighted Signed Y |
|---|---|---:|---|---:|---:|---:|
| `all` | `linear-alpha-0.1` | 40.1% | `linear-alpha-0.1-weight-vertical_edges` | 34.8% | 279.02 px | +183.48 px |
| `early` | `poly2-alpha-1.0-weight-screen_edges` | 44.5% | `poly2-alpha-1.0-weight-screen_edges` | 44.5% | 263.70 px | +171.94 px |
| `middle` | `poly2-alpha-1.0-weight-corners` | 31.7% | `poly2-alpha-1.0-weight-corners` | 31.7% | 244.26 px | +87.12 px |
| `late` | `poly2-alpha-1.0-weight-vertical_edges` | 30.1% | `poly2-alpha-1.0-weight-vertical_edges` | 30.1% | 239.58 px | +90.59 px |

Best weighted candidate was `early` + `poly2-alpha-1.0-weight-screen_edges` at `44.5%` grid accuracy. It beats the unweighted early replay candidate (`43.3%`) and the previous live baseline (`40.0%`) on grid accuracy, but it is still a retry-level model with high pixel error and severe top-row vertical bias.

Validation residuals for `early` + `poly2-alpha-1.0-weight-screen_edges`:

| Target | Mean Error | Signed X | Signed Y | Grid Accuracy |
|---|---:|---:|---:|---:|
| `v0` | 438.02 px | +112.28 px | +415.65 px | 0.0% |
| `v1` | 340.96 px | -34.54 px | +270.60 px | 0.0% |
| `v2` | 198.43 px | +96.80 px | +145.01 px | 57.8% |
| `v4` | 182.35 px | -6.22 px | +12.15 px | 78.1% |
| `v3` | 159.94 px | +153.36 px | +17.86 px | 85.9% |

Interpretation:

1. Weighting can move grid accuracy slightly in replay, but the best weighted result wins by shifting errors rather than solving the mapping.
2. The top validation row remains unusable: `v0` and `v1` both have `0.0%` grid accuracy and very large positive signed Y error.
3. Weighted screen-edge fitting improves lower validation targets (`v3`, `v4`) but worsens the top row, so it should not be promoted to live calibration.
4. The next higher-leverage change is calibration/validation geometry or explicit vertical-bias correction, not stronger edge weighting.

Decision: keep weighted candidates in the replay evaluator for comparison, but do not expose a live weighting config yet and do not change live defaults.

## Replay Vertical-Bias Correction Analysis

Date: 2026-05-15

The replay evaluator now includes evaluator-only vertical-bias correction candidates. The correction fits the base model first, predicts calibration samples, learns a one-dimensional linear residual correction from predicted normalized Y to Y residual, and applies that Y correction at prediction time while leaving X unchanged.

Latest-run results, isolated to lines `48430`-`53453`:

| Calibration Window | Best Overall Model | Overall Grid | Best Uncorrected/Unweighted | Uncorrected Grid | Best Weighted | Weighted Grid | Best Vertical-Bias | Vertical Grid |
|---|---|---:|---|---:|---|---:|---|---:|
| `all` | `poly2-alpha-10.0-vertical-bias-corrected` | 43.3% | `linear-alpha-0.1` | 40.1% | `linear-alpha-0.1-weight-vertical_edges` | 34.8% | `poly2-alpha-10.0-vertical-bias-corrected` | 43.3% |
| `early` | `poly2-alpha-1.0-weight-screen_edges` | 44.5% | `linear-alpha-0.1` | 43.3% | `poly2-alpha-1.0-weight-screen_edges` | 44.5% | `linear-alpha-0.1-vertical-bias-corrected` | 38.9% |
| `middle` | `poly2-alpha-10.0-vertical-bias-corrected` | 42.6% | `poly2-alpha-1.0-affine-corrected` | 29.8% | `poly2-alpha-1.0-weight-corners` | 31.7% | `poly2-alpha-10.0-vertical-bias-corrected` | 42.6% |
| `late` | `poly2-alpha-10.0-vertical-bias-corrected` | 37.9% | `poly2-alpha-1.0` | 28.5% | `poly2-alpha-1.0-weight-vertical_edges` | 30.1% | `poly2-alpha-10.0-vertical-bias-corrected` | 37.9% |

Best vertical-bias candidate was `all` + `poly2-alpha-10.0-vertical-bias-corrected` at `43.3%` grid accuracy. It improves over the uncorrected `all` baseline (`40.1%`) and reduces signed Y bias (`+81.53 px` vs `+147.64 px`), but it does not beat the best weighted replay result (`44.5%`) and remains retry-level.

Validation residuals for `all` + `poly2-alpha-10.0-vertical-bias-corrected`:

| Target | Mean Error | Signed X | Signed Y | Grid Accuracy |
|---|---:|---:|---:|---:|
| `v0` | 391.87 px | +59.28 px | +377.22 px | 0.0% |
| `v4` | 281.16 px | -90.80 px | -159.17 px | 7.8% |
| `v1` | 232.15 px | -58.35 px | +88.51 px | 50.8% |
| `v2` | 162.52 px | +78.12 px | +101.48 px | 71.9% |
| `v3` | 106.28 px | +84.40 px | -0.27 px | 85.9% |

Interpretation:

1. Vertical-bias correction helps some windows materially, especially `middle` (`29.8%` uncorrected best to `42.6%` vertical-bias corrected) and `late` (`28.5%` to `37.9%`).
2. It does not solve the failure. Top-left `v0` remains `0.0%` grid accuracy with `+377.22 px` signed Y error, and bottom-right `v4` remains weak at `7.8%` grid accuracy.
3. The correction reduces global signed Y bias but still leaves target-specific corner/edge failures. The remaining error is not a simple global Y bias.
4. Do not promote vertical-bias correction to live behavior yet.
5. Next evaluator slice should test geometry changes: denser top-row calibration, validation target placement, or per-band correction that can address top-left/top-right separately without overcorrecting bottom targets.

Decision: keep vertical-bias correction candidates in the replay evaluator for comparison, but do not expose live config and do not change live defaults.

## Replay Per-Band Correction Analysis

Date: 2026-05-15

The replay evaluator now includes evaluator-only per-band correction candidates. The correction fits the base model, predicts calibration samples, buckets calibration residuals by predicted normalized Y band, learns a mean Y residual per band, and applies the matching band's Y residual at prediction time while leaving X unchanged.

Latest-run results, isolated to lines `48430`-`53453`:

| Calibration Window | Best Overall Model | Overall Grid | Best Uncorrected/Unweighted | Uncorrected Grid | Best Weighted | Weighted Grid | Best Vertical-Bias | Vertical Grid | Best Per-Band | Per-Band Grid |
|---|---|---:|---|---:|---|---:|---|---:|---|---:|
| `all` | `poly2-alpha-10.0-vertical-bias-corrected` | 43.3% | `linear-alpha-0.1` | 40.1% | `linear-alpha-0.1-weight-vertical_edges` | 34.8% | `poly2-alpha-10.0-vertical-bias-corrected` | 43.3% | `linear-alpha-0.1-per-band-corrected` | 40.1% |
| `early` | `poly2-alpha-1.0-weight-screen_edges` | 44.5% | `linear-alpha-0.1` | 43.3% | `poly2-alpha-1.0-weight-screen_edges` | 44.5% | `linear-alpha-0.1-vertical-bias-corrected` | 38.9% | `linear-alpha-0.1-per-band-corrected` | 43.3% |
| `middle` | `poly2-alpha-10.0-vertical-bias-corrected` | 42.6% | `poly2-alpha-1.0-affine-corrected` | 29.8% | `poly2-alpha-1.0-weight-corners` | 31.7% | `poly2-alpha-10.0-vertical-bias-corrected` | 42.6% | `poly2-alpha-1.0-per-band-corrected` | 29.8% |
| `late` | `poly2-alpha-10.0-vertical-bias-corrected` | 37.9% | `poly2-alpha-1.0` | 28.5% | `poly2-alpha-1.0-weight-vertical_edges` | 30.1% | `poly2-alpha-10.0-vertical-bias-corrected` | 37.9% | `poly2-alpha-1.0-per-band-corrected` | 28.5% |

Best per-band candidate was `early` + `linear-alpha-0.1-per-band-corrected` at `43.3%` grid accuracy. That only matches the uncorrected early candidate, trails the best weighted replay result (`44.5%`), and trails the best global vertical-bias result for the default `all` window (`43.3%`) while carrying worse top-row residuals.

Validation residuals for `early` + `linear-alpha-0.1-per-band-corrected`:

| Target | Mean Error | Signed X | Signed Y | Grid Accuracy |
|---|---:|---:|---:|---:|
| `v0` | 423.70 px | +127.75 px | +399.18 px | 0.0% |
| `v1` | 285.63 px | -78.18 px | +222.85 px | 0.0% |
| `v3` | 182.68 px | +166.41 px | +44.34 px | 85.9% |
| `v4` | 173.10 px | -57.75 px | +16.43 px | 54.7% |
| `v2` | 154.77 px | +62.97 px | +106.83 px | 75.0% |

Interpretation:

1. Three-band residual correction does not fix top-row collapse. `v0` and `v1` remain `0.0%` grid accuracy.
2. Per-band correction mostly preserves the same grid result as its base early linear model while worsening signed Y bias (`+157.72 px` vs `+132.90 px`).
3. The remaining error is not a coarse top/middle/bottom Y-only correction problem; X residuals at corners are also large.
4. Do not promote per-band correction to live behavior.
5. Next high-leverage evaluator slice should test calibration geometry: denser top-row / edge targets or changed target placement, because correction-only wrappers are no longer removing the corner failures.

Decision: keep per-band candidates in the replay evaluator for comparison, but do not expose live config and do not change live defaults.

## Edge-Dense Live Validation

Date: 2026-05-15

A fresh logged run used the new 17-point edge-dense calibration path. The run spans lines `53454`-`59073` and includes `1300` calibration replay samples across targets `top0`-`top4`, `upper_left`, `upper_right`, `mid_left`, `mid_center`, `mid_right`, `lower_left`, `lower_right`, and `bottom0`-`bottom4`, plus `320` validation replay samples.

Live validation metrics:

| Metric | Result |
|---|---:|
| Sample count | `190` |
| Mean error | `181.13 px` |
| Median error | `129.06 px` |
| Max error | `518.95 px` |
| Mean X error | `88.46 px` |
| Mean Y error | `148.50 px` |
| Signed Y bias | `+17.53 px` |
| 4x3 grid accuracy | `41.6%` |
| Recommendation | `usable` |

Per-target validation behavior:

| Target | Mean Error | Signed Y | Grid Accuracy |
|---|---:|---:|---:|
| `v0` | `393.62 px` | `+357.71 px` | `0.0%` |
| `v1` | `61.13 px` | `+39.73 px` | `100.0%` |
| `v2` | `78.35 px` | `-2.32 px` | `89.5%` |
| `v3` | `120.51 px` | `-118.37 px` | `18.4%` |
| `v4` | `252.05 px` | `-189.11 px` | `0.0%` |

Replay evaluation of the same run shows the strongest offline signal from the `middle` calibration sample window:

| Calibration Window | Best Grid Candidate | Grid Accuracy | Mean Error | Recommendation |
|---|---|---:|---:|---|
| `all` | `poly2-alpha-10.0-vertical-bias-corrected` | `31.2%` | `197.18 px` | `usable` |
| `early` | `poly2-alpha-10.0-vertical-bias-corrected` | `17.5%` | `228.46 px` | `retry` |
| `middle` | `poly2-alpha-10.0-vertical-bias-corrected` | `46.6%` | `209.65 px` | `retry` |
| `late` | `poly2-alpha-10.0-vertical-bias-corrected` | `33.1%` | `223.52 px` | `retry` |

Interpretation:

1. Edge-dense live calibration improved over the latest late-window live run (`30.0%` grid, `296.41 px`) and slightly exceeded the previous best live grid run (`40.0%`). It also improved live mean error to `181.13 px`, reaching `usable`.
2. The gain is not broad enough to promote edge-dense calibration as the default. `v0` and `v4` remain `0.0%` grid accuracy, and `v3` is weak at `18.4%`.
3. The top-right validation target (`v1`) improved strongly, while top-left (`v0`) remains the dominant failure. The remaining issue is asymmetric corner/edge behavior rather than generic top-row coverage.
4. Middle-window replay is promising (`46.6%` grid), but the best replay candidate still has retry-level mean error and leaves `v0`/`v4` at `0.0%` in target residuals.
5. Next step should be a second logged edge-dense run to confirm reproducibility before another code change. If the same asymmetric `v0`/`v4` pattern repeats, the next implementation slice should test asymmetric geometry or target-specific diagnostics rather than another global correction wrapper.

Decision: keep Start Edge-Dense Calibration as an experimental option. Do not change the default 9-point calibration path, default sample window, or live model/correction policy yet.

## Second Edge-Dense Live Validation

Date: 2026-05-15

A second logged edge-dense run spans lines `59074`-`64477` and includes `1301` calibration replay samples across the same 17 calibration targets plus `317` validation replay samples.

Live validation metrics:

| Metric | First Edge-Dense | Second Edge-Dense |
|---|---:|---:|
| Mean error | `181.13 px` | `255.25 px` |
| Max error | `518.95 px` | `505.20 px` |
| Mean X error | `88.46 px` | `89.61 px` |
| Mean Y error | `148.50 px` | `225.77 px` |
| Signed Y bias | `+17.53 px` | `+130.06 px` |
| 4x3 grid accuracy | `41.6%` | `38.9%` |
| Recommendation | `usable` | `retry` |

Second-run per-target validation behavior:

| Target | Mean Error | Signed Y | Grid Accuracy |
|---|---:|---:|---:|
| `v0` | `471.18 px` | `+347.68 px` | `0.0%` |
| `v1` | `363.57 px` | `+361.64 px` | `0.0%` |
| `v2` | `183.33 px` | `+180.28 px` | `94.7%` |
| `v3` | `169.61 px` | `-166.62 px` | `0.0%` |
| `v4` | `88.57 px` | `-72.66 px` | `100.0%` |

Replay evaluation of the second run did not show a promotable offline candidate:

| Calibration Window | Best Grid Candidate | Grid Accuracy | Mean Error | Recommendation |
|---|---|---:|---:|---|
| `all` | `poly2-alpha-0.1` | `28.7%` | `323.55 px` | `retry` |
| `early` | `linear-alpha-0.1` | `23.7%` | `292.65 px` | `retry` |
| `middle` | `poly2-alpha-1.0` | `30.3%` | `241.45 px` | `retry` |
| `late` | `poly2-alpha-1.0` | `37.2%` | `232.78 px` | `retry` |

Interpretation:

1. The second edge-dense run did not reproduce the first run's `usable` result. Grid accuracy stayed near the previous live baseline, but mean error and signed Y bias regressed sharply.
2. The repeated pattern is not one stable failed corner: first run failed `v0`/`v4`; second run failed `v0`/`v1`/`v3` while `v4` became `100%`.
3. The stable failure class is large target-specific vertical error, especially on off-center validation targets, not a specific global correction problem.
4. Do not promote edge-dense calibration, middle/late sample windows, or any replay correction from these two runs.
5. Next implementation should add target-specific diagnostics that compare calibration target quality, feature distribution, and residuals across repeated live runs before more geometry changes. If adding behavior rather than diagnostics, make it an explicit asymmetric-geometry experiment behind a separate control.

Decision: keep edge-dense as an experimental manual option only. The next code slice should improve diagnosis/reproducibility, not tune another global model wrapper.

## Repeat-Run Target Diagnostics

Date: 2026-05-15

Added `tools/analyze_repeat_run_diagnostics.py` to compare line-bounded validation runs from scalar telemetry. The tool uses the latest `validation_metrics.sample_count` window per target so settle-phase `validation_sample` telemetry does not pollute the target summaries.

Command used for the two edge-dense runs:

```bash
uv run python tools/analyze_repeat_run_diagnostics.py metrics/demo.jsonl --run 53454:59073 --run 59074:64477 --screen-width 5120 --screen-height 1440 --grid-columns 4 --grid-rows 3
```

Per-target comparison:

| Run | Target | Samples | Mean Error | Signed X | Signed Y | Grid Accuracy | Predicted Cells |
|---|---|---:|---:|---:|---:|---:|---|
| First edge-dense | `v0` | 38 | 386.13 px | -158.82 px | +349.67 px | 0.0% | r1c0=38 |
| First edge-dense | `v1` | 38 | 59.84 px | +36.25 px | +40.84 px | 100.0% | r0c3=38 |
| First edge-dense | `v2` | 38 | 77.61 px | +59.83 px | -1.62 px | 92.1% | r1c1=3, r1c2=35 |
| First edge-dense | `v3` | 38 | 119.09 px | +10.94 px | -117.02 px | 18.4% | r1c0=4, r1c1=19, r2c0=8, r2c1=7 |
| First edge-dense | `v4` | 38 | 248.29 px | -161.81 px | -186.37 px | 0.0% | r1c2=38 |
| Second edge-dense | `v0` | 38 | 471.49 px | -316.87 px | +348.44 px | 0.0% | r1c0=38 |
| Second edge-dense | `v1` | 38 | 363.80 px | +19.49 px | +361.86 px | 0.0% | r1c2=13, r1c3=25 |
| Second edge-dense | `v2` | 38 | 184.02 px | +28.42 px | +180.97 px | 94.7% | r1c1=2, r1c2=36 |
| Second edge-dense | `v3` | 38 | 167.58 px | -19.67 px | -164.91 px | 0.0% | r1c0=29, r1c1=9 |
| Second edge-dense | `v4` | 38 | 88.48 px | +45.35 px | -73.42 px | 100.0% | r2c3=38 |

First-vs-second flags:

| Target | Signed Y Δ | Grid Accuracy Δ | Flags |
|---|---:|---:|---|
| `v1` | +321.03 px | -100.0% | grid-collapse, signed-y-shift |
| `v0` | -1.23 px | +0.0% | - |
| `v2` | +182.59 px | +2.6% | signed-y-shift |
| `v3` | -47.90 px | -18.4% | - |
| `v4` | +112.96 px | +100.0% | grid-recovery, signed-y-shift |

Interpretation:

1. The repeated stable failure is `v0`: both runs predict the top-left target into `r1c0`, about one row too low, with signed Y around `+349 px`.
2. The unstable failure is the right side: `v1` collapses from perfect top-right classification to middle/right cells, while `v4` recovers from `0.0%` to `100.0%`.
3. The second run's larger global signed Y bias is mainly a target-specific upward shift in `v1`, `v2`, and `v4`, not a uniform offset.
4. This points toward run/setup or feature instability plus asymmetric model/geometry behavior. It does not justify another global Y correction wrapper.

Decision: keep the repeat-run diagnostics tool and use it after future manual runs. Next implementation should add calibration-side repeat-run diagnostics, especially per-target feature drift between `v1`/`v4` and the stable `v0` failure.

## Calibration-Side Repeat-Run Feature Drift

Date: 2026-05-15

Extended `tools/analyze_repeat_run_diagnostics.py` to include scalar `calibration_replay_sample` summaries by target and first-vs-second feature-mean deltas. The same edge-dense line ranges were analyzed:

```bash
uv run python tools/analyze_repeat_run_diagnostics.py metrics/demo.jsonl --run 53454:59073 --run 59074:64477 --screen-width 5120 --screen-height 1440 --grid-columns 4 --grid-rows 3
```

Top calibration feature-drift targets:

| Calibration target | Samples A | Samples B | Max feature mean delta | Dominant feature index / signal |
|---|---:|---:|---:|---|
| `bottom4` | 76 | 76 | `0.113405` | index `13`, eye-line slope / vertical agreement; index `9`, right iris eye-relative Y also moved `+0.109561` |
| `bottom2` | 77 | 77 | `0.103482` | index `7`, left iris eye-relative Y |
| `mid_right` | 76 | 77 | `0.096476` | index `9`, right iris eye-relative Y |
| `top0` | 76 | 77 | `0.076690` | index `15`, face center Y |
| `bottom3` | 77 | 76 | `0.070156` | index `15`, face center Y; index `13`, eye-line slope / vertical agreement also moved `-0.065237` |

Feature-index reference for dominant signals:

| Index | Feature |
|---:|---|
| `7` | left iris eye-relative Y |
| `9` | right iris eye-relative Y |
| `12` | binocular eye-relative Y midpoint |
| `13` | left/right vertical agreement / eye-line slope proxy |
| `15` | normalized face center Y |
| `20` | roll proxy |

Interpretation:

1. Calibration sample counts are balanced across repeated runs (`76`-`77` per target), so the drift is not explained by sparse capture.
2. The largest repeat-run movement is in eye-relative vertical geometry and face-center/head-position features, not in raw target counts or validation windowing.
3. The stable top-left validation failure (`v0`) still shows nearly unchanged signed Y error (`-1.23 px` delta) even while the top-left calibration target `top0` has measurable face-center-Y drift. That suggests `v0` is a persistent model/geometry mismatch rather than only a between-run operator shift.
4. The right-side validation instability (`v1` collapse, `v4` recovery) coincides with larger right/edge calibration drift (`mid_right`, `upper_right`, `bottom4`), especially feature index `9` and slope/face-center signals. That points to setup/head-pose sensitivity before another global Y correction.

Decision: keep the calibration-side drift diagnostics in the repeat-run tool. Do not promote edge-dense calibration or add another global correction from these two runs. Next higher-leverage slice should make feature-drift output easier to interpret by naming feature indices in the report, then compare a fresh controlled repeat run with stricter head/camera posture.

## Decision Gate

Keep the added features only if manual evidence improves at least one of these without a clear regression:

1. vertical feature separability for top/middle/bottom targets,
2. mean Y error,
3. signed Y bias stability,
4. practical `4x3` grid accuracy,
5. red window-border usefulness.

The first head-pose run cleared the first three gates partially and improved grid accuracy only slightly. The replay-enabled run improved live 4x3 grid accuracy to `40.0%`, but pixel error regressed to `256.20 px`. Grid-first offline replay showed corrected candidates can improve over their base models, and sample-window replay showed middle/late calibration samples could slightly beat the latest live grid result offline. The first late-window live validation did not confirm that signal: grid accuracy regressed to `30.0%`. Target residual analysis now shows vertical compression at top/bottom and edge/corner targets. Replay target weighting can slightly improve offline grid accuracy (`44.5%`) but leaves the top validation row unusable. Vertical-bias correction reduces global signed Y bias and reaches `43.3%` grid accuracy, but still leaves `v0` at `0.0%`. Three-band correction also leaves `v0`/`v1` at `0.0%` and does not beat the base early linear result. The first edge-dense live validation improved mean error to `181.13 px` and grid accuracy to `41.6%`, but `v0` and `v4` remained `0.0%`; the second edge-dense run regressed to `255.25 px` mean error and `38.9%` grid accuracy with large signed Y bias. Repeat-run diagnostics show stable top-left collapse into `r1c0` plus unstable right-side vertical shifts (`v1` collapse, `v4` recovery). Calibration-side repeat-run drift now shows balanced sample counts but material movement in eye-relative vertical, face-center-Y, and slope/roll-related signals at edge targets. Keep the features, evaluator corrections, and repeat-run diagnostic tooling for now; do not change the default sample window, add live weighting, promote edge-dense calibration, or add another global correction. Prioritize named feature-drift reporting and a fresh controlled repeat run before more geometry/model tuning.

If feature separability improves but validation remains compressed, investigate model form, target weighting, or calibration/validation sampling windows before adding heavier features.

If feature separability does not improve, do not add eyebrow/chin expression features as standalone inputs. Next heavier candidate would be a real head-pose estimate such as solvePnP with canonical 3D face landmarks.
