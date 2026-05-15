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

Extended `tools/analyze_repeat_run_diagnostics.py` to include scalar `calibration_replay_sample` summaries by target and first-vs-second feature-mean deltas. The report now names the top dominant feature changes per calibration target so index lookups are no longer required for the highest-drift signals. The same edge-dense line ranges were analyzed:

```bash
uv run python tools/analyze_repeat_run_diagnostics.py metrics/demo.jsonl --run 53454:59073 --run 59074:64477 --screen-width 5120 --screen-height 1440 --grid-columns 4 --grid-rows 3
```

Top calibration feature-drift targets:

| Calibration target | Samples A | Samples B | Max feature mean delta | Dominant named feature deltas |
|---|---:|---:|---:|---|
| `bottom4` | 76 | 76 | `0.113405` | left-right eye-relative Y delta `-0.113405`; right iris eye-relative Y `+0.109561`; face center Y `-0.069195` |
| `bottom2` | 77 | 77 | `0.103482` | left iris eye-relative Y `-0.103482`; eye-relative Y midpoint `-0.074209`; face center Y `-0.066988` |
| `mid_right` | 76 | 77 | `0.096476` | right iris eye-relative Y `+0.096476`; eye-relative Y midpoint `+0.078830`; left iris eye-relative Y `+0.061185` |
| `top0` | 76 | 77 | `0.076690` | face center Y `-0.076690`; left iris eye-relative X `+0.044296`; right iris eye-relative X `+0.034231` |
| `bottom3` | 77 | 76 | `0.070156` | face center Y `-0.070156`; left-right eye-relative Y delta `-0.065237`; roll proxy `-0.044543` |

Feature-index reference for commonly dominant signals:

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

Decision: keep the calibration-side drift diagnostics in the repeat-run tool. Do not promote edge-dense calibration or add another global correction from these two runs. Next higher-leverage slice should compare a fresh controlled repeat run with stricter head/camera posture, now that dominant feature names are visible directly in the report.

## Third Edge-Dense Live Validation and Asymmetric Replay Check

Date: 2026-05-15

A third logged edge-dense run spans lines `64478`-`69737`. The live validation metrics at line `69737` were:

| Metric | Third Edge-Dense |
|---|---:|
| Sample count | `190` |
| Mean error | `221.95 px` |
| Median error | `237.66 px` |
| Max error | `499.99 px` |
| Mean X error | `143.69 px` |
| Mean Y error | `138.07 px` |
| Signed Y bias | `-33.76 px` |
| 4x3 grid accuracy | `42.1%` |
| Recommendation | `retry` |

Third-run per-target validation behavior from the live metrics:

| Target | Mean Error | Signed Y | Grid Accuracy |
|---|---:|---:|---:|
| `v0` | `312.50 px` | `+216.12 px` | `0.0%` |
| `v1` | `260.15 px` | `+39.85 px` | `100.0%` |
| `v2` | `72.03 px` | `-1.56 px` | `100.0%` |
| `v3` | `128.94 px` | `-116.68 px` | `10.5%` |
| `v4` | `336.14 px` | `-306.54 px` | `0.0%` |

Added evaluator-only `*-asymmetric-corrected` candidates. These fit the base model, learn quadrant-specific calibration residuals from scalar replay data, and apply the matching top/bottom + left/right residual at prediction time. This is intentionally replay-only; no live calibration behavior changed.

Replay comparison across the three edge-dense ranges:

| Run | Window | Best candidate | Best grid | Best mean error | Top asymmetric candidate | Asym grid | Asym mean error |
|---|---|---|---:|---:|---|---:|---:|
| `53454:59073` | `all` | `linear-alpha-1.0-asymmetric-corrected` | `32.2%` | `622.34 px` | `linear-alpha-1.0-asymmetric-corrected` | `32.2%` | `622.34 px` |
| `53454:59073` | `middle` | `poly2-alpha-10.0-vertical-bias-corrected` | `46.6%` | `495.89 px` | `poly2-alpha-1.0-asymmetric-corrected` | `37.8%` | `495.61 px` |
| `59074:64477` | `all` | `poly2-alpha-0.1` | `28.7%` | `649.82 px` | `linear-alpha-1.0-asymmetric-corrected` | `19.2%` | `693.98 px` |
| `59074:64477` | `late` | `poly2-alpha-1.0-bias-corrected` | `37.2%` | `516.08 px` | `poly2-alpha-1.0-asymmetric-corrected` | `18.9%` | `585.66 px` |
| `64478:69737` | `all` | `poly2-alpha-1.0-affine-corrected` | `56.9%` | `417.56 px` | `poly2-alpha-1.0-asymmetric-corrected` | `47.5%` | `597.37 px` |
| `64478:69737` | `late` | `linear-alpha-0.1-weight-vertical_edges` | `48.7%` | `457.30 px` | `poly2-alpha-1.0-asymmetric-corrected` | `43.4%` | `458.29 px` |

Interpretation:

1. The third live edge-dense run improves grid accuracy slightly over the first two runs (`42.1%` live), but remains `retry` and still leaves `v0` and `v4` at `0.0%` grid accuracy.
2. The stable top-left failure persists across all three runs. Repeat-run diagnostics show `v0` remains mostly one row too low: `r1c0=38`, `r1c0=38`, then `r0c0=7` / `r1c0=31`.
3. Asymmetric quadrant correction does not produce a repeatable promotion candidate. It wins one replay/window by grid only while carrying extreme pixel error, loses badly on the second run, and trails the best replay candidates on the third run.
4. The third run's best replay result (`all` + `poly2-alpha-1.0-affine-corrected`, `56.9%`) is interesting but not live-promotable from a single noisy run, especially with unresolved `v0`/`v4` live failures.

Decision: keep the asymmetric replay candidates for comparison only. Do not promote edge-dense calibration, asymmetric correction, affine correction, target weighting, or a sample-window change to live behavior yet. The next high-leverage code slice should test a true geometry experiment for the stable top-left collapse, or add a controlled head/camera posture gate before more model wrappers.

## Top-Left Focus Geometry Experiment Added

Date: 2026-05-15

Added a new non-default `25-point top-left focus calibration` path after replay-only correction candidates failed to produce a repeatable promotion signal. The pattern keeps broad edge-dense coverage and adds a 3x3 local calibration cluster around the held-out top-left validation region `(0.25, 0.25)`:

| Cluster target ids | Coordinates |
|---|---|
| `tl_upper_left`, `tl_upper_mid`, `tl_upper_right` | `(0.18, 0.18)`, `(0.25, 0.18)`, `(0.32, 0.18)` |
| `tl_center_left`, `tl_center`, `tl_center_right` | `(0.18, 0.25)`, `(0.25, 0.25)`, `(0.32, 0.25)` |
| `tl_lower_left`, `tl_lower_mid`, `tl_lower_right` | `(0.18, 0.32)`, `(0.25, 0.32)`, `(0.32, 0.32)` |

Rationale:

1. `v0` remains the stable failure across three edge-dense runs, mostly collapsing from top-left into `r1c0`.
2. Global correction, target weighting, vertical-bias correction, per-band correction, and asymmetric quadrant correction did not solve the repeat-run failure.
3. The next testable hypothesis is local calibration geometry around the failing held-out region, not another model wrapper.

Decision: expose the pattern through a separate `Start Top-Left Focus Calibration` control only. Keep the default 9-point path and the 17-point edge-dense path unchanged. Require a fresh logged manual validation run and repeat-run comparison before any promotion decision.

## Top-Left Focus Run 1 Analysis

Date: 2026-05-15

Run range: `69738:90649`; calibration replay samples: `83533:89287`; validation replay/sample window: `89696:90648`; validation metrics line: `90649`.

Live validation metric summary: mean error `328.50 px`, median error `397.05 px`, max error `553.71 px`, recommendation `retry`.

Repeat-run comparison against the third edge-dense run (`64478:69737`) shows the local top-left geometry fixed the large `v0` vertical collapse but displaced the failure to other targets:

| Target | Edge-dense Run 3 grid | Top-left focus grid | Edge-dense signed Y | Top-left signed Y | Top-left predicted cells |
|---|---:|---:|---:|---:|---|
| `v0` | `0.0%` | `39.5%` | `+221.85 px` | `-16.18 px` | `r0c0=23`, `r0c1=15` |
| `v1` | `100.0%` | `0.0%` | `+39.31 px` | `+188.75 px` | `r1c3=38` |
| `v2` | `100.0%` | `100.0%` | `-2.14 px` | `-152.77 px` | `r1c2=38` |
| `v3` | `10.5%` | `0.0%` | `-114.89 px` | `-462.22 px` | `r1c1=38` |
| `v4` | `0.0%` | `0.0%` | `-303.77 px` | `-366.35 px` | `r1c3=38` |

Calibration target sequence confirmed all 25 intended targets in order: top edge row, 3x3 top-left cluster, broad upper/middle/lower anchors, then bottom edge row.

Offline replay still shows candidate instability rather than a clean promotion path. With `--calibration-sample-window all`, the best grid candidate was `linear-alpha-0.1-asymmetric-corrected` at `69.8%`, but with high mean error (`648.85 px`) and `retry`. The live-equivalent `linear-alpha-1.0` candidate was only `22.0%` grid accuracy over the full replay sample set, also `retry`.

Decision: do not promote the top-left focus geometry to default and do not add another model correction on this evidence. Keep the opt-in path because it produced a useful diagnostic signal: local geometry can correct `v0` vertical bias, but current scalar features/modeling are not stable enough to preserve the rest of the grid. Next high-leverage step is posture/head-pose gating or explicit pose normalization before collecting more calibration geometries.

## Opt-In Posture Stability Gate Added

Date: 2026-05-15

Added an experimental calibration-capture gate controlled by `PUPIL_TRACKER_POSTURE_STABILITY_MAX_DELTA`. The gate is off by default. When enabled, each target uses its first accepted sample as the per-target reference and rejects later samples if any head-pose proxy feature drifts beyond the configured threshold.

Selected feature indices:

| Feature index | Meaning |
|---:|---|
| `20` | roll proxy |
| `21` | yaw proxy |
| `22` | pitch proxy |

Rationale:

1. Repeat-run diagnostics showed balanced sample counts but material drift in eye-relative vertical, face-center, and roll/slope-related signals.
2. Geometry experiments can move the failure (`v0` improved under top-left focus) but do not preserve the full validation grid.
3. The next controlled variable should be within-target posture stability, not another global model wrapper or denser target layout.

Decision: keep this gate opt-in only. Compare the same calibration geometry with and without `PUPIL_TRACKER_POSTURE_STABILITY_MAX_DELTA=0.08`; judge it by validation grid accuracy, per-target signed Y, and whether failures move to other targets.

## Posture Stability Gate Run 1 Analysis

Date: 2026-05-15

Run range: `90650:96278`; calibration replay samples: `90877:94778`; validation replay/sample window: `95329:96277`; validation metrics line: `96278`.

The run used the same 17-point edge-dense geometry as the third edge-dense baseline (`64478:69737`) with the posture stability gate intended at `PUPIL_TRACKER_POSTURE_STABILITY_MAX_DELTA=0.08`.

Live validation metric summary:

| Metric | Third Edge-Dense Baseline | Posture Gate Run 1 | Delta |
|---|---:|---:|---:|
| Sample count | `190` | `190` | `0` |
| Mean error | `221.95 px` | `149.08 px` | `-72.87 px` |
| Median error | `237.66 px` | `148.17 px` | `-89.49 px` |
| Max error | `499.99 px` | `383.98 px` | `-116.01 px` |
| Mean X error | `143.69 px` | `57.92 px` | `-85.77 px` |
| Mean Y error | `138.07 px` | `120.96 px` | `-17.11 px` |
| Signed Y bias | `-33.76 px` | `-46.62 px` | `-12.86 px` |
| 4x3 grid accuracy | `42.1%` | `34.7%` | `-7.4 pp` |
| Recommendation | `retry` | `usable` | improved threshold verdict |

Repeat-run target comparison against the third edge-dense baseline:

| Target | Baseline Grid | Gate Run Grid | Baseline Signed Y | Gate Run Signed Y | Gate Run Predicted Cells |
|---|---:|---:|---:|---:|---|
| `v0` | `0.0%` | `5.3%` | `+221.85 px` | `+155.04 px` | `r0c0=8`, `r0c1=2`, `r1c0=17`, `r1c1=11` |
| `v1` | `100.0%` | `71.1%` | `+39.31 px` | `+18.51 px` | `r0c2=11`, `r0c3=27` |
| `v2` | `100.0%` | `92.1%` | `-2.14 px` | `+2.40 px` | `r1c1=3`, `r1c2=35` |
| `v3` | `10.5%` | `7.9%` | `-114.89 px` | `-158.33 px` | `r1c1=35`, `r2c1=3` |
| `v4` | `0.0%` | `0.0%` | `-303.77 px` | `-251.90 px` | `r1c2=21`, `r1c3=17` |

Calibration quality events reported all 17 edge-dense targets advanced with zero rejected capture samples (`rejected_count: 0`). A replay approximation of head-pose proxy drift against the first logged sample per target found no sample above `0.08`; max observed proxy drift was about `0.0546`. This means the `0.08` threshold did not materially exercise the stability gate in this run. Lower thresholds would be needed to test actual rejection behavior; replay approximation suggests `0.05` would touch only a few edge targets, while `0.03` would be much more aggressive.

Interpretation:

1. The run improved pixel metrics substantially and moved the validator recommendation from `retry` to `usable`.
2. The product metric regressed: `4x3` grid accuracy fell from `42.1%` to `34.7%`.
3. The original `v0` collapse improved but did not resolve; `v4` remained unusable, and `v1`/`v2` lost previously perfect grid accuracy.
4. Because no capture samples were rejected, this is not strong evidence that the posture gate itself improved the model. It is better treated as another controlled edge-dense repeat run plus evidence that `0.08` is too permissive for the current head-pose proxy scale.

Decision: do not promote the posture gate threshold or edge-dense calibration. Keep the gate opt-in. Next controlled posture experiment should use a stricter threshold, likely `0.05` first; calibration starts now emit a scalar `calibration_config` event so the active path/window/model/threshold can be verified from the log before analysis.

## Posture Stability Gate Run 2 Analysis

Date: 2026-05-15

Run range: `105352:110399`; calibration config line: `105352`; validation metrics line: `110399`.

The run used `calibration_path: edge_dense`, `calibration_sample_window: all`, `LinearRidgeCalibrationModel`, and `PUPIL_TRACKER_POSTURE_STABILITY_MAX_DELTA=0.05` as confirmed by the new `calibration_config` event.

Live validation metric summary:

| Metric | Third Edge-Dense Baseline | Posture Gate 0.08 | Posture Gate 0.05 |
|---|---:|---:|---:|
| Sample count | `190` | `190` | `190` |
| Mean error | `221.95 px` | `149.08 px` | `260.53 px` |
| Median error | `237.66 px` | `148.17 px` | `252.29 px` |
| Max error | `499.99 px` | `383.98 px` | `962.99 px` |
| Mean X error | `143.69 px` | `57.92 px` | `145.07 px` |
| Mean Y error | `138.07 px` | `120.96 px` | `186.41 px` |
| Signed Y bias | `-33.76 px` | `-46.62 px` | `-28.39 px` |
| 4x3 grid accuracy | `42.1%` | `34.7%` | `36.8%` |
| Recommendation | `retry` | `usable` | `retry` |

Per-target validation behavior for the `0.05` run:

| Target | Mean Error | Signed Y | Grid Accuracy |
|---|---:|---:|---:|
| `v0` | `387.72 px` | `+304.82 px` | `0.0%` |
| `v1` | `254.32 px` | `+52.84 px` | `94.7%` |
| `v2` | `80.52 px` | `+37.13 px` | `81.6%` |
| `v3` | `161.18 px` | `-155.50 px` | `7.9%` |
| `v4` | `418.92 px` | `-381.23 px` | `0.0%` |

Calibration quality events again reported all 17 targets advanced with `rejected_count: 0`. This makes the `0.05` run a failed gate experiment: the active config was correct, but the capture path still did not reject any accepted calibration samples, and the validation metrics regressed versus the edge-dense baseline.

One telemetry caveat surfaced during analysis: `calibration_replay_sample` currently includes valid observations logged during settle/capture flow, not just accepted calibration samples. After this run, live telemetry was extended so future `calibration_replay_sample` events include scalar capture-decision fields: `capture_phase`, `sample_accepted`, and `decision_reason`. This should make the next posture-gate run directly auditable without inferring accepted/rejected status from aggregate target quality alone.

Decision: do not promote `PUPIL_TRACKER_POSTURE_STABILITY_MAX_DELTA=0.05`. Do not ask for another manual run until the decision-aware telemetry build has passed automated checks and been launched; the next validation should confirm whether any future threshold actually rejects capture samples before judging grid accuracy.

## Posture Stability Gate Run 3 Analysis

Date: 2026-05-15

Run range: `110622:115687`; calibration config line: `110622`; validation metrics line: `115687`.

The run used the decision-aware telemetry build with `calibration_path: edge_dense`, `calibration_sample_window: all`, `LinearRidgeCalibrationModel`, and `PUPIL_TRACKER_POSTURE_STABILITY_MAX_DELTA=0.05`.

Live validation metric summary:

| Metric | Third Edge-Dense Baseline | Posture Gate 0.05 Run 2 | Posture Gate 0.05 Run 3 |
|---|---:|---:|---:|
| Sample count | `190` | `190` | `189` |
| Mean error | `221.95 px` | `260.53 px` | `222.19 px` |
| Median error | `237.66 px` | `252.29 px` | `200.57 px` |
| Max error | `499.99 px` | `962.99 px` | `686.80 px` |
| Mean X error | `143.69 px` | `145.07 px` | `119.18 px` |
| Mean Y error | `138.07 px` | `186.41 px` | `181.06 px` |
| Signed Y bias | `-33.76 px` | `-28.39 px` | `+48.00 px` |
| 4x3 grid accuracy | `42.1%` | `36.8%` | `18.0%` |
| Recommendation | `retry` | `retry` | `retry` |

Decision-aware calibration replay showed:

| Signal | Count |
|---|---:|
| `calibration_replay_sample` events | `1297` |
| `sample_accepted: true` | `864` |
| `sample_accepted: false` | `433` |
| `capture_phase: settling` | `432` |
| `capture_phase: capturing` | `864` |
| `capture_phase: complete` | `1` |
| Target-quality rejected samples | `0` |

The `sample_accepted: false` samples were settle/not-evaluated observations, not posture-gate rejections. Accepted capture samples were very stable under the selected head-pose proxy indices: the largest accepted within-target drift against the first accepted sample was only `0.0175`, and a replay check found zero hypothetical accepted-sample rejections even at `0.02`.

Per-target validation behavior for the decision-aware `0.05` run:

| Target | Mean Error | Signed Y | Grid Accuracy |
|---|---:|---:|---:|
| `v0` | `348.57 px` | `+290.04 px` | `0.0%` |
| `v1` | `255.81 px` | `+196.19 px` | `5.3%` |
| `v2` | `99.26 px` | `+83.40 px` | `78.9%` |
| `v3` | `219.94 px` | `-176.98 px` | `0.0%` |
| `v4` | `187.31 px` | `-158.57 px` | `5.3%` |

Interpretation:

1. The posture gate is now auditable, and the evidence is clear: `0.05` does not reject capture samples.
2. The captured head-pose proxy features were already stable during accepted capture windows, but validation grid accuracy still collapsed to `18.0%`.
3. The dominant failure is not within-target head-pose drift under indices `20`, `21`, and `22`; it is still target/feature/model mismatch across screen regions, especially top/bottom and edge validation targets.

Decision: stop spending manual runs on `PUPIL_TRACKER_POSTURE_STABILITY_MAX_DELTA=0.05`. Keep the gate opt-in for future experiments, but do not promote it. The next higher-leverage slice should be evaluator-only pose normalization or a stronger head-pose estimate, not another threshold-only run.

## Evaluator-Only Pose Normalization

Date: 2026-05-15

The replay evaluator now includes opt-in candidate models that linearly normalize non-pose scalar features against the cheap head-pose proxy dimensions (`20`, `21`, `22`) before fitting the base calibration model. This is evaluator-only; no live calibration or tracking path changed.

Run range: `110622:115687`; command used the latest decision-aware posture-gate run with `--objective grid`, `--calibration-sample-window all`, and target residuals.

Pose-normalized replay results:

| Model | Mean Error | Mean X | Mean Y | Signed Y | 4x3 Grid Accuracy | Recommendation |
|---|---:|---:|---:|---:|---:|---|
| `linear-alpha-1.0-pose-normalized` | 338.67 px | 299.65 px | 135.16 px | -48.88 px | 22.4% | retry |
| `poly2-alpha-1.0-pose-normalized` | 344.70 px | 300.62 px | 147.18 px | -121.19 px | 21.1% | retry |

Top replay candidate from the same run remained `poly2-alpha-0.1` at `214.32 px` mean error and `51.1%` grid accuracy. The pose-normalized candidates reduce neither grid collapse nor pixel error; they introduce very large X error and trail the best replay candidate by about 29-30 percentage points of grid accuracy.

Decision: keep pose normalization evaluator-only and do not promote it to live calibration. The cheap proxy residualization is not the lever. Move next to stronger head-pose estimation or geometry/target sampling diagnostics rather than another correction wrapper.

## Validation Grid-Cell Collapse Diagnostics

Date: 2026-05-15

The replay evaluator now includes `Predicted Cells` in target residual tables. This is scalar-only and report-only: it counts which coarse grid cells each target's predicted gaze samples land in, making row/column collapse visible without images, frames, or raw landmark dumps.

Decision-aware posture-gate Run 3 (`110622:115687`) live validation, analyzed with `5120x1440`, shows row-collapse directly:

| Target | Grid Accuracy | Signed X | Signed Y | Predicted Cells |
|---|---:|---:|---:|---|
| `v0` | `0.0%` | `-189.79 px` | `+293.50 px` | `r1c0=37` |
| `v1` | `2.7%` | `+35.57 px` | `+200.46 px` | `r1c3=30`, `r1c2=6`, `r2c3=5`, `r0c3=1` |
| `v2` | `83.8%` | `+36.69 px` | `+84.81 px` | `r1c2=31`, `r1c1=6` |
| `v3` | `0.0%` | `+127.75 px` | `-174.98 px` | `r1c1=37` |
| `v4` | `8.1%` | `+95.75 px` | `-154.72 px` | `r1c3=34`, `r2c3=3` |

The best replay candidate for the same extracted run remains `poly2-alpha-0.1` (`51.1%` grid, `214.32 px` mean error), but target residuals show it still fails important held-out regions:

| Target | Grid Accuracy | Signed X | Signed Y | Predicted Cells |
|---|---:|---:|---:|---|
| `v0` | `7.8%` | `+78.50 px` | `+216.01 px` | `r1c1=33`, `r1c0=13`, `r2c2=9`, `r0c1=5`, `r0c0=2`, `r2c1=2` |
| `v1` | `0.0%` | `-92.83 px` | `+216.99 px` | `r1c3=47`, `r1c0=6`, `r1c1=4`, `r2c2=4`, `r1c2=2` |
| `v2` | `82.8%` | `+92.14 px` | `+49.66 px` | `r1c2=53`, `r1c3=9`, `r1c1=2` |
| `v3` | `85.5%` | `+140.13 px` | `+32.05 px` | `r2c1=53`, `r1c2=8`, `r1c1=1` |
| `v4` | `79.7%` | `-43.55 px` | `+103.45 px` | `r2c3=51`, `r2c1=10`, `r2c2=3` |

Interpretation:

1. The live linear model is not just noisy at individual corners; it compresses top and bottom validation rows into the middle row (`r1*`) for `v0`, `v3`, and most of `v4`.
2. The best offline polynomial candidate recovers `v3`/`v4` and the center but still sends top validation targets down into `r1*`, especially `v0` and `v1`.
3. This confirms the next lever should be geometry/target sampling or a stronger geometric pose signal. Another correction wrapper is unlikely to solve the held-out top-row collapse repeatably.

Decision: keep the predicted-cell diagnostic in evaluator reports and use it as the gate for the next experiment. Do not promote `poly2-alpha-0.1` live from this replay alone because it still leaves top-row held-out targets collapsed.

## Top-Row Focus Live Validation

Date: 2026-05-15

A logged manual run used the new opt-in `Start Top-Row Focus Calibration` path. This path is experimental and non-default; it adds 33 targets with local 3x3 clusters around both held-out top validation regions.

Run evidence:

- `calibration_config`: line `115925`
- `validation_metrics`: line `126151`
- run range analyzed: `115925:126151`
- calibration path: `top_row_focus`
- model: `LinearRidgeCalibrationModel`
- posture gate: disabled
- screen: `5120x1440`
- validation grid: `4x3`

Live validation metrics:

| Metric | Value |
|---|---:|
| Sample count | `190` |
| Mean error | `168.03 px` |
| Median error | `177.59 px` |
| Max error | `378.96 px` |
| Mean X error | `113.81 px` |
| Mean Y error | `89.15 px` |
| Signed Y bias | `+62.35 px` |
| 4x3 grid accuracy | `52.1%` |
| Recommendation | `usable` |

Per-target accepted validation behavior:

| Target | Mean Error | Signed X | Signed Y | Grid Accuracy | Predicted Cells |
|---|---:|---:|---:|---:|---|
| `v0` | `242.67 px` | `+25.43 px` | `+236.86 px` | `0.0%` | `r1c1=23`, `r1c0=15` |
| `v1` | `221.82 px` | `+184.15 px` | `+109.59 px` | `60.5%` | `r0c3=23`, `r1c3=15` |
| `v2` | `174.31 px` | `+172.96 px` | `+16.78 px` | `100.0%` | `r1c2=38` |
| `v3` | `63.44 px` | `-26.67 px` | `-43.52 px` | `28.9%` | `r2c0=27`, `r2c1=11` |
| `v4` | `131.79 px` | `+32.48 px` | `-6.11 px` | `65.8%` | `r2c3=25`, `r2c2=13` |

Compared against the previous decision-aware edge-dense run (`110622:115687`), top-row focus improved practical grid accuracy from `18.0%` to `52.1%` and reduced mean error from `222.19 px` to `168.03 px`. It also recovered `v1`, `v2`, and `v4` materially. The remaining hard failure is still `v0`, which continues to collapse from the top row into `r1c*` despite the new local top-left and top-right target clusters.

Replay-only evaluator on the isolated top-row run ranked `linear-alpha-1.0-vertical-bias-corrected` first by grid objective at `52.2%`, effectively matching live grid accuracy but with much worse pixel error (`528.90 px`) and a `retry` recommendation. That is not a promotion candidate; it confirms only that replay grid scoring sees the same practical ceiling.

Decision: keep top-row focus as an experimental manual path. It is the best live grid result so far, but `v0` remains `0.0%` and still collapses into the second row, so do not change the default 9-point calibration or live model. Next slice should explain why the top-left held-out region remains mis-mapped even after local geometry: inspect scalar feature separability for the top-left cluster versus `v0`, or add a stronger head-pose/geometry diagnostic before another calibration pattern.

## Top-Left Separability Diagnostic

Date: 2026-05-15

Command:

```bash
uv run python tools/analyze_top_left_separability.py metrics/demo.jsonl \
  --run 115925:126151 \
  --screen-width 5120 \
  --screen-height 1440 \
  --grid-columns 4 \
  --grid-rows 3
```

The diagnostic compares accepted scalar `calibration_replay_sample` rows inside the top-left local cluster around `(0.25, 0.25)` against the accepted `v0` validation replay window selected by the latest `validation_metrics` event.

| Metric | Value |
|---|---:|
| Top-left calibration cluster samples | `456` |
| Calibration targets | `tl_*` 3x3 cluster |
| `v0` validation samples | `38` |
| `v0` validation grid accuracy | `0.0%` |
| `v0` predicted cells | `r1c1=23`, `r1c0=15` |
| Separability assessment | `separable` |

Dominant signed feature shifts from the calibration cluster to `v0` validation:

| Feature | Calibration Mean | Validation Mean | Signed Δ | Normalized Δ |
|---|---:|---:|---:|---:|
| `18` face aspect ratio | `0.868917` | `0.887134` | `+0.018216` | `+3.10` |
| `22` pitch proxy | `0.267912` | `0.287717` | `+0.019805` | `+2.90` |
| `16` face width | `0.232940` | `0.237135` | `+0.004195` | `+2.56` |
| `15` face center Y | `0.511070` | `0.530978` | `+0.019907` | `+2.52` |
| `21` yaw proxy | `0.097451` | `0.063445` | `-0.034006` | `-2.40` |
| `19` interocular distance | `0.097223` | `0.101460` | `+0.004237` | `+2.17` |
| `20` roll proxy | `-0.067229` | `-0.036458` | `+0.030771` | `+1.61` |
| `12` eye-relative Y midpoint | `0.382197` | `0.426951` | `+0.044754` | `+1.46` |

Interpretation: the top-left held-out failure is not because scalar features are completely overlapping. The largest separation is dominated by face geometry and cheap pose proxies, especially aspect ratio, pitch proxy, face width/center Y, yaw proxy, and interocular distance. That points away from adding another calibration target pattern and toward model/geometry handling: the current linear live model is not using these separable posture/geometry shifts to keep `v0` in the top row.

Decision: keep top-row focus opt-in. Do not add more calibration geometry next. The next implementation slice should be evaluator-only: either test local/top-left-aware model candidates that can use the separable scalar signal without moving `v3`/`v4`, or add stronger head-pose features such as solvePnP-style scalar pose before live promotion.

## Decision Gate

Keep the added features only if manual evidence improves at least one of these without a clear regression:

1. vertical feature separability for top/middle/bottom targets,
2. mean Y error,
3. signed Y bias stability,
4. practical `4x3` grid accuracy,
5. red window-border usefulness.

The first head-pose run cleared the first three gates partially and improved grid accuracy only slightly. The replay-enabled run improved live 4x3 grid accuracy to `40.0%`, but pixel error regressed to `256.20 px`. Grid-first offline replay showed corrected candidates can improve over their base models, and sample-window replay showed middle/late calibration samples could slightly beat the latest live grid result offline. The first late-window live validation did not confirm that signal: grid accuracy regressed to `30.0%`. Target residual analysis now shows vertical compression at top/bottom and edge/corner targets. Replay target weighting can slightly improve offline grid accuracy (`44.5%`) but leaves the top validation row unusable. Vertical-bias correction reduces global signed Y bias and reaches `43.3%` grid accuracy, but still leaves `v0` at `0.0%`. Three-band correction also leaves `v0`/`v1` at `0.0%` and does not beat the base early linear result. The first edge-dense live validation improved mean error to `181.13 px` and grid accuracy to `41.6%`, but `v0` and `v4` remained `0.0%`; the second edge-dense run regressed to `255.25 px` mean error and `38.9%` grid accuracy with large signed Y bias. Repeat-run diagnostics show stable top-left collapse into `r1c0` plus unstable right-side vertical shifts (`v1` collapse, `v4` recovery). Calibration-side repeat-run drift now shows balanced sample counts but material movement in eye-relative vertical, face-center-Y, and slope/roll-related signals at edge targets; the report names the dominant drift features directly. The top-left focus geometry corrected `v0` signed Y bias in one run but regressed `v1`, `v3`, and overall live mean error to `328.50 px`, so it is diagnostic only. The first top-row focus run is the best live grid result so far (`52.1%`, `168.03 px`, `usable`) and recovered `v1`/`v4`, but `v0` remains `0.0%` and collapses into `r1c*`, so it is still diagnostic rather than a default-change signal. The first posture-gated edge-dense run improved pixel error to `149.08 px` and reached a `usable` recommendation, but grid accuracy regressed to `34.7%` and no capture samples were rejected at threshold `0.08`; treat that threshold as too permissive, not as validated gating. Keep the features, evaluator corrections, repeat-run diagnostic tooling, edge-dense path, top-left focus path, top-row focus path, and opt-in posture gate for now; do not change the default sample window, add live weighting, promote experimental geometry, promote a posture threshold, or add another global correction. Prioritize scalar feature-separability diagnostics for the top-left cluster versus `v0`, or move to a stronger head-pose/geometric pose estimate before adding more calibration targets.

If feature separability improves but validation remains compressed, investigate model form, target weighting, or calibration/validation sampling windows before adding heavier features.

If feature separability does not improve, do not add eyebrow/chin expression features as standalone inputs. Next heavier candidate would be a real head-pose estimate such as solvePnP with canonical 3D face landmarks.
