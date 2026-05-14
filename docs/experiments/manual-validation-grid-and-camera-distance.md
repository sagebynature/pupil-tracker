# Manual Validation: Grid Accuracy and Camera Distance

Date: 2026-05-14

## Purpose

We tested whether practical desktop gaze tracking improves when validation uses a product-shaped grid metric and when the user sits closer to the camera.

The product goal is coarse desktop/window selection, not pixel-perfect pointing. A run can have high radial error and still be useful if predicted gaze lands in the same practical screen cell or likely window. Conversely, good center accuracy is not enough if top/bottom screen regions collapse toward the middle.

## Method

Each run used explicit telemetry logging in `metrics/demo.jsonl`:

1. **Start Camera**
2. **Start Logging**
3. **Start Vertical Calibration** using the 15-point vertical pattern
4. **Start Validation**
5. Review the final `validation_metrics` event
6. Optionally move gaze across visible windows to check candidate-window feedback
7. **Stop Logging**

The validation grid was changed from a fixed `3x3` metric to a configurable grid. The desktop demo default is now `4x3` via:

- `PUPIL_TRACKER_VALIDATION_GRID_COLUMNS=4`
- `PUPIL_TRACKER_VALIDATION_GRID_ROWS=3`

## Results

| Run | Mean Error | Median Error | Max Error | Mean X | Mean Y | Signed Y | Grid | Grid Accuracy | Recommendation |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| Previous 4x3 baseline | 338.76 px | 209.03 px | 750.42 px | 145.14 px | 290.06 px | +146.96 px | 4x3 | 20.0% | retry |
| Closer to camera | 262.62 px | 262.50 px | 495.67 px | 124.92 px | 197.29 px | -17.23 px | 4x3 | 10.5% | retry |

Delta from the previous `4x3` baseline:

| Metric | Delta |
|---|---:|
| Mean error | -76.14 px |
| Max error | -254.75 px |
| Mean X error | -20.22 px |
| Mean Y error | -92.77 px |
| Signed Y bias | -164.18 px |
| Grid accuracy | -9.5 percentage points |

Compared with the prior best dense-polynomial run, the closer-camera run improved mean error, max error, and X error, but slightly worsened Y error:

| Metric | Closer-camera delta vs prior best dense-polynomial run |
|---|---:|
| Mean error | -44.15 px |
| Mean X error | -89.23 px |
| Mean Y error | +11.48 px |
| Signed Y bias | -22.94 px |
| Max error | -223.13 px |

## Per-Target Pattern in the Closer-Camera Run

| Target | Error | Signed Y | Grid Accuracy |
|---|---:|---:|---:|
| `v0` | 240.28 px | +238.07 px | 0.0% |
| `v1` | 344.51 px | +159.69 px | 10.5% |
| `v2` | 108.85 px | +52.40 px | 0.0% |
| `v3` | 229.51 px | -151.72 px | 42.1% |
| `v4` | 389.93 px | -384.59 px | 0.0% |

Top targets still tended to predict too low. Bottom targets still tended to predict too high. This is consistent with vertical compression toward the screen middle.

## Calibration Quality

Both recent runs had clean calibration capture:

- 15 calibration targets
- about 50–51 accepted samples per target
- 0 rejected samples
- mean confidence 1.0
- every target advanced

That rules out obvious sample dropout as the cause. It does not prove the features separate top/middle/bottom gaze well enough.

## Interpretation

Getting closer to the camera helped real pixel accuracy:

- lower mean error
- lower max error
- lower X error
- lower Y error
- signed vertical bias mostly removed

However, practical `4x3` grid accuracy stayed poor. Grid accuracy also remained sensitive to boundary placement and target geometry, so grid size should be treated as a decision metric, not a tuning knob.

The current evidence points to a feature/model limitation:

1. Detection confidence and accepted counts are clean.
2. Camera distance improves the signal.
3. Vertical extremes still collapse toward the middle.
4. Grid accuracy fails outside limited regions.

## Decisions

1. Keep the configurable grid metric. It exposed practical failure that radial error alone did not.
2. Use closer camera placement for future manual runs; it clearly improves pixel metrics.
3. Stop tuning grid dimensions until gaze estimates are more stable.
4. Add scalar feature-separability diagnostics next.

## Next Instrumentation

Add telemetry summaries that report, per calibration target:

- accepted sample count
- feature mean
- feature standard deviation
- target normalized x/y
- optional grouped top/middle/bottom deltas

Keep this telemetry scalar-only. Do not log frames, screenshots, full landmark dumps, or image payloads.

The next question is:

> Do the current features distinguish top/middle/bottom targets, or is the model fitting weak/ambiguous vertical signal?
