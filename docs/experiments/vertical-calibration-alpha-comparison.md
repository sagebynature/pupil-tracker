# Vertical Calibration Alpha Comparison

Date: 2026-05-14

## Purpose

We tested whether the desktop demo's weak vertical tracking was caused by ridge regularization compressing the fitted gaze range toward the screen center.

The practical goal is not perfect pixel tracking. The demo needs enough accuracy to support desktop use cases such as coarse region feedback and window selection.

## Method

1. Use explicit telemetry logging with `metrics/demo.jsonl`.
2. Run calibration and validation as a manual live loop:
   - **Start Camera**
   - **Start Logging**
   - run the selected calibration mode
   - **Start Validation**
   - wait for validation to complete
   - **Stop Logging**
3. Evaluate only emitted `validation_metrics` events.
4. Compare one variable at a time:
   - dense vertical polynomial calibration
   - linear vertical calibration with `alpha=1.0`
   - linear vertical calibration with `alpha=0.0`
5. Use per-axis and signed-bias metrics to separate vertical compression from vertical offset.

## Results

| Run | Mean Error | Mean X Error | Mean Y Error | Signed Y Bias | Interpretation |
|---|---:|---:|---:|---:|---|
| Dense vertical polynomial | 306.77 px | 214.15 px | 185.82 px | +5.71 px | Best overall so far; vertical bias mostly removed. |
| Linear vertical, `alpha=1.0` | 320.88 px | 107.30 px | 293.12 px | -2.20 px | Better horizontal error, but vertical range compression remained. |
| Linear vertical, `alpha=0.0` | 559.38 px | 109.02 px | 545.09 px | +545.09 px | Disabling regularization created a large downward vertical bias. |

## Decision

Revert the linear vertical mode to `alpha=1.0`.

The `alpha=0.0` experiment made vertical accuracy substantially worse while leaving horizontal error nearly unchanged. That rules out ridge shrinkage as the primary issue. The current evidence points more strongly at unstable or insufficient vertical feature signal than model regularization.

## Product-Oriented Evaluation Change

Pixel error is useful for debugging, but it is stricter than the desktop selection use case requires. We added a 3x3 grid-cell accuracy metric to validation so each sample also answers:

> Did the predicted gaze land in the same coarse screen cell as the validation target?

This better matches practical desktop behavior where the demo needs to identify the region or likely window, not necessarily land on the exact target pixel.

Validation telemetry now includes:

- `grid_cell_accuracy`
- `per_target_grid_cell_accuracy`

The debug label also reports grid accuracy after validation completes.

## Low-Lift UI Enhancement

The demo already enumerates visible macOS window candidates and logs the likely candidate under the gaze point. We added a low-lift visual affordance: the transparent gaze overlay draws a red border around the current candidate window.

This remains non-invasive:

- no focusing
- no raising
- no clicking
- no Accessibility permission requirement
- no mutation of desktop state

## Future Improvements

1. **Feature diagnostics:** log scalar summaries that show whether calibration and validation features separate top, middle, and bottom gaze targets.
2. **Vertical-sensitive features:** continue improving eye-relative iris position, aperture, binocular agreement, and face-pitch proxies.
3. **Window-selection metrics:** evaluate candidate-window hit rate when known windows are placed in target regions.
4. **Grid variants:** compare 3x3 versus denser practical grids only after 3x3 accuracy stabilizes.
5. **Smoothing:** add dwell/temporal stability before any future window-selection side effects.
6. **Manual protocol:** repeat metrics with stable camera angle, lighting, and posture before promoting a calibration mode to default.
