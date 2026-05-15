# Pupil Tracker MVP Manual Test Checklist

Use this checklist for live validation that intentionally touches local hardware, GUI windows, or macOS desktop state. Automated tests avoid these effects.

## Before Testing

- [ ] Run `make sync`.
- [ ] Run `make check`.
- [ ] Confirm the working tree only contains intentional changes.
- [ ] Place the webcam at the expected usage position, centered above or below the active display.
- [ ] Use stable lighting; avoid strong backlight and glare on glasses.
- [ ] Confirm you are comfortable granting macOS camera permission to the Python/terminal process.
- [ ] Do not grant Accessibility permission; it is not needed for the MVP because focus control is intentionally disabled.
- [ ] Confirm you have a MediaPipe FaceLandmarker `.task` model file for real tracker-backed calibration. To download the default asset, run `make download-model`.

## Launch

For the full end-to-end path, launch with the model asset path:

```bash
make download-model
PUPIL_TRACKER_MEDIAPIPE_MODEL=$(pwd)/models/face_landmarker.task make run-demo
```

You can also launch with another compatible `.task` file:

```bash
PUPIL_TRACKER_MEDIAPIPE_MODEL=/absolute/path/to/face_landmarker.task make run-demo
```

- [ ] Confirm the app window opens with the title `Pupil Tracker Demo`.
- [ ] Confirm the UI shows camera controls, calibration/validation controls, heatmap controls, debug text, and telemetry controls.
- [ ] Confirm the camera is not active until Start Camera is clicked.
- [ ] If the model variable is missing, confirm Start Calibration shows setup guidance mentioning `PUPIL_TRACKER_MEDIAPIPE_MODEL` instead of crashing.

## End-to-End Happy Path

- [ ] Click Start Camera.
- [ ] If macOS prompts for camera permission, approve it for this manual test.
- [ ] Confirm a live camera preview appears.
- [ ] Center your face in the preview.
- [ ] Confirm tracker annotations appear when face/iris landmarks are detected.
- [ ] Click Start Calibration.
- [ ] Follow the fullscreen calibration targets from `r0c0` through `r2c2`.
- [ ] Confirm each target shows a Settle phase before samples are counted.
- [ ] Confirm each target shows a Capture phase with accepted/rejected sample counts.
- [ ] Confirm low-confidence, lost-face, or missing-feature observations do not advance the target.
- [ ] Confirm calibration completes and debug text reports fit metrics.
- [ ] Click Start Validation.
- [ ] Look at each validation target until validation completes.
- [ ] Confirm the validation overlay shows target dot, predicted dot, and error line.
- [ ] Confirm validation metrics report mean error, mean X error, mean Y error, signed Y bias, configured grid accuracy (default `4x3`), max error, and a recommendation.
- [ ] If validation recommends `retry`, recalibrate after improving lighting, camera angle, or head stability.
- [ ] Move gaze across the screen.
- [ ] Confirm the transparent gaze overlay appears and tracks approximately.
- [ ] Enable Show Heatmap, stare at fixed points, and confirm the heatmap clusters near those points.
- [ ] Clear Heatmap and confirm old clusters disappear.
- [ ] Confirm the debug text updates the 3x3 region plausibly.
- [ ] Open one or more visible app windows and confirm the debug text shows a plausible window candidate.
- [ ] Confirm the transparent overlay draws a red border around the current candidate window.
- [ ] Confirm no app is focused, raised, clicked, or activated by gaze.

## Camera Preview

- [ ] Click Start Camera.
- [ ] Confirm the preview state changes from stopped to running.
- [ ] Click Stop Camera.
- [ ] Confirm the preview state changes back to stopped.
- [ ] Confirm the camera is released after Stop Camera.
- [ ] Start Camera again and confirm preview can restart cleanly.

## MediaPipe / Tracking Backend

- [ ] Confirm `PUPIL_TRACKER_MEDIAPIPE_MODEL` points to an existing `.task` file.
- [ ] Start camera/tracking in normal lighting.
- [ ] Confirm face/iris observations become valid when a face is centered.
- [ ] Cover the camera or turn away.
- [ ] Confirm invalid/low-confidence observations are reported rather than crashing.
- [ ] Confirm the camera preview remains usable even if tracker setup guidance is shown.

## Calibration

- [ ] Start the 9-point calibration flow.
- [ ] Confirm the target display covers the primary monitor, not just the embedded control panel.
- [ ] Confirm outer targets remain inset from the physical edges so they are stable to fixate.
- [ ] Follow each target in row-major order from `r0c0` through `r2c2`.
- [ ] During Settle, confirm samples are not counted yet.
- [ ] During Capture, confirm valid/confident observations increase the accepted count.
- [ ] Confirm invalid observations increase rejection feedback or leave the target unadvanced.
- [ ] Confirm low-quality targets are retried rather than silently accepted.
- [ ] Confirm calibration completion produces collected samples for all 9 targets.
- [ ] Confirm fit metrics include sample count, mean error, and max error.

## Validation

- [ ] Click Start Validation after calibration completes.
- [ ] Look directly at each validation target and keep your head stable.
- [ ] Confirm the target dot remains visible while validation samples are collected.
- [ ] Confirm the predicted gaze dot appears when calibrated gaze is valid.
- [ ] Confirm the error line connects the target dot to the predicted dot.
- [ ] Confirm final metrics include mean error, mean X error, mean Y error, signed Y bias, configured grid accuracy (default `4x3`), max error, and a recommendation.
- [ ] Compare mean X error and mean Y error; if mean Y is much worse, continue with the vertical checks below before tuning the model.
- [ ] Check signed Y bias: consistent positive/negative bias suggests camera angle, head pitch, or posture offset; high mean Y with near-zero bias suggests vertical noise/compression.
- [ ] Treat `excellent` and `good` as useful for continued testing.
- [ ] Treat `usable` as acceptable only for coarse region/window experiments.
- [ ] Treat `retry` as a failed calibration; improve conditions and recalibrate.

## Gaze Overlay, Trail, Heatmap, and Region Feedback

- [ ] Confirm the transparent overlay appears above other windows when valid gaze samples are available.
- [ ] Confirm the overlay does not intercept mouse input.
- [ ] Confirm high-confidence samples show a tighter halo than low-confidence samples.
- [ ] Confirm invalid samples hide or dim the cursor.
- [ ] Move gaze across the screen and confirm the trail follows recent gaze history.
- [ ] Enable Show Heatmap and stare at the center of the screen for several seconds.
- [ ] Confirm the heatmap intensifies near the stared-at point.
- [ ] Stare at a second fixed point and confirm a second cluster appears.
- [ ] Click Clear Heatmap and confirm accumulated cells disappear.
- [ ] Confirm the 3x3 region readout changes plausibly.

## macOS Visible-Window Candidate

- [ ] Open two or more visible app windows.
- [ ] Move estimated gaze over one window.
- [ ] Confirm the likely app/window title shown in debug output is plausible.
- [ ] Confirm a red border appears around the likely candidate window while valid gaze is over it.
- [ ] Overlap windows and confirm the frontmost/visible candidate is preferred.
- [ ] Confirm no app is focused, raised, clicked, or activated by gaze.

## Telemetry Privacy

- [ ] Confirm `metrics/` is ignored by git.
- [ ] Click Start Logging.
- [ ] Perform a short calibration/tracking interaction.
- [ ] Click Stop Logging.
- [ ] Open the generated JSONL file under `metrics/`.
- [ ] Confirm events are JSON objects with scalar payloads.
- [ ] Confirm telemetry includes expected event types such as `raw_observation`, `calibration_sample`, `calibration_replay_sample`, `calibration_target_quality`, `calibration_feature_diagnostics`, `validation_sample`, `validation_replay_sample`, `validation_metrics`, `gaze_sample`, and `window_candidate` when those actions occur.
- [ ] Run `uv run python tools/analyze_feature_diagnostics.py metrics/demo.jsonl` and confirm it reports the latest diagnostics line, feature count, target means/stds, and top/bottom-vs-center deltas.
- [ ] Run `uv run python tools/evaluate_calibration_models.py metrics/demo.jsonl --screen-width 1512 --screen-height 982 --grid-columns 4 --grid-rows 3 --objective grid --calibration-sample-window middle` with the screen dimensions from the manual run and confirm it ranks candidate calibration models. Use `--calibration-sample-window all|early|middle|late` to compare capture timing.
- [ ] Confirm telemetry contains no frame arrays, camera images, screenshots, video paths, or binary blobs.
- [ ] Confirm logging does not start until Start Logging is clicked.

## Calibration Troubleshooting Notes

If calibration feels wrong, record these details before changing code:

- Lighting conditions and whether glasses/glare were present.
- Webcam position relative to the active display.
- Whether the face stayed centered during Settle and Capture.
- Which targets retried or felt unstable.
- Validation mean/max error and recommendation.
- Validation mean X error, mean Y error, and signed Y bias.
- Whether the error line is vertically biased in one direction or random.
- Whether weak vertical tracking looks like consistent bias, vertical compression toward center, or jitter.
- Whether heatmap clusters are stable or drifting.

## Shutdown and Resource Cleanup

- [ ] Stop camera/tracking.
- [ ] Confirm the overlay disappears.
- [ ] Confirm telemetry logging stops if it was active.
- [ ] Close the app.
- [ ] Confirm the app exits without hanging.
- [ ] Confirm the camera LED turns off.
- [ ] Relaunch the demo and confirm the camera can be opened again.

## Known Acceptable MVP Issues

- Webcam gaze may drift and may only be reliable at region/window level.
- Accuracy may degrade with glasses glare, poor lighting, or large head movement.
- Multi-monitor behavior may be incomplete.
- The UI is developer-oriented and not product-polished.
- Actual window focusing is intentionally absent.
