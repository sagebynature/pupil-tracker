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
- [ ] Confirm you have a MediaPipe FaceLandmarker `.task` model file for real tracker-backed calibration.

## Launch

For the full end-to-end path, launch with the model asset path:

```bash
PUPIL_TRACKER_MEDIAPIPE_MODEL=/absolute/path/to/face_landmarker.task make run-demo
```

- [ ] Confirm the app window opens with the title `Pupil Tracker Demo`.
- [ ] Confirm the UI shows camera controls, calibration controls, debug text, and telemetry controls.
- [ ] Confirm the camera is not active until Start Camera is clicked.
- [ ] If the model variable is missing, confirm Start Calibration shows setup guidance mentioning `PUPIL_TRACKER_MEDIAPIPE_MODEL` instead of crashing.

## End-to-End Happy Path

- [ ] Click Start Camera.
- [ ] If macOS prompts for camera permission, approve it for this manual test.
- [ ] Confirm a live camera preview appears.
- [ ] Center your face in the preview.
- [ ] Confirm tracker annotations appear when face/iris landmarks are detected.
- [ ] Click Start Calibration.
- [ ] Follow the visible calibration targets from `r0c0` through `r2c2`.
- [ ] Confirm the target advances only while observations are valid.
- [ ] Confirm calibration completes and debug text reports fit metrics.
- [ ] Move gaze across the screen.
- [ ] Confirm the transparent gaze overlay appears and tracks approximately.
- [ ] Confirm the debug text updates the 3x3 region plausibly.
- [ ] Open one or more visible app windows and confirm the debug text shows a plausible window candidate.
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
- [ ] Follow each target in row-major order from `r0c0` through `r2c2`.
- [ ] Confirm invalid observations do not advance the current target.
- [ ] Confirm valid observations eventually advance to the next target.
- [ ] Confirm calibration completion produces collected samples for all 9 targets.
- [ ] Confirm fit metrics include sample count, mean error, and max error.

## Gaze Overlay and Region Feedback

- [ ] Confirm the transparent overlay appears above other windows when valid gaze samples are available.
- [ ] Confirm the overlay does not intercept mouse input.
- [ ] Confirm high-confidence samples show a tighter halo than low-confidence samples.
- [ ] Confirm invalid samples hide or dim the cursor.
- [ ] Move gaze across the screen and confirm the 3x3 region readout changes plausibly.

## macOS Visible-Window Candidate

- [ ] Open two or more visible app windows.
- [ ] Move estimated gaze over one window.
- [ ] Confirm the likely app/window title shown in debug output is plausible.
- [ ] Overlap windows and confirm the frontmost/visible candidate is preferred.
- [ ] Confirm no app is focused, raised, clicked, or activated by gaze.

## Telemetry Privacy

- [ ] Confirm `metrics/` is ignored by git.
- [ ] Click Start Logging.
- [ ] Perform a short calibration/tracking interaction.
- [ ] Click Stop Logging.
- [ ] Open the generated JSONL file under `metrics/`.
- [ ] Confirm events are JSON objects with scalar payloads.
- [ ] Confirm telemetry includes expected event types such as `raw_observation`, `calibration_sample`, `gaze_sample`, and `window_candidate` when those actions occur.
- [ ] Confirm telemetry contains no frame arrays, camera images, screenshots, video paths, or binary blobs.
- [ ] Confirm logging does not start until Start Logging is clicked.

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
