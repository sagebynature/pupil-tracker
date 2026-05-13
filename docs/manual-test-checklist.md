# Pupil Tracker MVP Manual Test Checklist

Use this checklist for live validation that intentionally touches local hardware, GUI windows, or macOS desktop state. Automated tests avoid these effects.

## Before Testing

- [ ] Run `make sync`.
- [ ] Run `make check`.
- [ ] Confirm the working tree only contains intentional changes.
- [ ] Place the webcam at the expected usage position.
- [ ] Use stable lighting; avoid strong backlight and glare on glasses.
- [ ] Confirm you are comfortable granting macOS camera permission to the Python/terminal process.
- [ ] Do not grant Accessibility permission; it is not needed for the MVP because focus control is intentionally disabled.

## Launch

- [ ] Run `make run-demo`.
- [ ] Confirm the app window opens with the title `Pupil Tracker Demo`.
- [ ] Confirm the UI shows camera controls, calibration controls, debug text, and telemetry controls.
- [ ] Confirm the camera is not active until Start Camera is clicked.

## Camera Preview

- [ ] Click Start Camera.
- [ ] If macOS prompts for camera permission, approve it for this manual test.
- [ ] Confirm the app does not crash after permission is granted.
- [ ] Confirm the preview state changes from stopped to running.
- [ ] Click Stop Camera.
- [ ] Confirm the preview state changes back to stopped.

## MediaPipe / Tracking Backend

- [ ] Provide a valid MediaPipe FaceLandmarker model asset path if the live backend requires one.
- [ ] Start camera/tracking in normal lighting.
- [ ] Confirm face/iris observations become valid when a face is centered.
- [ ] Cover the camera or turn away.
- [ ] Confirm invalid/low-confidence observations are reported rather than crashing.

## Calibration

- [ ] Start the 9-point calibration flow.
- [ ] Follow each target in row-major order from `r0c0` through `r2c2`.
- [ ] Confirm invalid observations do not advance the current target.
- [ ] Confirm valid observations eventually advance to the next target.
- [ ] Confirm calibration completion produces collected samples for all 9 targets.

## Gaze Overlay and Region Feedback

- [ ] Confirm the transparent overlay appears above other windows when gaze samples are available.
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
- [ ] Confirm telemetry contains no frame arrays, camera images, screenshots, video paths, or binary blobs.
- [ ] Confirm logging does not start until Start Logging is clicked.

## Shutdown

- [ ] Stop camera/tracking.
- [ ] Stop telemetry logging if active.
- [ ] Close the app.
- [ ] Confirm the app exits without hanging.

## Known Acceptable MVP Issues

- Webcam gaze may drift and may only be reliable at region/window level.
- Accuracy may degrade with glasses glare, poor lighting, or large head movement.
- Multi-monitor behavior may be incomplete.
- The UI is developer-oriented and not product-polished.
- Actual window focusing is intentionally absent.
