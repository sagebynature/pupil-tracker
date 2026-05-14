# Research Note: Additional Facial Features for Webcam Gaze Tracking

Date: 2026-05-14

## Question

Should the demo capture additional facial features such as face angle, chin position, eyebrow position, or other facial landmarks to improve webcam gaze tracking?

## Short Answer

Yes, but prioritize features that directly model gaze geometry:

1. **Head pose / face angle** — high value.
2. **Face geometry / face box context** — high value.
3. **Eye crops or richer eye-region landmarks** — high value, but larger implementation lift.
4. **Temporal stability / sequence features** — medium to high value after static features are diagnosed.
5. **Chin / eyebrow landmarks** — potentially useful as proxies for head pose or expression, but lower priority than explicit head pose and eye geometry.

Do not add many landmarks blindly. First add scalar feature diagnostics so we can measure whether each candidate feature separates top/middle/bottom calibration targets.

## Evidence from Literature

### MPIIGaze: everyday laptop use needs head-pose and appearance robustness

- Paper: *MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation* / *Appearance-Based Gaze Estimation in the Wild*
- arXiv: `1711.09017`, `1504.02863`
- Finding relevant to us: webcam/laptop gaze estimation in natural use varies with appearance, illumination, continuous gaze, and head pose.
- Implication: a pure normalized iris-position vector is likely underpowered for a desktop webcam demo where head distance and pose change.

### GazeCapture / iTracker: face context matters, not just eyes

- Paper: *Eye Tracking for Everyone*
- arXiv: `1606.05814`
- Finding relevant to us: iTracker uses eye information plus face/face-grid context on commodity hardware.
- Implication: adding face box / head-location context is justified. The model needs to know where the face is relative to the camera and screen, not just where the iris sits inside the face.

### Multi-modal appearance and shape cues improve head-pose-independent gaze

- Paper: *Recurrent CNN for 3D Gaze Estimation using Appearance and Shape Cues*
- arXiv: `1805.03064`
- Finding relevant to us: combining face, eye regions, and face landmarks improved 3D gaze estimation over prior state of the art on EYEDIAP.
- Implication: face landmarks and eye regions are complementary. This supports adding shape cues and head-pose information before trying more calibration-model tweaks.

### Gaze360 and ETH-XGaze: head pose and distance are central failure axes

- Papers: *Gaze360: Physically Unconstrained Gaze Estimation in the Wild* and *ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation*
- arXiv: `1910.10088`, `2007.15837`
- Finding relevant to us: robust gaze estimation is evaluated across wide ranges of head pose, distance, and gaze direction.
- Implication: our manual finding that camera distance helps is expected. We need to represent head pose/distance explicitly or normalize against it.

### Face orientation and eye orientation carry different signals

- Paper: *360-Degree Gaze Estimation in the Wild Using Multiple Zoom Scales*
- arXiv: `2009.06924`
- Finding relevant to us: gaze can be inferred from both face orientation and eye orientation, with usefulness depending on image quality and context.
- Implication: for coarse desktop/window selection, face orientation may provide a strong low-frequency prior, while iris/eye features provide fine correction.

## Current Feature State

The current feature vector includes:

- iris centers normalized to face bounds
- midpoint between both irises
- iris positions relative to eye boxes
- left/right eye aperture normalized by face height
- binocular vertical midpoint
- left/right vertical agreement

This was a good first step. It still does not explicitly encode:

- head yaw / pitch / roll
- face box position and size on camera frame
- nose/chin/forehead geometry as a pose proxy
- eye corner slope / eyelid contour shape
- temporal trend or stability
- raw eye appearance/crops

## Feature Candidates

### 1. Head pose: yaw, pitch, roll

**Value:** High  
**Lift:** Medium  
**Risk:** Moderate calibration complexity

Head pitch is the most directly relevant candidate for vertical compression. If the user tilts the head or changes camera distance, iris positions can shift in ways that mimic gaze changes. Explicit head pose gives the calibration model a way to separate eye rotation from head motion.

Potential inputs:

- solvePnP-derived face pose from stable landmarks
- MediaPipe facial transformation matrix if available through the Tasks API
- lightweight geometric proxies if full 3D pose is not available

Recommended scalar features:

- `head_yaw`
- `head_pitch`
- `head_roll`
- `face_center_x`
- `face_center_y`
- `face_width`
- `face_height`

### 2. Face position and scale

**Value:** High  
**Lift:** Low  
**Risk:** Low

Our closer-camera run improved pixel metrics. That means distance/scale matters. Face bounding box size is a cheap proxy for distance. Face center is a cheap proxy for camera alignment.

Recommended scalar features:

- face center normalized to frame
- face width/height normalized to frame
- face aspect ratio
- inter-ocular distance normalized to frame

These features are low-risk because we already compute face bounds.

### 3. Eye-corner and eyelid geometry

**Value:** High  
**Lift:** Medium  
**Risk:** Medium

Eye-relative iris y-position is useful, but vertical gaze also changes eyelid contour and aperture. Upper/lower eyelid landmarks can help distinguish actual vertical gaze from head pitch or squinting.

Recommended scalar features:

- iris y relative to eye-corner line, not just eye bounding box
- upper/lower eyelid aperture per eye
- eye-corner slope per eye
- normalized iris-to-upper-lid and iris-to-lower-lid distances
- left/right aperture ratio

### 4. Nose, chin, and brow landmarks

**Value:** Medium as pose/expression proxies  
**Lift:** Low to medium  
**Risk:** Higher noise / expression confounding

Chin and eyebrow landmarks can help estimate head pose or expression, but they are not direct gaze signals. Eyebrows move with expression and can add noise. Chin position is useful mainly as part of a head-pose model.

Recommendation:

- Do not add eyebrow/chin as standalone calibration features yet.
- Use stable face landmarks if they support head pose estimation.
- Prefer nose bridge/tip, eye corners, mouth corners, and chin as a pose set rather than many unrelated facial-expression features.

### 5. Temporal features

**Value:** Medium now, high for window selection  
**Lift:** Medium  
**Risk:** Can hide model errors if added too early

Temporal smoothing can make the red window border more stable. It will not fix vertical calibration by itself. Add it after feature separability is measurable.

Recommended later features:

- dwell duration per candidate window
- short-window mean and variance of gaze point
- velocity / jump rejection
- confidence-weighted smoothing

## Recommendation

Add feature diagnostics first, then add features in this order:

1. **Face position and scale** — low lift, directly supported by the camera-distance experiment.
2. **Head pose or head-pose proxy** — especially pitch, yaw, and roll.
3. **Eye-corner/eyelid geometry** — improve vertical gaze separability.
4. **Temporal window-selection stability** — after static gaze estimates improve.
5. **Chin/brow landmarks only as part of pose/expression diagnostics**, not as standalone first-class features.

## Proposed Next Experiment

Before changing the model, log scalar summaries per calibration target:

- mean and standard deviation for each existing feature
- grouped top/middle/bottom feature means
- top-vs-center and bottom-vs-center deltas
- left-vs-right deltas
- face scale / face center if available

Then run one close-camera validation. If existing vertical features barely move between top/middle/bottom targets, add richer features. If they separate well but the model still compresses, revisit model form and calibration target weighting.

## Decision for This Project

It is worth capturing more facial features, but the immediate next slice should not be “add every landmark.” The disciplined path is:

1. instrument feature separability,
2. add cheap face position/scale features,
3. add head pose / pose proxies,
4. evaluate with the same `4x3` grid and per-axis validation metrics.
