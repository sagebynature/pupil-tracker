# Session Checkpoint — 2026-05-15

Captured: 2026-05-15 13:59:32 EDT

## Repository State

- Branch: `main`
- Remote sync: `main...origin/main` clean
- Latest commit: `9452ace feat: add replay pose-normalized candidates`
- Full verification before checkpoint: `make check` passed with `325 passed`; `ruff` and `ty` passed; `git diff --check` passed.

## Completed Slice

Added evaluator-only pose-normalized replay candidates:

- `linear-alpha-1.0-pose-normalized`
- `poly2-alpha-1.0-pose-normalized`

Changed files in committed slice:

- `tools/evaluate_calibration_models.py`
- `tests/test_evaluate_calibration_models.py`
- `docs/experiments/feature-diagnostics-comparison.md`
- `docs/manual-test-checklist.md`

## Replay Evidence

Evaluation used the decision-aware posture-gate run range `110622:115687`.

| Model | Mean Error | Mean X | Mean Y | Signed Y | 4x3 Grid Accuracy | Decision |
|---|---:|---:|---:|---:|---:|---|
| `linear-alpha-1.0-pose-normalized` | 338.67 px | 299.65 px | 135.16 px | -48.88 px | 22.4% | Do not promote |
| `poly2-alpha-1.0-pose-normalized` | 344.70 px | 300.62 px | 147.18 px | -121.19 px | 21.1% | Do not promote |
| Best same-run replay candidate, `poly2-alpha-0.1` | 214.32 px | 129.90 px | 137.48 px | +123.92 px | 51.1% | Still evaluator-only |

Decision: cheap head-pose proxy residualization is not the lever. Keep pose normalization evaluator-only.

## Current Task State

All active todos are complete:

1. Inspect replay evaluator/model candidate structure and existing tests.
2. Add RED tests for evaluator-only pose-normalized replay candidate.
3. Implement minimal pose normalization candidate in evaluator.
4. Run focused/full checks, document outcome, commit and push.

## Recommended Resume Point

Next high-leverage step: stop adding post-fit correction wrappers and choose one of these paths:

1. Stronger head-pose estimation: build an evaluator-only solvePnP/canonical-3D-face-landmark experiment if sufficient scalar landmarks can be logged without images, or first add scalar-only telemetry for the required points.
2. Geometry/target sampling diagnostics: continue using existing scalar replay data to test calibration/validation geometry and target placement before touching live defaults.

Do not run another `PUPIL_TRACKER_POSTURE_STABILITY_MAX_DELTA=0.05` manual threshold test; decision-aware telemetry showed it rejected zero capture samples and did not improve grid accuracy.
