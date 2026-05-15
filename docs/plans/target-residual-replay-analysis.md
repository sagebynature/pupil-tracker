# Target Residual Replay Analysis Implementation Plan

> **For Hermes:** Use test-driven-development skill. Implement one slice at a time, verify, commit, and push.

## Goal

Explain the late-window live regression by adding target-specific residual reporting to the scalar replay evaluator. The report must show whether calibration and validation failures concentrate by target, vertical band, or screen geometry before changing defaults again.

## Context

The latest live run with `PUPIL_TRACKER_CALIBRATION_SAMPLE_WINDOW=late` regressed grid accuracy from `40.0%` to `30.0%`. Validation failures were concentrated at `v0`, `v3`, and `v4`. Offline replay on the same run also showed unstable winner changes across sample windows. We need residual evidence, not another live-policy guess.

## Task 1 — Add target residual summaries to evaluator

1. Add tests first in `tests/test_evaluate_calibration_models.py` for a wished-for residual API.
2. Expected behavior:
   - summarize calibration residuals per calibration target for a fitted model,
   - summarize validation residuals per validation target for a fitted model,
   - include target id, target x/y, sample count, mean error, mean absolute x/y error, mean signed x/y error,
   - include grid-cell accuracy for validation summaries,
   - sort worst targets first by mean error unless otherwise specified.
3. Keep replay payloads unchanged and scalar-only.
4. Keep live demo behavior unchanged.

Verification:

```bash
uv run pytest tests/test_evaluate_calibration_models.py -v
uv run ruff check tools/evaluate_calibration_models.py tests/test_evaluate_calibration_models.py
uv run ty check tools/evaluate_calibration_models.py tests/test_evaluate_calibration_models.py
```

Commit:

```bash
git commit -m "feat: report replay residuals by target"
```

## Task 2 — Add CLI reporting option

1. Add a CLI option such as `--include-target-residuals`.
2. Default output should remain the compact model table.
3. With the option enabled, append residual tables for the top-ranked model:
   - calibration residuals,
   - validation residuals.
4. The output should remain Markdown so it can be pasted into `docs/experiments/feature-diagnostics-comparison.md`.

Verification:

```bash
uv run python tools/evaluate_calibration_models.py metrics/demo.jsonl --screen-width 1512 --screen-height 982 --grid-columns 4 --grid-rows 3 --objective grid --calibration-sample-window all --include-target-residuals
make check
git diff --check
```

Commit:

```bash
git commit -m "feat: add replay target residual report"
```

## Task 3 — Analyze latest run and document decision

1. Isolate the latest manual run or otherwise make clear which line range is being analyzed.
2. Run the residual report for the latest run.
3. Document:
   - top-ranked model and score,
   - worst calibration targets,
   - worst validation targets,
   - whether failures align with vertical bands or specific target geometry,
   - recommended next implementation move.
4. Do not change default model/window policy in this task.

Verification:

```bash
git diff --check
```

Commit:

```bash
git commit -m "docs: record target residual replay analysis"
```

## Non-Goals

- No new landmarks.
- No default model promotion.
- No default sample-window change.
- No camera/UI automation in tests.
