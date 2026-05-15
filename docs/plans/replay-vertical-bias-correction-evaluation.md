# Replay Vertical-Bias Correction Evaluation Plan

> **For Hermes:** Use test-driven-development skill. Keep this evaluator-only until replay and live evidence justify promotion.

## Goal

Test whether explicit vertical-bias correction can fix the top-row / bottom-row compression pattern seen in replay without changing live defaults.

## Context

Recent replay analysis shows:

- Late-window live validation regressed to `30.0%` grid accuracy.
- Target residuals show vertical compression: top validation/calibration targets are predicted too low, bottom targets often too high.
- Replay target weighting reached `44.5%` grid accuracy, but left top validation targets `v0` and `v1` at `0.0%` grid accuracy.

Weighting shifted errors; it did not solve vertical bias. Next step is evaluator-only correction that learns a simple Y-axis residual relationship from calibration predictions.

## Task 1 — Add vertical-bias correction wrapper

1. Add tests first in `tests/test_evaluate_calibration_models.py`.
2. Add an evaluator-only wrapper in `tools/evaluate_calibration_models.py`.
3. The wrapper should:
   - fit the base model on calibration samples,
   - predict calibration samples,
   - learn a one-dimensional linear correction from predicted normalized Y to Y residual,
   - apply that Y correction at prediction time,
   - leave X unchanged.
4. Keep this separate from full affine correction so it is interpretable and specifically targets vertical compression.
5. Reject correction fitting with fewer than two valid calibration predictions.

Verification:

```bash
uv run pytest tests/test_evaluate_calibration_models.py::test_vertical_bias_correction_reduces_y_mapping_error -v
uv run pytest tests/test_evaluate_calibration_models.py -v
uv run ruff check tools/evaluate_calibration_models.py tests/test_evaluate_calibration_models.py
uv run ty check tools/evaluate_calibration_models.py tests/test_evaluate_calibration_models.py
```

Commit:

```bash
git commit -m "feat: evaluate vertical bias correction candidates"
```

## Task 2 — Add vertical-bias candidates

1. Add evaluator-only candidate names for likely families:
   - `linear-alpha-0.1-vertical-bias-corrected`
   - `linear-alpha-1.0-vertical-bias-corrected`
   - `poly2-alpha-1.0-vertical-bias-corrected`
   - `poly2-alpha-10.0-vertical-bias-corrected`
2. Do not combine vertical-bias correction with target weighting in this slice.
3. Ensure residual reporting works without changing output format.

Verification:

```bash
uv run python tools/evaluate_calibration_models.py metrics/demo.jsonl --screen-width 1512 --screen-height 982 --grid-columns 4 --grid-rows 3 --objective grid --calibration-sample-window all --include-target-residuals
make check
git diff --check
```

Commit:

```bash
git commit -m "feat: evaluate vertical bias correction candidates"
```

## Task 3 — Analyze latest run and document decision

1. Isolate latest run lines `48430`-`53453`.
2. Run replay evaluation across sample windows.
3. Compare:
   - best uncorrected/unweighted candidate,
   - best weighted candidate,
   - best vertical-bias corrected candidate,
   - previous live baseline (`40.0%`),
   - latest late-window live run (`30.0%`).
4. Document whether top validation row grid accuracy improves without bottom-row regression.
5. Do not change live defaults in this task.

Verification:

```bash
git diff --check
```

Commit:

```bash
git commit -m "docs: record replay vertical-bias correction analysis"
```

## Non-Goals

- No live default changes.
- No new landmarks.
- No target geometry changes yet.
- No weighted + vertical-bias combination candidates yet.
- No telemetry payload changes.
