# Replay Per-Band Correction Evaluation Plan

> **For Hermes:** Use test-driven-development skill. Keep this evaluator-only until replay and live evidence justify promotion.

## Goal

Test whether band-specific vertical correction can fix the remaining top-left/top-row collapse after global vertical-bias correction.

## Context

Recent evidence:

- Late-window live validation regressed to `30.0%` grid accuracy.
- Target weighting reached `44.5%` replay grid accuracy but left the top validation row unusable.
- Global vertical-bias correction reached `43.3%` replay grid accuracy and reduced signed Y bias, but still left `v0` at `0.0%` grid accuracy with very large positive signed Y error.

The remaining failure is target/region-specific, not a single global Y offset.

## Task 1 — Add per-band correction wrapper

1. Add tests first in `tests/test_evaluate_calibration_models.py`.
2. Add an evaluator-only wrapper in `tools/evaluate_calibration_models.py`.
3. The wrapper should:
   - fit the base model on calibration samples,
   - predict calibration samples,
   - bucket calibration residuals by predicted normalized Y band,
   - learn one mean Y residual per band,
   - apply the matching band's Y residual at prediction time,
   - leave X unchanged.
4. Use three bands by default: top, middle, bottom.
5. If a band has no calibration residuals, use the global mean Y residual as fallback.
6. Keep it separate from global vertical-bias correction for clear comparison.

Verification:

```bash
uv run pytest tests/test_evaluate_calibration_models.py::test_per_band_correction_applies_band_specific_y_residuals -v
uv run pytest tests/test_evaluate_calibration_models.py -v
uv run ruff check tools/evaluate_calibration_models.py tests/test_evaluate_calibration_models.py
uv run ty check tools/evaluate_calibration_models.py tests/test_evaluate_calibration_models.py
```

Commit:

```bash
git commit -m "feat: evaluate per-band correction candidates"
```

## Task 2 — Add per-band candidates

1. Add evaluator-only candidate names:
   - `linear-alpha-0.1-per-band-corrected`
   - `linear-alpha-1.0-per-band-corrected`
   - `poly2-alpha-1.0-per-band-corrected`
   - `poly2-alpha-10.0-per-band-corrected`
2. Do not combine per-band correction with target weighting in this slice.
3. Keep residual reporting unchanged.

Verification:

```bash
uv run python tools/evaluate_calibration_models.py metrics/demo.jsonl --screen-width 1512 --screen-height 982 --grid-columns 4 --grid-rows 3 --objective grid --calibration-sample-window all --include-target-residuals
make check
git diff --check
```

Commit:

```bash
git commit -m "feat: evaluate per-band correction candidates"
```

## Task 3 — Analyze latest run and document decision

1. Isolate latest run lines `48430`-`53453`.
2. Run replay evaluation across sample windows.
3. Compare:
   - best uncorrected/unweighted candidate,
   - best weighted candidate,
   - best global vertical-bias candidate,
   - best per-band candidate,
   - previous live baseline (`40.0%`),
   - latest late-window live run (`30.0%`).
4. Pay special attention to `v0`, `v1`, and `v4` residuals.
5. Do not change live defaults in this task.

Verification:

```bash
git diff --check
```

Commit:

```bash
git commit -m "docs: record replay per-band correction analysis"
```

## Non-Goals

- No live behavior changes.
- No target geometry changes yet.
- No new landmarks.
- No combined weighted + per-band candidates yet.
- No telemetry payload changes.
