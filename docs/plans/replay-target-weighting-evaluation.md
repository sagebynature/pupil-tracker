# Replay Target Weighting Evaluation Plan

> **For Hermes:** Use test-driven-development skill. Keep this evaluator-only until replay and live evidence justify promotion.

## Goal

Test whether target weighting can reduce top/bottom vertical compression and improve practical 4x3 grid accuracy without changing live defaults.

## Context

The late-window live validation regressed to `30.0%` grid accuracy. Target residual analysis showed top targets predicted too low, bottom targets predicted too high, and edge/corner targets worse than center. This suggests mapping compression / calibration geometry coverage. Before adding landmarks or changing live defaults, evaluate weighted calibration candidates offline from the scalar replay log.

## Task 1 — Add target weighting policies

1. Add tests first in `tests/test_evaluate_calibration_models.py`.
2. Add a small public helper in `tools/evaluate_calibration_models.py` that returns expanded calibration samples for a weighting policy.
3. Supported policies:
   - `none`: unchanged samples.
   - `vertical_edges`: increase top and bottom calibration targets.
   - `screen_edges`: increase left/right/top/bottom edge targets.
   - `corners`: increase corner targets.
4. Implement weighting by expanding/duplicating calibration samples for evaluator-only candidate fitting. This avoids changing source model APIs or live behavior.
5. Keep replay payloads unchanged and scalar-only.

Verification:

```bash
uv run pytest tests/test_evaluate_calibration_models.py -v
uv run ruff check tools/evaluate_calibration_models.py tests/test_evaluate_calibration_models.py
uv run ty check tools/evaluate_calibration_models.py tests/test_evaluate_calibration_models.py
```

Commit:

```bash
git commit -m "feat: add replay target weighting policies"
```

## Task 2 — Add weighted evaluator candidates

1. Add evaluator-only weighted model candidates for the highest-signal model families:
   - linear-alpha-0.1
   - poly2-alpha-1.0
2. Candidate names should include the policy, for example:
   - `linear-alpha-0.1-weight-vertical_edges`
   - `poly2-alpha-1.0-weight-screen_edges`
3. Weighting must affect only the calibration samples passed into that candidate's `fit` call.
4. Residual reports should work for weighted candidates without changing output format.

Verification:

```bash
uv run python tools/evaluate_calibration_models.py metrics/demo.jsonl --screen-width 1512 --screen-height 982 --grid-columns 4 --grid-rows 3 --objective grid --calibration-sample-window all --include-target-residuals
make check
git diff --check
```

Commit:

```bash
git commit -m "feat: evaluate weighted replay candidates"
```

## Task 3 — Analyze latest run and document decision

1. Isolate latest run lines `48430`-`53453`.
2. Run weighted replay candidates across relevant calibration sample windows.
3. Document:
   - best unweighted candidate,
   - best weighted candidate,
   - whether weighting beats the previous live baseline (`40.0%`) and latest late-window live run (`30.0%`),
   - residual interpretation,
   - recommended next move.
4. Do not change live defaults in this task.

Verification:

```bash
git diff --check
```

Commit:

```bash
git commit -m "docs: record replay target weighting analysis"
```

## Non-Goals

- No live default changes.
- No new landmarks.
- No camera/UI automation.
- No changes to telemetry payload format.
