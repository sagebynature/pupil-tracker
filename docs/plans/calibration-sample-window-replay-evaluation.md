# Calibration Sample-Window Replay Evaluation Implementation Plan

> **For Hermes:** Use test-driven-development skill to implement this plan task-by-task.

**Goal:** Determine whether calibration sample timing/selection, not model form alone, is limiting 4x3 window-cell accuracy.

**Architecture:** Keep scalar replay telemetry unchanged. Extend the offline evaluator to filter calibration replay samples per target by capture window (`all`, `early`, `middle`, `late`) before fitting candidate models. Run those windows against the latest manual run, sorted by grid objective, and document whether a live calibration collector change is justified.

**Tech Stack:** Python 3.11, uv, pytest, ruff, ty, existing replay evaluator and calibration models.

---

## Task 1: Add Per-Target Sample-Window Filtering

**Objective:** Allow replay evaluation to fit models using only early/middle/late calibration samples per target.

**Files:**
- Modify: `tools/evaluate_calibration_models.py`
- Modify: `tests/test_evaluate_calibration_models.py`
- Modify: `README.md`
- Modify: `docs/manual-test-checklist.md`

**TDD:**
1. Add tests for a public `filter_calibration_samples_by_window(samples, window="middle")` helper.
2. Build synthetic samples with two target ids and six ordered samples per target.
3. Expected behavior:
   - `all` returns all samples unchanged.
   - `early` returns the first third per target.
   - `middle` returns the middle third per target.
   - `late` returns the final third per target.
   - each target retains at least one sample when possible.
4. Verify RED:
   ```bash
   uv run pytest tests/test_evaluate_calibration_models.py::test_filter_calibration_samples_by_window_keeps_same_window_per_target -v
   ```
5. Implement `SampleWindow = Literal["all", "early", "middle", "late"]`, helper function, and `--calibration-sample-window` CLI option defaulting to `all`.
6. Apply filtering inside `evaluate_replay_models` before fitting candidates.
7. Update README/checklist commands to show optional `--calibration-sample-window middle` for sampling diagnostics.
8. Verify:
   ```bash
   uv run ruff check tools/evaluate_calibration_models.py tests/test_evaluate_calibration_models.py
   uv run ty check tools/evaluate_calibration_models.py tests/test_evaluate_calibration_models.py
   uv run pytest tests/test_evaluate_calibration_models.py -v
   make check
   git diff --check
   ```
9. Commit and push:
   ```bash
   git add tools/evaluate_calibration_models.py tests/test_evaluate_calibration_models.py README.md docs/manual-test-checklist.md
   git commit -m "feat: evaluate calibration sample windows"
   git push origin HEAD
   ```

## Task 2: Replay Latest Manual Run Across Sample Windows

**Objective:** Compare all/early/middle/late sample windows on the latest `metrics/demo.jsonl` and document the next live-calibration decision.

**Files:**
- Modify: `docs/experiments/feature-diagnostics-comparison.md`

**Steps:**
1. Run grid-first evaluator for all four windows:
   ```bash
   for window in all early middle late; do
     uv run python tools/evaluate_calibration_models.py metrics/demo.jsonl \
       --screen-width 1512 \
       --screen-height 982 \
       --grid-columns 4 \
       --grid-rows 3 \
       --objective grid \
       --calibration-sample-window "$window"
   done
   ```
2. Record top candidate per window and compare against latest live validation grid accuracy (`40.0%`).
3. Decision rule:
   - If a filtered window beats live grid accuracy with acceptable pixel error, plan a live collector policy change.
   - If no window beats live grid accuracy, next move is calibration geometry/target placement or validation sampling, not model promotion.
4. Verify doc diff:
   ```bash
   git diff --check
   ```
5. Commit and push:
   ```bash
   git add docs/experiments/feature-diagnostics-comparison.md
   git commit -m "docs: record calibration sample-window evaluation"
   git push origin HEAD
   ```
