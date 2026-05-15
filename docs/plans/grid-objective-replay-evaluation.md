# Grid-Objective Replay Evaluation Implementation Plan

> **For Hermes:** Use test-driven-development skill to implement this plan task-by-task.

**Goal:** Make replay evaluation optimize the product objective — coarse 4x3 window-cell accuracy — before promoting any live calibration model changes.

**Architecture:** Keep the existing scalar replay pipeline. Extend the offline evaluator with objective-aware sorting and post-fit correction candidates that can be measured against the same replay samples without touching live camera code first. Only promote a live model after the evaluator shows a clear grid-cell win without unacceptable pixel-error regression.

**Tech Stack:** Python 3.11, uv, pytest, ruff, ty, existing ridge calibration models, JSONL replay telemetry.

---

## Task 1: Add Grid-First Objective Sorting

**Objective:** Make `tools/evaluate_calibration_models.py` rank by grid-cell accuracy first when requested, while preserving error-first sorting as an explicit option.

**Files:**
- Modify: `tools/evaluate_calibration_models.py`
- Modify: `tests/test_evaluate_calibration_models.py`
- Modify: `README.md`
- Modify: `docs/manual-test-checklist.md`

**TDD:**
1. Add a test for a public `sort_model_results(results, objective="grid")` helper.
2. Expected grid objective order: highest `grid_cell_accuracy`, then lowest `mean_error_px`.
3. Expected error objective order: lowest `mean_error_px`.
4. Verify RED:
   ```bash
   uv run pytest tests/test_evaluate_calibration_models.py::test_sort_model_results_supports_grid_first_objective -v
   ```
5. Implement `EvaluationObjective = Literal["error", "grid"]`, `sort_model_results`, and `--objective` CLI option defaulting to `grid`.
6. Update report text/docs so manual runs use `--objective grid`.
7. Verify:
   ```bash
   uv run ruff check tools/evaluate_calibration_models.py tests/test_evaluate_calibration_models.py
   uv run ty check tools/evaluate_calibration_models.py tests/test_evaluate_calibration_models.py
   uv run pytest tests/test_evaluate_calibration_models.py -v
   make check
   git diff --check
   ```
8. Commit and push:
   ```bash
   git add tools/evaluate_calibration_models.py tests/test_evaluate_calibration_models.py README.md docs/manual-test-checklist.md
   git commit -m "feat: rank replay models by grid objective"
   git push origin HEAD
   ```

## Task 2: Add Post-Fit Correction Candidates

**Objective:** Add evaluator-only candidate models that learn a simple second-stage correction from calibration residuals, without changing the live model yet.

**Files:**
- Modify: `tools/evaluate_calibration_models.py`
- Modify: `tests/test_evaluate_calibration_models.py`

**TDD:**
1. Add a synthetic replay test where a base linear model has a consistent y compression and a corrected candidate improves grid accuracy or mean Y error.
2. Verify RED:
   ```bash
   uv run pytest tests/test_evaluate_calibration_models.py::test_corrected_candidates_can_reduce_axis_bias -v
   ```
3. Implement evaluator-only wrappers:
   - `*-bias-corrected`: subtract mean calibration residual x/y.
   - `*-affine-corrected`: fit a simple 2D affine correction from predicted calibration coordinates to target coordinates.
4. Keep candidate count small initially: corrected variants for `linear-alpha-1.0` and `poly2-alpha-1.0` only.
5. Verify focused tests, `make check`, and `git diff --check`.
6. Commit and push:
   ```bash
   git add tools/evaluate_calibration_models.py tests/test_evaluate_calibration_models.py
   git commit -m "feat: evaluate corrected calibration candidates"
   git push origin HEAD
   ```

## Task 3: Replay Latest Manual Run and Document Decision

**Objective:** Run the grid-first evaluator on `metrics/demo.jsonl`, record whether a candidate should be promoted to the live demo, and specify the next implementation decision.

**Files:**
- Modify: `docs/experiments/feature-diagnostics-comparison.md`

**Steps:**
1. Run:
   ```bash
   uv run python tools/evaluate_calibration_models.py metrics/demo.jsonl --screen-width 1512 --screen-height 982 --grid-columns 4 --grid-rows 3 --objective grid
   ```
2. Record top grid candidates and compare to the latest live `validation_metrics` event.
3. Decision rule:
   - If an offline candidate beats live grid accuracy and has acceptable pixel error, plan a live-model promotion slice.
   - If no candidate beats live grid accuracy, plan calibration geometry/target sampling changes instead.
4. Verify doc diff:
   ```bash
   git diff --check
   ```
5. Commit and push:
   ```bash
   git add docs/experiments/feature-diagnostics-comparison.md
   git commit -m "docs: record grid-objective replay evaluation"
   git push origin HEAD
   ```
