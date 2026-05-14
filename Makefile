.PHONY: help sync test typecheck lint format check download-model test-model run-demo clean

UV ?= uv
PYTHON ?= $(UV) run python
PYTEST ?= $(UV) run pytest
RUFF ?= $(UV) run ruff
TY ?= $(UV) run ty
MODEL_DIR ?= models
FACE_LANDMARKER_MODEL_URL ?= https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

help:
	@echo "Available targets:"
	@echo "  sync       Install/update the uv-managed environment"
	@echo "  test       Run pytest"
	@echo "  typecheck  Run ty type checking"
	@echo "  lint       Run ruff lint checks"
	@echo "  format     Format Python files with ruff"
	@echo "  check      Run lint, typecheck, and tests"
	@echo "  download-model  Download the MediaPipe FaceLandmarker model"
	@echo "  run-demo   Launch the desktop demo"
	@echo "  clean      Remove local caches"

sync:
	$(UV) sync --dev

test:
	$(PYTEST) -v

typecheck:
	$(TY) check src apps tests

lint:
	$(RUFF) check src apps tests

format:
	$(RUFF) format src apps tests

check: lint typecheck test

download-model: $(MODEL_DIR)/face_landmarker.task
	@echo "Downloaded MediaPipe FaceLandmarker model: $<"
	@echo "Run with: PUPIL_TRACKER_MEDIAPIPE_MODEL=$$(pwd)/$< make run-demo"

$(MODEL_DIR)/face_landmarker.task:
	mkdir -p $(MODEL_DIR)
	curl -L --fail --output $@.tmp $(FACE_LANDMARKER_MODEL_URL)
	mv $@.tmp $@

test-model:
	@test -s $(MODEL_DIR)/face_landmarker.task

run-demo:
	$(PYTHON) apps/desktop_demo/main.py

clean:
	rm -rf .pytest_cache .ruff_cache .ty_cache htmlcov .coverage dist build *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
