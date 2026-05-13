.PHONY: help sync test typecheck lint format check run-demo clean

UV ?= uv
PYTHON ?= $(UV) run python
PYTEST ?= $(UV) run pytest
RUFF ?= $(UV) run ruff
TY ?= $(UV) run ty

help:
	@echo "Available targets:"
	@echo "  sync       Install/update the uv-managed environment"
	@echo "  test       Run pytest"
	@echo "  typecheck  Run ty type checking"
	@echo "  lint       Run ruff lint checks"
	@echo "  format     Format Python files with ruff"
	@echo "  check      Run lint, typecheck, and tests"
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

run-demo:
	$(PYTHON) apps/desktop_demo/main.py

clean:
	rm -rf .pytest_cache .ruff_cache .ty_cache htmlcov .coverage dist build *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
