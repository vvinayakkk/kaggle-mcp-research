.PHONY: install test lint format build publish clean

# ── Developer setup ──────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"

# ── Quality ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -v --tb=short -x

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

# ── Build & release ───────────────────────────────────────────────────────────
build: clean
	python -m build

publish: build
	twine check dist/*
	twine upload dist/*

# Usage: make tag VER=2.0.0
tag:
	git tag v$(VER)
	git push origin v$(VER)

# ── Housekeeping ──────────────────────────────────────────────────────────────
clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
