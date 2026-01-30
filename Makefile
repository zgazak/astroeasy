# Astroeasy Makefile
# Common development commands

.PHONY: help install install-dev test coverage lint format build build-docker clean

# Use uv run to ensure we're using the virtual environment
PYTHON := uv run python

# Default target
help:
	@echo "Available targets:"
	@echo "  install       - Install package in current environment"
	@echo "  install-dev   - Install package with dev dependencies"
	@echo "  test          - Run tests"
	@echo "  coverage      - Run tests with coverage report"
	@echo "  lint          - Run linter (ruff)"
	@echo "  format        - Format code (ruff)"
	@echo "  build         - Build package (wheel and sdist)"
	@echo "  build-docker  - Build astrometry-cli Docker image"
	@echo "  clean         - Remove build artifacts"
	@echo ""
	@echo "Index management:"
	@echo "  indices-download SERIES=5200_LITE OUTPUT=/path       - Download indices"
	@echo "  indices-examine  SERIES=5200_LITE INDEX_PATH=/path   - Examine indices"
	@echo ""
	@echo "Testing:"
	@echo "  test-install-local   - Test local astrometry.net installation"
	@echo "  test-install-docker  - Test Docker astrometry.net installation"

# Installation
install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -v
	uv run genbadge tests -o tests.svg

coverage:
	pytest tests/ -v \
    	--junitxml=reports/junit/junit.xml \
		--cov=astroeasy \
		--cov-report=term-missing \
		--cov-report=xml \
		--cov-report=html
	uv run genbadge tests -o tests.svg
	uv run genbadge coverage -i coverage.xml -o coverage.svg
	@echo "Coverage report: htmlcov/index.html"

badges: coverage
	@echo "Badges generated:"
	@ls -1 *.svg

# Code quality
lint:
	ruff check astroeasy/

format:
	ruff format astroeasy/
	ruff check --fix astroeasy/

# Build
build:
	$(PYTHON) -m build

build-docker:
	docker build -t astrometry-cli astroeasy/dotnet/

# Clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf reports/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Index management
indices-download:
	@if [ -z "$(OUTPUT)" ]; then echo "Usage: make indices-download SERIES=5200_LITE OUTPUT=/path"; exit 1; fi
	$(PYTHON) -m astroeasy.cli indices download --series $(SERIES) --output $(OUTPUT)

indices-examine:
	@if [ -z "$(INDEX_PATH)" ]; then echo "Usage: make indices-examine SERIES=5200_LITE INDEX_PATH=/path"; exit 1; fi
	$(PYTHON) -m astroeasy.cli indices examine --series $(SERIES) --path $(INDEX_PATH)

# Installation verification
test-install-local:
	$(PYTHON) -m astroeasy.cli test-install --local

test-install-docker:
	$(PYTHON) -m astroeasy.cli test-install --docker astrometry-cli
