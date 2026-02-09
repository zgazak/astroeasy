# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_models.py -v

# Run a single test function
pytest tests/test_models.py::test_detection_creation -v

# Run tests with coverage
pytest tests/ -v --cov=astroeasy --cov-report=term-missing

# Lint
ruff check astroeasy/

# Format
ruff format astroeasy/
ruff check --fix astroeasy/

# Build Docker image for astrometry.net
docker build -t astrometry-cli astroeasy/dotnet/

# Test installation
python -m astroeasy.cli test-install --docker astrometry-cli
python -m astroeasy.cli test-install --local
```

## Architecture

This package wraps astrometry.net's `solve-field` command with a Python API, supporting both local and Docker-based execution.

### Data Flow

```
User detections (x, y, flux) → solve_field() → FITS file → astrometry.net → WCS solution
```

### Key Layers

1. **Public API** (`astroeasy/runner.py`): `solve_field()` validates inputs and dispatches to the dotnet layer. This is the main entry point.

2. **Dotnet Orchestration** (`astroeasy/dotnet/runner.py`): Creates temp directory, writes sources to FITS, generates astrometry.cfg, runs solve-field, parses results. The `solve_field()` here does the actual work.

3. **Execution Backends** (`astroeasy/dotnet/docker.py`, `astroeasy/dotnet/local.py`): Handle subprocess execution for Docker containers vs local installation. Both expose `run_against_*()` and `test_dotnet_install()`.

4. **Models** (`astroeasy/models.py`): Dataclasses for `Detection`, `ImageMetadata`, `WCSResult`, `MatchedStar`, `SolveResult`. `WCSResult` stores raw FITS header and provides `to_astropy_wcs()` conversion.

5. **Configuration** (`astroeasy/config.py`): `AstrometryConfig` dataclass with YAML serialization. Required field: `indices_path`. Key optional: `docker_image` (None = local execution).

### Solve Flow

When `solve_field()` is called:
1. `astroeasy/runner.py:solve_field()` checks minimum sources, then calls `dotnet/runner.py:solve_field()`
2. Dotnet runner creates temp dir with `sources.fits` and `astrometry.cfg`
3. Dispatches to `docker.py` or `local.py` based on `config.docker_image`
4. Backend runs `solve-field` command with appropriate mounts/paths
5. On success, parses `.wcs` file into `WCSResult` and `.rdls` into `MatchedStar` list

### Index Management

`astroeasy/indices.py` handles downloading and verifying astrometry.net index files. `astroeasy/constants.py` defines `AstrometryIndexSeries` enum with expected files for each series (5200_LITE, 5200, 4100, etc.).
