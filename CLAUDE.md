# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This package contains the central slit-decomposition algorithm from PyReduce (see [Piskunov, Wehrhahn & Marquart 2021](https://doi.org/10.1051/0004-6361/202038293)), extracted for further development. It extracts 1D spectra from 2D detector frames captured by echelle spectrographs.

### Progression Beyond Polynomial Slit Shapes

The code here extends the original polynomial-based slit shape model (described in the paper) to support **array-based slit descriptions**. Instead of polynomial coefficients, the slit is now described by an array of `delta_x` values—pixel shifts from vertical—that specify how the slit curves from row to row in each spectral order.

### Workflow

1. **Generate test data**: `make_testdata.py` creates synthetic detector frames
2. **Measure slit deltas**: Use either `make_slitdeltas.py` (Gaussian fitting) or `make_slitdeltas_xcorr.py` (cross-correlation) to measure the `delta_x` values
3. **Extract spectra**: Run `uv run py.test` to perform extraction using both the data and measured deltas


## Development Setup

This project uses **uv** for fast, modern Python package management. All Python commands should use `uv run` instead of direct Python invocation.

```bash
# Install dependencies (use uv, not pip)
uv sync

# Install with development dependencies
uv sync --all-extras

# Install pre-commit hooks (IMPORTANT: run this once after cloning)
uv run pre-commit install
```

## Common Commands

### Using uv
**IMPORTANT: Always use `uv run` to execute Python commands.** This ensures the correct environment and dependencies.

```bash
# Run tests
uv run py.test

```

### Building Locally

```bash
# Build platform-specific wheel for local testing
uv build

# For development: rebuild C/C++ extensions after code changes
uv sync --reinstall-package charslit
```

**IMPORTANT:** The package is installed in editable mode by `uv sync`. However:
- **Python code changes** are picked up automatically (no rebuild needed)
- **C/C++ code changes** require `uv sync --reinstall-package charslit` to recompile the extension

**Note:** See "Release Process" section below for publishing to PyPI.

### Code Quality
```bash
# Format and lint with Ruff (replaces black, isort, flake8)
uv run ruff format .
uv run ruff check .
uv run ruff check --fix .

# Run pre-commit hooks (runs automatically on commit, or manually)
uv run pre-commit run --all-files

```

## Build System

### Modern Tooling Stack (2025)
- **Package manager**: uv (fast, modern alternative to pip/poetry)
- **Build backend**: Hatchling (PEP 517 compliant, replaces setuptools)
- **Linter/formatter**: Ruff (replaces black, isort, flake8, pyupgrade)
- **Python version**: 3.11+ (specified in pyproject.toml)


## Slit Delta Measurement Methods

Two approaches are available for measuring slit deltas:

### 1. Gaussian Fitting (`make_slitdeltas.py`)
- **Method**: Detects peaks in each row, fits Gaussians for sub-pixel precision, tracks peaks by proximity, uses median of offsets
- **Strengths**: Detailed diagnostics, handles varying peak counts, detects individual peak issues
- **Key features**:
  - Proximity-based peak matching (max 2px shift between rows)
  - Robust statistics (MAD-based variance)
  - Quality flags for fit failures, outliers, weak correlations
  - Reasonable peak count filtering (±2 from most common)

### 2. Cross-Correlation (`make_slitdeltas_xcorr.py`) - **Recommended**
- **Method**: Cross-correlates each row with reference (median of middle 5 rows), finds shift from correlation peak
- **Strengths**: Simpler, faster, naturally handles all peaks together, no peak detection needed
- **Key features**:
  - 10x upsampling for 0.1 pixel precision
  - Reference: median of middle 5 rows (sharper, less smearing)
  - Automatic quality detection via correlation strength
  - Flags bad rows (e.g., row 0 with spurious peaks) via weak correlation

**Performance**: Cross-correlation achieves <0.06 px RMS agreement with Gaussian fitting for most test cases.

**When to use which**:
- Use **cross-correlation** for normal slit tilt measurements (all peaks shift together)
- Use **Gaussian fitting** when peaks appear/disappear or need detailed individual peak diagnostics

**Comparison**: Run `uv run python compare_slitdelta_methods.py` to compare both methods side-by-side.

## Important Implementation Notes

### Slit Delta Measurement
- **Single offset per row**: Both methods assume all peaks in a row shift together (true for instrumental slit tilt)
- **Output format**: NPZ files with `filename`, `avg_offset`, `std_offset`, `median_offsets` arrays
- **Interpolation**: Missing/bad rows are filled by linear interpolation from valid neighbors
- **Quality detection**:
  - Gaussian method: peak count validation, fit quality, trajectory outliers
  - XCorr method: correlation strength (>1e7 is good, <50% of mean is weak)

### General Development
- Always use `uv run` for Python commands to ensure correct environment
- Pre-commit hooks enforce code quality (runs Ruff automatically)
- The C extensions must compile successfully for extraction to work
- After modifying C/C++ code, use `uv sync --reinstall-package charslit` to rebuild
- Do not commit changes without asking unless you are sure this is intended. NEVER push until asked explicitly.