# CharSlit

Python wrapper for the `slitdec` C library, providing slit decomposition for astronomical spectrograph data.

## What It Does

CharSlit separates a 2D spectral image into its components:
- **1D Spectrum**: Intensity vs wavelength
- **1D Slit function**: Spatial illumination pattern along the slit
- **2D Model**: Reconstructed image from spectrum ⊗ slit function

The algorithm handles slit curvature, pixel-to-pixel variations, outlier rejection, and applies smoothing constraints during iterative optimization.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd CharSlit

# Install with uv
uv pip install -e .
```

## Quick Start

```python
import numpy as np
import slitchar

# Load your spectral image and prepare inputs
im = ...              # 2D spectral image (nrows, ncols)
pix_unc = ...         # Pixel uncertainties (same shape)
mask = ...            # Pixel mask, uint8 (1=good, 0=bad)
ycen = ...            # Order center offsets (ncols,)
slitcurve = ...       # Curvature coefficients (ncols, 3)
slitdeltas = ...      # Horizontal offsets (nrows,) - auto-interpolated

# Run decomposition
result = slitchar.slitdec(
    im, pix_unc, mask, ycen, slitcurve, slitdeltas,
    osample=6,        # Oversampling factor
    lambda_sP=0.0,    # Spectrum smoothing
    lambda_sL=0.1,    # Slit function smoothing
    maxiter=20        # Max iterations
)

# Access results
spectrum = result['spectrum']          # Extracted 1D spectrum
slitfunc = result['slitfunction']      # Slit illumination function
model = result['model']                # Reconstructed 2D image
uncertainty = result['uncertainty']    # Spectrum uncertainties
info = result['info']                  # [success, cost, status, iter, delta_x]
```

## Testing

```bash
# Run fast tests (24 tests, ~0.2s)
uv run pytest

# Run visualization tests (8 tests, ~5s, generates plots in test_data_output/)
uv run pytest -m save_output
```

## Developer Documentation

For detailed information about the build system, algorithm internals, testing infrastructure, and common issues, see [CLAUDE.md](CLAUDE.md).

## Requirements

- Python >= 3.11
- numpy >= 1.20.0
- matplotlib >= 3.5.0 (for visualization)
- astropy >= 5.0.0 (for FITS file support)
- scipy >= 1.7.0

Build dependencies:
- scikit-build-core
- nanobind
- CMake
