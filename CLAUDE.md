# CLAUDE.md - Developer Notes for Future Sessions

This document contains important information about the CharSlit codebase, build system, and key implementation details that were discovered during development.

## Project Overview

CharSlit is a Python wrapper for the `slitdec` C library, which performs slit decomposition for astronomical spectrograph data. It separates a 2D spectral image into:
- **Spectrum**: 1D array representing intensity vs wavelength
- **Slit function**: 1D array representing the spatial illumination pattern
- **Model**: Reconstructed 2D image from spectrum � slit function

## Build System

### Technology Stack
- **Wrapper**: nanobind (modern C++/Python binding library)
- **Build system**: scikit-build-core with CMake
- **Package manager**: uv (used in development)
- **Testing**: pytest

### Build Configuration

**CMakeLists.txt**: The C code is compiled **directly** with the extension module, not as a separate static library:
```cmake
nanobind_add_module(
  slitchar
  STABLE_ABI
  NB_STATIC
  slitdec/slitdec_wrapper.cpp
  slitdec/slitdec.c  # Compiled together
)
set_source_files_properties(slitdec/slitdec.c PROPERTIES LANGUAGE C)
```

**Important**: Attempting to build slitdec.c as a separate static library and link it causes symbol resolution issues. The direct compilation approach avoids these problems.

**C++ Linkage**: Must use `extern "C"` wrapper around the C header:
```cpp
extern "C" {
#include "slitdec.h"
}
```
Without this, C++ name mangling prevents the linker from finding the C symbols.

### Build Commands
```bash
# Build and install in development mode
uv pip install -e .

# Force rebuild (useful after C code changes)
uv pip install -e . --force-reinstall --no-deps

# Clean cache if old binaries persist
uv cache clean
```

### Testing
```bash
# Run fast tests (24 tests in ~0.2s, excludes save_output marked tests)
uv run pytest

# Run visualization tests (8 tests in ~5s, generates plots)
uv run pytest -m save_output

# Run specific test class
uv run pytest tests/test_slitdec.py::TestRealData -v
```

## The slitdec Algorithm

### What It Does

The `slitdec` function performs iterative optimization to decompose a 2D spectral image into:
1. A 1D spectrum (length: ncols)
2. A 1D oversampled slit illumination function (length: ny)
3. A 2D model reconstruction

The algorithm:
- Uses subpixel oversampling (typically 6x) for the slit function
- Handles slit curvature via polynomial coefficients
- Handles pixel-to-pixel variations in slit illumination (slitdeltas)
- Performs outlier rejection by modifying the input mask
- Applies smoothing constraints to spectrum and slit function

### Input Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `im` | ndarray (nrows, ncols) | Input 2D spectral image |
| `pix_unc` | ndarray (nrows, ncols) | Pixel uncertainties (typically from Poisson noise) |
| `mask` | ndarray (nrows, ncols), uint8 | Pixel mask (0=bad, 1=good). **Modified during execution** |
| `ycen` | ndarray (ncols,) | Order center line offset from pixel boundary. **Modified during execution** |
| `slitcurve` | ndarray (ncols, 3) | Polynomial coefficients [c0, c1, c2] for slit curvature |
| `slitdeltas` | ndarray (nrows,) or (ny,) | Pixel-to-pixel horizontal offsets (see below) |
| `osample` | int | Oversampling factor (default: 6) |
| `lambda_sP` | float | Smoothing parameter for spectrum (default: 0.0) |
| `lambda_sL` | float | Smoothing parameter for slit function (default: 0.1) |
| `maxiter` | int | Maximum iterations (default: 20) |

### Important: slitdeltas

**What it is**: Per-pixel horizontal displacement corrections for the slit function.

**Length flexibility**: The wrapper accepts two formats:
- **Length nrows**: One value per detector row (most common, matches real data format)
- **Length ny**: One value per oversampled subpixel where `ny = osample * (nrows + 1) + 1`

**Wrapper behavior**: If length is nrows, the wrapper **automatically interpolates** to length ny using linear interpolation before passing to the C function.

**Real data format**: The `.npz` files in `data/` contain `median_offsets` with length nrows, which are loaded and interpolated by the wrapper.

### Critical Implementation Detail: delta_x Calculation

**Issue discovered**: The C code allocates memory based on `delta_x`, the maximum horizontal pixel shift. Originally, `delta_x` was calculated **only** from the curve polynomial:

```c
delta_x = max(ceil(abs(y * curve[1] + y� * curve[2])))
```

However, `slitdeltas` adds **additional** horizontal shift! With real data ranging from -3.0 to +2.6 pixels, this caused **out-of-bounds memory access** and crashes.

**Fix applied** (slitdec.c lines 956-961):
```c
// Account for additional shift from slitdeltas
for (int iy = 0; iy < ny; iy++)
{
    tmp = ceil(fabs(slitdeltas[iy]));
    delta_x = max(delta_x, tmp);
}
```

This ensures `nx = 4 * delta_x + 1` is large enough to allocate sufficient memory for arrays like `MAX_PAIJ = ncols * nx`.

**Lesson**: When adding new input parameters that affect spatial coordinates, always check if they need to be included in the `delta_x` calculation.

### How slitdeltas is Used in C Code

In the geometry calculation (slitdec.c:574-576):
```c
delta = (slitcurve[...1] + slitcurve[...2] * (dy - ycen[x])) * (dy - ycen[x])
        + slitdeltas[iy];
ix1 = delta;  // Truncated to int
ix2 = ix1 + signum(delta);
```

Then used as column offset:
```c
if (x + ix1 >= 0 && x + ix2 < ncols) {
    xx = x + ix1;  // Shifted column index
    ...
}
```

So `delta` represents **horizontal pixel displacement**, and `slitdeltas[iy]` is added to the total shift from the curve polynomial.

## Testing Infrastructure

### Test Organization

32 tests organized in 8 classes:
- **TestBasicFunctionality**: Import, callable, basic execution
- **TestOutputStructure**: Output format validation
- **TestInputValidation**: Error handling for invalid inputs
- **TestNumericalBehavior**: Normalization, reconstruction quality
- **TestEdgeCases**: Minimal images, curvature, smoothing variations
- **TestDefaultParameters**: Default values verification
- **TestMemoryManagement**: Input preservation, multiple calls
- **TestRealData**: Tests on real FITS files from `data/` directory

### Running Tests

**Fast automated testing** (24 tests in ~0.2s):
```bash
pytest
```
Default runs exclude `@pytest.mark.save_output` tests for speed (configured in `pyproject.toml`).

**Generate visualization plots** (8 tests in ~5s):
```bash
pytest -m save_output
```
Tests marked with `@pytest.mark.save_output` automatically save outputs to `test_data_output/`:
- **NPZ file**: Contains im, model, spectrum, slitfunction, uncertainty
- **PNG file**: 5-panel plot with fixed positions to prevent axes shifting
  - Top row: Input image, Model reconstruction, Difference (all viridis/seismic colormaps)
  - Bottom row: Extracted spectrum with ±1σ uncertainty, Slit function (both as line plots)

Currently 4 marked tests (8 total with parametrization):
- `test_basic_execution`
- `test_with_custom_parameters`
- `test_mask_modification`
- `test_real_data_files` (parametrized over 5 FITS files)

### Real Data Fixtures

The `real_data_files` fixture (tests/conftest.py):
1. Loads all FITS files from `data/*.fits`
2. Looks for corresponding `slitdeltas_{basename}.npz` files
3. Loads `median_offsets` array from NPZ (length nrows)
4. Wrapper automatically interpolates slitdeltas from nrows → ny before passing to C
5. Falls back to zeros if NPZ doesn't exist
6. Hardcodes: `ycen=0.5` (middle of row), `slitcurve=zeros` (no curvature)
7. Computes `pix_unc` from Poisson noise: `sqrt(abs(im) + 1.0)`

Currently 5 real data files:
- `Hsim.fits` (90�53)
- `Rsim.fits` (140�84)
- `discontinuous.fits` (100�150)
- `fixedslope.fits` (100�150)
- `multislope.fits` (100�150)

## Common Issues and Solutions

### Issue: Undefined symbols during linking
**Symptom**: `undefined symbol: _Z7slitdeciiPdS_PhS_S_S_iddiS_S_S_S_S_`

**Cause**: C++ name mangling trying to find slitdec as a C++ function.

**Solution**: Use `extern "C"` wrapper around the C header include.

### Issue: Static library linking fails
**Symptom**: Symbols not found even with extern "C"

**Solution**: Don't use static library. Compile slitdec.c directly with the extension module.

### Issue: Crashes with real slitdeltas data
**Symptom**: Segfault or "munmap_chunk(): invalid pointer" with non-zero slitdeltas

**Cause**: `delta_x` calculation didn't include slitdeltas, causing insufficient memory allocation.

**Solution**: Add slitdeltas to delta_x calculation (see "Critical Implementation Detail" above).

### Issue: Old binary loads after code changes
**Symptom**: Changes to C code don't take effect

**Solution**:
```bash
uv cache clean
uv pip install -e . --force-reinstall --no-deps
```

## Interface Documentation Issues

**interface.md line 16** states:
```
double *slitdeltas, // exposed, array length nrows
```

This comment is **incorrect**. The C code requires `ny = osample * (nrows + 1) + 1`, not `nrows`.

However, the **wrapper** now accepts both lengths:
- Length `nrows`: Interpolated to `ny` (matches real data format)
- Length `ny`: Used directly

So the interface.md comment is now effectively correct for the **Python interface**, but misleading about the **C interface**.

## Memory Management Notes

### Arrays Modified by C Code

These arrays are **modified in place** by slitdec:
- `mask`: Updated during outlier rejection
- `ycen`: Integer offsets extracted and stored separately

The wrapper creates **copies** of these before passing to C, so Python inputs remain unchanged.

### Output Array Ownership

The wrapper allocates output arrays with `new[]` and transfers ownership to Python using nanobind capsules:
```cpp
double* sP = new double[ncols];
nb::capsule sP_owner(sP, [](void *p) noexcept { delete[] (double *) p; });
auto sP_array = nb::ndarray<nb::numpy, double>(sP, 1, sP_shape, sP_owner);
```

This ensures proper cleanup when Python garbage-collects the arrays.

## Data Files

### FITS Files
Located in `data/*.fits`, gitignored (data files are large).

Download from: https://tmy.se/t/tmpdata.tar.gz (contains both FITS and NPZ files)

### NPZ Files
Located in `data/slitdeltas_{basename}.npz`, gitignored.

Structure:
```python
{
    'filename': str,           # Original FITS filename
    'avg_offset': float,       # Average offset
    'std_offset': float,       # Standard deviation
    'median_offsets': ndarray  # Shape (nrows,) - the actual data we use
}
```

## Development Workflow

1. **After modifying C code**:
   ```bash
   uv pip install -e . --force-reinstall --no-deps
   uv run pytest -v
   ```

2. **After modifying Python/wrapper code**:
   ```bash
   uv pip install -e .
   uv run pytest -v
   ```

3. **To visualize results**:
   ```bash
   uv run pytest -m save_output
   # Check test_data_output/*.png
   ```

4. **To test specific real data file**:
   ```bash
   uv run pytest tests/test_slitdec.py::TestRealData::test_real_data_files[data/Hsim.fits] -v
   ```

## Key Takeaways for Future Work

1. **slitdeltas** affects memory allocation via `delta_x` - always check bounds calculations when modifying geometry code

2. **The wrapper provides convenience** by accepting both nrows and ny lengths for slitdeltas - maintain this flexibility

3. **interface.md may be outdated** - trust the actual C code implementation over comments

4. **Real data is essential** - synthetic tests passed but real data exposed the delta_x bug

5. **Build system quirks** - direct compilation of C code with extension works better than static library approach

6. **Memory management** - the C code modifies mask and ycen in place; wrapper creates copies to preserve Python inputs

## Useful References

- nanobind documentation: https://nanobind.readthedocs.io/
- scikit-build-core: https://scikit-build-core.readthedocs.io/
- C code location: `slitdec/slitdec.c` (main algorithm), `slitdec/slitdec.h` (interface)
- Wrapper code: `slitdec/slitdec_wrapper.cpp`
- Test suite: `tests/test_slitdec.py`
- Test fixtures: `tests/conftest.py`
