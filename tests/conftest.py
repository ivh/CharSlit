"""Pytest configuration and fixtures for charslit tests."""

import sys
from pathlib import Path

# Add parent directory to path so we can import slitcurve_plotting
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import warnings

from plotting import overlay_slitcurve_trajectories, plot_slitdec_results


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "save_output: mark test to save output data as .npz and .png files (excluded from default runs)"
    )


@pytest.fixture
def save_test_data(request):
    """
    Fixture to save test data for tests marked with @pytest.mark.save_output.

    Tests with this marker will automatically save their outputs to test_data_output/
    as .npz (data arrays) and .png (visualization) files.

    Usage in tests:
        @pytest.mark.save_output
        def test_something(self, data, save_test_data):
            result = charslit.slitdec(...)
            save_test_data(data['im'], result['model'],
                          spectrum=result['spectrum'],
                          slitfunction=result['slitfunction'],
                          uncertainty=result['uncertainty'])
    """
    # Check if this test has the save_output marker
    save_enabled = request.node.get_closest_marker("save_output") is not None

    if not save_enabled:
        # Return a no-op function if test doesn't have the marker
        return lambda *args, **kwargs: None

    # Create output directory if it doesn't exist
    output_dir = Path("test_data_output")
    output_dir.mkdir(exist_ok=True)

    def save_data(im, model, **extra_arrays):
        """
        Save input image and output model to .npz file and create a plot.

        Parameters
        ----------
        im : ndarray
            Input image
        model : ndarray
            Output model from slitdec
        **extra_arrays : additional arrays to save (optional)
        """
        # Get test name and sanitize for filesystem
        test_name = request.node.name
        # Replace characters that aren't valid in filenames
        test_name = test_name.replace('[', '_').replace(']', '').replace('/', '_')

        # Create filenames
        npz_filename = output_dir / f"{test_name}.npz"
        png_filename = output_dir / f"{test_name}.png"

        # Prepare data to save
        save_dict = {"im": im, "model": model}
        save_dict.update(extra_arrays)

        # Save to npz file
        np.savez(npz_filename, **save_dict)
        print(f"\nSaved test data to: {npz_filename}")

        # Call plotting function
        plot_slitdec_results(
            im,
            model,
            spectrum=extra_arrays.get('spectrum'),
            slitfunction=extra_arrays.get('slitfunction'),
            uncertainty=extra_arrays.get('uncertainty'),
            slitcurve_data=extra_arrays.get('slitcurve_data'),
            output_filename=png_filename
        )

    return save_data


@pytest.fixture
def simple_image_data():
    """
    Create a simple synthetic image with a curved slit.

    Returns a dictionary with all necessary inputs for slitdec.
    """
    nrows = 20
    ncols = 50
    osample = 6

    # Create a simple spectrum (Gaussian-like)
    x = np.arange(ncols)
    spectrum = np.exp(-0.5 * ((x - ncols/2) / 5)**2) * 100 + 10

    # Create a simple slit function (Gaussian)
    ny = osample * (nrows + 1) + 1
    y = np.arange(ny) / osample - nrows / 2
    slitfunc = np.exp(-0.5 * (y / 2)**2)
    slitfunc /= slitfunc.sum() / osample  # Normalize

    # Create synthetic image from outer product
    # Simplified model without curvature for basic tests
    im = np.zeros((nrows, ncols))
    ycen = np.full(ncols, 0.5)  # Center line at middle of pixels

    # Simple model: each column gets spectrum value weighted by slit function
    for i in range(ncols):
        for j in range(nrows):
            # Get slit function values for this row
            iy_start = j * osample
            iy_end = (j + 1) * osample + 1
            slit_contrib = slitfunc[iy_start:iy_end].sum()
            im[j, i] = spectrum[i] * slit_contrib

    # Add some noise
    np.random.seed(42)
    im += np.random.normal(0, 1, im.shape)

    # Create pixel uncertainties
    pix_unc = np.ones_like(im)

    # Create mask (all valid initially)
    mask = np.ones(im.shape, dtype=np.uint8)

    # Slit curve (no curvature for simple case)
    slitcurve = np.zeros((ncols, 3))

    # Slit deltas (no shifts)
    slitdeltas = np.zeros(ny)

    return {
        'im': im,
        'pix_unc': pix_unc,
        'mask': mask,
        'ycen': ycen,
        'slitcurve': slitcurve,
        'slitdeltas': slitdeltas,
        'nrows': nrows,
        'ncols': ncols,
        'osample': osample,
        'ny': ny,
        'true_spectrum': spectrum,
        'true_slitfunc': slitfunc
    }


@pytest.fixture
def curved_image_data():
    """
    Create a synthetic image with slit curvature.

    Returns a dictionary with all necessary inputs for slitdec.
    """
    nrows = 15
    ncols = 40
    osample = 4

    # Create image data
    im = np.random.randn(nrows, ncols) * 5 + 100
    pix_unc = np.ones_like(im) * 2
    mask = np.ones(im.shape, dtype=np.uint8)

    # Add ycen with some variation
    ycen = np.linspace(0.3, 0.7, ncols)

    # Add slight curvature
    slitcurve = np.zeros((ncols, 3))
    slitcurve[:, 1] = 0.01  # Linear term
    slitcurve[:, 2] = 0.001  # Quadratic term

    ny = osample * (nrows + 1) + 1
    slitdeltas = np.zeros(ny)

    return {
        'im': im,
        'pix_unc': pix_unc,
        'mask': mask,
        'ycen': ycen,
        'slitcurve': slitcurve,
        'slitdeltas': slitdeltas,
        'nrows': nrows,
        'ncols': ncols,
        'osample': osample,
        'ny': ny
    }


@pytest.fixture
def minimal_image_data():
    """
    Create minimal valid input for testing edge cases.
    """
    nrows = 5
    ncols = 10
    osample = 3

    im = np.ones((nrows, ncols)) * 100
    pix_unc = np.ones_like(im)
    mask = np.ones(im.shape, dtype=np.uint8)
    ycen = np.full(ncols, 0.5)
    slitcurve = np.zeros((ncols, 3))

    ny = osample * (nrows + 1) + 1
    slitdeltas = np.zeros(ny)

    return {
        'im': im,
        'pix_unc': pix_unc,
        'mask': mask,
        'ycen': ycen,
        'slitcurve': slitcurve,
        'slitdeltas': slitdeltas,
        'nrows': nrows,
        'ncols': ncols,
        'osample': osample,
        'ny': ny
    }


@pytest.fixture(params=glob.glob('data/*.fits'))
def real_data_files(request):
    """
    Load real data from FITS files in data/ directory.

    Looks for corresponding curvedelta_*.npz file first (contains slitcurve, slitdeltas, ycen),
    otherwise falls back to slitdeltas_*.npz (legacy format), or uses zeros.

    Returns a dictionary with all necessary inputs for slitdec.
    """
    fits_path = Path(request.param)

    # Look for curvedelta_{basename}.npz (new format with curve polynomials)
    curvedelta_path = fits_path.parent / f"curvedelta_{fits_path.stem}.npz"
    # Fallback to slitdeltas_{basename}.npz (legacy format)
    slitdeltas_path = fits_path.parent / f"slitdeltas_{fits_path.stem}.npz"

    # Load FITS image
    with fits.open(fits_path) as hdul:
        im = hdul[0].data.astype(np.float64)

    # Skip non-2D data (e.g., ycen arrays)
    if im.ndim != 2:
        pytest.skip(f"Skipping {fits_path.name}: not a 2D image (shape={im.shape})")

    nrows, ncols = im.shape
    osample = 6  # Default oversampling

    # Calculate ny
    ny = osample * (nrows + 1) + 1

    # Default values
    slitcurve = np.zeros((ncols, 3))
    slitdeltas = np.zeros(nrows)
    ycen = np.full(ncols, nrows / 2.0)
    lambda_sP = 0.0  # Default spectrum smoothing
    lambda_sL = 1.0  # Default slit function smoothing

    # Load from curvedelta NPZ if available (preferred)
    slitcurve_data = {}
    if curvedelta_path.exists():
        with np.load(curvedelta_path) as data:
            if 'slitcurve' in data:
                slitcurve = data['slitcurve'].astype(np.float64)
                if slitcurve.shape[0] != ncols:
                    raise ValueError(f"slitcurve shape mismatch: expected ({ncols}, n_coeffs), got {slitcurve.shape}")

            if 'slitdeltas' in data:
                slitdeltas = data['slitdeltas'].astype(np.float64)
                if slitdeltas.shape[0] != nrows:
                    raise ValueError(f"slitdeltas shape mismatch: expected {nrows}, got {slitdeltas.shape[0]}")

            if 'ycen' in data:
                ycen = data['ycen'].astype(np.float64)
                if ycen.shape[0] != ncols:
                    raise ValueError(f"ycen shape mismatch: expected {ncols}, got {ycen.shape[0]}")

            if 'lambda_sP' in data:
                lambda_sP = float(data['lambda_sP'])
            if 'lambda_sL' in data:
                lambda_sL = float(data['lambda_sL'])

            # Load trajectory data for plotting (copy arrays to avoid memory-mapped issues)
            # Load ALL keys from the NPZ file to match standalone script behavior
            slitcurve_data = {k: data[k] for k in data.files}
            
            # Ensure we have the explicit arrays we need as copies (not memory mapped)
            # This is critical because the data context manager closes the file
            keys_to_copy = ['slitcurve', 'slitdeltas', 'slitcurve_coeffs', 'x_refs', 'y_refs']
            for key in keys_to_copy:
                if key in slitcurve_data:
                    slitcurve_data[key] = np.array(slitcurve_data[key])
            
            # Fallback for slitcurve/slitdeltas if not in file (legacy support)
            if 'slitcurve' not in slitcurve_data:
                slitcurve_data['slitcurve'] = slitcurve
            if 'slitdeltas' not in slitcurve_data:
                slitcurve_data['slitdeltas'] = slitdeltas

    # Fallback to legacy slitdeltas NPZ
    elif slitdeltas_path.exists():
        with np.load(slitdeltas_path) as data:
            if 'median_offsets' in data:
                slitdeltas = data['median_offsets']
                if slitdeltas.shape[0] != nrows:
                    raise ValueError(f"median_offsets shape mismatch: expected {nrows}, got {slitdeltas.shape[0]}")

            if 'ycen' in data:
                ycen = data['ycen'].astype(np.float64)
                if ycen.shape[0] != ncols:
                    raise ValueError(f"ycen shape mismatch: expected {ncols}, got {ycen.shape[0]}")

        slitcurve_data = {
            'slitcurve': slitcurve,
            'slitdeltas': slitdeltas,
        }
    else:
        slitcurve_data = {
            'slitcurve': slitcurve,
            'slitdeltas': slitdeltas,
        }

    # Handle NaN pixels (bad pixels)
    nan_mask = np.isnan(im)
    n_nans = nan_mask.sum()

    # Create mask (0=bad, 1=good)
    mask = np.ones(im.shape, dtype=np.uint8)
    mask[nan_mask] = 0  # Mark NaN pixels as bad

    # Replace NaN pixels with 0 for numerical stability
    # (the mask will tell slitdec to ignore these pixels)
    im[nan_mask] = 0.0

    if n_nans > 0:
        print(f"  {fits_path.name}: Found {n_nans} NaN pixels ({100*n_nans/im.size:.1f}%), marked as bad in mask")

    # Create pixel uncertainties (assuming Poisson noise)
    # Set uncertainty to large value for bad pixels
    pix_unc = np.sqrt(np.abs(im) + 1.0)
    pix_unc[nan_mask] = 1e10  # Large uncertainty for bad pixels

    # Debug: check loaded data
    print(f"  {fits_path.name}: slitcurve shape={slitcurve.shape}, ycen shape={ycen.shape}, slitdeltas shape={slitdeltas.shape}")
    n_coeffs = slitcurve.shape[1]
    coeffs_str = ", ".join([f"c{i}={np.sum(np.abs(slitcurve[:,i])>1e-10)}" for i in range(n_coeffs)])
    print(f"  {fits_path.name}: slitcurve non-zero coeffs (poly degree {n_coeffs-1}): {coeffs_str}")
    print(f"  {fits_path.name}: ycen range=[{ycen.min():.3f}, {ycen.max():.3f}], mean={ycen.mean():.3f}")
    print(f"  {fits_path.name}: lambda_sP={lambda_sP}, lambda_sL={lambda_sL}")

    return {
        'im': im,
        'pix_unc': pix_unc,
        'mask': mask,
        'ycen': ycen,
        'slitcurve': slitcurve,
        'slitdeltas': slitdeltas,
        'nrows': nrows,
        'ncols': ncols,
        'osample': osample,
        'ny': ny,
        'lambda_sP': lambda_sP,
        'lambda_sL': lambda_sL,
        'filename': fits_path.name,
        'slitcurve_data': slitcurve_data,
    }
