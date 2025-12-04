"""Pytest configuration and fixtures for charslit tests."""

import numpy as np
import pytest
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import warnings


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

        # Create 5-panel plot with fixed positions to prevent movement
        # Top row: 3 equal panels (input, model, difference)
        # Bottom row: 2 panels with 2:1 ratio (spectrum, slit function)
        fig = plt.figure(figsize=(15, 7.5))

        # Calculate difference
        diff = im - model

        # Define fixed positions [left, bottom, width, height] in figure coordinates
        # Top row (height 2/3 of plotting area)
        top_height = 0.50
        bottom_height = 0.25
        vertical_gap = 0.08
        top_bottom = 0.08 + bottom_height + vertical_gap

        # Horizontal positions for 3 panels in top row
        panel_width = 0.26
        colorbar_width = 0.015
        h_gap = 0.04
        left_margin = 0.06

        pos1 = [left_margin, top_bottom, panel_width, top_height]
        pos2 = [left_margin + panel_width + h_gap, top_bottom, panel_width, top_height]
        pos3 = [left_margin + 2*(panel_width + h_gap), top_bottom, panel_width, top_height]

        # Bottom row positions
        bottom_left_width = 2 * panel_width + h_gap
        bottom_right_width = panel_width
        pos4 = [left_margin, 0.08, bottom_left_width, bottom_height]
        pos5 = [left_margin + bottom_left_width + h_gap, 0.08, bottom_right_width, bottom_height]

        # Top row - Panel 1: Input image
        ax1 = fig.add_axes(pos1)
        im1 = ax1.imshow(im, cmap='viridis', aspect='auto')
        ax1.set_title('Input Image')
        ax1.set_xticks([])
        ax1.set_yticks([])
        cax1 = fig.add_axes([pos1[0] + pos1[2] + 0.005, pos1[1], colorbar_width, pos1[3]])
        fig.colorbar(im1, cax=cax1, label='Intensity')

        # Top row - Panel 2: Model
        ax2 = fig.add_axes(pos2)
        im2 = ax2.imshow(model, cmap='viridis', aspect='auto')
        ax2.set_title('Model')
        ax2.set_xticks([])
        ax2.set_yticks([])
        cax2 = fig.add_axes([pos2[0] + pos2[2] + 0.005, pos2[1], colorbar_width, pos2[3]])
        fig.colorbar(im2, cax=cax2, label='Intensity')

        # Top row - Panel 3: Difference (im - model)
        ax3 = fig.add_axes(pos3)
        vmax_diff = np.max(np.abs(diff))
        im3 = ax3.imshow(diff, cmap='seismic', aspect='auto',
                         vmin=-vmax_diff, vmax=vmax_diff)
        ax3.set_title('Difference (Input - Model)')
        ax3.set_xticks([])
        ax3.set_yticks([])
        cax3 = fig.add_axes([pos3[0] + pos3[2] + 0.005, pos3[1], colorbar_width, pos3[3]])
        fig.colorbar(im3, cax=cax3, label='Residual')

        # Bottom row - Panel 4: Spectrum (spans 2/3 width)
        if 'spectrum' in extra_arrays and 'uncertainty' in extra_arrays:
            ax4 = fig.add_axes(pos4)
            spectrum = extra_arrays['spectrum']
            uncertainty = extra_arrays['uncertainty']
            x = np.arange(len(spectrum))

            # Plot spectrum as solid line
            ax4.plot(x, spectrum, 'b-', linewidth=1.5, label='Spectrum')

            # Plot uncertainty as shaded region
            ax4.fill_between(x, spectrum - uncertainty, spectrum + uncertainty,
                            alpha=0.3, color='blue', label='±1σ uncertainty')

            ax4.set_xlabel('Column (wavelength direction)')
            ax4.set_ylabel('Intensity')
            ax4.set_title('Extracted Spectrum')
            ax4.legend(loc='best')
            ax4.grid(True, alpha=0.3)

        # Bottom row - Panel 5: Slit function (spans 1/3 width)
        if 'slitfunction' in extra_arrays:
            ax5 = fig.add_axes(pos5)
            slitfunction = extra_arrays['slitfunction']
            x = np.arange(len(slitfunction))

            # Plot slit function horizontally for better detail visibility
            ax5.plot(x, slitfunction, 'r-', linewidth=1.5)
            ax5.set_xlabel('Oversampled subpixel')
            ax5.set_ylabel('Intensity')
            ax5.set_title('Slit Function')
            ax5.grid(True, alpha=0.3)

        # Suppress tight_layout warning for fixed-position axes
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*not compatible with tight_layout.*')
            plt.savefig(png_filename, dpi=100)
        plt.close(fig)
        print(f"Saved plot to: {png_filename}")

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

    nrows, ncols = im.shape
    osample = 6  # Default oversampling

    # Calculate ny
    ny = osample * (nrows + 1) + 1

    # Default values
    slitcurve = np.zeros((ncols, 3))
    slitdeltas = np.zeros(nrows)
    ycen = np.full(ncols, nrows / 2.0)

    # Load from curvedelta NPZ if available (preferred)
    if curvedelta_path.exists():
        with np.load(curvedelta_path) as data:
            if 'slitcurve' in data:
                slitcurve = data['slitcurve'].astype(np.float64)
                if slitcurve.shape[0] != ncols:
                    raise ValueError(f"slitcurve shape mismatch: expected ({ncols}, 3), got {slitcurve.shape}")

            if 'slitdeltas' in data:
                slitdeltas = data['slitdeltas'].astype(np.float64)
                if slitdeltas.shape[0] != nrows:
                    raise ValueError(f"slitdeltas shape mismatch: expected {nrows}, got {slitdeltas.shape[0]}")

            if 'ycen' in data:
                ycen = data['ycen'].astype(np.float64)
                if ycen.shape[0] != ncols:
                    raise ValueError(f"ycen shape mismatch: expected {ncols}, got {ycen.shape[0]}")

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

    # Create pixel uncertainties (assuming Poisson noise)
    pix_unc = np.sqrt(np.abs(im) + 1.0)

    # Create mask (all valid initially)
    mask = np.ones(im.shape, dtype=np.uint8)

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
        'filename': fits_path.name
    }
