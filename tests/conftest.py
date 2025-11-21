"""Pytest configuration and fixtures for slitchar tests."""

import numpy as np
import pytest
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--save-data",
        action="store_true",
        default=False,
        help="Save input images and output models from slitdec tests as .npz files"
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "uses_slitdec: mark test as calling slitdec() with image data"
    )


@pytest.fixture
def save_test_data(request):
    """
    Fixture to save test data when --save-data option is enabled.

    Usage in tests:
        result = slitchar.slitdec(...)
        save_test_data(data['im'], result['model'])
    """
    save_enabled = request.config.getoption("--save-data")

    if not save_enabled:
        # Return a no-op function if saving is disabled
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
        # Get test name
        test_name = request.node.name

        # Create filenames
        npz_filename = output_dir / f"{test_name}.npz"
        png_filename = output_dir / f"{test_name}.png"

        # Prepare data to save
        save_dict = {"im": im, "model": model}
        save_dict.update(extra_arrays)

        # Save to npz file
        np.savez(npz_filename, **save_dict)
        print(f"\nSaved test data to: {npz_filename}")

        # Create 3-panel plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Calculate difference
        diff = im - model

        # Plot 1: Input image
        im1 = axes[0].imshow(im, cmap='viridis', aspect='auto')
        axes[0].set_title('Input Image')
        axes[0].set_xlabel('Column')
        axes[0].set_ylabel('Row')
        plt.colorbar(im1, ax=axes[0], label='Intensity')

        # Plot 2: Model
        im2 = axes[1].imshow(model, cmap='viridis', aspect='auto')
        axes[1].set_title('Model')
        axes[1].set_xlabel('Column')
        axes[1].set_ylabel('Row')
        plt.colorbar(im2, ax=axes[1], label='Intensity')

        # Plot 3: Difference (im - model)
        # Use symmetric colormap limits for difference
        vmax_diff = np.max(np.abs(diff))
        im3 = axes[2].imshow(diff, cmap='seismic', aspect='auto',
                            vmin=-vmax_diff, vmax=vmax_diff)
        axes[2].set_title('Difference (Input - Model)')
        axes[2].set_xlabel('Column')
        axes[2].set_ylabel('Row')
        plt.colorbar(im3, ax=axes[2], label='Residual')

        plt.tight_layout()
        plt.savefig(png_filename, dpi=100, bbox_inches='tight')
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
