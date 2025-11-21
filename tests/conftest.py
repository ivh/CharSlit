"""Pytest configuration and fixtures for slitchar tests."""

import numpy as np
import pytest


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
