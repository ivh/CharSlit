"""
Test script for the pure Python slit decomposition implementation.

This demonstrates how to use the Python version and compares it with the C version.
"""

import numpy as np
import time
from extract_python import extract as extract_python

try:
    import charslit
    has_c_version = True
except ImportError:
    has_c_version = False
    print("C version not available, will only test Python version")


def create_test_data(ncols=50, nrows=20, osample=5):
    """Create synthetic test data for extraction."""
    ny = osample * (nrows + 1) + 1

    # Create a simple spectrum (Gaussian-like)
    x = np.arange(ncols)
    spectrum = 1000 * np.exp(-((x - ncols/2)**2) / (2 * (ncols/10)**2))

    # Create a simple slit function (normalized Gaussian)
    iy = np.arange(ny)
    slit_func = np.exp(-((iy - ny/2)**2) / (2 * (ny/6)**2))
    slit_func /= np.sum(slit_func) / osample

    # Create synthetic image by convolving spectrum with slit function
    # (simplified - just outer product for now)
    im = np.zeros((nrows, ncols))

    # Simple slit deltas (slight curvature)
    slitdeltas = np.zeros(ny)
    for i in range(ny):
        # Quadratic curvature
        y_norm = (i - ny/2) / (ny/2)
        slitdeltas[i] = 0.5 * y_norm**2  # Small curvature

    # Create ycen (center line position)
    ycen = np.ones(ncols) * 0.5  # Centered at 0.5

    # Create image (rough approximation)
    for x in range(ncols):
        for y in range(nrows):
            # Sum contributions from nearby subpixels
            for iy_offset in range(-osample, osample+1):
                iy = (y - nrows//2) * osample + osample//2 + iy_offset + ny//2
                if 0 <= iy < ny:
                    im[y, x] += spectrum[x] * slit_func[iy] / osample

    # Add some noise
    noise = np.random.normal(0, np.sqrt(np.maximum(im, 1)), im.shape)
    im += noise

    # Create uncertainty array (Poisson noise)
    pix_unc = np.sqrt(np.maximum(im, 1))

    # Create mask (all good pixels)
    mask = np.ones((nrows, ncols), dtype=np.uint8)

    return im, pix_unc, mask, ycen, slitdeltas, spectrum, slit_func


def test_python_version():
    """Test the pure Python implementation."""
    print("\n" + "="*70)
    print("Testing Pure Python Implementation")
    print("="*70)

    # Create test data
    ncols = 50
    nrows = 20
    osample = 5

    print(f"\nCreating test data: ncols={ncols}, nrows={nrows}, osample={osample}")
    im, pix_unc, mask, ycen, slitdeltas, true_spec, true_slit = create_test_data(
        ncols, nrows, osample
    )

    print(f"Image shape: {im.shape}")
    print(f"Image range: [{im.min():.2f}, {im.max():.2f}]")
    print(f"Image mean: {im.mean():.2f}")

    # Run extraction
    lambda_sP = 0.1
    lambda_sL = 0.1
    maxiter = 20

    print(f"\nRunning extraction with:")
    print(f"  lambda_sP = {lambda_sP}")
    print(f"  lambda_sL = {lambda_sL}")
    print(f"  maxiter = {maxiter}")

    start_time = time.time()
    sP, sL, model, unc, info = extract_python(
        ncols, nrows, im, pix_unc, mask, ycen, slitdeltas,
        osample, lambda_sP, lambda_sL, maxiter
    )
    elapsed_time = time.time() - start_time

    print(f"\nExtraction completed in {elapsed_time:.3f} seconds")
    print(f"\nExtraction info:")
    print(f"  Success: {info[0]}")
    print(f"  Cost (reduced chi-square): {info[1]:.6f}")
    print(f"  Status: {info[2]}")
    print(f"  Iterations: {int(info[3])}")
    print(f"  Delta_x: {int(info[4])}")

    print(f"\nExtracted spectrum:")
    print(f"  Shape: {sP.shape}")
    print(f"  Range: [{sP.min():.2f}, {sP.max():.2f}]")
    print(f"  Mean: {sP.mean():.2f}")

    print(f"\nExtracted slit function:")
    print(f"  Shape: {sL.shape}")
    print(f"  Range: [{sL.min():.6f}, {sL.max():.6f}]")
    print(f"  Sum/osample: {np.sum(sL)/osample:.6f} (should be ~1.0)")

    print(f"\nModel image:")
    print(f"  Shape: {model.shape}")
    print(f"  Range: [{model.min():.2f}, {model.max():.2f}]")

    print(f"\nResiduals (data - model):")
    residuals = im - model
    print(f"  Mean: {residuals.mean():.6f}")
    print(f"  Std: {residuals.std():.2f}")
    print(f"  RMS: {np.sqrt(np.mean(residuals**2)):.2f}")

    return sP, sL, model, unc, info


def compare_with_c_version():
    """Compare Python and C implementations."""
    if not has_c_version:
        print("\nC version not available for comparison")
        return

    print("\n" + "="*70)
    print("Comparing Python vs C Implementation")
    print("="*70)

    # Create test data
    ncols = 50
    nrows = 20
    osample = 5

    im, pix_unc, mask, ycen, slitdeltas, _, _ = create_test_data(
        ncols, nrows, osample
    )

    # Run Python version
    print("\nRunning Python version...")
    start_time = time.time()
    sP_py, sL_py, model_py, unc_py, info_py = extract_python(
        ncols, nrows, im, pix_unc, mask, ycen, slitdeltas,
        osample, 0.1, 0.1, 20
    )
    time_py = time.time() - start_time

    # Run C version
    print("Running C version...")
    start_time = time.time()
    result_c = charslit.extract(
        im, pix_unc, mask.astype(np.int32), ycen, slitdeltas,
        osample, 0.1, 0.1, 20
    )
    time_c = time.time() - start_time

    _, sL_c, sP_c, model_c, unc_c, info_c, _, _ = result_c

    print(f"\nTiming:")
    print(f"  Python: {time_py:.3f} seconds")
    print(f"  C:      {time_c:.3f} seconds")
    print(f"  Speedup: {time_py/time_c:.1f}x")

    print(f"\nSpectrum comparison:")
    print(f"  Max difference: {np.max(np.abs(sP_py - sP_c)):.6e}")
    print(f"  Mean difference: {np.mean(np.abs(sP_py - sP_c)):.6e}")
    print(f"  RMS difference: {np.sqrt(np.mean((sP_py - sP_c)**2)):.6e}")

    print(f"\nSlit function comparison:")
    print(f"  Max difference: {np.max(np.abs(sL_py - sL_c)):.6e}")
    print(f"  Mean difference: {np.mean(np.abs(sL_py - sL_c)):.6e}")
    print(f"  RMS difference: {np.sqrt(np.mean((sL_py - sL_c)**2)):.6e}")

    print(f"\nModel comparison:")
    print(f"  Max difference: {np.max(np.abs(model_py - model_c)):.6e}")
    print(f"  Mean difference: {np.mean(np.abs(model_py - model_c)):.6e}")
    print(f"  RMS difference: {np.sqrt(np.mean((model_py - model_c)**2)):.6e}")

    print(f"\nInfo comparison:")
    for i, name in enumerate(['success', 'cost', 'status', 'iterations', 'delta_x']):
        print(f"  {name:12s}: Python={info_py[i]:10.6f}, C={info_c[i]:10.6f}")


if __name__ == "__main__":
    # Test Python version
    test_python_version()

    # Compare with C version if available
    if has_c_version:
        compare_with_c_version()

    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70)
