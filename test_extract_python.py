"""
Test suite for slit decomposition implementations using pytest fixtures.

This module tests both the Python and C implementations, comparing their
results and performance.
"""

import numpy as np
import pytest
import time
from typing import Tuple, Callable, Optional
from extract_python import extract as extract_python

try:
    import charslit
    HAS_C_VERSION = True
except ImportError:
    HAS_C_VERSION = False


def extract_c_wrapper(
    ncols: int,
    nrows: int,
    im: np.ndarray,
    pix_unc: np.ndarray,
    mask: np.ndarray,
    ycen: np.ndarray,
    slitdeltas: np.ndarray,
    osample: int,
    lambda_sP: float,
    lambda_sL: float,
    maxiter: int,
    slit_func_in: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper for C implementation to match Python interface.

    Returns
    -------
    sP : np.ndarray
        Extracted spectrum, shape (ncols,)
    sL : np.ndarray
        Slit function, shape (ny,)
    model : np.ndarray
        Model image, shape (nrows, ncols)
    unc : np.ndarray
        Spectrum uncertainties, shape (ncols,)
    info : np.ndarray
        Fit information [success, cost, status, iterations, delta_x]
    """
    result = charslit.extract(
        im, pix_unc, mask.astype(np.int32), ycen, slitdeltas,
        osample, lambda_sP, lambda_sL, maxiter, slit_func_in
    )
    # Unpack: (result_code, sL, sP, model, unc, info, img_mad, img_mad_mask)
    _, sL, sP, model, unc, info, _, _ = result
    return sP, sL, model, unc, info


# Fixture for extraction implementations
@pytest.fixture(params=['python', 'c'] if HAS_C_VERSION else ['python'])
def extract_impl(request):
    """Fixture providing different extraction implementations."""
    if request.param == 'python':
        return {
            'name': 'Python',
            'func': extract_python,
            'available': True
        }
    elif request.param == 'c':
        return {
            'name': 'C',
            'func': extract_c_wrapper,
            'available': HAS_C_VERSION
        }


# Test data fixtures
@pytest.fixture
def test_data_small():
    """Small test dataset for quick testing."""
    return create_test_data(ncols=30, nrows=15, osample=5, noise_level=0.1)


@pytest.fixture
def test_data_medium():
    """Medium test dataset."""
    return create_test_data(ncols=50, nrows=20, osample=5, noise_level=0.1)


@pytest.fixture
def test_data_large():
    """Large test dataset for performance testing."""
    return create_test_data(ncols=100, nrows=40, osample=5, noise_level=0.1)


@pytest.fixture
def test_data_curved():
    """Test dataset with significant slit curvature."""
    return create_test_data(ncols=50, nrows=20, osample=5,
                           noise_level=0.1, curvature=2.0)


@pytest.fixture
def test_data_noisy():
    """Test dataset with high noise."""
    return create_test_data(ncols=50, nrows=20, osample=5, noise_level=0.5)


def create_test_data(
    ncols: int = 50,
    nrows: int = 20,
    osample: int = 5,
    noise_level: float = 0.1,
    curvature: float = 0.5
) -> dict:
    """
    Create synthetic test data for extraction.

    Parameters
    ----------
    ncols : int
        Number of columns
    nrows : int
        Number of rows
    osample : int
        Oversampling factor
    noise_level : float
        Relative noise level (fraction of signal)
    curvature : float
        Slit curvature strength

    Returns
    -------
    dict with keys:
        im, pix_unc, mask, ycen, slitdeltas,
        true_spectrum, true_slit, ncols, nrows, osample
    """
    ny = osample * (nrows + 1) + 1

    # Create a simple spectrum (Gaussian-like)
    x = np.arange(ncols)
    spectrum = 1000 * np.exp(-((x - ncols/2)**2) / (2 * (ncols/10)**2))

    # Create a simple slit function (normalized Gaussian)
    iy = np.arange(ny)
    slit_func = np.exp(-((iy - ny/2)**2) / (2 * (ny/6)**2))
    slit_func /= np.sum(slit_func) / osample

    # Slit deltas with curvature
    slitdeltas = np.zeros(ny)
    for i in range(ny):
        y_norm = (i - ny/2) / (ny/2)
        slitdeltas[i] = curvature * y_norm**2

    # Create ycen (center line position)
    ycen = np.ones(ncols) * 0.5

    # Create image (rough approximation)
    im = np.zeros((nrows, ncols))
    for x_idx in range(ncols):
        for y_idx in range(nrows):
            # Sum contributions from nearby subpixels
            for iy_offset in range(-osample, osample+1):
                iy_idx = (y_idx - nrows//2) * osample + osample//2 + iy_offset + ny//2
                if 0 <= iy_idx < ny:
                    im[y_idx, x_idx] += spectrum[x_idx] * slit_func[iy_idx] / osample

    # Add Poisson noise
    signal_level = np.maximum(im, 1)
    noise = np.random.normal(0, noise_level * np.sqrt(signal_level), im.shape)
    im += noise

    # Create uncertainty array
    pix_unc = noise_level * np.sqrt(np.maximum(im, 1))

    # Create mask (all good pixels)
    mask = np.ones((nrows, ncols), dtype=np.uint8)

    return {
        'im': im,
        'pix_unc': pix_unc,
        'mask': mask,
        'ycen': ycen,
        'slitdeltas': slitdeltas,
        'true_spectrum': spectrum,
        'true_slit': slit_func,
        'ncols': ncols,
        'nrows': nrows,
        'osample': osample
    }


def run_extraction(extract_func, test_data, lambda_sP=0.1, lambda_sL=0.1, maxiter=20):
    """
    Run extraction and return results with timing.

    Returns
    -------
    dict with keys: sP, sL, model, unc, info, runtime
    """
    start_time = time.time()
    sP, sL, model, unc, info = extract_func(
        test_data['ncols'], test_data['nrows'],
        test_data['im'], test_data['pix_unc'], test_data['mask'],
        test_data['ycen'], test_data['slitdeltas'],
        test_data['osample'], lambda_sP, lambda_sL, maxiter
    )
    runtime = time.time() - start_time

    return {
        'sP': sP,
        'sL': sL,
        'model': model,
        'unc': unc,
        'info': info,
        'runtime': runtime
    }


# ============================================================================
# Tests
# ============================================================================

class TestExtraction:
    """Test suite for slit decomposition extraction."""

    def test_extraction_runs(self, extract_impl, test_data_small):
        """Test that extraction runs without errors."""
        if not extract_impl['available']:
            pytest.skip(f"{extract_impl['name']} implementation not available")

        result = run_extraction(extract_impl['func'], test_data_small)

        # Check that outputs have correct shapes
        assert result['sP'].shape == (test_data_small['ncols'],)
        assert result['sL'].shape == (
            test_data_small['osample'] * (test_data_small['nrows'] + 1) + 1,
        )
        assert result['model'].shape == (
            test_data_small['nrows'], test_data_small['ncols']
        )
        assert result['unc'].shape == (test_data_small['ncols'],)
        assert result['info'].shape == (5,)

    def test_extraction_converges(self, extract_impl, test_data_medium):
        """Test that extraction converges successfully."""
        if not extract_impl['available']:
            pytest.skip(f"{extract_impl['name']} implementation not available")

        result = run_extraction(extract_impl['func'], test_data_medium)

        # Check convergence
        success = result['info'][0]
        iterations = int(result['info'][3])

        assert success == 1.0, f"Extraction failed with {extract_impl['name']}"
        assert 1 <= iterations <= 20, f"Iterations out of range: {iterations}"

    def test_slit_function_normalized(self, extract_impl, test_data_medium):
        """Test that slit function is properly normalized."""
        if not extract_impl['available']:
            pytest.skip(f"{extract_impl['name']} implementation not available")

        result = run_extraction(extract_impl['func'], test_data_medium)

        # Slit function should sum to osample (approximately)
        sL_sum = np.sum(result['sL'])
        expected_sum = test_data_medium['osample']

        assert np.abs(sL_sum - expected_sum) < 1e-6, \
            f"Slit function not normalized: sum={sL_sum}, expected={expected_sum}"

    def test_model_matches_data(self, extract_impl, test_data_medium):
        """Test that model reasonably matches input data."""
        if not extract_impl['available']:
            pytest.skip(f"{extract_impl['name']} implementation not available")

        result = run_extraction(extract_impl['func'], test_data_medium)

        # Calculate residuals
        residuals = test_data_medium['im'] - result['model']
        rms = np.sqrt(np.mean(residuals**2))
        signal_level = np.mean(np.abs(test_data_medium['im']))

        # RMS should be reasonably small compared to signal
        relative_rms = rms / signal_level
        assert relative_rms < 2.0, \
            f"Model fit poor: relative RMS = {relative_rms:.2f}"

    def test_handles_curved_slit(self, extract_impl, test_data_curved):
        """Test extraction with significant slit curvature."""
        if not extract_impl['available']:
            pytest.skip(f"{extract_impl['name']} implementation not available")

        result = run_extraction(extract_impl['func'], test_data_curved)

        # Should still converge
        success = result['info'][0]
        assert success == 1.0, "Failed to handle curved slit"

    def test_handles_noisy_data(self, extract_impl, test_data_noisy):
        """Test extraction with high noise levels."""
        if not extract_impl['available']:
            pytest.skip(f"{extract_impl['name']} implementation not available")

        result = run_extraction(extract_impl['func'], test_data_noisy)

        # Should still converge (though may take more iterations)
        success = result['info'][0]
        assert success == 1.0, "Failed to handle noisy data"


@pytest.mark.skipif(not HAS_C_VERSION, reason="C version not available")
class TestComparison:
    """Compare Python and C implementations."""

    def test_results_match(self, test_data_medium):
        """Test that Python and C implementations give similar results."""
        # Run both implementations
        result_py = run_extraction(extract_python, test_data_medium)
        result_c = run_extraction(extract_c_wrapper, test_data_medium)

        # Compare spectra
        sP_diff = np.abs(result_py['sP'] - result_c['sP'])
        assert np.max(sP_diff) < 1e-6, \
            f"Spectra differ: max diff = {np.max(sP_diff):.2e}"

        # Compare slit functions
        sL_diff = np.abs(result_py['sL'] - result_c['sL'])
        assert np.max(sL_diff) < 1e-6, \
            f"Slit functions differ: max diff = {np.max(sL_diff):.2e}"

        # Compare models
        model_diff = np.abs(result_py['model'] - result_c['model'])
        assert np.max(model_diff) < 1e-6, \
            f"Models differ: max diff = {np.max(model_diff):.2e}"

    def test_convergence_matches(self, test_data_medium):
        """Test that both implementations converge to same cost."""
        result_py = run_extraction(extract_python, test_data_medium)
        result_c = run_extraction(extract_c_wrapper, test_data_medium)

        cost_py = result_py['info'][1]
        cost_c = result_c['info'][1]

        # Costs should be very similar
        assert np.abs(cost_py - cost_c) < 1e-3, \
            f"Costs differ: Python={cost_py:.6f}, C={cost_c:.6f}"

    def test_performance_comparison(self, test_data_large):
        """Compare performance of Python vs C implementation."""
        # Run both implementations
        result_py = run_extraction(extract_python, test_data_large)
        result_c = run_extraction(extract_c_wrapper, test_data_large)

        # Calculate speedup
        speedup = result_py['runtime'] / result_c['runtime']

        print(f"\nPerformance comparison:")
        print(f"  Python: {result_py['runtime']:.3f} seconds")
        print(f"  C:      {result_c['runtime']:.3f} seconds")
        print(f"  Speedup: {speedup:.1f}x")

        # C should be significantly faster
        assert speedup > 10, \
            f"C version not fast enough: only {speedup:.1f}x faster"


# ============================================================================
# Detailed comparison report
# ============================================================================

def generate_comparison_report(test_data_name: str, test_data: dict):
    """Generate detailed comparison report between implementations."""
    if not HAS_C_VERSION:
        print("\nC version not available - skipping comparison")
        return

    print(f"\n{'='*70}")
    print(f"Comparison Report: {test_data_name}")
    print(f"{'='*70}")
    print(f"Size: {test_data['ncols']} × {test_data['nrows']} pixels")
    print(f"Oversampling: {test_data['osample']}x")

    # Run both implementations
    print("\nRunning Python implementation...")
    result_py = run_extraction(extract_python, test_data)

    print("Running C implementation...")
    result_c = run_extraction(extract_c_wrapper, test_data)

    # Timing comparison
    print(f"\n{'Timing':-^70}")
    print(f"  Python: {result_py['runtime']:.4f} seconds")
    print(f"  C:      {result_c['runtime']:.4f} seconds")
    print(f"  Speedup: {result_py['runtime']/result_c['runtime']:.1f}x")

    # Convergence comparison
    print(f"\n{'Convergence':-^70}")
    print(f"  {'':20s} {'Python':>15s} {'C':>15s} {'Difference':>15s}")
    print(f"  {'Success':20s} {result_py['info'][0]:15.0f} "
          f"{result_c['info'][0]:15.0f} {result_py['info'][0]-result_c['info'][0]:15.6f}")
    print(f"  {'Cost':20s} {result_py['info'][1]:15.6f} "
          f"{result_c['info'][1]:15.6f} {result_py['info'][1]-result_c['info'][1]:15.6e}")
    print(f"  {'Status':20s} {result_py['info'][2]:15.0f} "
          f"{result_c['info'][2]:15.0f} {result_py['info'][2]-result_c['info'][2]:15.6f}")
    print(f"  {'Iterations':20s} {result_py['info'][3]:15.0f} "
          f"{result_c['info'][3]:15.0f} {result_py['info'][3]-result_c['info'][3]:15.6f}")

    # Spectrum comparison
    print(f"\n{'Spectrum Comparison':-^70}")
    sP_diff = result_py['sP'] - result_c['sP']
    print(f"  Max absolute difference: {np.max(np.abs(sP_diff)):15.6e}")
    print(f"  Mean absolute difference: {np.mean(np.abs(sP_diff)):15.6e}")
    print(f"  RMS difference: {np.sqrt(np.mean(sP_diff**2)):15.6e}")
    print(f"  Max relative difference: {np.max(np.abs(sP_diff/np.maximum(result_c['sP'],1e-10))):15.6e}")

    # Slit function comparison
    print(f"\n{'Slit Function Comparison':-^70}")
    sL_diff = result_py['sL'] - result_c['sL']
    print(f"  Max absolute difference: {np.max(np.abs(sL_diff)):15.6e}")
    print(f"  Mean absolute difference: {np.mean(np.abs(sL_diff)):15.6e}")
    print(f"  RMS difference: {np.sqrt(np.mean(sL_diff**2)):15.6e}")

    # Model comparison
    print(f"\n{'Model Comparison':-^70}")
    model_diff = result_py['model'] - result_c['model']
    print(f"  Max absolute difference: {np.max(np.abs(model_diff)):15.6e}")
    print(f"  Mean absolute difference: {np.mean(np.abs(model_diff)):15.6e}")
    print(f"  RMS difference: {np.sqrt(np.mean(model_diff**2)):15.6e}")

    # Residuals comparison
    print(f"\n{'Residual Analysis':-^70}")
    residuals_py = test_data['im'] - result_py['model']
    residuals_c = test_data['im'] - result_c['model']
    print(f"  {'':20s} {'Python':>15s} {'C':>15s}")
    print(f"  {'RMS residual':20s} {np.sqrt(np.mean(residuals_py**2)):15.2f} "
          f"{np.sqrt(np.mean(residuals_c**2)):15.2f}")
    print(f"  {'Mean residual':20s} {np.mean(residuals_py):15.6f} "
          f"{np.mean(residuals_c):15.6f}")
    print(f"  {'Std residual':20s} {np.std(residuals_py):15.2f} "
          f"{np.std(residuals_c):15.2f}")


# ============================================================================
# Main execution for manual testing
# ============================================================================

if __name__ == "__main__":
    # Run pytest
    print("Running pytest suite...")
    pytest.main([__file__, "-v", "--tb=short"])

    # Generate detailed comparison reports
    if HAS_C_VERSION:
        print("\n" + "="*70)
        print("Generating Detailed Comparison Reports")
        print("="*70)

        test_cases = {
            'Small Dataset': create_test_data(30, 15, 5),
            'Medium Dataset': create_test_data(50, 20, 5),
            'Large Dataset': create_test_data(100, 40, 5),
            'Curved Slit': create_test_data(50, 20, 5, curvature=2.0),
            'Noisy Data': create_test_data(50, 20, 5, noise_level=0.5),
        }

        for name, data in test_cases.items():
            generate_comparison_report(name, data)

    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70)
