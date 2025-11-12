"""
Show detailed statistics for the Python slit decomposition implementation.
"""

import numpy as np
import time
from extract_python import extract
from test_extract_python import create_test_data


def analyze_extraction(test_name, test_data, lambda_sP=0.1, lambda_sL=0.1, maxiter=20):
    """Run extraction and show detailed statistics."""
    print(f"\n{'='*70}")
    print(f"Test Case: {test_name}")
    print(f"{'='*70}")

    # Dataset info
    print(f"\nDataset Information:")
    print(f"  Size: {test_data['ncols']} × {test_data['nrows']} pixels")
    print(f"  Oversampling: {test_data['osample']}x")
    print(f"  Subpixel array size (ny): {test_data['osample'] * (test_data['nrows'] + 1) + 1}")
    print(f"  Image statistics:")
    print(f"    Min: {test_data['im'].min():.2f}")
    print(f"    Max: {test_data['im'].max():.2f}")
    print(f"    Mean: {test_data['im'].mean():.2f}")
    print(f"    Std: {test_data['im'].std():.2f}")
    print(f"  Total flux: {test_data['im'].sum():.2f}")

    # Run extraction
    print(f"\nExtraction Parameters:")
    print(f"  lambda_sP (spectrum smoothing): {lambda_sP}")
    print(f"  lambda_sL (slit smoothing): {lambda_sL}")
    print(f"  Max iterations: {maxiter}")

    print(f"\nRunning extraction...")
    start_time = time.time()

    sP, sL, model, unc, info = extract(
        test_data['ncols'], test_data['nrows'],
        test_data['im'], test_data['pix_unc'], test_data['mask'],
        test_data['ycen'], test_data['slitdeltas'],
        test_data['osample'], lambda_sP, lambda_sL, maxiter
    )

    runtime = time.time() - start_time

    # Extraction results
    print(f"\n{'Extraction Results':-^70}")
    print(f"  Runtime: {runtime:.4f} seconds")
    print(f"  Success: {info[0]} (1.0 = converged)")
    print(f"  Final cost (reduced χ²): {info[1]:.6f}")
    print(f"  Status: {info[2]} (1.0 = normal, -1 = max iter)")
    print(f"  Iterations: {int(info[3])}")
    print(f"  Delta_x (max curvature shift): {int(info[4])} pixels")

    # Spectrum statistics
    print(f"\n{'Extracted Spectrum':-^70}")
    print(f"  Shape: {sP.shape}")
    print(f"  Non-zero pixels: {np.count_nonzero(sP)}/{len(sP)}")
    print(f"  Statistics:")
    non_zero_sP = sP[sP > 0]
    if len(non_zero_sP) > 0:
        print(f"    Min (non-zero): {non_zero_sP.min():.2f}")
        print(f"    Max: {sP.max():.2f}")
        print(f"    Mean (non-zero): {non_zero_sP.mean():.2f}")
        print(f"    Std (non-zero): {non_zero_sP.std():.2f}")
        print(f"  Total flux: {sP.sum():.2f}")
        print(f"  SNR (mean/std): {non_zero_sP.mean()/non_zero_sP.std():.2f}")

    # Slit function statistics
    print(f"\n{'Extracted Slit Function':-^70}")
    print(f"  Shape: {sL.shape}")
    print(f"  Statistics:")
    print(f"    Min: {sL.min():.6f}")
    print(f"    Max: {sL.max():.6f}")
    print(f"    Mean: {sL.mean():.6f}")
    print(f"    Sum/osample: {sL.sum()/test_data['osample']:.6f} (should be ~1.0)")
    print(f"  Peak position: subpixel {np.argmax(sL)}")
    print(f"  FWHM estimate: {estimate_fwhm(sL):.2f} subpixels")

    # Model quality
    print(f"\n{'Model Quality':-^70}")
    residuals = test_data['im'] - model
    print(f"  Model statistics:")
    print(f"    Min: {model.min():.2f}")
    print(f"    Max: {model.max():.2f}")
    print(f"    Mean: {model.mean():.2f}")
    print(f"    Total flux: {model.sum():.2f}")
    print(f"  Residuals (data - model):")
    print(f"    Mean: {residuals.mean():.6f}")
    print(f"    Std: {residuals.std():.2f}")
    print(f"    RMS: {np.sqrt(np.mean(residuals**2)):.2f}")
    print(f"    Max absolute: {np.abs(residuals).max():.2f}")

    # Relative errors
    rel_flux_error = abs(model.sum() - test_data['im'].sum()) / test_data['im'].sum() * 100
    print(f"  Flux conservation:")
    print(f"    Input flux: {test_data['im'].sum():.2f}")
    print(f"    Model flux: {model.sum():.2f}")
    print(f"    Relative error: {rel_flux_error:.3f}%")

    # Goodness of fit
    chi2 = np.sum((residuals / np.maximum(test_data['pix_unc'], 1))**2)
    dof = test_data['ncols'] * test_data['nrows'] - (test_data['ncols'] + len(sL))
    reduced_chi2 = chi2 / dof
    print(f"  Goodness of fit:")
    print(f"    χ²: {chi2:.2f}")
    print(f"    Degrees of freedom: {dof}")
    print(f"    Reduced χ²: {reduced_chi2:.4f}")

    # Uncertainty statistics
    print(f"\n{'Uncertainty Estimates':-^70}")
    non_zero_unc = unc[unc > 0]
    if len(non_zero_unc) > 0:
        print(f"  Uncertainty statistics:")
        print(f"    Min (non-zero): {non_zero_unc.min():.2f}")
        print(f"    Max: {unc.max():.2f}")
        print(f"    Mean (non-zero): {non_zero_unc.mean():.2f}")
        print(f"  SNR estimates (spectrum / uncertainty):")
        valid = (sP > 0) & (unc > 0)
        if np.any(valid):
            snr = sP[valid] / unc[valid]
            print(f"    Mean SNR: {snr.mean():.2f}")
            print(f"    Median SNR: {np.median(snr):.2f}")
            print(f"    Max SNR: {snr.max():.2f}")

    return {
        'runtime': runtime,
        'iterations': int(info[3]),
        'cost': info[1],
        'rms_residual': np.sqrt(np.mean(residuals**2)),
        'flux_error': rel_flux_error
    }


def estimate_fwhm(profile):
    """Estimate FWHM of a profile."""
    half_max = profile.max() / 2
    above_half = profile > half_max
    if np.any(above_half):
        indices = np.where(above_half)[0]
        return indices[-1] - indices[0]
    return 0


def main():
    """Run analysis on multiple test cases."""
    print("\n" + "="*70)
    print("PYTHON SLIT DECOMPOSITION STATISTICS")
    print("="*70)

    # Define test cases with varying parameters
    test_cases = [
        ("Small Dataset (30×15)", create_test_data(30, 15, 5, noise_level=0.1)),
        ("Medium Dataset (50×20)", create_test_data(50, 20, 5, noise_level=0.1)),
        ("Large Dataset (100×40)", create_test_data(100, 40, 5, noise_level=0.1)),
        ("High Oversampling (50×20, os=10)", create_test_data(50, 20, 10, noise_level=0.1)),
        ("Low Curvature (50×20)", create_test_data(50, 20, 5, noise_level=0.1, curvature=0.1)),
        ("High Curvature (50×20)", create_test_data(50, 20, 5, noise_level=0.1, curvature=2.0)),
        ("Low Noise (50×20)", create_test_data(50, 20, 5, noise_level=0.05)),
        ("High Noise (50×20)", create_test_data(50, 20, 5, noise_level=0.5)),
    ]

    # Run all test cases and collect summary statistics
    summary = []
    for name, data in test_cases:
        results = analyze_extraction(name, data)
        summary.append((name, results))

    # Print summary table
    print(f"\n{'='*70}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Test Case':<35s} {'Time':>8s} {'Iter':>5s} {'Cost':>10s} {'RMS':>8s}")
    print("-"*70)
    for name, results in summary:
        print(f"{name:<35s} {results['runtime']:>7.3f}s "
              f"{results['iterations']:>5d} "
              f"{results['cost']:>10.2f} "
              f"{results['rms_residual']:>8.2f}")

    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
