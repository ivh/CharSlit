#!/usr/bin/env python3
"""
Measure slit deltas using cross-correlation between rows.

This script uses a cross-correlation approach to determine horizontal shifts
between rows in echelle spectrograph data. Unlike the Gaussian-fitting approach,
this method:
  1. Uses a reference row (typically the middle row)
  2. Cross-correlates each row with the reference
  3. Finds the sub-pixel shift from the correlation peak

This naturally handles all peaks together and doesn't require peak detection.
"""

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.signal import correlate


@dataclass
class XCorrConfig:
    """Configuration for cross-correlation analysis."""

    # Cross-correlation parameters
    upsample_factor: int = 10  # Upsample for sub-pixel precision
    max_shift: float = 20.0  # Maximum expected shift (pixels)
    reference_row: str = "median"  # 'median', 'middle', or row index

    # Directory configuration
    data_dir: str = "data"
    plots_dir: str = "plots"


DEFAULT_CONFIG = XCorrConfig()


def create_reference_row(
    data: np.ndarray, config: XCorrConfig
) -> tuple[np.ndarray, int]:
    """
    Create a reference row for cross-correlation.

    Args:
        data: 2D array (rows x cols)
        config: Configuration

    Returns:
        Tuple of (reference_row, reference_row_index)
    """
    num_rows = data.shape[0]

    if config.reference_row == "median":
        # Use median of middle 5 rows (sharper reference, less smearing)
        mid_idx = num_rows // 2
        start_idx = max(0, mid_idx - 2)
        end_idx = min(num_rows, mid_idx + 3)
        reference = np.median(data[start_idx:end_idx], axis=0)
        ref_idx = mid_idx  # For display purposes
    elif config.reference_row == "middle":
        # Use middle row
        ref_idx = num_rows // 2
        reference = data[ref_idx]
    elif isinstance(config.reference_row, int):
        # Use specified row
        ref_idx = config.reference_row
        reference = data[ref_idx]
    else:
        raise ValueError(f"Invalid reference_row: {config.reference_row}")

    return reference, ref_idx


def find_correlation_peak(
    correlation: np.ndarray, upsample_factor: int
) -> tuple[float, float]:
    """
    Find the peak of the correlation function with sub-pixel precision.

    Args:
        correlation: Cross-correlation array
        upsample_factor: Factor by which lags were upsampled

    Returns:
        Tuple of (peak_position, peak_strength)
    """
    # Find the peak
    peak_idx = np.argmax(correlation)
    peak_value = correlation[peak_idx]

    # Convert to shift in original pixel units
    # The correlation is computed with upsampled data, so we need to scale back
    center = len(correlation) // 2
    shift_upsampled = peak_idx - center
    shift_pixels = shift_upsampled / upsample_factor

    return shift_pixels, peak_value


def cross_correlate_rows(
    data: np.ndarray, reference: np.ndarray, config: XCorrConfig
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cross-correlate each row with the reference to find shifts.

    Args:
        data: 2D array (rows x cols)
        reference: 1D reference row
        config: Configuration

    Returns:
        Tuple of (shifts, correlation_strengths)
    """
    num_rows, num_cols = data.shape
    shifts = np.zeros(num_rows)
    correlation_strengths = np.zeros(num_rows)

    # Upsample for sub-pixel precision
    upsampled_length = num_cols * config.upsample_factor
    max_lag_upsampled = int(config.max_shift * config.upsample_factor)

    # Prepare upsampled reference
    ref_upsampled = np.interp(
        np.linspace(0, num_cols - 1, upsampled_length),
        np.arange(num_cols),
        reference,
    )

    for row_idx in range(num_rows):
        row_data = data[row_idx]

        # Upsample the row
        row_upsampled = np.interp(
            np.linspace(0, num_cols - 1, upsampled_length),
            np.arange(num_cols),
            row_data,
        )

        # Cross-correlate (mode='same' keeps same length as inputs)
        correlation = correlate(row_upsampled, ref_upsampled, mode="same")

        # Restrict to region around zero lag (within max_shift)
        center = len(correlation) // 2
        start = max(0, center - max_lag_upsampled)
        end = min(len(correlation), center + max_lag_upsampled + 1)
        correlation_restricted = correlation[start:end]

        # Find peak in restricted region
        peak_idx_restricted = np.argmax(correlation_restricted)
        peak_value = correlation_restricted[peak_idx_restricted]

        # Convert back to shift in original pixels
        lag_upsampled = peak_idx_restricted - (len(correlation_restricted) // 2)
        shift_pixels = lag_upsampled / config.upsample_factor

        shifts[row_idx] = shift_pixels
        correlation_strengths[row_idx] = peak_value

    return shifts, correlation_strengths


def normalize_shifts_to_median(shifts: np.ndarray) -> np.ndarray:
    """
    Convert absolute shifts to offsets from median.

    This makes the output compatible with the Gaussian-fitting approach,
    where offsets are relative to the median position.

    Args:
        shifts: Array of absolute shifts

    Returns:
        Array of offsets from median
    """
    median_shift = np.median(shifts)
    return shifts - median_shift


def calculate_quality_metrics(
    shifts: np.ndarray, correlation_strengths: np.ndarray
) -> dict:
    """
    Calculate quality metrics for the cross-correlation analysis.

    Args:
        shifts: Array of shifts
        correlation_strengths: Array of correlation peak strengths

    Returns:
        Dictionary of quality metrics
    """
    # Robust statistics
    median_shift = np.median(shifts)
    mad = np.median(np.abs(shifts - median_shift))
    robust_std = 1.4826 * mad

    # Correlation strength statistics
    min_corr = np.min(correlation_strengths)
    max_corr = np.max(correlation_strengths)
    mean_corr = np.mean(correlation_strengths)

    # Identify weak correlations (potential problems)
    threshold = mean_corr * 0.5  # Less than 50% of mean
    weak_rows = np.where(correlation_strengths < threshold)[0]

    return {
        "median_shift": median_shift,
        "mad": mad,
        "robust_std": robust_std,
        "mean_std": np.std(shifts),
        "min_correlation": min_corr,
        "max_correlation": max_corr,
        "mean_correlation": mean_corr,
        "weak_correlation_rows": weak_rows.tolist(),
        "num_weak_rows": len(weak_rows),
    }


def process_fits_file(filename: str, config: XCorrConfig = DEFAULT_CONFIG) -> dict:
    """
    Process a FITS file using cross-correlation.

    Args:
        filename: Path to FITS file
        config: Configuration

    Returns:
        Dictionary containing results
    """
    print(f"Processing {filename}...")

    with fits.open(filename) as hdul:
        data = hdul[0].data

    # Create reference row
    reference, ref_row_idx = create_reference_row(data, config)

    # Cross-correlate all rows
    shifts, correlation_strengths = cross_correlate_rows(data, reference, config)

    # Convert to offsets from median (for compatibility)
    median_offsets = normalize_shifts_to_median(shifts)

    # Calculate quality metrics
    quality_metrics = calculate_quality_metrics(median_offsets, correlation_strengths)

    # Calculate statistics for output
    avg_offset = np.mean(median_offsets)
    std_offset = np.std(median_offsets)

    print(f"  Average offset: {avg_offset:.4f} ± {std_offset:.4f} pixels")
    print(f"  Robust std: {quality_metrics['robust_std']:.4f} pixels")
    print(
        f"  Correlation: min={quality_metrics['min_correlation']:.2e}, "
        f"max={quality_metrics['max_correlation']:.2e}, "
        f"mean={quality_metrics['mean_correlation']:.2e}"
    )
    if quality_metrics["num_weak_rows"] > 0:
        print(f"  ⚠️  {quality_metrics['num_weak_rows']} rows with weak correlation")

    return {
        "filename": filename,
        "median_offsets": median_offsets,
        "shifts": shifts,
        "correlation_strengths": correlation_strengths,
        "reference_row": reference,
        "reference_row_index": ref_row_idx,
        "quality_metrics": quality_metrics,
        "avg_offset": avg_offset,
        "std_offset": std_offset,
    }


def plot_results(results: list[dict], config: XCorrConfig = DEFAULT_CONFIG) -> None:
    """
    Generate diagnostic plots for cross-correlation analysis.

    Args:
        results: List of result dictionaries
        config: Configuration
    """
    os.makedirs(config.plots_dir, exist_ok=True)

    for result in results:
        filename = result["filename"]
        basename = os.path.basename(filename).replace(".fits", "")
        median_offsets = result["median_offsets"]
        correlation_strengths = result["correlation_strengths"]
        quality_metrics = result["quality_metrics"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Cross-Correlation Analysis - {basename}", fontsize=14)

        # Plot 1: Median offsets (slit deltas)
        ax1 = axes[0, 0]
        ax1.plot(np.arange(len(median_offsets)), median_offsets, "o-", markersize=3)
        ax1.set_xlabel("Row Index")
        ax1.set_ylabel("Slit Delta (pixels)")
        ax1.set_title("Slit Deltas (Offsets from Median)")
        ax1.grid(True, alpha=0.3)

        # Mark weak correlation rows
        weak_rows = quality_metrics["weak_correlation_rows"]
        if weak_rows:
            ax1.plot(
                weak_rows,
                median_offsets[weak_rows],
                "rx",
                markersize=8,
                label="Weak correlation",
                markeredgewidth=2,
            )
            ax1.legend()

        # Plot 2: Correlation strengths
        ax2 = axes[0, 1]
        ax2.plot(
            np.arange(len(correlation_strengths)),
            correlation_strengths,
            "o-",
            markersize=3,
        )
        ax2.set_xlabel("Row Index")
        ax2.set_ylabel("Correlation Peak Strength")
        ax2.set_title("Cross-Correlation Quality")
        ax2.grid(True, alpha=0.3)

        # Mark threshold
        threshold = quality_metrics["mean_correlation"] * 0.5
        ax2.axhline(
            y=threshold, color="r", linestyle="--", label="Weak threshold", linewidth=1
        )
        ax2.legend()

        # Plot 3: Distribution of offsets
        ax3 = axes[1, 0]
        ax3.hist(median_offsets, bins=50, alpha=0.7, edgecolor="black")
        ax3.axvline(
            x=quality_metrics["median_shift"],
            color="r",
            linestyle="--",
            label="Median",
            linewidth=2,
        )
        ax3.set_xlabel("Offset (pixels)")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Offset Distribution")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        ax4.axis("off")

        summary_text = "Quality Metrics:\n\n"
        summary_text += f"Total rows: {len(median_offsets)}\n"
        summary_text += f"Median shift: {quality_metrics['median_shift']:.4f} px\n"
        summary_text += f"MAD: {quality_metrics['mad']:.4f} px\n"
        summary_text += f"Robust std: {quality_metrics['robust_std']:.4f} px\n"
        summary_text += f"Mean std: {quality_metrics['mean_std']:.4f} px\n\n"
        summary_text += "Correlation strength:\n"
        summary_text += f"  Min: {quality_metrics['min_correlation']:.2e}\n"
        summary_text += f"  Max: {quality_metrics['max_correlation']:.2e}\n"
        summary_text += f"  Mean: {quality_metrics['mean_correlation']:.2e}\n\n"
        summary_text += f"Weak rows: {quality_metrics['num_weak_rows']}\n"

        ax4.text(
            0.05,
            0.95,
            summary_text,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(config.plots_dir, f"{basename}_xcorr_diagnostics.png"),
            dpi=150,
        )
        plt.close()


def save_results(
    results: list[dict], config: XCorrConfig = DEFAULT_CONFIG
) -> list[str]:
    """
    Save results to NPZ files (same format as Gaussian approach).

    Args:
        results: List of result dictionaries
        config: Configuration

    Returns:
        List of saved file paths
    """
    saved_files = []

    for result in results:
        basename = os.path.splitext(os.path.basename(result["filename"]))[0]
        output_file = os.path.join(config.data_dir, f"slitdeltas_{basename}.npz")

        # Save with same structure as original script for compatibility
        np.savez(
            output_file,
            filename=result["filename"],
            avg_offset=result["avg_offset"],
            std_offset=result["std_offset"],
            median_offsets=result["median_offsets"],
        )
        saved_files.append(output_file)

    return saved_files


def main():
    """Main function to process all test data files."""
    config = DEFAULT_CONFIG

    # Check for data directory
    if not os.path.exists(config.data_dir):
        print(f"Error: {config.data_dir}/ directory not found!")
        return

    # Find all FITS files
    fits_files = [
        os.path.join(config.data_dir, f)
        for f in os.listdir(config.data_dir)
        if f.endswith(".fits")
    ]

    if not fits_files:
        print(f"No test data FITS files found in {config.data_dir}/!")
        return

    print(f"Found {len(fits_files)} FITS files to process\n")
    print("Using cross-correlation method")
    print(f"  Upsample factor: {config.upsample_factor}x")
    print(f"  Max shift: ±{config.max_shift} pixels")
    print(f"  Reference: {config.reference_row}\n")

    # Process each file
    results = []
    for fits_file in fits_files:
        result = process_fits_file(fits_file, config)
        results.append(result)
        print()

    # Generate plots
    print("Generating plots...")
    plot_results(results, config)

    # Save results
    print("Saving results...")
    saved_files = save_results(results, config)

    print(f"\nResults saved to {len(saved_files)} NPZ files:")
    for file in saved_files:
        print(f"  - {file}")

    print(f"\nPlots saved to {config.plots_dir}/")


if __name__ == "__main__":
    main()
