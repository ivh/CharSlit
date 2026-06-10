#!/usr/bin/env python3
"""
Analyze FITS files to find and track peak positions across rows.

This script identifies spectral peaks in each row of a FITS image, fits Gaussians
to determine precise peak positions, and calculates how peaks shift between rows.
The median offsets are saved for use in spectral extraction algorithms.
"""

import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PeakFindingConfig:
    """Configuration parameters for peak finding and fitting."""

    # Peak detection parameters
    height_multiplier: float = 1.5  # Minimum peak height as multiple of mean
    min_peak_distance: int = 10  # Minimum distance between peaks (pixels)

    # Gaussian fitting parameters
    fit_window_size: int = 15  # Window size around peak for fitting (pixels)
    initial_sigma: float = 5.0  # Initial guess for Gaussian sigma (pixels)

    # Directory configuration
    data_dir: str = "data"
    plots_dir: str = "plots"


DEFAULT_CONFIG = PeakFindingConfig()


# =============================================================================
# Core Functions
# =============================================================================


def gaussian(
    x: np.ndarray, amplitude: float, mean: float, sigma: float, offset: float
) -> np.ndarray:
    """Gaussian function with vertical offset."""
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma**2)) + offset


def fit_gaussian_to_peak(
    x_vals: np.ndarray,
    row_data: np.ndarray,
    peak_loc: int,
    window_size: int,
    initial_sigma: float,
) -> tuple[float, dict]:
    """
    Fit a Gaussian to a peak and return the fitted position.

    Args:
        x_vals: X-axis values (pixel indices)
        row_data: Intensity values for the row
        peak_loc: Initial peak location (pixel index)
        window_size: Window size around peak for fitting
        initial_sigma: Initial guess for Gaussian sigma

    Returns:
        Tuple of (fitted_position, fit_parameters_dict)
    """
    num_cols = len(row_data)

    # Define fitting window
    left_bound = max(0, peak_loc - window_size)
    right_bound = min(num_cols, peak_loc + window_size + 1)

    x_window = x_vals[left_bound:right_bound]
    y_window = row_data[left_bound:right_bound]

    try:
        # Initial parameter guess [amplitude, mean, sigma, offset]
        p0 = [
            row_data[peak_loc] - np.min(y_window),  # amplitude
            peak_loc,  # mean
            initial_sigma,  # sigma
            np.min(y_window),  # offset
        ]

        # Fit the Gaussian
        popt, pcov = curve_fit(gaussian, x_window, y_window, p0=p0)

        # Calculate goodness of fit (reduced chi-squared)
        y_fit = gaussian(x_window, *popt)
        residuals = y_window - y_fit
        dof = len(x_window) - 4  # 4 parameters
        chi2_reduced = np.sum(residuals**2) / dof if dof > 0 else np.inf

        # Calculate uncertainties from covariance matrix
        perr = np.sqrt(np.diag(pcov))

        # Check for unphysical fit results
        position_shift = abs(popt[1] - peak_loc)
        sigma_reasonable = 0.5 < popt[2] < 20.0  # reasonable sigma range
        amplitude_positive = popt[0] > 0

        fit_params = {
            "amplitude": popt[0],
            "position": popt[1],
            "sigma": popt[2],
            "offset": popt[3],
            "fit_failed": False,
            "position_uncertainty": perr[1],
            "sigma_uncertainty": perr[2],
            "chi2_reduced": chi2_reduced,
            "position_shift_from_initial": position_shift,
            "fit_quality_flags": {
                "sigma_reasonable": sigma_reasonable,
                "amplitude_positive": amplitude_positive,
                "large_position_shift": position_shift > window_size / 2,
                "high_chi2": chi2_reduced > 5.0,
            },
        }

        return popt[1], fit_params

    except (RuntimeError, ValueError) as e:
        # If fitting fails, use the original peak location
        fit_params = {
            "amplitude": row_data[peak_loc] - np.min(y_window),
            "position": float(peak_loc),
            "sigma": initial_sigma,
            "offset": np.min(y_window),
            "fit_failed": True,
            "position_uncertainty": np.nan,
            "sigma_uncertainty": np.nan,
            "chi2_reduced": np.inf,
            "position_shift_from_initial": 0.0,
            "fit_quality_flags": {
                "sigma_reasonable": False,
                "amplitude_positive": False,
                "large_position_shift": False,
                "high_chi2": True,
            },
            "fit_error": str(e),
        }

        return float(peak_loc), fit_params


def find_and_fit_peaks(
    data: np.ndarray, config: PeakFindingConfig = DEFAULT_CONFIG
) -> tuple:
    """
    Process FITS data row by row to find and fit peaks.

    1. Identify peaks in each row
    2. Fit a Gaussian to each peak to get the precise position
    3. Calculate absolute offsets from the median position of each peak

    Args:
        data: 2D numpy array (rows x columns)
        config: Configuration for peak finding

    Returns:
        Tuple of (peak_positions, median_offsets, all_peak_fits, quality_diagnostics)
        - peak_positions: List of arrays containing fitted peak positions for each row
        - median_offsets: Array of absolute offsets from median for each row
        - all_peak_fits: List of lists containing fit parameters for each peak
        - quality_diagnostics: Dictionary containing quality metrics and diagnostics
    """
    num_rows, num_cols = data.shape
    x_vals = np.arange(num_cols)

    # Store peak positions for each row
    peak_positions = []
    all_peak_fits = []

    # Find peaks in each row
    for row_idx in range(num_rows):
        row_data = data[row_idx]

        # Find peaks
        peaks, _ = find_peaks(
            row_data,
            height=np.mean(row_data) * config.height_multiplier,
            distance=config.min_peak_distance,
        )

        if len(peaks) == 0:
            peak_positions.append(np.array([]))
            all_peak_fits.append([])
            continue

        # Fit Gaussian to each peak
        fitted_positions = []
        row_fits = []

        for peak_idx, peak_loc in enumerate(peaks):
            fitted_pos, fit_params = fit_gaussian_to_peak(
                x_vals, row_data, peak_loc, config.fit_window_size, config.initial_sigma
            )

            fitted_positions.append(fitted_pos)
            fit_params.update({"peak_idx": peak_idx, "row": row_idx})
            row_fits.append(fit_params)

        # Sort peaks by position to maintain consistent order across rows
        sort_idx = np.argsort(fitted_positions)
        fitted_positions = np.array(fitted_positions)[sort_idx]
        row_fits = [row_fits[i] for i in sort_idx]

        peak_positions.append(fitted_positions)
        all_peak_fits.append(row_fits)

    # Calculate median offsets and get quality diagnostics
    median_offsets, quality_diagnostics = calculate_median_offsets_with_diagnostics(
        peak_positions, num_rows, all_peak_fits
    )

    return peak_positions, median_offsets, all_peak_fits, quality_diagnostics


def match_peaks_across_rows(
    peak_positions: list, valid_rows: list, max_shift: float = 2.0
) -> list[list[tuple[int, float]]]:
    """
    Match peaks across rows by proximity to create continuous trajectories.

    Instead of assuming peak index i in row N corresponds to peak index i in row N+1,
    we match peaks by finding the nearest neighbor within max_shift distance.

    Args:
        peak_positions: List of arrays containing peak positions for each row
        valid_rows: List of row indices that have peaks
        max_shift: Maximum allowed shift between consecutive rows (pixels)

    Returns:
        List of trajectories, where each trajectory is a list of (row_idx, peak_position) tuples
    """
    if not valid_rows:
        return []

    # Initialize trajectories with peaks from first valid row
    first_row = valid_rows[0]
    trajectories = [[(first_row, pos)] for pos in peak_positions[first_row]]

    # Process each subsequent row
    for row_idx in valid_rows[1:]:
        current_peaks = list(peak_positions[row_idx])

        if not current_peaks:
            continue

        # For each existing trajectory, try to extend it with the closest peak
        trajectory_extended = [False] * len(trajectories)
        peak_used = [False] * len(current_peaks)

        # Match trajectories to peaks by proximity
        for traj_idx, trajectory in enumerate(trajectories):
            # Get last known position in this trajectory
            last_row, last_pos = trajectory[-1]

            # Find closest unused peak
            best_peak_idx = None
            best_distance = float("inf")

            for peak_idx, peak_pos in enumerate(current_peaks):
                if peak_used[peak_idx]:
                    continue

                distance = abs(peak_pos - last_pos)

                if distance < best_distance and distance <= max_shift * (
                    row_idx - last_row
                ):
                    best_distance = distance
                    best_peak_idx = peak_idx

            # Extend trajectory if a good match was found
            if best_peak_idx is not None:
                trajectory.append((row_idx, current_peaks[best_peak_idx]))
                trajectory_extended[traj_idx] = True
                peak_used[best_peak_idx] = True

        # Start new trajectories for unmatched peaks
        for peak_idx, peak_pos in enumerate(current_peaks):
            if not peak_used[peak_idx]:
                trajectories.append([(row_idx, peak_pos)])

    # Filter out very short trajectories (< 3 points)
    trajectories = [traj for traj in trajectories if len(traj) >= 3]

    return trajectories


def detect_outliers_in_peak_trajectories(
    peak_positions_by_index: list[list[float]],
    row_indices_by_peak: list[list[int]],
    sigma_threshold: float = 3.0,
) -> dict:
    """
    Detect outlier peak positions using sigma clipping.

    Args:
        peak_positions_by_index: List of peak positions for each peak index
        row_indices_by_peak: Corresponding row indices
        sigma_threshold: Threshold in standard deviations for outlier detection

    Returns:
        Dictionary containing outlier information per peak
    """
    outlier_info = {}

    for peak_idx in range(len(peak_positions_by_index)):
        positions = np.array(peak_positions_by_index[peak_idx])
        rows = np.array(row_indices_by_peak[peak_idx])

        if len(positions) < 3:
            outlier_info[peak_idx] = {
                "outlier_rows": [],
                "outlier_positions": [],
                "num_outliers": 0,
            }
            continue

        # Calculate robust statistics
        median_pos = np.median(positions)
        mad = np.median(np.abs(positions - median_pos))
        # Convert MAD to std deviation estimate (for normal distribution)
        robust_std = 1.4826 * mad

        # Detect outliers
        deviations = np.abs(positions - median_pos)
        is_outlier = deviations > sigma_threshold * robust_std

        outlier_rows = rows[is_outlier].tolist()
        outlier_positions = positions[is_outlier].tolist()

        outlier_info[peak_idx] = {
            "outlier_rows": outlier_rows,
            "outlier_positions": outlier_positions,
            "outlier_deviations": deviations[is_outlier].tolist(),
            "num_outliers": int(np.sum(is_outlier)),
            "median_position": median_pos,
            "robust_std": robust_std,
            "threshold_used": sigma_threshold * robust_std,
        }

    return outlier_info


def calculate_median_offsets_with_diagnostics(
    peak_positions: list, num_rows: int, all_peak_fits: list
) -> tuple[np.ndarray, dict]:
    """
    Calculate absolute offsets from median for each peak across all rows,
    and provide comprehensive quality diagnostics.

    Args:
        peak_positions: List of arrays containing peak positions for each row
        num_rows: Total number of rows
        all_peak_fits: List of lists containing fit parameters for each peak

    Returns:
        Tuple of (median_offsets, quality_diagnostics)
    """
    median_offsets = np.zeros(num_rows)

    # Initialize diagnostics dictionary
    quality_diagnostics = {
        "fit_failures": [],
        "fit_quality_issues": [],
        "peak_count_histogram": {},
        "rows_with_inconsistent_peak_count": [],
    }

    # Collect fit quality issues
    for row_idx, row_fits in enumerate(all_peak_fits):
        for fit in row_fits:
            if fit.get("fit_failed", False):
                quality_diagnostics["fit_failures"].append(
                    {
                        "row": row_idx,
                        "peak_idx": fit.get("peak_idx", -1),
                        "position": fit.get("position", np.nan),
                        "error": fit.get("fit_error", "Unknown"),
                    }
                )

            # Check for quality flag issues
            flags = fit.get("fit_quality_flags", {})
            if any(
                not flags.get(k, True)
                for k in ["sigma_reasonable", "amplitude_positive"]
            ) or any(
                flags.get(k, False) for k in ["large_position_shift", "high_chi2"]
            ):
                quality_diagnostics["fit_quality_issues"].append(
                    {
                        "row": row_idx,
                        "peak_idx": fit.get("peak_idx", -1),
                        "position": fit.get("position", np.nan),
                        "chi2_reduced": fit.get("chi2_reduced", np.nan),
                        "sigma": fit.get("sigma", np.nan),
                        "flags": flags,
                    }
                )

    # Find the most common number of peaks across rows
    peak_counts = [len(pos) for pos in peak_positions if len(pos) > 0]
    if not peak_counts:
        quality_diagnostics["warning"] = "No peaks found in any row"
        return median_offsets, quality_diagnostics

    # Build peak count histogram
    from collections import Counter

    peak_count_histogram = Counter(peak_counts)
    quality_diagnostics["peak_count_histogram"] = dict(peak_count_histogram)

    most_common_peak_count = max(set(peak_counts), key=peak_counts.count)
    quality_diagnostics["most_common_peak_count"] = most_common_peak_count
    quality_diagnostics["num_rows_with_most_common_count"] = peak_count_histogram[
        most_common_peak_count
    ]

    # Determine reasonable range for peak count
    # Allow peaks to vary by ±2 from the most common count (to handle real variations)
    # But exclude rows with wildly different counts (e.g., 8 peaks when typical is 2)
    min_reasonable_peaks = max(1, most_common_peak_count - 2)
    max_reasonable_peaks = most_common_peak_count + 2

    # Use rows that have a reasonable number of peaks
    valid_rows = [
        i
        for i, pos in enumerate(peak_positions)
        if min_reasonable_peaks <= len(pos) <= max_reasonable_peaks
    ]

    # Track rows with no peaks (truly missing data)
    quality_diagnostics["rows_with_no_peaks"] = [
        {"row": i} for i in range(num_rows) if len(peak_positions[i]) == 0
    ]

    # Track rows with unreasonable peak counts (excluded as likely spurious)
    quality_diagnostics["rows_with_unreasonable_peak_count"] = [
        {"row": i, "peak_count": len(peak_positions[i])}
        for i in range(num_rows)
        if len(peak_positions[i]) > 0
        and (
            len(peak_positions[i]) < min_reasonable_peaks
            or len(peak_positions[i]) > max_reasonable_peaks
        )
    ]

    # Track varying peak counts within reasonable range (informational)
    quality_diagnostics["rows_with_varying_peak_count"] = [
        {"row": i, "peak_count": len(peak_positions[i])}
        for i in range(num_rows)
        if min_reasonable_peaks <= len(peak_positions[i]) <= max_reasonable_peaks
        and len(peak_positions[i]) != most_common_peak_count
    ]

    if len(valid_rows) < 2:
        print("Warning: Not enough valid rows with peaks")
        quality_diagnostics["warning"] = "Not enough valid rows with peaks"
        return median_offsets, quality_diagnostics

    # Match peaks across rows by proximity
    # Peaks shift slowly (< 2 pixels per row), so we can match by nearest neighbor
    peak_trajectories = match_peaks_across_rows(
        peak_positions, valid_rows, max_shift=2.0
    )

    # Organize by trajectory
    max_peak_count = len(peak_trajectories)
    peak_positions_by_index = [[] for _ in range(max_peak_count)]
    row_indices_by_peak = [[] for _ in range(max_peak_count)]

    for trajectory_idx, trajectory in enumerate(peak_trajectories):
        for row_idx, peak_pos in trajectory:
            peak_positions_by_index[trajectory_idx].append(peak_pos)
            row_indices_by_peak[trajectory_idx].append(row_idx)

    # Detect outliers in peak trajectories
    outlier_info = detect_outliers_in_peak_trajectories(
        peak_positions_by_index, row_indices_by_peak
    )
    quality_diagnostics["peak_trajectory_outliers"] = outlier_info

    # Calculate median position for each peak
    peak_medians = [
        np.median(positions) if positions else None
        for positions in peak_positions_by_index
    ]

    # Calculate offsets from median for each peak in each row
    # Also track variance across peaks for diagnostic purposes
    offset_variance_by_row = {}

    # First pass: collect all offsets
    for peak_idx in range(max_peak_count):
        if peak_medians[peak_idx] is not None:
            for i, row_idx in enumerate(row_indices_by_peak[peak_idx]):
                offset = peak_positions_by_index[peak_idx][i] - peak_medians[peak_idx]

                # Track individual peak offsets for variance calculation
                if row_idx not in offset_variance_by_row:
                    offset_variance_by_row[row_idx] = []
                offset_variance_by_row[row_idx].append(offset)

    # Second pass: calculate median offset per row, excluding outliers
    for row_idx, offsets in offset_variance_by_row.items():
        if len(offsets) == 1:
            median_offsets[row_idx] = offsets[0]
        elif len(offsets) > 1:
            # Use median of offsets to be robust against outliers
            # (e.g., a 5th peak that has a very different offset)
            median_offsets[row_idx] = np.median(offsets)

    # Calculate offset variance across peaks for each row
    # Use MAD (Median Absolute Deviation) as a robust measure of spread
    # This is consistent with using median for the final offset
    quality_diagnostics["offset_variance_by_row"] = {}
    for row_idx, offsets in offset_variance_by_row.items():
        if len(offsets) > 1:
            median_offset = np.median(offsets)
            mad = np.median(np.abs(np.array(offsets) - median_offset))
            robust_std = 1.4826 * mad  # Convert MAD to std equivalent

            quality_diagnostics["offset_variance_by_row"][row_idx] = {
                "offsets": offsets,
                "median": median_offset,
                "mad": mad,
                "robust_std": robust_std,
                "std": np.std(offsets),  # Keep regular std for reference
                "mean": np.mean(offsets),
            }

    # Track which rows had valid offsets calculated (before interpolation)
    # Use a separate flag array rather than checking if offset==0, since 0.0 is a valid offset
    rows_with_calculated_offsets = np.zeros(num_rows, dtype=bool)
    for row_idx in offset_variance_by_row.keys():
        rows_with_calculated_offsets[row_idx] = True

    # Interpolate missing offsets
    median_offsets = interpolate_missing_offsets(median_offsets)

    # Identify interpolated rows: those not in valid_rows
    all_rows = set(range(num_rows))
    valid_rows_set = set(valid_rows)
    interpolated_rows = sorted(all_rows - valid_rows_set)
    quality_diagnostics["interpolated_rows"] = interpolated_rows

    # Detect outliers in final median_offsets
    if len(median_offsets[median_offsets != 0]) > 3:
        valid_offsets = median_offsets[median_offsets != 0]
        median_of_offsets = np.median(valid_offsets)
        mad_of_offsets = np.median(np.abs(valid_offsets - median_of_offsets))
        robust_std_of_offsets = 1.4826 * mad_of_offsets

        offset_outliers = []
        for row_idx in range(num_rows):
            if median_offsets[row_idx] != 0:
                deviation = abs(median_offsets[row_idx] - median_of_offsets)
                if deviation > 3.0 * robust_std_of_offsets:
                    offset_outliers.append(
                        {
                            "row": row_idx,
                            "offset": median_offsets[row_idx],
                            "deviation": deviation,
                            "threshold": 3.0 * robust_std_of_offsets,
                        }
                    )

        quality_diagnostics["median_offset_outliers"] = offset_outliers
        quality_diagnostics["median_offset_stats"] = {
            "median": median_of_offsets,
            "mad": mad_of_offsets,
            "robust_std": robust_std_of_offsets,
        }

    return median_offsets, quality_diagnostics


def calculate_median_offsets(peak_positions: list, num_rows: int) -> np.ndarray:
    """
    Calculate absolute offsets from median for each peak across all rows.

    This is a simplified version for backward compatibility.
    For diagnostics, use calculate_median_offsets_with_diagnostics instead.

    Args:
        peak_positions: List of arrays containing peak positions for each row
        num_rows: Total number of rows

    Returns:
        Array of median offsets for each row
    """
    median_offsets = np.zeros(num_rows)

    # Find the most common number of peaks across rows
    peak_counts = [len(pos) for pos in peak_positions if len(pos) > 0]
    if not peak_counts:
        return median_offsets

    most_common_peak_count = max(set(peak_counts), key=peak_counts.count)

    # Use only rows with the most common number of peaks
    valid_rows = [
        i for i, pos in enumerate(peak_positions) if len(pos) == most_common_peak_count
    ]

    if len(valid_rows) < 2:
        print("Warning: Not enough valid rows with consistent peak counts")
        return median_offsets

    # Organize peak positions by peak index across all rows
    peak_positions_by_index = [[] for _ in range(most_common_peak_count)]
    row_indices_by_peak = [[] for _ in range(most_common_peak_count)]

    for row_idx in valid_rows:
        for peak_idx in range(most_common_peak_count):
            if peak_idx < len(peak_positions[row_idx]):
                peak_positions_by_index[peak_idx].append(
                    peak_positions[row_idx][peak_idx]
                )
                row_indices_by_peak[peak_idx].append(row_idx)

    # Calculate median position for each peak
    peak_medians = [
        np.median(positions) if positions else None
        for positions in peak_positions_by_index
    ]

    # Calculate offsets from median for each peak in each row
    for peak_idx in range(most_common_peak_count):
        if peak_medians[peak_idx] is not None:
            for i, row_idx in enumerate(row_indices_by_peak[peak_idx]):
                offset = peak_positions_by_index[peak_idx][i] - peak_medians[peak_idx]
                if median_offsets[row_idx] == 0:  # Only set if not already set
                    median_offsets[row_idx] = offset
                else:  # Average with existing offset
                    median_offsets[row_idx] = (median_offsets[row_idx] + offset) / 2

    # Interpolate missing offsets
    median_offsets = interpolate_missing_offsets(median_offsets)

    return median_offsets


def interpolate_missing_offsets(median_offsets: np.ndarray) -> np.ndarray:
    """
    Fill in missing (zero) offsets using linear interpolation.

    Args:
        median_offsets: Array of offsets (zeros indicate missing values)

    Returns:
        Array with interpolated values filled in
    """
    num_rows = len(median_offsets)
    valid_indices = np.where(median_offsets != 0)[0]

    if len(valid_indices) == 0:
        return median_offsets

    for i in range(num_rows):
        if median_offsets[i] == 0:
            # Find nearest valid indices
            left_indices = valid_indices[valid_indices < i]
            right_indices = valid_indices[valid_indices > i]

            if len(left_indices) > 0 and len(right_indices) > 0:
                # Linear interpolation
                left_idx = left_indices[-1]
                right_idx = right_indices[0]
                left_val = median_offsets[left_idx]
                right_val = median_offsets[right_idx]
                weight = (i - left_idx) / (right_idx - left_idx)
                median_offsets[i] = left_val + weight * (right_val - left_val)
            elif len(left_indices) > 0:
                # Use left value
                median_offsets[i] = median_offsets[left_indices[-1]]
            elif len(right_indices) > 0:
                # Use right value
                median_offsets[i] = median_offsets[right_indices[0]]

    return median_offsets


# =============================================================================
# File Processing
# =============================================================================


def print_quality_diagnostics(diagnostics: dict, filename: str) -> None:
    """
    Print quality diagnostics in a human-readable format.

    Args:
        diagnostics: Dictionary containing quality diagnostics
        filename: Name of the file being processed
    """
    print(f"\n  === Quality Diagnostics for {os.path.basename(filename)} ===")

    # Fit failures
    if diagnostics.get("fit_failures"):
        print(f"  ⚠️  {len(diagnostics['fit_failures'])} Gaussian fit failures:")
        for fail in diagnostics["fit_failures"][:5]:  # Show first 5
            print(
                f"      Row {fail['row']}, Peak {fail['peak_idx']}: {fail.get('error', 'Unknown')}"
            )
        if len(diagnostics["fit_failures"]) > 5:
            print(f"      ... and {len(diagnostics['fit_failures']) - 5} more")

    # Fit quality issues
    if diagnostics.get("fit_quality_issues"):
        print(
            f"  ⚠️  {len(diagnostics['fit_quality_issues'])} fits with quality issues:"
        )
        for issue in diagnostics["fit_quality_issues"][:5]:
            flags = issue["flags"]
            issues_list = [
                k
                for k, v in flags.items()
                if (k in ["high_chi2", "large_position_shift"] and v)
                or (k in ["sigma_reasonable", "amplitude_positive"] and not v)
            ]
            print(
                f"      Row {issue['row']}, Peak {issue['peak_idx']}: {', '.join(issues_list)}"
            )
            print(f"        χ²_red={issue['chi2_reduced']:.2f}, σ={issue['sigma']:.2f}")
        if len(diagnostics["fit_quality_issues"]) > 5:
            print(f"      ... and {len(diagnostics['fit_quality_issues']) - 5} more")

    # Peak count distribution
    if diagnostics.get("peak_count_histogram"):
        print("  Peak count distribution:")
        for count, freq in sorted(diagnostics["peak_count_histogram"].items()):
            marker = " ✓" if count == diagnostics.get("most_common_peak_count") else ""
            print(f"      {count} peaks: {freq} rows{marker}")

    # Rows with unreasonable peak counts (excluded)
    if diagnostics.get("rows_with_unreasonable_peak_count"):
        unreasonable = diagnostics["rows_with_unreasonable_peak_count"]
        print(
            f"  ⚠️  {len(unreasonable)} rows with unreasonable peak count (excluded from analysis)"
        )
        if unreasonable:
            for item in unreasonable[:5]:
                print(
                    f"      Row {item['row']}: {item['peak_count']} peaks (expected ~{diagnostics.get('most_common_peak_count')})"
                )
            if len(unreasonable) > 5:
                print(f"      ... and {len(unreasonable) - 5} more")

    # Rows with varying peak counts (informational, still included)
    if diagnostics.get("rows_with_varying_peak_count"):
        varying = diagnostics["rows_with_varying_peak_count"]
        if varying:
            print(
                f"  ℹ️  {len(varying)} rows with varying peak count (still included in analysis)"
            )

    # Rows with no peaks (truly missing data)
    if diagnostics.get("rows_with_no_peaks"):
        no_peaks = diagnostics["rows_with_no_peaks"]
        if no_peaks:
            print(f"  ⚠️  {len(no_peaks)} rows with no peaks detected")

    # Peak trajectory outliers
    if diagnostics.get("peak_trajectory_outliers"):
        total_outliers = sum(
            info["num_outliers"]
            for info in diagnostics["peak_trajectory_outliers"].values()
        )
        if total_outliers > 0:
            print(f"  ⚠️  {total_outliers} outliers detected in peak trajectories:")
            for peak_idx, info in diagnostics["peak_trajectory_outliers"].items():
                if info["num_outliers"] > 0:
                    print(
                        f"      Peak {peak_idx}: {info['num_outliers']} outliers in rows {info['outlier_rows'][:5]}"
                    )
                    if info["num_outliers"] > 5:
                        print(f"        ... and {info['num_outliers'] - 5} more")

    # Offset variance across peaks
    if diagnostics.get("offset_variance_by_row"):
        high_variance_rows = [
            (row, stats)
            for row, stats in diagnostics["offset_variance_by_row"].items()
            if stats["robust_std"] > 0.5
        ]
        if high_variance_rows:
            print(
                f"  ⚠️  {len(high_variance_rows)} rows with high offset variance across peaks (>0.5 px):"
            )
            for row, stats in sorted(
                high_variance_rows, key=lambda x: x[1]["robust_std"], reverse=True
            )[:5]:
                print(
                    f"      Row {row}: robust_std={stats['robust_std']:.3f} (MAD={stats['mad']:.3f}), offsets={[f'{o:.2f}' for o in stats['offsets']]}"
                )

    # Interpolated rows
    if diagnostics.get("interpolated_rows"):
        interp_rows = diagnostics["interpolated_rows"]
        if interp_rows:
            print(f"  ℹ️  {len(interp_rows)} rows had offsets filled by interpolation")

    # Final offset outliers
    if diagnostics.get("median_offset_outliers"):
        outliers = diagnostics["median_offset_outliers"]
        print(f"  ⚠️  {len(outliers)} rows with outlier median offsets:")
        for out in outliers[:5]:
            print(
                f"      Row {out['row']}: offset={out['offset']:.3f}, deviation={out['deviation']:.3f}"
            )
        if len(outliers) > 5:
            print(f"      ... and {len(outliers) - 5} more")

    if diagnostics.get("median_offset_stats"):
        stats = diagnostics["median_offset_stats"]
        print(
            f"  Median offset stats: median={stats['median']:.3f}, MAD={stats['mad']:.3f}, robust_std={stats['robust_std']:.3f}"
        )

    print()


def process_fits_file(
    filename: str, config: PeakFindingConfig = DEFAULT_CONFIG
) -> Optional[dict]:
    """
    Process a FITS file and return peak finding results.

    Args:
        filename: Path to FITS file
        config: Configuration for peak finding

    Returns:
        Dictionary containing analysis results, or None if processing fails
    """
    print(f"Processing {filename}...")
    try:
        with fits.open(filename) as hdul:
            data = hdul[0].data

        # Find and fit peaks
        peak_positions, median_offsets, all_peak_fits, quality_diagnostics = (
            find_and_fit_peaks(data, config)
        )

        # Calculate statistics
        avg_offset = np.mean(median_offsets)
        std_offset = np.std(median_offsets)

        print(
            f"  Average offset from median: {avg_offset:.4f} ± {std_offset:.4f} pixels"
        )

        # Print quality diagnostics
        print_quality_diagnostics(quality_diagnostics, filename)

        return {
            "filename": filename,
            "peak_positions": peak_positions,
            "median_offsets": median_offsets,
            "all_peak_fits": all_peak_fits,
            "quality_diagnostics": quality_diagnostics,
            "avg_offset": avg_offset,
            "std_offset": std_offset,
        }
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        import traceback

        traceback.print_exc()
        return None


# =============================================================================
# Plotting
# =============================================================================


def plot_results(
    results: list[dict], config: PeakFindingConfig = DEFAULT_CONFIG
) -> None:
    """
    Generate plots for the analysis results, including quality diagnostics.

    Args:
        results: List of result dictionaries from process_fits_file
        config: Configuration (for output directory)
    """
    os.makedirs(config.plots_dir, exist_ok=True)

    for result in results:
        if not result:
            continue

        filename = result["filename"]
        basename = os.path.basename(filename).replace(".fits", "")
        median_offsets = result["median_offsets"]
        peak_positions = result["peak_positions"]
        all_peak_fits = result["all_peak_fits"]
        diagnostics = result.get("quality_diagnostics", {})

        # Create comprehensive multi-panel diagnostic plot
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Plot 1: Slit deltas (offsets from median)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(np.arange(len(median_offsets)), median_offsets, "o-", markersize=3)

        # Mark interpolated rows
        if diagnostics.get("interpolated_rows"):
            interp_rows = diagnostics["interpolated_rows"]
            if interp_rows:
                ax1.plot(
                    interp_rows,
                    median_offsets[interp_rows],
                    "rx",
                    markersize=8,
                    label="Interpolated",
                    markeredgewidth=2,
                )

        ax1.set_xlabel("Row Index")
        ax1.set_ylabel("Slit Delta (pixels)")
        ax1.set_title(f"Slit Deltas - {basename}")
        ax1.grid(True, alpha=0.3)
        if diagnostics.get("interpolated_rows"):
            ax1.legend()

        # Plot 2: Peak positions across rows (trajectories)
        ax2 = fig.add_subplot(gs[1, 0])
        rows_with_peaks = [i for i, pos in enumerate(peak_positions) if len(pos) > 0]
        max_peaks = max([len(pos) for pos in peak_positions], default=0)

        if max_peaks > 0:
            for peak_idx in range(max_peaks):
                positions = []
                rows = []

                for row_idx in rows_with_peaks:
                    if peak_idx < len(peak_positions[row_idx]):
                        positions.append(peak_positions[row_idx][peak_idx])
                        rows.append(row_idx)

                if positions:
                    ax2.plot(
                        rows,
                        positions,
                        "o-",
                        label=f"Peak {peak_idx+1}",
                        markersize=2,
                        alpha=0.7,
                    )

            # Mark trajectory outliers
            if diagnostics.get("peak_trajectory_outliers"):
                for peak_idx, info in diagnostics["peak_trajectory_outliers"].items():
                    if info["num_outliers"] > 0:
                        outlier_rows = info["outlier_rows"]
                        outlier_positions = info["outlier_positions"]
                        ax2.plot(
                            outlier_rows,
                            outlier_positions,
                            "rx",
                            markersize=8,
                            markeredgewidth=2,
                        )

            ax2.set_xlabel("Row Index")
            ax2.set_ylabel("Peak Position (pixel)")
            ax2.set_title("Peak Trajectories")
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)

        # Plot 3: Within-row variance (key diagnostic!)
        ax3 = fig.add_subplot(gs[1, 1])
        if diagnostics.get("offset_variance_by_row"):
            variance_data = diagnostics["offset_variance_by_row"]
            rows = sorted(variance_data.keys())
            robust_stds = [variance_data[r]["robust_std"] for r in rows]

            ax3.bar(rows, robust_stds, width=1.0, alpha=0.7)
            ax3.axhline(
                y=0.5,
                color="r",
                linestyle="--",
                label="Threshold (0.5 px)",
                linewidth=1,
            )
            ax3.set_xlabel("Row Index")
            ax3.set_ylabel("Robust Std Dev of Offsets (pixels)")
            ax3.set_title("Within-Row Peak Agreement (MAD-based)")
            ax3.grid(True, alpha=0.3, axis="y")
            ax3.legend()

            # Highlight problematic rows
            high_var_rows = [r for r in rows if variance_data[r]["robust_std"] > 0.5]
            if high_var_rows:
                ax3.bar(
                    high_var_rows,
                    [variance_data[r]["robust_std"] for r in high_var_rows],
                    width=1.0,
                    color="red",
                    alpha=0.7,
                )

        # Plot 4: Fit quality - Chi-squared
        ax4 = fig.add_subplot(gs[1, 2])
        chi2_values = []
        chi2_rows = []
        for row_idx, row_fits in enumerate(all_peak_fits):
            for fit in row_fits:
                if not fit.get("fit_failed", False):
                    chi2 = fit.get("chi2_reduced", np.nan)
                    if np.isfinite(chi2) and chi2 < 20:  # Cap at 20 for visualization
                        chi2_values.append(chi2)
                        chi2_rows.append(row_idx)

        if chi2_values:
            ax4.scatter(chi2_rows, chi2_values, alpha=0.5, s=10)
            ax4.axhline(
                y=5.0, color="r", linestyle="--", label="High χ² threshold", linewidth=1
            )
            ax4.set_xlabel("Row Index")
            ax4.set_ylabel("Reduced χ²")
            ax4.set_title("Fit Quality (χ²)")
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            ax4.set_ylim(bottom=0)

        # Plot 5: Fitted sigma values
        ax5 = fig.add_subplot(gs[2, 0])
        sigma_values = []
        sigma_rows = []
        for row_idx, row_fits in enumerate(all_peak_fits):
            for fit in row_fits:
                if not fit.get("fit_failed", False):
                    sigma = fit.get("sigma", np.nan)
                    if np.isfinite(sigma):
                        sigma_values.append(sigma)
                        sigma_rows.append(row_idx)

        if sigma_values:
            ax5.scatter(sigma_rows, sigma_values, alpha=0.5, s=10)
            ax5.axhline(
                y=config.initial_sigma,
                color="g",
                linestyle="--",
                label=f"Initial guess ({config.initial_sigma})",
                linewidth=1,
            )
            ax5.set_xlabel("Row Index")
            ax5.set_ylabel("Fitted σ (pixels)")
            ax5.set_title("Gaussian Width")
            ax5.grid(True, alpha=0.3)
            ax5.legend()

        # Plot 6: Position uncertainties
        ax6 = fig.add_subplot(gs[2, 1])
        pos_unc_values = []
        pos_unc_rows = []
        for row_idx, row_fits in enumerate(all_peak_fits):
            for fit in row_fits:
                if not fit.get("fit_failed", False):
                    unc = fit.get("position_uncertainty", np.nan)
                    if np.isfinite(unc) and unc < 1.0:  # Cap for visualization
                        pos_unc_values.append(unc)
                        pos_unc_rows.append(row_idx)

        if pos_unc_values:
            ax6.scatter(pos_unc_rows, pos_unc_values, alpha=0.5, s=10)
            ax6.set_xlabel("Row Index")
            ax6.set_ylabel("Position Uncertainty (pixels)")
            ax6.set_title("Fit Position Uncertainty")
            ax6.grid(True, alpha=0.3)
            ax6.set_ylim(bottom=0)

        # Plot 7: Summary statistics text
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis("off")

        summary_text = "Quality Summary:\n\n"
        summary_text += f"Total rows: {len(median_offsets)}\n"

        if diagnostics.get("fit_failures"):
            summary_text += f"Fit failures: {len(diagnostics['fit_failures'])}\n"
        else:
            summary_text += "Fit failures: 0\n"

        if diagnostics.get("fit_quality_issues"):
            summary_text += (
                f"Quality issues: {len(diagnostics['fit_quality_issues'])}\n"
            )
        else:
            summary_text += "Quality issues: 0\n"

        if diagnostics.get("interpolated_rows"):
            summary_text += (
                f"Interpolated rows: {len(diagnostics['interpolated_rows'])}\n"
            )
        else:
            summary_text += "Interpolated rows: 0\n"

        if diagnostics.get("offset_variance_by_row"):
            high_var = sum(
                1
                for v in diagnostics["offset_variance_by_row"].values()
                if v["robust_std"] > 0.5
            )
            summary_text += f"High within-row variance: {high_var}\n"

        if diagnostics.get("peak_trajectory_outliers"):
            total_traj_outliers = sum(
                info["num_outliers"]
                for info in diagnostics["peak_trajectory_outliers"].values()
            )
            summary_text += f"Trajectory outliers: {total_traj_outliers}\n"

        if diagnostics.get("peak_count_histogram"):
            summary_text += "\nPeak count distribution:\n"
            for count, freq in sorted(diagnostics["peak_count_histogram"].items()):
                marker = (
                    " ✓" if count == diagnostics.get("most_common_peak_count") else ""
                )
                summary_text += f"  {count} peaks: {freq} rows{marker}\n"

        ax7.text(
            0.05,
            0.95,
            summary_text,
            transform=ax7.transAxes,
            fontsize=9,
            verticalalignment="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        plt.savefig(
            os.path.join(config.plots_dir, f"{basename}_diagnostics.png"), dpi=150
        )
        plt.close("all")


def save_results(
    results: list[dict], config: PeakFindingConfig = DEFAULT_CONFIG
) -> list[str]:
    """
    Save numerical results to NPZ files.

    Args:
        results: List of result dictionaries
        config: Configuration (for output directory)

    Returns:
        List of saved file paths
    """
    saved_files = []

    for result in results:
        if not result:
            continue

        # Create filename based on original FITS file
        basename = os.path.splitext(os.path.basename(result["filename"]))[0]
        output_file = os.path.join(config.data_dir, f"slitdeltas_{basename}.npz")

        # Save individual dataset
        np.savez(
            output_file,
            filename=result["filename"],
            avg_offset=result["avg_offset"],
            std_offset=result["std_offset"],
            median_offsets=result["median_offsets"],
        )
        saved_files.append(output_file)

    return saved_files


# =============================================================================
# Main
# =============================================================================


def main():
    """Main function to process all test data files."""
    config = DEFAULT_CONFIG

    # Check for data directory
    if not os.path.exists(config.data_dir):
        print(f"Error: {config.data_dir}/ directory not found!")
        return

    # Find all FITS files in data directory
    fits_files = [
        os.path.join(config.data_dir, f)
        for f in os.listdir(config.data_dir)
        if f.endswith(".fits")
    ]

    if not fits_files:
        print(f"No test data FITS files found in {config.data_dir}/!")
        return

    print(f"Found {len(fits_files)} FITS files to process\n")

    # Process each file
    results = []
    for fits_file in fits_files:
        result = process_fits_file(fits_file, config)
        if result:
            results.append(result)

    if not results:
        print("No files were successfully processed!")
        return

    # Generate plots
    print("\nGenerating plots...")
    plot_results(results, config)

    # Save numerical results
    print("\nSaving results...")
    saved_files = save_results(results, config)

    print(f"\nResults saved to {len(saved_files)} NPZ files:")
    for file in saved_files:
        print(f"  - {file}")

    print(f"\nPlots saved to {config.plots_dir}/")


if __name__ == "__main__":
    main()
