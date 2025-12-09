#!/usr/bin/env python3
"""
Analyze FITS files to find spectral line curvature and residual slit deltas.

This script:
1. Identifies spectral peaks in each row and tracks them as trajectories
2. Fits polynomials x = P(y) to each line's position across rows
3. Fits the polynomial coefficients as functions of column position P(x)
4. Calculates residual slitdeltas after removing polynomial curvature
5. Saves slitcurve polynomials and residual slitdeltas for use in slitdec

Usage:
    python make_curvedelta.py [options] file1.fits file2.fits ...
    python make_curvedelta.py --height-multiplier 2.0 data/Hsim.fits
"""

import argparse
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
class CurveDeltaConfig:
    """Configuration parameters for curvature and delta analysis."""

    # Peak detection parameters
    height_multiplier: float = 1.5  # Minimum peak height as multiple of mean
    min_peak_distance: int = 10  # Minimum distance between peaks (pixels)

    # Gaussian fitting parameters
    fit_window_size: int = 15  # Window size around peak for fitting (pixels)
    initial_sigma: float = 5.0  # Initial guess for Gaussian sigma (pixels)

    # Polynomial fitting parameters
    poly_degree: int = 2  # Degree of polynomial for x=P(y) fit (default: parabola)
    coeff_poly_degree: int = 2  # Degree of polynomial for coefficient interpolation

    # ycen parameter (offset from pixel boundary, 0-1)
    ycen_value: float = 0.5  # Default: middle of pixel

    # Directory configuration
    data_dir: str = "data"
    plots_dir: str = "plots"


DEFAULT_CONFIG = CurveDeltaConfig()


# =============================================================================
# Core Functions (from make_slitdeltas.py)
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

        # Calculate goodness of fit
        y_fit = gaussian(x_window, *popt)
        residuals = y_window - y_fit
        dof = len(x_window) - 4
        chi2_reduced = np.sum(residuals**2) / dof if dof > 0 else np.inf

        perr = np.sqrt(np.diag(pcov))

        fit_params = {
            "amplitude": popt[0],
            "position": popt[1],
            "sigma": popt[2],
            "offset": popt[3],
            "fit_failed": False,
            "position_uncertainty": perr[1],
            "chi2_reduced": chi2_reduced,
        }

        return popt[1], fit_params

    except (RuntimeError, ValueError) as e:
        fit_params = {
            "position": float(peak_loc),
            "fit_failed": True,
            "fit_error": str(e),
        }
        return float(peak_loc), fit_params


def match_peaks_across_rows(
    peak_positions: list, valid_rows: list, max_shift: float = 2.0
) -> list[list[tuple[int, float]]]:
    """
    Match peaks across rows by proximity to create continuous trajectories.

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


def find_and_track_peaks(
    data: np.ndarray, config: CurveDeltaConfig = DEFAULT_CONFIG
) -> tuple:
    """
    Process FITS data row by row to find and track peaks.

    Args:
        data: 2D numpy array (rows x columns)
        config: Configuration for peak finding

    Returns:
        Tuple of (peak_positions, all_peak_fits)
    """
    num_rows, num_cols = data.shape
    x_vals = np.arange(num_cols)

    # Store peak positions for each row
    peak_positions = []
    all_peak_fits = []

    # Find peaks in each row
    for row_idx in range(num_rows):
        row_data = data[row_idx]

        # Calculate mean, handling masked arrays
        if np.ma.is_masked(row_data):
            mean_val = np.ma.mean(row_data)
            # Convert masked array to regular array for find_peaks (fill masked with mean)
            row_data_filled = row_data.filled(mean_val)
        else:
            mean_val = np.mean(row_data)
            row_data_filled = row_data

        # Find peaks
        peaks, _ = find_peaks(
            row_data_filled,
            height=mean_val * config.height_multiplier,
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
                x_vals, row_data_filled, peak_loc, config.fit_window_size, config.initial_sigma
            )

            fitted_positions.append(fitted_pos)
            fit_params.update({"peak_idx": peak_idx, "row": row_idx})
            row_fits.append(fit_params)

        # Sort peaks by position
        sort_idx = np.argsort(fitted_positions)
        fitted_positions = np.array(fitted_positions)[sort_idx]
        row_fits = [row_fits[i] for i in sort_idx]

        peak_positions.append(fitted_positions)
        all_peak_fits.append(row_fits)

    return peak_positions, all_peak_fits


# =============================================================================
# Polynomial Fitting Functions
# =============================================================================


def fit_trajectory_polynomial(
    trajectory: list[tuple[int, float]], poly_degree: int, nrows: int, ycen: float
) -> tuple[np.ndarray, dict]:
    """
    Fit a polynomial (x - x_ref) = a0 + a1*y + a2*y^2 to a peak trajectory.

    The polynomial is centered at y=0 (row index 0) because the C code slitdec
    evaluates delta = c1*(dy - ycen) + c2*(dy - ycen)^2 where (dy - ycen) ≈ y.

    Args:
        trajectory: List of (row_idx, x_position) tuples
        poly_degree: Degree of polynomial to fit
        nrows: Number of rows in image
        ycen: Fractional offset (not used for centering, kept for API compatibility)

    Returns:
        Tuple of (coefficients, fit_info)
        coefficients: [a0, a1, a2, ...] for (x - x_ref) = a0 + a1*y + a2*y^2
        fit_info: Dictionary with fit quality metrics including x_ref, y_ref
    """
    if len(trajectory) < poly_degree + 1:
        return None, {"fit_failed": True, "reason": "Insufficient data points"}

    rows = np.array([row for row, _ in trajectory])
    x_positions = np.array([x for _, x in trajectory])

    try:
        # Reference point: y=0 (row index 0)
        # This matches the C code which computes (dy - ycen) ≈ y
        y_ref = 0.0

        # Find x position at y=0 (extrapolate)
        coeffs_abs = np.polyfit(rows, x_positions, poly_degree)
        x_ref = np.polyval(coeffs_abs, y_ref)

        # Fit centered at y=0: (x - x_ref) vs y
        # Since y_ref = 0, y_centered = rows - 0 = rows
        x_centered = x_positions - x_ref

        coeffs = np.polyfit(rows, x_centered, poly_degree)
        coeffs = coeffs[::-1]  # Reverse to get [a0, a1, a2, ...]

        # Calculate residuals
        y_fit = np.polyval(coeffs[::-1], rows)
        residuals = x_centered - y_fit
        rms_residual = np.sqrt(np.mean(residuals**2))

        # Calculate average column position
        avg_col = np.mean(x_positions)

        fit_info = {
            "fit_failed": False,
            "num_points": len(trajectory),
            "rms_residual": rms_residual,
            "max_residual": np.max(np.abs(residuals)),
            "avg_col": avg_col,
            "x_ref": x_ref,
            "y_ref": y_ref,
            "row_range": (rows.min(), rows.max()),
            "residuals": residuals,
            "rows": rows,
            "x_positions": x_positions,
        }

        return coeffs, fit_info

    except Exception as e:
        return None, {"fit_failed": True, "reason": str(e)}


def convert_to_slitcurve_coeffs(
    poly_coeffs: np.ndarray, ycen: float
) -> np.ndarray:
    """
    Convert polynomial coefficients from x = a0 + a1*y + a2*y^2
    to slitcurve format: delta_x = c0 + c1*(y - ycen) + c2*(y - ycen)^2

    Args:
        poly_coeffs: [a0, a1, a2, ...] coefficients
        ycen: Center line offset (0-1)

    Returns:
        [c0, c1, c2] coefficients in slitcurve format
    """
    # For now, we only support quadratic polynomials
    if len(poly_coeffs) < 3:
        poly_coeffs = np.pad(poly_coeffs, (0, 3 - len(poly_coeffs)), mode="constant")

    a0, a1, a2 = poly_coeffs[:3]

    # Coordinate transformation:
    # x = a0 + a1*y + a2*y^2
    # x = a0 + a1*(dy + ycen) + a2*(dy + ycen)^2  where dy = y - ycen
    # x = (a0 + a1*ycen + a2*ycen^2) + (a1 + 2*a2*ycen)*dy + a2*dy^2

    c0 = a0 + a1 * ycen + a2 * ycen**2
    c1 = a1 + 2 * a2 * ycen
    c2 = a2

    return np.array([c0, c1, c2])


def fit_coefficient_interpolation(
    avg_cols: np.ndarray, coeffs_array: np.ndarray, ncols: int, poly_degree: int
) -> tuple[np.ndarray, dict]:
    """
    Fit polynomial interpolation for each coefficient as a function of column position.

    Args:
        avg_cols: Array of average column positions for each line
        coeffs_array: Array of shape (n_lines, n_coeffs) with polynomial coefficients
        ncols: Total number of columns in the image
        poly_degree: Degree of polynomial for interpolation

    Returns:
        Tuple of (slitcurve, fit_info)
        slitcurve: Array of shape (ncols, n_coeffs) with coefficients for all columns
        fit_info: Dictionary with interpolation quality metrics
    """
    n_coeffs = coeffs_array.shape[1]
    slitcurve = np.zeros((ncols, n_coeffs))
    fit_info = {}

    x_eval = np.arange(ncols)

    for i in range(n_coeffs):
        coeff_values = coeffs_array[:, i]

        # a0 (i=0) should be ~0, so don't interpolate it - just set to 0
        if i == 0:
            slitcurve[:, 0] = 0.0
            fit_info[f"c{i}_fit"] = {
                "poly_coeffs": None,
                "rms_residual": 0.0,
                "input_points": len(avg_cols),
                "method": "fixed_zero",
            }
            continue

        if len(avg_cols) >= poly_degree + 1:
            # Fit polynomial c_i(x)
            poly_coeffs = np.polyfit(avg_cols, coeff_values, poly_degree)
            slitcurve[:, i] = np.polyval(poly_coeffs, x_eval)

            # Calculate fit quality
            y_fit = np.polyval(poly_coeffs, avg_cols)
            residuals = coeff_values - y_fit
            rms_residual = np.sqrt(np.mean(residuals**2))

            fit_info[f"c{i}_fit"] = {
                "poly_coeffs": poly_coeffs,
                "rms_residual": rms_residual,
                "input_points": len(avg_cols),
            }
        else:
            # Not enough points, use constant (median) value
            median_val = np.median(coeff_values)
            slitcurve[:, i] = median_val

            fit_info[f"c{i}_fit"] = {
                "poly_coeffs": None,
                "rms_residual": 0.0,
                "input_points": len(avg_cols),
                "method": "constant",
            }

    return slitcurve, fit_info


def calculate_residual_slitdeltas(
    trajectories: list[list[tuple[int, float]]],
    trajectory_poly_coeffs: list[np.ndarray],
    trajectory_fit_info: list[dict],
    nrows: int,
) -> tuple[np.ndarray, dict]:
    """
    Calculate residual slitdeltas after removing polynomial curvature.

    Args:
        trajectories: List of peak trajectories
        trajectory_poly_coeffs: List of polynomial coefficients for each trajectory
        trajectory_fit_info: List of fit info dicts containing x_ref and y_ref
        nrows: Total number of rows

    Returns:
        Tuple of (slitdeltas, diagnostics)
        slitdeltas: Array of shape (nrows,) with residual offsets
        diagnostics: Dictionary with quality metrics
    """
    # Initialize arrays to collect residuals
    residuals_by_row = {row: [] for row in range(nrows)}

    # Calculate residuals for each trajectory
    for traj_idx, (trajectory, poly_coeffs, fit_info) in enumerate(
        zip(trajectories, trajectory_poly_coeffs, trajectory_fit_info)
    ):
        if poly_coeffs is None or fit_info.get("fit_failed", False):
            continue

        x_ref = fit_info["x_ref"]
        y_ref = fit_info["y_ref"]

        for row, x_measured in trajectory:
            # Evaluate polynomial: (x - x_ref) = a0 + a1*(y - y_ref) + a2*(y - y_ref)^2
            y_centered = row - y_ref
            x_offset = np.polyval(poly_coeffs[::-1], y_centered)
            x_predicted = x_ref + x_offset

            # Calculate residual
            residual = x_measured - x_predicted
            residuals_by_row[row].append(residual)

    # For each row, take median of residuals
    slitdeltas = np.zeros(nrows)
    rows_with_data = []
    residual_stds = []

    for row in range(nrows):
        if residuals_by_row[row]:
            slitdeltas[row] = np.median(residuals_by_row[row])
            rows_with_data.append(row)

            if len(residuals_by_row[row]) > 1:
                residual_stds.append(np.std(residuals_by_row[row]))

    # Interpolate missing rows
    if len(rows_with_data) > 0:
        slitdeltas = interpolate_missing_offsets(slitdeltas, rows_with_data)

    diagnostics = {
        "rows_with_data": len(rows_with_data),
        "rows_interpolated": nrows - len(rows_with_data),
        "median_residual_std": np.median(residual_stds) if residual_stds else 0.0,
        "slitdelta_range": (slitdeltas.min(), slitdeltas.max()),
    }

    return slitdeltas, diagnostics


def interpolate_missing_offsets(
    offsets: np.ndarray, valid_rows: list
) -> np.ndarray:
    """
    Interpolate missing offsets using linear interpolation.

    Args:
        offsets: Array with some zero values to be interpolated
        valid_rows: List of row indices with valid data

    Returns:
        Array with interpolated values
    """
    if len(valid_rows) == 0:
        return offsets

    nrows = len(offsets)
    valid_rows = np.array(valid_rows)
    valid_values = offsets[valid_rows]

    for i in range(nrows):
        if i not in valid_rows:
            # Find nearest valid indices
            left_indices = valid_rows[valid_rows < i]
            right_indices = valid_rows[valid_rows > i]

            if len(left_indices) > 0 and len(right_indices) > 0:
                # Linear interpolation
                left_idx = left_indices[-1]
                right_idx = right_indices[0]
                left_val = offsets[left_idx]
                right_val = offsets[right_idx]
                weight = (i - left_idx) / (right_idx - left_idx)
                offsets[i] = left_val + weight * (right_val - left_val)
            elif len(left_indices) > 0:
                offsets[i] = offsets[left_indices[-1]]
            elif len(right_indices) > 0:
                offsets[i] = offsets[right_indices[0]]

    return offsets


# =============================================================================
# Main Processing Function
# =============================================================================


def load_ycen_file(filename: str) -> Optional[np.ndarray]:
    """
    Load ycen array from NPZ or FITS file.

    Args:
        filename: Path to NPZ or FITS file containing ycen array

    Returns:
        ycen array as 1D numpy array, or None if loading fails
    """
    try:
        if filename.endswith('.npz'):
            # Load from NPZ
            ycen_data = np.load(filename)
            ycen_array = ycen_data['ycen']
        elif filename.endswith('.fits') or filename.endswith('.fit'):
            # Load from FITS
            with fits.open(filename) as hdul:
                ycen_array = hdul[0].data
        else:
            print(f"  Warning: Unknown ycen file format: {filename}")
            return None

        # Ensure 1D array
        ycen_array = np.atleast_1d(ycen_array).astype(np.float64)

        # Flatten if needed (in case FITS has extra dimensions)
        if ycen_array.ndim > 1:
            ycen_array = ycen_array.flatten()

        return ycen_array

    except Exception as e:
        print(f"  Warning: Could not load ycen from {filename}: {e}")
        return None


def process_fits_file(
    filename: str, config: CurveDeltaConfig = DEFAULT_CONFIG, ycen_file: Optional[str] = None
) -> Optional[dict]:
    """
    Process a FITS file to extract slitcurve and residual slitdeltas.

    Args:
        filename: Path to FITS file
        config: Configuration for processing
        ycen_file: Optional path to NPZ file containing ycen array

    Returns:
        Dictionary containing analysis results, or None if processing fails
    """
    print(f"Processing {filename}...")
    try:
        with fits.open(filename) as hdul:
            raw_data = hdul[0].data

        nrows, ncols = raw_data.shape

        # Handle NaN values (bad pixels) using masked array
        nan_mask = np.isnan(raw_data)
        n_nans = nan_mask.sum()
        if n_nans > 0:
            print(f"  Found {n_nans} NaN pixels ({100*n_nans/raw_data.size:.1f}%), using masked array")
            data = np.ma.masked_array(raw_data, mask=nan_mask)
        else:
            data = raw_data

        # Load or compute ycen
        ycen_array = None
        if ycen_file:
            # Explicit ycen file provided
            ycen_array = load_ycen_file(ycen_file)
            if ycen_array is not None:
                print(f"  Loaded ycen from {ycen_file}")
        else:
            # Try to find ycen_{basename}.npz or ycen_{basename}.fits
            basename = os.path.splitext(os.path.basename(filename))[0]
            dirname = os.path.dirname(filename) or '.'

            # Try NPZ first
            auto_ycen_npz = os.path.join(dirname, f"ycen_{basename}.npz")
            if os.path.exists(auto_ycen_npz):
                ycen_array = load_ycen_file(auto_ycen_npz)
                if ycen_array is not None:
                    print(f"  Loaded ycen from {auto_ycen_npz}")

            # Try FITS if NPZ not found
            if ycen_array is None:
                auto_ycen_fits = os.path.join(dirname, f"ycen_{basename}.fits")
                if os.path.exists(auto_ycen_fits):
                    ycen_array = load_ycen_file(auto_ycen_fits)
                    if ycen_array is not None:
                        print(f"  Loaded ycen from {auto_ycen_fits}")

        # Determine ycen_value for polynomial coordinate transformation
        # NOTE: ycen_value here is the ORDER CENTER in row coordinates (not fractional offset!)
        # This is used for polynomial fitting: x = c0 + c1*(y - ycen_value) + c2*(y - ycen_value)^2
        # where y is in row indices (0 to nrows-1)
        #
        # The ycen_array (0-1 fractional offsets) is saved separately for slitdec
        ycen_value = nrows / 2.0  # Always use center row for polynomial reference
        print(f"  Using order center ycen={ycen_value:.1f} for polynomial fitting")

        # Find and track peaks
        peak_positions, all_peak_fits = find_and_track_peaks(data, config)

        # Match peaks across rows to create trajectories
        valid_rows = [
            i for i, pos in enumerate(peak_positions) if len(pos) > 0
        ]
        trajectories = match_peaks_across_rows(peak_positions, valid_rows, max_shift=2.0)

        print(f"  Found {len(trajectories)} spectral line trajectories")

        if len(trajectories) == 0:
            print("  Warning: No trajectories found!")
            return None

        # Fit polynomials to each trajectory
        trajectory_poly_coeffs = []
        trajectory_fit_info = []

        # Use ycen=0 so reference is at nrows/2 + 0 = nrows/2
        ycen_fit = 0.0

        for traj_idx, trajectory in enumerate(trajectories):
            coeffs, fit_info = fit_trajectory_polynomial(trajectory, config.poly_degree, nrows, ycen_fit)
            trajectory_poly_coeffs.append(coeffs)
            trajectory_fit_info.append(fit_info)

            if not fit_info.get("fit_failed", False):
                print(
                    f"    Line {traj_idx}: avg_col={fit_info['avg_col']:.1f}, "
                    f"rms_residual={fit_info['rms_residual']:.3f} px"
                )

        # Collect slitcurve coefficients and reference positions with quality filtering
        # Coefficients are already in the correct format: (x - x_ref) = a0 + a1*(y - y_ref) + a2*(y - y_ref)^2
        slitcurve_coeffs = []
        avg_cols = []
        x_refs = []
        y_refs = []

        # Quality filtering thresholds
        min_points_fraction = 0.8  # Require detection in >80% of rows
        min_points = int(nrows * min_points_fraction)
        max_rms = 2.0    # Filter out poor fits
        min_rms = 0.01   # Filter out suspiciously perfect fits (likely 2-3 points)

        print(f"  Quality filter: requiring >= {min_points} points ({min_points_fraction*100:.0f}% of {nrows} rows)")

        n_filtered = 0
        for coeffs, fit_info in zip(trajectory_poly_coeffs, trajectory_fit_info):
            if coeffs is None or fit_info.get("fit_failed", False):
                n_filtered += 1
                continue

            # Apply quality filters
            num_points = fit_info.get("num_points", 0)
            rms = fit_info.get("rms_residual", 999)

            if num_points < min_points:
                n_filtered += 1
                continue
            if rms < min_rms or rms > max_rms:
                n_filtered += 1
                continue

            # Good trajectory - keep it
            slitcurve_coeffs.append(coeffs)
            avg_cols.append(fit_info["avg_col"])
            x_refs.append(fit_info["x_ref"])
            y_refs.append(fit_info["y_ref"])

        if n_filtered > 0:
            print(f"  Filtered out {n_filtered} low-quality trajectories")

        slitcurve_coeffs = np.array(slitcurve_coeffs)
        avg_cols = np.array(avg_cols)
        x_refs = np.array(x_refs)
        y_refs = np.array(y_refs)

        print(f"  Successfully fit {len(slitcurve_coeffs)} lines")

        # Interpolate coefficients across all columns
        slitcurve, interp_fit_info = fit_coefficient_interpolation(
            avg_cols, slitcurve_coeffs, ncols, config.coeff_poly_degree
        )

        print(f"  Interpolated slitcurve coefficients across {ncols} columns")

        # Calculate residual slitdeltas
        slitdeltas, delta_diagnostics = calculate_residual_slitdeltas(
            trajectories, trajectory_poly_coeffs, trajectory_fit_info, nrows
        )

        print(
            f"  Calculated residual slitdeltas: range=[{delta_diagnostics['slitdelta_range'][0]:.3f}, "
            f"{delta_diagnostics['slitdelta_range'][1]:.3f}] px"
        )

        return {
            "filename": filename,
            "nrows": nrows,
            "ncols": ncols,
            "slitcurve": slitcurve,
            "slitdeltas": slitdeltas,
            "ycen_value": ycen_value,
            "ycen_array": ycen_array,
            "poly_degree": config.poly_degree,
            "trajectories": trajectories,
            "trajectory_poly_coeffs": trajectory_poly_coeffs,
            "trajectory_fit_info": trajectory_fit_info,
            "slitcurve_coeffs": slitcurve_coeffs,
            "avg_cols": avg_cols,
            "x_refs": x_refs,
            "y_refs": y_refs,
            "interp_fit_info": interp_fit_info,
            "delta_diagnostics": delta_diagnostics,
        }

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        import traceback

        traceback.print_exc()
        return None


# =============================================================================
# Plotting and Saving
# =============================================================================


def plot_results(results: list[dict], config: CurveDeltaConfig = DEFAULT_CONFIG) -> None:
    """
    Generate diagnostic plots for the analysis results.

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

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Plot 1: Slitcurve coefficients
        ax1 = fig.add_subplot(gs[0, :])
        x_cols = np.arange(result["ncols"])
        ax1.plot(x_cols, result["slitcurve"][:, 0], label="c0", alpha=0.7)
        ax1.plot(x_cols, result["slitcurve"][:, 1], label="c1", alpha=0.7)
        ax1.plot(x_cols, result["slitcurve"][:, 2] * 10, label="c2 × 10", alpha=0.7)

        # Mark the positions where we have actual line measurements
        for avg_col, coeffs in zip(result["avg_cols"], result["slitcurve_coeffs"]):
            ax1.axvline(avg_col, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)

        ax1.set_xlabel("Column")
        ax1.set_ylabel("Coefficient Value")
        ax1.set_title(f"Slitcurve Coefficients - {basename}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Residual slitdeltas
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(np.arange(result["nrows"]), result["slitdeltas"], "o-", markersize=3)
        ax2.set_xlabel("Row")
        ax2.set_ylabel("Residual Slit Delta (pixels)")
        ax2.set_title(f"Residual Slit Deltas - {basename}")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Trajectory fits
        ax3 = fig.add_subplot(gs[2, 0])
        for traj_idx, (trajectory, fit_info) in enumerate(
            zip(result["trajectories"][:10], result["trajectory_fit_info"][:10])
        ):
            if fit_info.get("fit_failed", False):
                continue

            rows = fit_info["rows"]
            x_positions = fit_info["x_positions"]
            ax3.plot(rows, x_positions, "o", markersize=2, alpha=0.5)

        ax3.set_xlabel("Row")
        ax3.set_ylabel("Column Position")
        ax3.set_title("Line Trajectories (first 10)")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Fit residuals
        ax4 = fig.add_subplot(gs[2, 1])
        all_residuals = []
        for fit_info in result["trajectory_fit_info"]:
            if not fit_info.get("fit_failed", False):
                all_residuals.extend(fit_info["residuals"])

        if all_residuals:
            ax4.hist(all_residuals, bins=50, alpha=0.7, edgecolor="black")
            ax4.set_xlabel("Polynomial Fit Residual (pixels)")
            ax4.set_ylabel("Count")
            ax4.set_title("Distribution of Polynomial Fit Residuals")
            ax4.grid(True, alpha=0.3, axis="y")

        plt.savefig(
            os.path.join(config.plots_dir, f"{basename}_curvedelta.png"), dpi=150
        )
        plt.close("all")


def save_results(
    results: list[dict], config: CurveDeltaConfig = DEFAULT_CONFIG
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

        basename = os.path.splitext(os.path.basename(result["filename"]))[0]
        output_file = os.path.join(config.data_dir, f"curvedelta_{basename}.npz")

        # Use ycen array if provided, otherwise create from ycen_value
        # IMPORTANT: slitdec expects ycen in absolute coordinates (nrows/2 + fractional_offset)
        # not just fractional (0-1) coordinates
        if result["ycen_array"] is not None:
            ycen = result["nrows"] / 2.0 + result["ycen_array"]
        else:
            ycen = np.full(result["ncols"], result["nrows"] / 2.0 + config.ycen_value)

        # The slitcurve coefficients are already in the correct format for the C code
        # Polynomial: delta = c0 + c1*y + c2*y^2 where y is the row index
        # C code computes: delta = c1*(dy - ycen) + c2*(dy - ycen)^2 where (dy - ycen) ≈ y
        # Note: c0 should be ~0 since the polynomial is centered at y=0

        # Save both slitcurve and slitdeltas
        np.savez(
            output_file,
            filename=result["filename"],
            slitcurve=result["slitcurve"],
            slitdeltas=result["slitdeltas"],
            ycen=ycen,
            poly_degree=result["poly_degree"],
            # Trajectory fit information (individual emission lines)
            avg_cols=result["avg_cols"],
            slitcurve_coeffs=result["slitcurve_coeffs"],
            x_refs=result["x_refs"],
            y_refs=result["y_refs"],
            delta_range=result["delta_diagnostics"]["slitdelta_range"],
        )
        saved_files.append(output_file)

        print(f"  Saved: {output_file}")

    return saved_files


# =============================================================================
# Main
# =============================================================================


def main():
    """Main function to process FITS files specified on command line."""
    parser = argparse.ArgumentParser(
        description="Analyze FITS files to find spectral line curvature and residual slit deltas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config parameters as command-line options
    parser.add_argument(
        '--height-multiplier', type=float, default=DEFAULT_CONFIG.height_multiplier,
        help='Minimum peak height as multiple of mean'
    )
    parser.add_argument(
        '--min-peak-distance', type=int, default=DEFAULT_CONFIG.min_peak_distance,
        help='Minimum distance between peaks (pixels)'
    )
    parser.add_argument(
        '--fit-window-size', type=int, default=DEFAULT_CONFIG.fit_window_size,
        help='Window size around peak for Gaussian fitting (pixels)'
    )
    parser.add_argument(
        '--initial-sigma', type=float, default=DEFAULT_CONFIG.initial_sigma,
        help='Initial guess for Gaussian sigma (pixels)'
    )
    parser.add_argument(
        '--poly-degree', type=int, default=DEFAULT_CONFIG.poly_degree,
        help='Degree of polynomial for x=P(y) fit'
    )
    parser.add_argument(
        '--coeff-poly-degree', type=int, default=DEFAULT_CONFIG.coeff_poly_degree,
        help='Degree of polynomial for coefficient interpolation'
    )
    parser.add_argument(
        '--ycen-value', type=float, default=DEFAULT_CONFIG.ycen_value,
        help='ycen parameter (offset from pixel boundary, 0-1)'
    )
    parser.add_argument(
        '--ycen-file', type=str, default=None,
        help='NPZ or FITS file containing ycen array (only for single input file). '
             'If not specified, looks for ycen_{basename}.npz or ycen_{basename}.fits automatically.'
    )
    parser.add_argument(
        '--data-dir', type=str, default=DEFAULT_CONFIG.data_dir,
        help='Directory for output NPZ files'
    )
    parser.add_argument(
        '--plots-dir', type=str, default=DEFAULT_CONFIG.plots_dir,
        help='Directory for output plot files'
    )

    # Positional arguments: input FITS files
    parser.add_argument(
        'files', nargs='+', metavar='FILE',
        help='FITS files to process'
    )

    args = parser.parse_args()

    # Create config from command-line arguments
    config = CurveDeltaConfig(
        height_multiplier=args.height_multiplier,
        min_peak_distance=args.min_peak_distance,
        fit_window_size=args.fit_window_size,
        initial_sigma=args.initial_sigma,
        poly_degree=args.poly_degree,
        coeff_poly_degree=args.coeff_poly_degree,
        ycen_value=args.ycen_value,
        data_dir=args.data_dir,
        plots_dir=args.plots_dir,
    )

    # Check that all input files exist
    fits_files = []
    for filename in args.files:
        if not os.path.exists(filename):
            print(f"Error: File not found: {filename}")
            return
        if not filename.endswith('.fits'):
            print(f"Warning: {filename} does not have .fits extension, processing anyway...")
        fits_files.append(filename)

    # Validate ycen_file usage
    if args.ycen_file and len(fits_files) > 1:
        print("Error: --ycen-file can only be used with a single input file")
        return

    if args.ycen_file and not os.path.exists(args.ycen_file):
        print(f"Error: ycen file not found: {args.ycen_file}")
        return

    print(f"Processing {len(fits_files)} FITS file(s)\n")

    # Process each file
    results = []
    for fits_file in fits_files:
        # Use explicit ycen_file only if processing single file
        ycen_file_to_use = args.ycen_file if len(fits_files) == 1 else None
        result = process_fits_file(fits_file, config, ycen_file=ycen_file_to_use)
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

    print(f"\nResults saved to {len(saved_files)} NPZ files in {config.data_dir}/")
    print(f"Plots saved to {config.plots_dir}/")


if __name__ == "__main__":
    main()
