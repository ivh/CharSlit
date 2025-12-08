"""
Shared plotting utilities for slitcurve visualization.

This module contains functions used by both the test suite and plot_curvedelta.py
to overlay slitcurve trajectories on spectral images.
"""

import numpy as np


def evaluate_trajectory_fit(coeffs, x_ref, y_ref, nrows):
    """
    Evaluate a trajectory fit: x = x_ref + a0 + a1*(y - y_ref) + a2*(y - y_ref)^2

    Args:
        coeffs: Array [a0, a1, a2] polynomial coefficients
        x_ref: Reference x position (where the line crosses y_ref)
        y_ref: Reference y position (usually nrows/2)
        nrows: Number of rows in image

    Returns:
        Tuple of (y_positions, x_positions)
        y_positions: Array of row indices
        x_positions: Array of x positions along the trajectory
    """
    a0, a1, a2 = coeffs

    # Evaluate at each row position
    y_positions = np.arange(nrows)

    # Calculate offset from reference
    dy = y_positions - y_ref

    # Evaluate polynomial: x = x_ref + a0 + a1*dy + a2*dy^2
    x_positions = x_ref + a0 + a1 * dy + a2 * dy**2

    return y_positions, x_positions


def overlay_slitcurve_trajectories(
    ax,
    nrows,
    ncols,
    slitcurve_data,
    num_lines=5,
    show_fitted=True,
    show_interpolated=True,
    fitted_color="red",
    fitted_alpha=0.7,
    fitted_linewidth=1.5,
    interp_color="white",
    interp_alpha=0.8,
    interp_linewidth=1.0,
    interp_linestyle="--",
):
    """
    Overlay slitcurve trajectories on an image axes.

    Args:
        ax: Matplotlib axes object to plot on
        nrows: Number of rows in the image
        ncols: Number of columns in the image
        slitcurve_data: Dictionary containing:
            - 'slitcurve_coeffs': Raw trajectory coefficients (n_lines, 3) [optional]
            - 'x_refs': Reference x positions for each trajectory (n_lines,) [optional]
            - 'y_refs': Reference y positions for each trajectory (n_lines,) [optional]
            - 'slitcurve': Interpolated coefficients (ncols, 3)
            - 'slitdeltas': Per-row horizontal offsets (nrows,)
        num_lines: Number of lines to plot (evenly spaced)
        show_fitted: Whether to show fitted trajectories (only if x_refs/y_refs available)
        show_interpolated: Whether to show interpolated slitcurve + slitdeltas
        fitted_color: Color for fitted trajectories
        fitted_alpha: Alpha for fitted trajectories
        fitted_linewidth: Line width for fitted trajectories
        interp_color: Color for interpolated curves
        interp_alpha: Alpha for interpolated curves
        interp_linewidth: Line width for interpolated curves
        interp_linestyle: Line style for interpolated curves
    """
    slitcurve = slitcurve_data["slitcurve"]
    slitdeltas = slitcurve_data.get("slitdeltas", np.zeros(nrows))

    # Check if we have fitted trajectory data
    has_fitted = ("slitcurve_coeffs" in slitcurve_data and
                  "x_refs" in slitcurve_data and
                  "y_refs" in slitcurve_data)

    if has_fitted:
        slitcurve_coeffs = slitcurve_data["slitcurve_coeffs"]
        x_refs = slitcurve_data["x_refs"]
        y_refs = slitcurve_data["y_refs"]
        n_trajectories = len(x_refs)

        # Use all fitted trajectories (ignore num_lines)
        plot_indices = np.arange(n_trajectories)
        x_positions_to_plot = x_refs
        y_positions_to_plot = y_refs
    else:
        # No fitted trajectories, use num_lines evenly spaced vertical lines
        plot_indices = np.linspace(0, ncols - 1, num_lines, dtype=int)
        x_positions_to_plot = plot_indices.astype(float)
        y_positions_to_plot = np.full(len(plot_indices), nrows / 2.0)

    # Plot fitted trajectories (individual emission lines) if available
    if show_fitted and has_fitted:
        for i, idx in enumerate(plot_indices):
            coeffs = slitcurve_coeffs[idx]
            x_ref = x_refs[idx]
            y_ref = y_refs[idx]

            # Evaluate the trajectory fit
            y_positions, x_positions = evaluate_trajectory_fit(coeffs, x_ref, y_ref, nrows)

            # Plot the fitted trajectory
            ax.plot(
                x_positions,
                y_positions,
                color=fitted_color,
                linewidth=fitted_linewidth,
                alpha=fitted_alpha,
                label="Fitted lines" if i == 0 else None,
            )

    # Plot interpolated slitcurve + slitdeltas
    if show_interpolated:
        for i, (x_ref, y_ref) in enumerate(zip(x_positions_to_plot, y_positions_to_plot)):
            # Get interpolated coefficients at this x_ref position
            x_col = int(np.round(x_ref))
            if 0 <= x_col < ncols:
                interp_coeffs = slitcurve[x_col]  # [a0=0, a1, a2]

                # Evaluate the interpolated curve
                y_positions, x_positions_interp = evaluate_trajectory_fit(
                    interp_coeffs, x_ref, y_ref, nrows
                )

                # Add slitdeltas to the x positions
                x_positions_with_deltas = x_positions_interp + slitdeltas

                # Plot the interpolated curve + deltas
                ax.plot(
                    x_positions_with_deltas,
                    y_positions,
                    color=interp_color,
                    linewidth=interp_linewidth,
                    alpha=interp_alpha,
                    linestyle=interp_linestyle,
                    label="Poly + deltas" if i == 0 else None,
                )
