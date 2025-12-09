"""
Shared plotting utilities for slitcurve visualization.

This module contains functions used by both the test suite and plot_curvedelta.py
to overlay slitcurve trajectories on spectral images.
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings


def evaluate_trajectory_fit(coeffs, x_ref, y_ref, nrows):
    """
    Evaluate a trajectory fit: x = x_ref + P(y - y_ref)
    where P is a polynomial with coefficients [a0, a1, a2, ...].

    Args:
        coeffs: Array [a0, a1, a2, ...] polynomial coefficients (arbitrary degree)
        x_ref: Reference x position (where the line crosses y_ref)
        y_ref: Reference y position (usually nrows/2)
        nrows: Number of rows in image

    Returns:
        Tuple of (y_positions, x_positions)
        y_positions: Array of row indices
        x_positions: Array of x positions along the trajectory
    """
    # Evaluate at each row position
    y_positions = np.arange(nrows)

    # Calculate offset from reference
    dy = y_positions - y_ref

    # Evaluate polynomial: x = x_ref + sum(a_i * dy^i)
    # np.polyval expects coefficients in descending order [a_n, ..., a2, a1, a0]
    # but we have them in ascending order [a0, a1, a2, ..., a_n]
    x_positions = x_ref + np.polyval(coeffs[::-1], dy)

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
            - 'slitcurve_coeffs': Raw trajectory coefficients (n_lines, n_coeffs) [optional]
            - 'x_refs': Reference x positions for each trajectory (n_lines,) [optional]
            - 'y_refs': Reference y positions for each trajectory (n_lines,) [optional]
            - 'slitcurve': Interpolated coefficients (ncols, n_coeffs)
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
                interp_coeffs = slitcurve[x_col]  # [a0, a1, a2, ...] (arbitrary degree)

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

def plot_slitdec_results(im, model, spectrum=None, slitfunction=None, uncertainty=None, slitcurve_data=None, output_filename=None, show=False):
    """
    Create a 5-panel plot showing input, model, residual, spectrum, and slit function.

    Parameters
    ----------
    im : ndarray
        Input image
    model : ndarray
        Output model from slitdec
    spectrum : ndarray, optional
        Extracted spectrum
    slitfunction : ndarray, optional
        Extracted slit function
    uncertainty : ndarray, optional
        Spectrum uncertainty
    slitcurve_data : dict, optional
        Data for overlaying slitcurve trajectories
    output_filename : str, optional
        Path to save the plot. If None, plot is not saved.
    show : bool, optional
        Whether to display the plot interactively. Default is False.
    """
    # Calculate difference
    diff = im - model

    # Calculate percentile limits for color scaling
    im_vmin = np.nanpercentile(im, 10)
    im_vmax = np.nanpercentile(im, 90)
    diff_vmin = np.nanpercentile(diff, 5)
    diff_vmax = np.nanpercentile(diff, 95)

    # Check aspect ratio to determine layout
    nrows, ncols = im.shape
    aspect_ratio = ncols / nrows
    wide_image = aspect_ratio > 3

    if wide_image:
        # Vertical layout for wide images: each image panel gets its own row
        fig = plt.figure(figsize=(22, 9))

        # Define positions [left, bottom, width, height]
        img_panel_width = 0.80
        img_panel_height = 0.15
        colorbar_width = 0.015
        left_margin = 0.06
        vertical_gap = 0.02

        # Three image panels stacked vertically
        top_start = 0.82
        pos1 = [left_margin, top_start, img_panel_width, img_panel_height]
        pos2 = [left_margin, top_start - img_panel_height - vertical_gap, img_panel_width, img_panel_height]
        pos3 = [left_margin, top_start - 2*(img_panel_height + vertical_gap), img_panel_width, img_panel_height]

        # Colorbar for input image (right side, spanning im and model panels)
        im_cbar_left = left_margin + img_panel_width + 0.005
        im_cbar_bottom = pos2[1]  # Bottom of model panel
        im_cbar_height = pos1[1] + pos1[3] - pos2[1]  # From model bottom to im top
        pos_im_cbar = [im_cbar_left, im_cbar_bottom, colorbar_width, im_cbar_height]

        # Position for residual colorbar (aligned with im_cbar)
        residual_cbar_x = im_cbar_left

        # Bottom row positions (spectrum and slit function)
        bottom_height = 0.25
        bottom_gap = 0.04
        bottom_left_width = 0.52
        bottom_right_width = 0.28
        pos4 = [left_margin, 0.08, bottom_left_width, bottom_height]
        pos5 = [left_margin + bottom_left_width + bottom_gap, 0.08, bottom_right_width, bottom_height]
    else:
        # Horizontal layout for normal images: 3 panels in top row
        fig = plt.figure(figsize=(15, 7.5))

        # Define fixed positions [left, bottom, width, height] in figure coordinates
        top_height = 0.50
        bottom_height = 0.25
        vertical_gap = 0.08
        top_bottom = 0.08 + bottom_height + vertical_gap

        # Horizontal positions for 3 panels in top row
        panel_width = 0.26
        colorbar_width = 0.015
        h_gap = 0.04
        left_margin = 0.08  # Extra margin for left colorbar

        pos1 = [left_margin, top_bottom, panel_width, top_height]
        pos2 = [left_margin + panel_width + h_gap, top_bottom, panel_width, top_height]
        pos3 = [left_margin + 2*(panel_width + h_gap), top_bottom, panel_width, top_height]

        # Colorbar for input image (left side of first panel)
        pos_im_cbar = [left_margin - colorbar_width - 0.01, top_bottom, colorbar_width, top_height]

        # Position for residual colorbar (right side of third panel)
        residual_cbar_x = pos3[0] + pos3[2] + 0.005

        # Bottom row positions
        bottom_left_width = 2 * panel_width + h_gap
        bottom_right_width = panel_width
        pos4 = [left_margin, 0.08, bottom_left_width, bottom_height]
        pos5 = [left_margin + bottom_left_width + h_gap, 0.08, bottom_right_width, bottom_height]

    # Panel 1: Input image with percentile scaling
    ax1 = fig.add_axes(pos1)
    im1 = ax1.imshow(im, cmap='viridis', aspect='equal', vmin=im_vmin, vmax=im_vmax,
                     origin='lower', extent=[0, ncols, 0, nrows])
    if wide_image:
        ax1.set_ylabel('Input Image', rotation=0, ha='right', va='center')
    else:
        ax1.set_title('Input Image')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Colorbar for input image - position computed to match actual image height
    # Need to account for figure aspect ratio when computing image bounds
    bbox1 = ax1.get_position()
    fig_width, fig_height = fig.get_size_inches()
    fig_aspect = fig_width / fig_height
    # Convert axes dimensions to display aspect ratio
    axes_aspect = (bbox1.width * fig_aspect) / bbox1.height
    img_aspect = ncols / nrows  # width/height of image
    
    if img_aspect > axes_aspect:
        # Image is limited by width, centered vertically
        img_height_frac = axes_aspect / img_aspect
        cbar1_bottom = bbox1.y0 + bbox1.height * (1 - img_height_frac) / 2
        cbar1_height = bbox1.height * img_height_frac
    else:
        # Image is limited by height
        cbar1_bottom = bbox1.y0
        cbar1_height = bbox1.height

    if wide_image:
        # Wide layout: colorbar on right, spanning im and model panels
        cax1 = fig.add_axes(pos_im_cbar)
        fig.colorbar(im1, cax=cax1)
    else:
        # Normal layout: colorbar on left of first panel, aligned with image
        cax1 = fig.add_axes([bbox1.x0 - colorbar_width - 0.005, cbar1_bottom, colorbar_width, cbar1_height])
        cbar1 = fig.colorbar(im1, cax=cax1)
        cax1.yaxis.set_ticks_position('left')
        cax1.yaxis.set_label_position('left')

    # Panel 2: Model with same scaling as input (no colorbar)
    ax2 = fig.add_axes(pos2)
    im2 = ax2.imshow(model, cmap='viridis', aspect='equal', vmin=im_vmin, vmax=im_vmax,
                     origin='lower', extent=[0, ncols, 0, nrows])
    if wide_image:
        ax2.set_ylabel('Model', rotation=0, ha='right', va='center')
    else:
        ax2.set_title('Model')
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Panel 3: Difference with percentile scaling (with colorbar matching image height)
    ax3 = fig.add_axes(pos3)
    im3 = ax3.imshow(diff, cmap='bwr', aspect='equal', vmin=diff_vmin, vmax=diff_vmax,
                     origin='lower', extent=[0, ncols, 0, nrows])
    if wide_image:
        ax3.set_ylabel('Residual', rotation=0, ha='right', va='center')
    else:
        ax3.set_title('Difference (Input - Model)')
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Colorbar positioned to match the actual image height in the axes
    # Need to account for figure aspect ratio when computing image bounds
    bbox = ax3.get_position()
    axes_aspect3 = (bbox.width * fig_aspect) / bbox.height
    img_aspect3 = ncols / nrows  # width/height of image
    
    if img_aspect3 > axes_aspect3:
        # Image is limited by width, centered vertically
        img_height_frac = axes_aspect3 / img_aspect3
        cbar_bottom = bbox.y0 + bbox.height * (1 - img_height_frac) / 2
        cbar_height = bbox.height * img_height_frac
    else:
        # Image is limited by height
        cbar_bottom = bbox.y0
        cbar_height = bbox.height

    cax3 = fig.add_axes([residual_cbar_x, cbar_bottom, colorbar_width, cbar_height])
    fig.colorbar(im3, cax=cax3, label='Residual')

    # Overlay slitcurve trajectories if available
    if slitcurve_data is not None:
        # Overlay on all three image panels
        for ax in [ax1, ax2, ax3]:
            overlay_slitcurve_trajectories(
                ax,
                nrows,
                ncols,
                slitcurve_data,
                num_lines=5,
                show_fitted=True,
                show_interpolated=True,
            )
            # Set axis limits to match image extent
            ax.set_xlim(0, ncols)
            ax.set_ylim(0, nrows)

    # Bottom row - Panel 4: Spectrum (spans 2/3 width)
    if spectrum is not None and uncertainty is not None:
        ax4 = fig.add_axes(pos4)
        x = np.arange(len(spectrum))

        # Plot spectrum as solid line
        ax4.plot(x, spectrum, 'b-', linewidth=1.5, label='Spectrum')

        # Plot uncertainty as shaded region
        ax4.fill_between(x, spectrum - uncertainty, spectrum + uncertainty,
                        alpha=0.3, color='blue', label='±1σ uncertainty')

        ax4.set_xlabel('Column (wavelength direction)')
        ax4.set_title('Extracted Spectrum')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)

    # Bottom row - Panel 5: Slit function (spans 1/3 width)
    if slitfunction is not None:
        ax5 = fig.add_axes(pos5)
        x = np.arange(len(slitfunction))

        # Plot slit function horizontally for better detail visibility
        ax5.plot(x, slitfunction, 'r-', linewidth=1.5)
        ax5.set_xlabel('Oversampled subpixel')
        ax5.set_title('Slit Function')
        ax5.grid(True, alpha=0.3)

    if output_filename:
        # Suppress tight_layout warning for fixed-position axes
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*not compatible with tight_layout.*')
            plt.savefig(output_filename, dpi=100)
        print(f"Saved plot to: {output_filename}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
