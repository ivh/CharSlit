#!/usr/bin/env python3
"""
Visualize slitcurve fits overlaid on FITS images.

This script plots FITS images with slitcurve predictions overlaid as white lines
at regularly spaced column positions. Also shows the effect of residual slitdeltas.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from slitcurve_plotting import overlay_slitcurve_trajectories


def plot_curvedelta(
    fits_file,
    curvedelta_file,
    num_lines=5,
    output_dir="plots",
):
    """
    Plot FITS image with slitcurve overlays.

    Args:
        fits_file: Path to FITS file
        curvedelta_file: Path to curvedelta NPZ file
        num_lines: Number of lines to plot (evenly spaced in x)
        output_dir: Directory to save plots
    """
    # Load FITS data
    with fits.open(fits_file) as hdul:
        im = hdul[0].data

    nrows, ncols = im.shape

    # Load curvedelta results
    slitcurve_data = dict(np.load(curvedelta_file))

    # Create plot with aspect ratio matching image (square pixels)
    # Calculate figsize to maintain square pixels
    aspect_ratio = ncols / nrows
    base_height = 8  # inches
    fig_width = base_height * aspect_ratio
    fig_height = base_height

    # Limit figure width to reasonable bounds
    if fig_width > 30:
        fig_width = 30
        fig_height = fig_width / aspect_ratio

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Calculate vmin/vmax from percentiles for better contrast
    # Handle NaN values if present
    valid_pixels = im[~np.isnan(im)] if np.any(np.isnan(im)) else im.flatten()
    vmin = np.percentile(valid_pixels, 10)
    vmax = np.percentile(valid_pixels, 90)

    # Plot image as background
    im_plot = ax.imshow(
        im,
        origin="lower",
        cmap="viridis",
        aspect="equal",  # Changed from "auto" to "equal" for square pixels
        extent=[0, ncols, 0, nrows],
        vmin=vmin,
        vmax=vmax,
    )

    # Overlay slitcurve trajectories using shared function
    overlay_slitcurve_trajectories(
        ax,
        nrows,
        ncols,
        slitcurve_data,
        num_lines=num_lines,
        show_fitted=True,
        show_interpolated=True,
    )

    cbar = plt.colorbar(im_plot, ax=ax, label="Intensity")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)

    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(fits_file).replace(".fits", "")
    output_file = os.path.join(output_dir, f"{basename}_slitcurve_overlay.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {output_file}")


def main():
    """Main function to process all files or specified file."""
    parser = argparse.ArgumentParser(
        description="Plot FITS images with slitcurve overlays"
    )
    parser.add_argument(
        "-n",
        "--num-lines",
        type=int,
        default=5,
        help="Number of slitcurve lines to plot (default: 5)",
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing FITS and NPZ files (default: data)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save plots (default: plots)",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific FITS files to process (default: all in data_dir)",
    )

    args = parser.parse_args()

    # Find files to process
    if args.files:
        fits_files = args.files
    else:
        # Find all FITS files in data directory
        if not os.path.exists(args.data_dir):
            print(f"Error: {args.data_dir}/ directory not found!")
            sys.exit(1)

        fits_files = [
            os.path.join(args.data_dir, f)
            for f in os.listdir(args.data_dir)
            if f.endswith(".fits")
        ]

    if not fits_files:
        print(f"No FITS files found!")
        sys.exit(1)

    print(f"Processing {len(fits_files)} files with {args.num_lines} slitcurve lines...\n")

    # Process each file
    for fits_file in sorted(fits_files):
        basename = os.path.basename(fits_file).replace(".fits", "")
        curvedelta_file = os.path.join(args.data_dir, f"curvedelta_{basename}.npz")

        if not os.path.exists(curvedelta_file):
            print(f"Warning: {curvedelta_file} not found, skipping {fits_file}")
            continue

        print(f"Processing {basename}...")
        plot_curvedelta(fits_file, curvedelta_file, args.num_lines, args.output_dir)

    print(f"\nDone! Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
