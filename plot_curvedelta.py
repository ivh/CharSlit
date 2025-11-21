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


def evaluate_slitcurve(x_col, slitcurve, ycen_value, nrows):
    """
    Evaluate slitcurve at a given column position.

    Args:
        x_col: Column position (integer)
        slitcurve: Array of shape (ncols, 3) with [c0, c1, c2] coefficients
        ycen_value: Reference y offset (0-1)
        nrows: Number of rows in image

    Returns:
        Tuple of (y_positions, x_offsets)
        y_positions: Array of row indices
        x_offsets: Array of horizontal offsets from x_col
    """
    c0, c1, c2 = slitcurve[x_col]

    # Evaluate at each row position
    y_positions = np.arange(nrows)

    # Calculate offset from central line for each row
    # The ycen is a fractional offset within each row, but for plotting
    # purposes we evaluate the polynomial at integer row positions
    dy = y_positions - ycen_value

    # Evaluate polynomial: delta_x = c0 + c1*dy + c2*dy^2
    x_offsets = c0 + c1 * dy + c2 * dy**2

    return y_positions, x_offsets


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
    data = np.load(curvedelta_file)
    slitcurve = data["slitcurve"]
    slitdeltas = data["slitdeltas"]
    ycen_value = float(data["ycen_value"])

    # Select column positions for plotting (evenly spaced)
    x_positions = np.linspace(0, ncols - 1, num_lines, dtype=int)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot image as background
    im_plot = ax.imshow(
        im,
        origin="lower",
        cmap="viridis",
        aspect="auto",
        extent=[0, ncols, 0, nrows],
    )

    # Plot slitcurves at selected positions
    for x_col in x_positions:
        y_positions, x_offsets = evaluate_slitcurve(x_col, slitcurve, ycen_value, nrows)

        # Curve position: x_col + offset
        x_curve = x_col + x_offsets

        # Plot pure slitcurve (no deltas)
        ax.plot(
            x_curve,
            y_positions,
            color="white",
            linewidth=1.5,
            alpha=0.8,
            label=f"Slitcurve @ x={x_col}" if x_col == x_positions[0] else None,
        )

        # Plot slitcurve + slitdeltas
        x_curve_with_delta = x_curve + slitdeltas
        ax.plot(
            x_curve_with_delta,
            y_positions,
            color="red",
            linewidth=1.5,
            alpha=0.8,
            linestyle="--",
            label=f"Slitcurve + deltas @ x={x_col}"
            if x_col == x_positions[0]
            else None,
        )

    # Add colorbar
    cbar = plt.colorbar(im_plot, ax=ax, label="Intensity")

    # Labels and title
    basename = os.path.basename(fits_file).replace(".fits", "")
    ax.set_xlabel("Column (x)")
    ax.set_ylabel("Row (y)")
    ax.set_title(
        f"Slitcurve Visualization - {basename}\n"
        f"White: pure polynomial, Red dashed: polynomial + residual deltas"
    )
    ax.legend(loc="upper right", fontsize=8)

    # Set limits
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
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
