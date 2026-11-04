#!/usr/bin/env python3
"""
Compare original fitted line curves with re-evaluated curves after interpolation.

This shows:
- Green: Original polynomial fit to each detected emission line
- Magenta: Re-evaluated polynomial using interpolated slitcurve coefficients

This helps verify that the coefficient interpolation preserves the original fits.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys

def compare_fits(fits_file, curvedelta_file, output_file=None):
    """Compare original and interpolated curve fits."""

    # Load FITS data
    with fits.open(fits_file) as hdul:
        im = hdul[0].data

    nrows, ncols = im.shape

    # Load curvedelta results
    data = np.load(curvedelta_file)
    slitcurve = data['slitcurve']
    ycen_value = data['ycen_value']
    slitcurve_coeffs = data['slitcurve_coeffs']  # Original fits
    avg_cols = data['avg_cols']  # Where lines were detected

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot image as background
    ax.imshow(im, origin='lower', cmap='viridis', aspect='auto',
              extent=[0, ncols, 0, nrows])

    # For each detected line, plot both the original fit and the re-evaluated curve
    y_positions = np.arange(nrows)
    dy = y_positions - ycen_value

    for i, avg_col in enumerate(avg_cols):
        x_col = int(avg_col)

        # Original fit (from slitcurve_coeffs before interpolation)
        c0_orig, c1_orig, c2_orig = slitcurve_coeffs[i]
        x_orig = c0_orig + c1_orig * dy + c2_orig * dy**2

        # Re-evaluated from interpolated slitcurve
        c0_interp, c1_interp, c2_interp = slitcurve[x_col]
        x_interp = c0_interp + c1_interp * dy + c2_interp * dy**2

        # Plot original fit
        ax.plot(x_orig, y_positions, 'g-', linewidth=2, alpha=0.8,
                label='Original fit' if i == 0 else None)

        # Plot re-evaluated curve
        ax.plot(x_interp, y_positions, 'm--', linewidth=1.5, alpha=0.8,
                label='After interpolation' if i == 0 else None)

        # Mark the average column position
        ax.axvline(avg_col, color='cyan', linestyle=':', alpha=0.3, linewidth=0.8)

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.set_xlabel('Column (x)')
    ax.set_ylabel('Row (y)')
    ax.set_title(f'Comparison: Original Fits vs Interpolated Curves\\n'
                 f'Green=original polynomial fits, Magenta=after coeff interpolation')
    ax.legend(loc='upper right')

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f'Saved: {output_file}')
    else:
        plt.show()

    plt.close()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: compare_curve_fits.py <fits_file> <curvedelta_npz>')
        print('Example: compare_curve_fits.py data/Rsim.fits data/curvedelta_Rsim.npz')
        sys.exit(1)

    fits_file = sys.argv[1]
    curvedelta_file = sys.argv[2]

    import os
    basename = os.path.basename(fits_file).replace('.fits', '')
    output_file = f'plots/{basename}_fit_comparison.png'

    compare_fits(fits_file, curvedelta_file, output_file)
