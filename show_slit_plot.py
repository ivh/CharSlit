#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import numpy as np
from astropy.io import fits
import charslit
from plotting import plot_slitdec_results
import matplotlib.pyplot as plt

def load_data(fits_path):
    fits_path = Path(fits_path)
    # Look for curvedelta_{basename}.npz (new format with curve polynomials)
    curvedelta_path = fits_path.parent / f"curvedelta_{fits_path.stem}.npz"
    # Fallback to slitdeltas_{basename}.npz (legacy format)
    slitdeltas_path = fits_path.parent / f"slitdeltas_{fits_path.stem}.npz"

    # Load FITS image
    with fits.open(fits_path) as hdul:
        im = hdul[0].data.astype(np.float64)

    if im.ndim != 2:
        raise ValueError(f"Not a 2D image (shape={im.shape})")

    nrows, ncols = im.shape
    osample = 6  # Default oversampling

    # Calculate ny
    ny = osample * (nrows + 1) + 1

    # Default values
    slitcurve = np.zeros((ncols, 3))
    slitdeltas = np.zeros(nrows)
    ycen = np.full(ncols, nrows / 2.0)
    lambda_sP = 0.0  # Default spectrum smoothing
    lambda_sL = 1.0  # Default slit function smoothing

    # Load from curvedelta NPZ if available (preferred)
    if curvedelta_path.exists():
        with np.load(curvedelta_path) as data:
            if 'slitcurve' in data:
                slitcurve = data['slitcurve'].astype(np.float64)
            if 'slitdeltas' in data:
                slitdeltas = data['slitdeltas'].astype(np.float64)
            if 'ycen' in data:
                ycen = data['ycen'].astype(np.float64)
            if 'lambda_sP' in data:
                lambda_sP = float(data['lambda_sP'])
            if 'lambda_sL' in data:
                lambda_sL = float(data['lambda_sL'])

    # Fallback to legacy slitdeltas NPZ
    elif slitdeltas_path.exists():
        with np.load(slitdeltas_path) as data:
            if 'median_offsets' in data:
                slitdeltas = data['median_offsets']
            if 'ycen' in data:
                ycen = data['ycen'].astype(np.float64)

    # Create slitcurve_data dictionary for plotting
    slitcurve_data = {
        'slitcurve': slitcurve,
        'slitdeltas': slitdeltas,
    }

    # Add trajectory data if available (from curvedelta file)
    if curvedelta_path.exists():
        with np.load(curvedelta_path) as data:
            if 'slitcurve_coeffs' in data and 'x_refs' in data and 'y_refs' in data:
                slitcurve_data['slitcurve_coeffs'] = data['slitcurve_coeffs']
                slitcurve_data['x_refs'] = data['x_refs']
                slitcurve_data['y_refs'] = data['y_refs']

    # Handle NaN pixels (bad pixels)
    nan_mask = np.isnan(im)
    
    # Create mask (0=bad, 1=good)
    mask = np.ones(im.shape, dtype=np.uint8)
    mask[nan_mask] = 0  # Mark NaN pixels as bad

    # Replace NaN pixels with 0 for numerical stability
    im[nan_mask] = 0.0

    # Create pixel uncertainties (assuming Poisson noise)
    pix_unc = np.sqrt(np.abs(im) + 1.0)
    pix_unc[nan_mask] = 1e10  # Large uncertainty for bad pixels

    return {
        'im': im,
        'pix_unc': pix_unc,
        'mask': mask,
        'ycen': ycen,
        'slitcurve': slitcurve,
        'slitdeltas': slitdeltas,
        'nrows': nrows,
        'ncols': ncols,
        'osample': osample,
        'ny': ny,
        'lambda_sP': lambda_sP,
        'lambda_sL': lambda_sL,
        'slitcurve_data': slitcurve_data,
    }

def main():
    parser = argparse.ArgumentParser(description="Run slitdec on a FITS file and show the results.")
    parser.add_argument("fits_file", help="Path to the input FITS file")
    args = parser.parse_args()

    fits_path = Path(args.fits_file)
    if not fits_path.exists():
        print(f"Error: File {fits_path} not found.")
        sys.exit(1)

    print(f"Loading data from {fits_path}...")
    try:
        data = load_data(fits_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    print("Running slitdec...")
    try:
        result = charslit.slitdec(
            data["im"],
            data["pix_unc"],
            data["mask"],
            data["ycen"],
            data["slitcurve"],
            data["slitdeltas"],
            osample=data["osample"],
            lambda_sP=data["lambda_sP"],
            lambda_sL=data["lambda_sL"],
        )
    except Exception as e:
        print(f"Error running slitdec: {e}")
        sys.exit(1)

    if result["return_code"] != 0:
        print("Error: slitdec failed.")
        sys.exit(1)

    print("Plotting results...")
    plot_slitdec_results(
        data["im"],
        result["model"],
        spectrum=result["spectrum"],
        slitfunction=result["slitfunction"],
        uncertainty=result["uncertainty"],
        slitcurve_data=data["slitcurve_data"],
        show=True
    )

if __name__ == "__main__":
    main()
