#!/usr/bin/env python3
"""
cutnshift.py - Align and extract spectral order strip from 2D image

Takes a FITS image and corresponding ycen array, aligns all columns to a common
center position by integer pixel shifts, then extracts a strip of specified height.
"""

import numpy as np
from astropy.io import fits
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Align spectral order columns and extract strip'
    )
    parser.add_argument('fits_file', type=str, help='Input FITS image file')
    parser.add_argument('npz_file', type=str, help='NPZ file containing ycen array')
    parser.add_argument('height', type=int, help='Height of extracted strip')
    parser.add_argument('start_col', type=int, nargs='?', default=None, help='Start column (optional)')
    parser.add_argument('end_col', type=int, nargs='?', default=None, help='End column (optional)')
    args = parser.parse_args()

    with fits.open(args.fits_file) as hdul:
        im = hdul[0].data
        header = hdul[0].header

    nrows, ncols = im.shape

    npz_data = np.load(args.npz_file)
    ycen = npz_data['ycen']

    if len(ycen) != ncols:
        raise ValueError(f"ycen length {len(ycen)} doesn't match image columns {ncols}")

    ycen_int = np.floor(ycen).astype(int)
    ycen_frac = ycen - ycen_int

    median_ycen_int = int(np.median(ycen_int))

    height = args.height
    half_height = height // 2

    start_col = args.start_col if args.start_col is not None else 0
    end_col = args.end_col if args.end_col is not None else ncols

    if start_col < 0 or end_col > ncols or start_col >= end_col:
        raise ValueError(f"Invalid column range [{start_col}, {end_col}) for image with {ncols} columns")

    ncols_out = end_col - start_col
    output = np.full((height, ncols_out), np.nan, dtype=im.dtype)

    for col in range(start_col, end_col):
        src_center = ycen_int[col]
        src_start = src_center - half_height
        src_end = src_start + height

        out_col = col - start_col
        for i in range(height):
            src_row = src_start + i
            if 0 <= src_row < nrows:
                output[i, out_col] = im[src_row, col]

    input_path = Path(args.fits_file)
    col_suffix = f"_c{start_col}-{end_col}" if args.start_col is not None or args.end_col is not None else ""
    output_filename = f"{input_path.stem}_y{median_ycen_int}_h{height}{col_suffix}.fits"

    hdu = fits.PrimaryHDU(output, header=header)
    hdu.writeto(output_filename, overwrite=True)

    if args.start_col is not None or args.end_col is not None:
        npz_path = Path(args.npz_file)
        ycen_output_filename = f"{npz_path.stem}{col_suffix}.npz"
        ycen_sliced = ycen[start_col:end_col]
        np.savez(ycen_output_filename, ycen=ycen_sliced)
        print(f"Saved: {ycen_output_filename}")

    print(f"Saved: {output_filename}")
    print(f"Input shape: {im.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Column range: [{start_col}, {end_col})")
    print(f"Median ycen (integer): {median_ycen_int}")
    print(f"ycen range: [{ycen_int.min()}, {ycen_int.max()}]")


if __name__ == '__main__':
    main()
