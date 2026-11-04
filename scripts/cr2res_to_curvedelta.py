#!/usr/bin/env python3
"""
Convert CR2RES trace wave table FITS files to curvedelta NPZ format.

CR2RES trace tables contain slit curvature polynomials for each order/trace.
This script extracts them and converts to the format used by CharSlit:
- slitcurve: interpolated polynomial coefficients (ncols, 3)
- slitdeltas: per-row offsets (set to zeros)
- ycen: center y position for each column
- x_refs, y_refs, slitcurve_coeffs: sampled trajectory points

Each detector/order/trace combination is saved as a separate NPZ file.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits


def convert_cr2res_trace(fits_path, output_dir="data", sample_interval=200):
    """
    Convert CR2RES trace wave table to curvedelta NPZ files.

    Args:
        fits_path: Path to CR2RES trace wave FITS file
        output_dir: Directory to save NPZ files
        sample_interval: Column spacing for trajectory sampling
    """
    fits_path = Path(fits_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Converting {fits_path.name}...")

    with fits.open(fits_path) as hdul:
        # Process each detector chip
        for det in [1, 2, 3]:
            ext_name = f"CHIP{det}.INT1"

            try:
                tdata = hdul[ext_name].data
            except KeyError:
                print(f"  Extension {ext_name} not found, skipping")
                continue

            if tdata is None or len(tdata) == 0:
                print(f"  Extension {ext_name} is empty, skipping")
                continue

            print(f"  Processing {ext_name}: {len(tdata)} traces")

            # Process each trace in this detector
            for trace in tdata:
                order = trace["Order"]
                trace_nb = trace["TraceNb"]

                # Get center line polynomial (defines ycen)
                center_poly = trace["All"]  # Shape (degree+1,), highest degree first

                # Get curvature polynomials
                poly_a = trace["SlitPolyA"]
                poly_b = trace["SlitPolyB"]
                poly_c = trace["SlitPolyC"]

                # Define column positions (CR2RES uses 2048 columns)
                ncols = 2048
                x_cols = np.arange(ncols)

                # Evaluate center line position at each column
                ycen = np.polyval(center_poly[::-1], x_cols)

                # Check if this order has valid data
                if np.isnan(ycen[ncols // 2]):
                    print(f"    Detector {det}, Order {order}, Trace {trace_nb}: Invalid center line, skipping")
                    continue

                # For nrows, we need to know the detector height
                # CR2RES detectors are typically 2048x2048
                nrows = 2048

                # Evaluate SlitPolyB and SlitPolyC directly at each column
                # CR2RES uses: x = x_col + (y - yc) * b + (y - yc)^2 * c
                # Our format: x = x_ref + a0 + a1*(y - y_ref) + a2*(y - y_ref)^2
                # With y_ref = yc, we have: a0 = 0, a1 = b, a2 = c

                c1_values = np.polyval(poly_b[::-1], x_cols)
                c2_values = np.polyval(poly_c[::-1], x_cols)
                slitcurve = np.column_stack([np.zeros(ncols), c1_values, c2_values])

                # Sample trajectory points for visualization/verification
                sample_cols = np.arange(sample_interval // 2, ncols, sample_interval)
                n_samples = len(sample_cols)

                x_refs = sample_cols.astype(float)
                y_refs = ycen[sample_cols]
                slitcurve_coeffs = slitcurve[sample_cols]

                # Create zero slitdeltas (no per-row corrections)
                slitdeltas = np.zeros(nrows)

                # Create output filename
                basename = fits_path.stem
                out_filename = f"curvedelta_{basename}_det{det}_o{order}_t{trace_nb}.npz"
                out_path = output_dir / out_filename

                # Save to NPZ
                np.savez(
                    out_path,
                    filename=str(fits_path.name),
                    detector=det,
                    order=order,
                    trace_nb=trace_nb,
                    slitcurve=slitcurve,
                    slitdeltas=slitdeltas,
                    ycen=ycen,
                    x_refs=x_refs,
                    y_refs=y_refs,
                    slitcurve_coeffs=slitcurve_coeffs,
                    avg_cols=x_refs,  # For compatibility
                )

                print(f"    Saved: {out_filename}")
                print(f"      Detector {det}, Order {order}, Trace {trace_nb}")
                print(f"      {n_samples} sampled points, ncols={ncols}, nrows={nrows}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CR2RES trace wave tables to curvedelta NPZ format"
    )
    parser.add_argument(
        "fits_file",
        help="Path to CR2RES trace wave FITS file"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="data",
        help="Output directory for NPZ files (default: data)"
    )
    parser.add_argument(
        "-s", "--sample-interval",
        type=int,
        default=200,
        help="Column spacing for trajectory sampling (default: 200)"
    )

    args = parser.parse_args()

    try:
        convert_cr2res_trace(
            args.fits_file,
            output_dir=args.output_dir,
            sample_interval=args.sample_interval
        )
        print("\nConversion complete!")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
