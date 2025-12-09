#!/usr/bin/env python3
"""
Analyze curvedelta NPZ files: plot slit geometry and compute statistics.

This script visualizes and analyzes the slit curvature data:
- Center line position (ycen)
- Per-row offsets (slitdeltas)
- Polynomial curvature (slitcurve coefficients)
- Trajectory shapes at sampled positions
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def analyze_curvedelta(npz_path, output_dir=None, show=False):
    """
    Analyze and visualize a curvedelta NPZ file.

    Args:
        npz_path: Path to curvedelta NPZ file
        output_dir: Directory to save plots (if None, same as input)
        show: Whether to display plots interactively
    """
    npz_path = Path(npz_path)

    if output_dir is None:
        output_dir = npz_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

    # Load data
    data = np.load(npz_path)

    print(f"Analyzing {npz_path.name}")
    print("=" * 70)

    # Extract arrays
    slitcurve = data['slitcurve']
    slitdeltas = data['slitdeltas']
    ycen = data['ycen']

    ncols = len(ycen)
    nrows = len(slitdeltas)

    # Get optional metadata
    has_trajectories = 'x_refs' in data and 'y_refs' in data
    filename = data.get('filename', 'unknown')
    detector = data.get('detector', 'N/A')
    order = data.get('order', 'N/A')
    trace_nb = data.get('trace_nb', 'N/A')

    print(f"Source file: {filename}")
    print(f"Detector: {detector}, Order: {order}, Trace: {trace_nb}")
    print(f"Dimensions: {nrows} rows × {ncols} columns")
    print()

    # Extract polynomial coefficients (arbitrary degree)
    n_coeffs = slitcurve.shape[1]
    poly_degree = n_coeffs - 1

    # ========== Statistics ==========
    print("SLITDELTAS (per-row horizontal offsets):")
    print(f"  Mean:   {slitdeltas.mean():8.4f} pixels")
    print(f"  Median: {np.median(slitdeltas):8.4f} pixels")
    print(f"  Std:    {slitdeltas.std():8.4f} pixels")
    print(f"  Range:  [{slitdeltas.min():7.4f}, {slitdeltas.max():7.4f}]")
    print(f"  RMS:    {np.sqrt(np.mean(slitdeltas**2)):8.4f} pixels")
    print()

    print("YCEN (center line position):")
    print(f"  Mean:   {ycen.mean():8.4f} pixels")
    print(f"  Range:  [{ycen.min():7.4f}, {ycen.max():7.4f}]")
    print(f"  Span:   {ycen.max() - ycen.min():8.4f} pixels")
    ycen_frac = ycen - np.floor(ycen)
    if np.all(ycen > 10):
        print(f"  Integer offset: ~{int(np.floor(ycen.mean()))}")
        print(f"  Fractional range: [{ycen_frac.min():.4f}, {ycen_frac.max():.4f}]")
    print()

    print(f"SLITCURVE POLYNOMIAL (degree {poly_degree}):")
    for i in range(n_coeffs):
        c = slitcurve[:, i]
        term_name = ["constant", "linear", "quadratic", "cubic", "quartic", "quintic"][i] if i < 6 else f"degree-{i}"
        units = "" if i == 0 else f" pixels/row^{i}"
        print(f"  c{i} ({term_name} term):")
        if i <= 2:
            print(f"    Mean:   {c.mean():10.6f}{units}")
            print(f"    Range:  [{c.min():10.6f}, {c.max():10.6f}]")
        else:
            print(f"    Mean:   {c.mean():10.6e}{units}")
            print(f"    Range:  [{c.min():10.6e}, {c.max():10.6e}]")

        if i == 1:
            print(f"    Variation: {c.max() - c.min():10.6f} pixels/row over {ncols} columns")
            print(f"    Rate of change: {(c[-1] - c[0]) / ncols:.3e} per column")
            angle_mean = np.arctan(c.mean()) * 180 / np.pi
            angle_min = np.arctan(c.min()) * 180 / np.pi
            angle_max = np.arctan(c.max()) * 180 / np.pi
            print(f"    Angle from vertical:")
            print(f"      Mean: {angle_mean:7.3f}°")
            print(f"      Range: [{angle_min:7.3f}°, {angle_max:7.3f}°]")
            tilt_mean = c.mean() * nrows
            tilt_range = (c.max() - c.min()) * nrows
            print(f"    Total tilt over {nrows} rows:")
            print(f"      Mean: {tilt_mean:7.3f} pixels")
            print(f"      Range variation: {tilt_range:7.3f} pixels")
        elif i == 2 and c.max() > 0:
            curv_mean = c.mean() * nrows**2
            print(f"    Total curvature over {nrows} rows: {curv_mean:7.3f} pixels")
    print()

    if has_trajectories:
        x_refs = data['x_refs']
        y_refs = data['y_refs']
        print(f"Trajectory samples: {len(x_refs)} positions")
        print(f"  Column range: [{x_refs.min():.1f}, {x_refs.max():.1f}]")
        print()

    # ========== Create plots ==========
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Create title
    title_parts = [npz_path.stem]
    if detector != 'N/A':
        title_parts.append(f"Det{detector} O{order} T{trace_nb}")
    fig.suptitle(" | ".join(title_parts), fontsize=14, fontweight='bold')

    # Panel 1: ycen
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ycen, 'b-', linewidth=1)
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row position')
    ax1.set_title('Center line position (ycen)')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, f'Range: [{ycen.min():.2f}, {ycen.max():.2f}]\nSpan: {ycen.max()-ycen.min():.2f} px',
             transform=ax1.transAxes, va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel 2: slitdeltas
    ax2 = fig.add_subplot(gs[0, 1])
    row_indices = np.arange(nrows)
    ax2.plot(row_indices, slitdeltas, 'r-', linewidth=1)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Row')
    ax2.set_ylabel('Horizontal offset (pixels)')
    ax2.set_title('Per-row offsets (slitdeltas)')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.98, f'Mean: {slitdeltas.mean():.3f}\nStd: {slitdeltas.std():.3f}\nRMS: {np.sqrt(np.mean(slitdeltas**2)):.3f}',
             transform=ax2.transAxes, va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel 3: Polynomial coefficients
    ax3 = fig.add_subplot(gs[1, 0])
    ax3_twin = ax3.twinx()

    l1 = ax3.plot(c1, 'b-', linewidth=1.5, label='c1 (linear)')
    ax3.set_xlabel('Column')
    ax3.set_ylabel('c1 (pixels/row)', color='b')
    ax3.tick_params(axis='y', labelcolor='b')
    ax3.grid(True, alpha=0.3)

    l2 = ax3_twin.plot(c2, 'r-', linewidth=1.5, label='c2 (quadratic)')
    ax3_twin.set_ylabel('c2 (pixels/row²)', color='r')
    ax3_twin.tick_params(axis='y', labelcolor='r')

    ax3.set_title('Polynomial coefficients vs column')

    # Combine legends
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='best')

    # Panel 4: Angle from vertical
    ax4 = fig.add_subplot(gs[1, 1])
    angles = np.arctan(c1) * 180 / np.pi
    ax4.plot(angles, 'g-', linewidth=1.5)
    ax4.set_xlabel('Column')
    ax4.set_ylabel('Angle from vertical (degrees)')
    ax4.set_title('Slit tilt angle')
    ax4.grid(True, alpha=0.3)
    ax4.text(0.02, 0.98, f'Mean: {angle_mean:.3f}°\nRange: [{angle_min:.3f}°, {angle_max:.3f}°]',
             transform=ax4.transAxes, va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel 5: Slit shapes (trajectories)
    ax5 = fig.add_subplot(gs[2, :])

    # Plot ycen as background reference
    ax5.plot(np.arange(ncols), ycen, 'k--', linewidth=1, alpha=0.3, label='Center line (ycen)')

    if has_trajectories:
        x_refs = data['x_refs']
        y_refs = data['y_refs']
        slitcurve_coeffs = data.get('slitcurve_coeffs', slitcurve[x_refs.astype(int)])

        # Plot trajectories
        y_positions = np.arange(nrows)

        for i, (x_ref, y_ref) in enumerate(zip(x_refs, y_refs)):
            # Get coefficients at this position
            x_col = int(np.round(x_ref))
            if 0 <= x_col < ncols:
                c1_val = slitcurve[x_col, 1]
                c2_val = slitcurve[x_col, 2]

                # Evaluate trajectory: x = x_ref + c1*(y - y_ref) + c2*(y - y_ref)^2
                dy = y_positions - y_ref
                x_positions = x_ref + c1_val * dy + c2_val * dy**2

                # Add slitdeltas
                x_positions += slitdeltas

                # Plot
                color = plt.cm.viridis(i / max(len(x_refs) - 1, 1))
                ax5.plot(x_positions, y_positions, '-', color=color, linewidth=1.5, alpha=0.7,
                        label=f'x={x_ref:.0f}' if i % max(len(x_refs)//5, 1) == 0 else None)

    ax5.set_xlabel('Column (x)')
    ax5.set_ylabel('Row (y)')
    ax5.set_title('Slit shape trajectories (polynomial + deltas)')
    ax5.grid(True, alpha=0.3)
    if has_trajectories and len(x_refs) <= 20:
        ax5.legend(loc='best', fontsize=8, ncol=min(len(x_refs)//5 + 1, 5))
    ax5.set_xlim(0, ncols)
    ax5.set_ylim(0, nrows)
    ax5.set_aspect('equal')

    # Save plot
    output_file = output_dir / f"{npz_path.stem}_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_file}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze curvedelta NPZ files: plot geometry and compute statistics"
    )
    parser.add_argument(
        "npz_files",
        nargs="+",
        help="Path(s) to curvedelta NPZ file(s)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Output directory for plots (default: same as input)"
    )
    parser.add_argument(
        "-s", "--show",
        action="store_true",
        help="Display plots interactively"
    )

    args = parser.parse_args()

    try:
        for npz_file in args.npz_files:
            analyze_curvedelta(npz_file, output_dir=args.output_dir, show=args.show)
            print()
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
