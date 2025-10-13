#!/usr/bin/env python3
"""
Compare results from Gaussian fitting vs cross-correlation methods.

This script loads the original results (from Gaussian fitting) and the
new results (from cross-correlation) and creates comparison plots.
"""

import os

import matplotlib.pyplot as plt
import numpy as np


def load_npz_data(filepath: str) -> dict:
    """Load data from NPZ file."""
    data = np.load(filepath)
    return {
        "filename": str(data["filename"]),
        "avg_offset": float(data["avg_offset"]),
        "std_offset": float(data["std_offset"]),
        "median_offsets": data["median_offsets"],
    }


def compare_methods(data_dir: str = "data", plots_dir: str = "plots") -> None:
    """
    Compare Gaussian fitting vs cross-correlation methods.

    Args:
        data_dir: Directory containing NPZ files
        plots_dir: Directory to save comparison plots
    """
    os.makedirs(plots_dir, exist_ok=True)

    # Find all original files
    original_files = [f for f in os.listdir(data_dir) if f.endswith("_original.npz")]

    if not original_files:
        print("No original files found for comparison!")
        return

    print(f"Comparing {len(original_files)} datasets...\n")

    for orig_file in sorted(original_files):
        # Load original (Gaussian fitting) data
        orig_path = os.path.join(data_dir, orig_file)
        orig_data = load_npz_data(orig_path)

        # Load corresponding cross-correlation data
        base_name = orig_file.replace("_original.npz", ".npz")
        xcorr_path = os.path.join(data_dir, base_name)

        if not os.path.exists(xcorr_path):
            print(f"Warning: No cross-correlation data for {orig_file}")
            continue

        xcorr_data = load_npz_data(xcorr_path)

        # Extract base name for plots
        plot_base = base_name.replace("slitdeltas_", "").replace(".npz", "")

        # Calculate differences
        diff = xcorr_data["median_offsets"] - orig_data["median_offsets"]
        max_diff = np.max(np.abs(diff))
        rms_diff = np.sqrt(np.mean(diff**2))

        print(f"Dataset: {plot_base}")
        print(
            f"  Original: avg={orig_data['avg_offset']:.4f}, std={orig_data['std_offset']:.4f}"
        )
        print(
            f"  XCorr:    avg={xcorr_data['avg_offset']:.4f}, std={xcorr_data['std_offset']:.4f}"
        )
        print(f"  Difference: max={max_diff:.4f} px, RMS={rms_diff:.4f} px")
        print()

        # Create comparison plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f"Method Comparison - {plot_base}", fontsize=14)

        # Plot 1: Both methods overlaid
        ax1 = axes[0]
        rows = np.arange(len(orig_data["median_offsets"]))
        ax1.plot(
            rows,
            orig_data["median_offsets"],
            "o-",
            markersize=3,
            label="Gaussian Fitting",
            alpha=0.7,
        )
        ax1.plot(
            rows,
            xcorr_data["median_offsets"],
            "s-",
            markersize=3,
            label="Cross-Correlation",
            alpha=0.7,
        )
        ax1.set_xlabel("Row Index")
        ax1.set_ylabel("Slit Delta (pixels)")
        ax1.set_title("Slit Deltas - Both Methods")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Difference (XCorr - Gaussian)
        ax2 = axes[1]
        ax2.plot(rows, diff, "o-", markersize=3, color="red")
        ax2.axhline(y=0, color="black", linestyle="--", linewidth=1)
        ax2.set_xlabel("Row Index")
        ax2.set_ylabel("Difference (pixels)")
        ax2.set_title(
            f"XCorr - Gaussian (max={max_diff:.4f} px, RMS={rms_diff:.4f} px)"
        )
        ax2.grid(True, alpha=0.3)

        # Plot 3: Scatter plot (correlation)
        ax3 = axes[2]
        ax3.scatter(
            orig_data["median_offsets"],
            xcorr_data["median_offsets"],
            alpha=0.5,
            s=20,
        )

        # Add diagonal line (perfect agreement)
        all_offsets = np.concatenate(
            [orig_data["median_offsets"], xcorr_data["median_offsets"]]
        )
        min_offset = np.min(all_offsets)
        max_offset = np.max(all_offsets)
        ax3.plot(
            [min_offset, max_offset],
            [min_offset, max_offset],
            "r--",
            linewidth=2,
            label="Perfect agreement",
        )

        ax3.set_xlabel("Gaussian Fitting (pixels)")
        ax3.set_ylabel("Cross-Correlation (pixels)")
        ax3.set_title("Method Correlation")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        plt.savefig(
            os.path.join(plots_dir, f"{plot_base}_method_comparison.png"), dpi=150
        )
        plt.close()

    print(f"Comparison plots saved to {plots_dir}/")


def main():
    """Main function."""
    compare_methods()


if __name__ == "__main__":
    main()
