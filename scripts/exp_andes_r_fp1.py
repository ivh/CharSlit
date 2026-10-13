"""Slit-kernel self-calibration on the E2E frame data/ANDES_R_FP1.fits, swath by swath.

Reports chi2/pix without and with kernels per swath, and the per-slit-position kernel
moments (shift k1, variance k2, skew k3) of the calibrated kernels, to see the fibre-
scale structure the single shared LSF misses.

uv run --python 3.13 scripts/exp_andes_r_fp1.py [--swath 256] [--out DIR]
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import slitkernel as sk  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swath", type=int, default=256)
    ap.add_argument("--nswath", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    im = fits.getdata(root / "data/ANDES_R_FP1.fits").astype(float)
    g = np.load(root / "data/curvedelta_ANDES_R_FP1.npz")
    nrows, ncols = im.shape
    starts = list(range(0, ncols - a.swath + 1, a.swath))
    if a.nswath:
        starts = starts[:: max(1, len(starts) // a.nswath)][: a.nswath]
    kw = dict(osample=6, lambda_sP=0.0, lambda_sL=0.1, maxiter=30, kappa=0)
    out = {}
    print(f"ANDES_R_FP1 {im.shape}, swath {a.swath}")
    for s0 in starts:
        sl = slice(s0, s0 + a.swath)
        c = sk.selfcal_u(im[:, sl], g["ycen"][sl], g["slitcurve"][sl], g["slitdeltas"], niter=6, **kw)
        K = c["kernel_nodes"]
        o = np.arange(K.shape[1]) - K.shape[1] // 2
        k1 = (K * o).sum(1)
        k2 = (K * o**2).sum(1) - k1**2
        k3 = (K * (o - k1[:, None]) ** 3).sum(1)
        print(f"  cols {s0:4d}-{s0 + a.swath - 1:4d}: chi2 {[round(x, 1) for x in c['chi2']]}")
        out[s0] = dict(nodes=c["nodes"], K=K, k1=k1, k2=k2, k3=k3, chi2=np.array(c["chi2"]), lit=c["lit"])
    np.savez(Path(a.out) / "andes_r_fp1_kernels.npz", **{f"{s}_{k}": v for s, d in out.items() for k, v in d.items()})

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for s0, d in out.items():
        for i, k in enumerate(["k1", "k2", "k3"]):
            ax[i].plot(d["nodes"], np.where(d["lit"], d[k], np.nan), lw=0.8, label=f"cols {s0}")
    for i, lab in enumerate(["kernel shift k1 [px]", "kernel variance k2 [px^2]", "kernel 3rd moment k3 [px^3]"]):
        ax[i].set_ylabel(lab)
    ax[2].set_xlabel("slit coordinate u [rows]")
    ax[0].legend(fontsize=7, ncol=4)
    ax[0].set_title("ANDES_R_FP1: calibrated per-slit-position kernels (relative to the swath's common LSF)")
    fig.tight_layout()
    fig.savefig(Path(a.out) / "andes_r_fp1_kernels.png", dpi=110)


if __name__ == "__main__":
    main()
