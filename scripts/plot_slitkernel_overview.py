"""One-page illustration of what slit kernels add to slitdec, on a synthetic fibre pseudo-slit.

18 round fibres, PSF width and skew growing along the slit, tilted slit, FP-like lines.
Calibrate slitdeltas and kernels on a uniformly lit frame, then extract frames with
other fibre illuminations using that calibration.

uv run --python 3.13 scripts/plot_slitkernel_overview.py   -> plots/slitkernel_overview.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import charslit  # noqa: E402
import fibreslit_sim as fs  # noqa: E402
import slitkernel as sk  # noqa: E402

NC, NR, NFIB, TILT, RN, YC = 240, 120, 18, 0.05, 10.0, 1e-3
KW = dict(osample=6, lambda_sP=0.0, lambda_sL=0.1, maxiter=30, kappa=0)
PX_MS = 750.0
I = np.linspace(0, 1, NFIB)
PER = dict(psf_sx=0.6 + 0.4 * I, skew_eps=0.06 * I, skew_d=1.2)
LINES = fs.fp_lines(NC, spacing=20.0, amp=3e6)
ILLUM = {
    "uniform": np.ones(NFIB),
    "gradient": 1 + 0.4 * np.linspace(-1, 1, NFIB),
    "lower half": np.where(np.linspace(-1, 1, NFIB) < 0, 1.0, 0.15),
    "odd/even": 1 + 0.3 * np.cos(np.arange(NFIB) * np.pi),
}
C_PLAIN, C_DELTA, C_KERN, C_DATA = "#9aa6b6", "#5b8cd6", "#1f3f99", "#c2410c"


def render(flux):
    fib = fs.make_fibres(NFIB, nrows=NR, **PER)
    for f, fl in zip(fib, flux):
        f["flux"] = fl
    return fs.render(NC, NR, fib, LINES, tilt=TILT, ycentre=NR / 2 + YC, continuum=2e3)


def unc(im):
    return np.sqrt(np.abs(im) + RN**2)


def run(im, sd, kern):
    sc = np.zeros((NC, 2))
    sc[:, 1] = TILT
    return charslit.slitdec(im, unc(im), np.ones(im.shape, np.uint8), np.full(NC, YC), sc, sd, slitkernel=kern, **KW)


def main():
    frames = {k: render(v) for k, v in ILLUM.items()}
    uni = frames["uniform"]
    sc = np.zeros((NC, 2))
    sc[:, 1] = TILT
    yc = np.full(NC, YC)
    c1 = sk.selfcal(uni, yc, sc, np.zeros(NR), niter=6, fit_k2=False, fit_k3=False, **KW)
    sd = sk.rows_to_subpix(c1["slitdeltas"], KW["osample"], YC)
    cu = sk.selfcal_u(uni, yc, sc, sd, niter=6, rn=RN, **KW)
    variants = {"slitdec": (np.zeros(NR), None, C_PLAIN), "+ slitdeltas": (sd, None, C_DELTA),
                "+ kernels": (sd, cu["kernel"], C_KERN)}
    res = {k: run(uni, s, kern) for k, (s, kern, _) in variants.items()}
    peaks, half = sk.line_windows(res["slitdec"]["spectrum"])

    # illumination test: change of line centroid and variance in the extracted spectrum
    ill = {}
    for name, (s, kern, _) in variants.items():
        m = {k: sk.row_moments(run(im, s, kern)["spectrum"][None, :], peaks, half)[:, 0] for k, im in frames.items()}
        ill[name] = {k: np.nanmean(m[k] - m["uniform"], axis=1) for k in frames if k != "uniform"}

    fig = plt.figure(figsize=(15, 11.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.05, 1, 0.9], hspace=0.42, wspace=0.32)
    fig.suptitle("Slit kernels in slitdec (CharSlit branch slitkernel)", fontsize=15, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.935,
             r"slitdec:   $M(x,y) = s_L(y)\;s_P\left(x-\delta(y)-\Delta(y)\right)$" "      "
             r"+ kernels:   $M(x,y) = s_L(y)\;\left(K_y \ast s_P\right)\left(x-\delta(y)-\Delta(y)\right)$",
             fontsize=12.5)
    fig.text(0.02, 0.912,
             "δ: slit curve; Δ: per-row shift (slitdeltas); K_y: small kernel per slit position (9 taps on 0.5-row nodes) that "
             "widens, narrows or skews the shared line profile. Both self-calibrated on one uniformly lit FP frame; no PSF model.",
             fontsize=10, color="#444")

    # (a) data zoom
    ax = fig.add_subplot(gs[0, 0])
    p = peaks[len(peaks) // 2]
    ax.imshow(np.sqrt(uni[:, p - 8 : p + 9]), origin="lower", aspect="auto", cmap="magma", extent=(-8.5, 8.5, 0, NR))
    ax.set(title="(a) one FP line through 18 fibres", xlabel="x − line [px]", ylabel="slit row y")

    # (b) per-row line width: data vs models
    ax = fig.add_subplot(gs[0, 1:3])
    k2 = lambda img: np.nanmedian(sk.row_moments(img, peaks, half)[1], axis=1)  # noqa: E731
    lit = uni.sum(1) > 0.2 * uni.sum(1).max()
    y = np.arange(NR) + 0.5
    ax.plot(y[lit], k2(uni)[lit], color=C_DATA, lw=2.4, label="data")
    for name, (_, _, col) in variants.items():
        if name != "+ slitdeltas":
            ax.plot(y[lit], k2(res[name]["model"])[lit], color=col, lw=1.6, ls="--" if name == "slitdec" else "-",
                    label=f"model, {name}")
    ax.set(title="(b) per-row line variance (fibre disks + PSF trend)",
           xlabel="slit row y", ylabel="row LSF variance [px²]")
    ax.legend(frameon=False, fontsize=9)

    # (c) kernels
    ax = fig.add_subplot(gs[0, 3])
    K, nodes, lk = cu["kernel_nodes"], cu["nodes"], cu["lit"]
    o = np.arange(K.shape[1]) - K.shape[1] // 2
    idx = np.flatnonzero(lk)
    cmap = plt.get_cmap("viridis")
    for j, i in enumerate(idx[:: max(1, len(idx) // 6)]):
        ax.plot(o, K[i], "-o", ms=3, color=cmap(j / 6), label=f"y = {nodes[i]:.0f}")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set(title="(c) calibrated kernels K_y", xlabel="tap [px]", ylabel="weight")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # (d) residuals
    for j, (name, (_, _, col)) in enumerate(variants.items()):
        ax = fig.add_subplot(gs[1, j])
        r = (uni - res[name]["model"]) / unc(uni)
        chi2 = float((r[:, 20:-20] ** 2).mean())
        ax.imshow(r[:, 70:130], origin="lower", aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -80, 80),
                  extent=(69.5, 129.5, 0, NR))
        ax.set_title(f"(d) residual, {name}:  χ²/pix {chi2:.0f}", color=col if name != "slitdec" else "#333")
        ax.set(xlabel="x [px]", ylabel="slit row y" if j == 0 else None)
    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    ax.text(0, 0.95, "Residual (data − model)/σ, ±80σ,\nnoiseless frame: χ² is model error only.\n\n"
            "Plain slitdec misses the per-row\ncentroid shifts (PSF skew trend) and\nthe per-row width changes (fibre disks,\n"
            "PSF width trend).\n\nslitdeltas fix the centroids,\nkernels the widths and shapes.\nχ² is not the target: slitdeltas alone\nraise it here, yet remove the bias (e).\n\n"
            "Cost: about linear in kernel length,\n0.03 → 0.27 s per 404×256 swath (9 taps).",
            va="top", fontsize=10, color="#333")

    # (e) illumination test
    names = [k for k in ILLUM if k != "uniform"]
    for j, (mi, lab, scale) in enumerate(((0, "line position change [m/s]", PX_MS), (1, "LSF variance change [px²]", 1.0))):
        ax = fig.add_subplot(gs[2, 2 * j : 2 * j + 2])
        w = 0.26
        for v, (name, (_, _, col)) in enumerate(variants.items()):
            vals = [ill[name][n][mi] * scale for n in names]
            b = ax.bar(np.arange(len(names)) + (v - 1) * w, vals, w, color=col, label=name)
            ax.bar_label(b, fmt="%+.1f" if mi == 0 else "%+.3f", fontsize=8, padding=2)
        ax.axhline(0, color="#666", lw=0.8)
        ax.set_xticks(range(len(names)), [f"illumination: {n}" for n in names])
        ax.set(ylabel=lab)
        ax.set_title(f"(e) calibrate on uniform illumination, extract others: {lab.split(' [')[0]}", fontsize=11)
        if j == 0:
            ax.legend(frameon=False, fontsize=9)
    for a in fig.axes:
        a.spines[["top", "right"]].set_visible(False)
    out = ROOT / "plots" / "slitkernel_overview.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)
    for name in variants:
        print(name, {n: np.round(ill[name][n][:2] * [PX_MS, 1], 3).tolist() for n in names})


if __name__ == "__main__":
    main()
