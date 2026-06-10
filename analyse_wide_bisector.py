"""Bisector + wing-flux analysis for the wide-LSF s-sweep.

Reads `experiment_synthetic_lsf_wide_ssweep.npz` (run that script first with
`--tophat-w 1.6 --sigma 0.5 --osamps 1 2 3 4 5 6 --fringes 0 --tag _wide_ssweep`),
computes the bisector of the truth and of each recovered profile, and plots:

  1. bisector x(h) vs level h, truth + each s
  2. BIS (bisector inverse slope) vs s
  3. blue / red wing flux vs s

The bisector is the canonical RV-pipeline shape diagnostic: asymmetric
distortions in the LSF *look exactly like a Doppler shift* in CCF analysis,
and BIS is what HARPS/ESPRESSO use to disentangle stellar activity from
real velocity changes. A pipeline that gets FWHM right but BIS wrong will
generate fake planets.
"""

import os

import matplotlib
matplotlib.use("Agg") if os.environ.get("SHOW", "1") == "0" else None
import matplotlib.pyplot as plt
import numpy as np


def bisector(x, y, levels):
    """For each fractional level h in `levels`, return the bisector midpoint
    x_B(h) = (x_L + x_R)/2 of the curve `y(x)` at y = h * max(y). Profile is
    assumed to have a single peak; left/right crossings are linearly
    interpolated. Returns NaN where the level is not bracketed on both sides.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    pk = int(np.argmax(y))
    ymax = y[pk]
    out = np.full_like(levels, np.nan, dtype=float)
    for k, h in enumerate(levels):
        lvl = h * ymax
        xl = xr = np.nan
        for i in range(pk, 0, -1):
            if (y[i - 1] - lvl) * (y[i] - lvl) < 0:
                f = (lvl - y[i - 1]) / (y[i] - y[i - 1])
                xl = x[i - 1] + f * (x[i] - x[i - 1])
                break
        for i in range(pk, len(x) - 1):
            if (y[i] - lvl) * (y[i + 1] - lvl) < 0:
                f = (lvl - y[i]) / (y[i + 1] - y[i])
                xr = x[i] + f * (x[i + 1] - x[i])
                break
        if np.isfinite(xl) and np.isfinite(xr):
            out[k] = 0.5 * (xl + xr)
    return out


def bis_value(levels, bis_x):
    """HARPS-style BIS: mean bisector position in the top of the line
    (h = 0.6..0.85) minus mean in the bottom (h = 0.1..0.4). Units: pixels."""
    top = (levels >= 0.6) & (levels <= 0.85)
    bot = (levels >= 0.1) & (levels <= 0.4)
    return np.nanmean(bis_x[top]) - np.nanmean(bis_x[bot])


def area_normalise(x, y, x0, win=5.0):
    sel = np.abs(x - x0) < win
    x, y = x[sel], y[sel].astype(float)
    area = np.trapezoid(y, x)
    return x, y / area if area != 0 else y


def wing_fluxes(x, y, x0):
    dx = np.gradient(x)
    blue = np.sum(y[(x > x0 - 4.0) & (x < x0 - 1.0)]
                  * dx[(x > x0 - 4.0) & (x < x0 - 1.0)])
    red = np.sum(y[(x > x0 + 1.0) & (x < x0 + 4.0)]
                 * dx[(x > x0 + 1.0) & (x < x0 + 4.0)])
    return blue, red


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    d = np.load(os.path.join(here, "experiment_synthetic_lsf_wide_ssweep.npz"))
    u_grid = d["u_grid"]; lsf = d["lsf"]; line_x0 = float(d["line_x0"])

    # ground truth on its native fine grid
    xt = u_grid + line_x0
    xtw, tnorm = area_normalise(xt, lsf, line_x0, win=5.0)

    levels = np.linspace(0.10, 0.95, 35)
    bis_truth = bisector(xtw, tnorm, levels) - line_x0
    bis_truth_val = bis_value(levels, bis_truth)

    # recover s from saved keys: "<case>_s<s>_lf<lf>_xf|sp"
    ss = sorted({int(k.split("_s")[1].split("_lf")[0])
                 for k in d.files if k.startswith("noisy_s")})

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colours = plt.cm.viridis(np.linspace(0, 0.85, len(ss)))

    ax = axes[0]
    ax.plot(bis_truth * 1000, levels, color="0.4", lw=2.5, ls="--",
            label=f"truth (BIS={bis_truth_val*1000:.1f} mpx)")

    bis_vals = []; blue_recov = []; red_recov = []; bias_vs_truth = []
    for col, s in zip(colours, ss):
        xf = d[f"noisy_s{s}_lf0_xf"]
        sp = d[f"noisy_s{s}_lf0_sp"]
        xw, yw = area_normalise(xf, sp, line_x0, win=5.0)
        b = bisector(xw, yw, levels) - line_x0
        bv = bis_value(levels, b)
        bis_vals.append(bv)
        bias_vs_truth.append(bv - bis_truth_val)
        bw, rw = wing_fluxes(xw, yw, line_x0)
        blue_recov.append(bw); red_recov.append(rw)
        ax.plot(b * 1000, levels, color=col, lw=1.6,
                label=f"s={s} (BIS={bv*1000:.1f} mpx)")
    ax.set_xlabel("bisector position relative to line centre [mpx]")
    ax.set_ylabel("fractional level of peak")
    ax.set_title("bisector curves: truth vs recovered")
    ax.legend(fontsize=8, loc="lower right")
    ax.axvline(0, color="0.85", lw=0.5)

    bt_truth, rt_truth = wing_fluxes(xtw, tnorm, line_x0)
    ax = axes[1]
    ax.axhline(bis_truth_val * 1000, color="0.4", ls="--", lw=2,
               label=f"truth BIS = {bis_truth_val*1000:.1f} mpx")
    ax.plot(ss, np.asarray(bis_vals) * 1000, "o-", lw=1.5, ms=7,
            label="recovered BIS")
    ax.plot(ss, np.asarray(bias_vs_truth) * 1000, "s--", lw=1, ms=6,
            color="C3", label="BIS bias (recov - truth)")
    ax.set_xlabel("osamp_spec  s")
    ax.set_ylabel("BIS  [mpx]   (top 0.6-0.85  minus  bottom 0.1-0.4)")
    ax.set_title("bisector inverse slope vs s")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="0.85", lw=0.5)

    ax = axes[2]
    ax.axhline(bt_truth, color="C0", ls="--", lw=1.5,
               label=f"truth blue = {bt_truth:.3f}")
    ax.axhline(rt_truth, color="C3", ls="--", lw=1.5,
               label=f"truth red  = {rt_truth:.3f}")
    ax.plot(ss, blue_recov, "o-", color="C0", lw=1.5, ms=7,
            label="recovered blue")
    ax.plot(ss, red_recov, "s-", color="C3", lw=1.5, ms=7,
            label="recovered red")
    ax.set_xlabel("osamp_spec  s")
    ax.set_ylabel("wing flux (1-4 px)")
    ax.set_title("wing flux convergence vs s")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(here, "analyse_wide_bisector.png")
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")

    print(f"\nTruth: BIS = {bis_truth_val*1000:+.1f} mpx   "
          f"blue/red wing = {bt_truth:.3f} / {rt_truth:.3f}")
    print(f"{'s':>3} {'BIS[mpx]':>9} {'bias[mpx]':>10} "
          f"{'blue':>7} {'red':>7}")
    for s, b, bw, rw in zip(ss, bis_vals, blue_recov, red_recov):
        print(f"{s:>3} {b*1000:>+9.1f} {(b-bis_truth_val)*1000:>+10.1f} "
              f"{bw:>7.3f} {rw:>7.3f}")


if __name__ == "__main__":
    main()
