"""Standalone experiment: how much of an asymmetric LSF can osamp_spec recover?

Synthetic setup
---------------
A single *unresolved* emission line is rendered onto an 80-row slit image.
The line profile in the dispersion direction is the instrument LSF, built to
be slightly asymmetric:

  - core: top-hat (full width ~0.8 px) convolved with a Gaussian
    (sigma ~0.25 px)  -> a slightly flat-topped, under-sampled core
  - blue wing: a fraction of the flux is convolved with a one-sided
    exponential extending to lower x (blue = lower column index here)

The slit is tilted, and the tilt (px of horizontal shift per row) varies
*linearly with row* from 0.1 px/row at the bottom (row 0) to 0.001 px/row at
the top (row 79). Integrating that linear tilt gives a quadratic horizontal
displacement delta(y), which is exactly representable by the degree-2
slitcurve polynomial that slitdec evaluates around the central row
(t = y - nrows/2):

    d(delta)/dy = s_bot + (s_top - s_bot) * y / (nrows - 1)
 => delta(t)    = c1*t + c2*t^2
    c2 = (s_top - s_bot) / (2*(nrows - 1))
    c1 = s_bot + (s_top - s_bot) * (nrows//2) / (nrows - 1)

The varying tilt is the point of the experiment: the bottom rows sweep the
sub-pixel phase quickly (good super-resolution leverage), the top rows barely
move (no leverage), and the extraction sees the average constraint.

The image is then extracted with the osamp_spec branch at s = 1, 2, 3 and a
small grid of lambda_fringe values, and the recovered fine spectra are
compared against the true LSF binned onto each fine grid.

Run:
    uv run experiment_synthetic_lsf.py
    SHOW=0 uv run experiment_synthetic_lsf.py   # no window, just PNG + table

Outputs:
    experiment_synthetic_lsf.png   five-panel summary figure
    experiment_synthetic_lsf.npz   all recovered spectra + truth + metrics
"""

import argparse
import os

import matplotlib

if os.environ.get("SHOW", "1") == "0":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import charslit

Q = 50  # sub-pixel render resolution (sub-bins per detector pixel)


# ----------------------------------------------------------------------------
# Truth: asymmetric LSF
# ----------------------------------------------------------------------------

def make_lsf(tophat_w=0.8, sigma=0.25, wing_frac=0.15, wing_tau=0.8,
             half_span=8.0):
    """Asymmetric LSF on a fine grid: slight top hat + blue (low-x) wing.

    Returns (u, lsf) with u in detector pixels relative to line centre and
    lsf normalised to unit integral.
    """
    du = 1.0 / Q
    u = np.arange(-half_span, half_span + du / 2, du)

    core = np.where(np.abs(u) <= tophat_w / 2, 1.0, 0.0)
    g = np.exp(-0.5 * (u / sigma) ** 2)
    core = np.convolve(core, g, mode="same")
    core /= core.sum() * du

    # one-sided exponential toward negative u (blue)
    blue = np.where(u <= 0, np.exp(u / wing_tau), 0.0)
    blue /= blue.sum() * du
    wing = np.convolve(core, blue, mode="same") * du

    lsf = (1 - wing_frac) * core + wing_frac * wing
    lsf /= lsf.sum() * du
    return u, lsf


def lsf_profile(u_grid, lsf, x_sub, centre):
    """Evaluate the LSF at sub-pixel columns x_sub for a line at `centre`."""
    return np.interp(x_sub - centre, u_grid, lsf, left=0.0, right=0.0)


# ----------------------------------------------------------------------------
# Synthetic image
# ----------------------------------------------------------------------------

def tilt_coeffs(nrows, slope_bot, slope_top):
    """slitcurve (c1, c2) for tilt varying linearly with row, in the
    t = y - nrows//2 frame that slitdec uses."""
    jc = nrows // 2
    c2 = (slope_top - slope_bot) / (2.0 * (nrows - 1))
    c1 = slope_bot + (slope_top - slope_bot) * jc / (nrows - 1)
    return c1, c2, jc


def make_image(nrows, ncols, line_x0, slope_bot, slope_top, u_grid, lsf,
               amp=4000.0, read_noise=2.0, noise=True, seed=42, subrows=8):
    """Render the tilted unresolved line. Each detector row is integrated
    over `subrows` sub-row positions so the within-row shear of the varying
    tilt is included, matching a real continuous exposure."""
    c1, c2, jc = tilt_coeffs(nrows, slope_bot, slope_top)
    x_sub = (np.arange(ncols * Q) + 0.5) / Q

    # smooth, fairly flat slit illumination with soft edges
    yy_all = np.arange(nrows)
    slit = np.exp(-0.5 * ((yy_all - (nrows - 1) / 2) / (0.30 * nrows)) ** 4)

    model = np.zeros((nrows, ncols))
    for j in range(nrows):
        row = np.zeros(ncols)
        for k in range(subrows):
            t = j + (k + 0.5) / subrows - 0.5 - jc
            delta = t * (c1 + c2 * t)
            prof = lsf_profile(u_grid, lsf, x_sub, line_x0 + delta)
            row += prof.reshape(ncols, Q).mean(axis=1)
        model[j] = row / subrows
    model *= amp * slit[:, None]

    rng = np.random.RandomState(seed)
    if noise:
        im = rng.poisson(np.maximum(model, 0)).astype(float)
        im += rng.normal(0.0, read_noise, im.shape)
        unc = np.sqrt(np.abs(model) + read_noise ** 2)
    else:
        # same weighting as the noisy case, just a zero-noise realisation
        im = model.copy()
        unc = np.sqrt(np.abs(model) + read_noise ** 2)
    return im, unc, model, (c1, c2)


# ----------------------------------------------------------------------------
# Extraction + comparison
# ----------------------------------------------------------------------------

def extract(im, unc, ycen, slitcurve, slitdeltas, osample, s, lambda_fringe,
            lambda_sP=0.0):
    r = charslit.slitdec(
        im, unc, np.ones(im.shape, dtype=np.uint8), ycen.copy(),
        slitcurve, slitdeltas, osample=osample, osamp_spec=s,
        lambda_sP=lambda_sP, lambda_fringe=lambda_fringe, kappa=0.0,
    )
    if r["return_code"] != 0:
        return None, None
    sp = r["spectrum"]
    xf = (np.arange(sp.size) + 0.5) / s
    return xf, sp


def binned_truth(u_grid, lsf, line_x0, ncols, s):
    """True LSF integrated into the fine bins of grid s (unit total area)."""
    x_sub = (np.arange(ncols * Q) + 0.5) / Q
    prof = lsf_profile(u_grid, lsf, x_sub, line_x0)
    # mean flux density per fine bin via the cumulative integral, so Q need
    # not be divisible by s
    cum = np.concatenate([[0.0], np.cumsum(prof) / Q])
    x_edges = np.arange(ncols * Q + 1) / Q
    bin_edges = np.arange(ncols * s + 1) / s
    cum_at = np.interp(bin_edges, x_edges, cum)
    return np.diff(cum_at) * s


def norm_window(xf, y, x0, win):
    """Area-normalise y over |xf - x0| < win; returns (x, y_norm) in window."""
    sel = np.abs(xf - x0) < win
    x, yy = xf[sel], y[sel].astype(float)
    dx = x[1] - x[0]
    area = np.sum(yy) * dx
    return x, yy / area if area != 0 else yy


def half_crossings(x, y, level):
    """Left/right crossings of y == level around the global peak."""
    pk = int(np.argmax(y))
    left = right = np.nan
    for i in range(pk, 0, -1):
        if (y[i - 1] - level) * (y[i] - level) < 0:
            f = (level - y[i - 1]) / (y[i] - y[i - 1])
            left = x[i - 1] + f * (x[i] - x[i - 1])
            break
    for i in range(pk, len(x) - 1):
        if (y[i] - level) * (y[i + 1] - level) < 0:
            f = (level - y[i]) / (y[i + 1] - y[i])
            right = x[i] + f * (x[i + 1] - x[i])
            break
    return left, right


def rms_vs_continuous(xw, yw, s, xtw, tnorm):
    """RMS of the recovered staircase against the continuous truth, both
    area-normalised, evaluated on the truth's fine grid. Directly comparable
    across different osamp_spec values (includes each grid's binning error)."""
    edges0 = xw[0] - 0.5 / s
    idx = np.floor((xtw - edges0) * s).astype(int)
    ok = (idx >= 0) & (idx < yw.size)
    return float(np.sqrt(np.mean((yw[idx[ok]] - tnorm[ok]) ** 2)))


def lsf_metrics(x, y, x0):
    """Shape metrics on an area-normalised profile."""
    pk = float(np.max(y))
    fl, fr = half_crossings(x, y, pk / 2)
    tl, tr = half_crossings(x, y, pk / 5)  # 20% level captures the wing
    dx = x[1] - x[0]
    centroid = np.sum(x * y) * dx / (np.sum(y) * dx)
    blue = np.sum(y[(x > x0 - 4.0) & (x < x0 - 1.0)]) * dx
    red = np.sum(y[(x > x0 + 1.0) & (x < x0 + 4.0)]) * dx
    return {
        "fwhm": fr - fl if np.isfinite(fl) and np.isfinite(fr) else np.nan,
        "asym50": (x[np.argmax(y)] - fl) - (fr - x[np.argmax(y)]),
        "asym20": (x[np.argmax(y)] - tl) - (tr - x[np.argmax(y)]),
        "blue_wing_flux": blue,
        "red_wing_flux": red,
        "centroid_err": centroid - x0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nrows", type=int, default=80)
    p.add_argument("--ncols", type=int, default=60)
    p.add_argument("--osample", type=int, default=6)
    p.add_argument("--slope-bot", type=float, default=0.1)
    p.add_argument("--slope-top", type=float, default=0.001)
    p.add_argument("--amp", type=float, default=4000.0)
    p.add_argument("--tophat-w", type=float, default=0.8,
                   help="full width of the top-hat core in detector px")
    p.add_argument("--sigma", type=float, default=0.25,
                   help="Gaussian sigma the top-hat is convolved with [px]")
    p.add_argument("--osamps", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--fringes", type=float, nargs="+",
                   default=[0.0, 0.001, 0.01, 0.1, 1.0])
    p.add_argument("--tag", default="",
                   help="suffix for output filenames, e.g. _notilt")
    args = p.parse_args()

    nrows, ncols = args.nrows, args.ncols
    line_x0 = ncols / 2 + 0.37  # deliberately off pixel centre
    win = 5.0

    u_grid, lsf = make_lsf(tophat_w=args.tophat_w, sigma=args.sigma)
    c1, c2, jc = tilt_coeffs(nrows, args.slope_bot, args.slope_top)
    print(f"tilt: {args.slope_bot} -> {args.slope_top} px/row over {nrows} "
          f"rows  =>  slitcurve c1={c1:.6f}, c2={c2:.3e} (t = y - {jc})")
    tspan = np.array([-jc, nrows - 1 - jc], dtype=float)
    dspan = tspan * (c1 + c2 * tspan)
    print(f"total horizontal shift across slit: {dspan[1] - dspan[0]:+.2f} px")

    ny = args.osample * (nrows + 1) + 1
    ycen = np.full(ncols, 0.5)
    slitcurve = np.zeros((ncols, 3))
    slitcurve[:, 1] = c1
    slitcurve[:, 2] = c2
    slitdeltas = np.zeros(ny)

    # truth metrics from the continuous LSF
    xt = u_grid + line_x0
    xtw, tnorm = norm_window(xt, lsf, line_x0, win)
    mt = lsf_metrics(xtw, tnorm, line_x0)

    results = {}
    hdr = (f"{'case':>10} {'s':>2} {'l_fr':>5} {'FWHM':>6} {'asym50':>7} "
           f"{'asym20':>7} {'blue':>6} {'red':>6} {'cen_err':>8} {'rms':>7} "
           f"{'rms_cont':>8}")

    for label, noise in [("noisy", True), ("noiseless", False)]:
        im, unc, model, _ = make_image(nrows, ncols, line_x0, args.slope_bot,
                                       args.slope_top, u_grid, lsf,
                                       amp=args.amp, noise=noise)
        print(f"\n--- {label} (peak pixel {im.max():.0f} counts) ---")
        print(hdr)
        print(f"{'truth':>10} {'-':>2} {'-':>5} {mt['fwhm']:>6.3f} "
              f"{mt['asym50']:>7.3f} {mt['asym20']:>7.3f} "
              f"{mt['blue_wing_flux']:>6.3f} {mt['red_wing_flux']:>6.3f} "
              f"{mt['centroid_err']:>8.4f} {'-':>7}")

        for s in args.osamps:
            truth_s = binned_truth(u_grid, lsf, line_x0, ncols, s)
            xf_t = (np.arange(ncols * s) + 0.5) / s
            for lf in args.fringes:
                if s == 1 and lf > 0:
                    continue  # lambda_fringe inactive at s=1
                xf, sp = extract(im, unc, ycen, slitcurve, slitdeltas,
                                 args.osample, s, lf)
                if xf is None:
                    print(f"{label:>10} {s:>2} {lf:>5} extraction FAILED")
                    continue
                xw, yw = norm_window(xf, sp, line_x0, win)
                _, tw = norm_window(xf_t, truth_s, line_x0, win)
                rms = float(np.sqrt(np.mean((yw - tw) ** 2)))
                rms_c = rms_vs_continuous(xw, yw, s, xtw, tnorm)
                m = lsf_metrics(xw, yw, line_x0)
                results[(label, s, lf)] = dict(
                    xf=xf, sp=sp, xw=xw, yw=yw, tw=tw, rms=rms,
                    rms_cont=rms_c, **m)
                print(f"{label:>10} {s:>2} {lf:>5} {m['fwhm']:>6.3f} "
                      f"{m['asym50']:>7.3f} {m['asym20']:>7.3f} "
                      f"{m['blue_wing_flux']:>6.3f} {m['red_wing_flux']:>6.3f} "
                      f"{m['centroid_err']:>8.4f} {rms:>7.4f} {rms_c:>8.4f}")
        if noise:
            im_plot = im

    # pick, per (case, s), the lambda_fringe with lowest rms vs continuous truth
    best = {}
    for (label, s, lf), r in results.items():
        k = (label, s)
        if k not in best or r["rms_cont"] < results[best[k]]["rms_cont"]:
            best[k] = (label, s, lf)

    # ------------------------------------------------------------------ plot
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.3])

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(im_plot, origin="lower", aspect="auto", cmap="viridis")
    ax.set_title("synthetic image (noisy)")
    ax.set_xlabel("column"); ax.set_ylabel("row")

    ax = fig.add_subplot(gs[0, 1])
    yyy = np.arange(nrows, dtype=float)
    tt = yyy - jc
    ax.plot(tt * (c1 + c2 * tt) + line_x0, yyy, "k-")
    ax.set_title("line trajectory: tilt 0.1 -> 0.001 px/row")
    ax.set_xlabel("line centre [column]"); ax.set_ylabel("row")

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(u_grid, lsf / lsf.max(), "k-", lw=2)
    ax.set_xlim(-4, 4)
    ax.set_title("true LSF: top-hat core + blue wing")
    ax.set_xlabel("offset [px]"); ax.set_ylabel("normalised")
    ax.axvline(0, color="0.7", lw=0.5)

    for col, label in enumerate(["noisy", "noiseless"]):
        ax = fig.add_subplot(gs[1, col])
        ax.plot(xtw - line_x0, tnorm, color="0.6", lw=2.5, ls="--",
                label="true LSF")
        for s in args.osamps:
            k = best.get((label, s))
            if k is None:
                continue
            r = results[k]
            ax.step(r["xw"] - line_x0, r["yw"], where="mid", lw=1.3,
                    label=f"s={s}, l_fr={k[2]:g} (rms {r['rms_cont']:.3f})")
        ax.set_xlim(-win, win)
        ax.set_title(f"recovered LSF, {label} (best lambda_fringe per s)")
        ax.set_xlabel("offset from line centre [px]")
        ax.set_ylabel("area-normalised")
        ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    for s in args.osamps:
        k = best.get(("noisy", s))
        if k is None:
            continue
        r = results[k]
        ax.step(r["xw"] - line_x0, r["yw"] - r["tw"], where="mid", lw=1.2,
                label=f"s={s}")
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_xlim(-win, win)
    ax.set_title("residual vs binned truth (noisy)")
    ax.set_xlabel("offset [px]"); ax.set_ylabel("residual")
    ax.legend(fontsize=8)

    fig.tight_layout()
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, f"experiment_synthetic_lsf{args.tag}.png")
    fig.savefig(out, dpi=120)
    print(f"\nSaved {out}")

    np.savez(os.path.join(here, f"experiment_synthetic_lsf{args.tag}.npz"),
             u_grid=u_grid, lsf=lsf, line_x0=line_x0, c1=c1, c2=c2,
             **{f"{lab}_s{s}_lf{lf:g}_{key}": results[(lab, s, lf)][key]
                for (lab, s, lf) in results for key in ("xf", "sp")})
    if os.environ.get("SHOW", "1") != "0":
        plt.show()


if __name__ == "__main__":
    main()
