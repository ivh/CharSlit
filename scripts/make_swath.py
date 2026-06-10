"""Cut a single-order swath from a cr2res full-frame + trace-wave file.

Input
-----
- image: e.g. data/CRIRES_UNE_J.fits with CHIPn.INT1 / CHIPnERR.INT1
  extensions (2048 x 2048 each)
- trace:  e.g. data/J1228_tw.fits, BinTable per chip with one row per
  order trace: All/Upper/Lower (y(x) polynomials, increasing powers),
  SlitPolyA/B/C (a(x), b(x), c(x) of the tilt polynomial
  x'(y) = a + b*y + c*y^2 in absolute detector coordinates)

cr2res polynomials use 1-based pixel coordinates; we evaluate at
x+1 / keep y 1-based until the final integer shift.

Swath construction (slitdec conventions)
----------------------------------------
For each column x: yc = All(x). The column is shifted vertically by
int(yc) so the trace lands on the swath centre row jc = nrows//2, and
only frac(yc) is passed to slitdec as ycen. slitdec evaluates the tilt
polynomial at t = (slit position relative to trace centre), so

    c1(x) = b(x) + 2*c(x)*yc(x)      c2(x) = c(x)

Usage
-----
    uv run make_swath.py --list                 # enumerate chips/orders
    uv run make_swath.py --chip 1 --order 4     # cut + first-look plot

Outputs data/swath_{image-stem}_c{chip}o{order}.npz with im, unc,
ycen (fractional), slitcurve, plus provenance, and a diagnostic PNG
with the swath and the line-population (width vs amplitude) scatter.
"""

import argparse
import os

import matplotlib

if os.environ.get("SHOW", "1") == "0":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def polyval_inc(coeffs, x):
    """Evaluate polynomial with coefficients in increasing powers."""
    return sum(c * x ** i for i, c in enumerate(coeffs))


def list_orders(tw_path):
    with fits.open(tw_path) as h:
        for ext in range(1, len(h)):
            t = h[ext].data
            name = h[ext].name
            print(f"--- {name}")
            xc = 1024.0
            for r in range(len(t)):
                yc = polyval_inc(t["All"][r], xc)
                ylo = polyval_inc(t["Lower"][r], xc)
                yhi = polyval_inc(t["Upper"][r], xc)
                b = polyval_inc(t["SlitPolyB"][r], xc)
                c = polyval_inc(t["SlitPolyC"][r], xc)
                tilt = b + 2 * c * yc
                wl = t["Wavelength"][r]
                wlc = polyval_inc(wl, xc)
                print(f"  order {t['Order'][r]:2d}: yc={yc:7.1f}  "
                      f"height={yhi - ylo:5.1f}  tilt(centre)={tilt:+.4f} "
                      f"px/row  wl(centre)={wlc:.1f} nm")


def cut_swath(img_path, tw_path, chip, order, nrows=None, trim=4,
              x0=0, width=None, verbose=True, xbin=1):
    """x0/width select a column block (0-based detector columns).

    With xbin=2 the block is binned pairwise in x: both native columns of
    a pair are cut with a single integer trace shift (from the pair-centre
    trace value) and summed; uncertainties add in quadrature. The tilt
    coefficients are divided by xbin since delta is then measured in
    binned pixels, while t (rows) is unchanged."""
    with fits.open(img_path) as h:
        im_full = h[f"CHIP{chip}.INT1"].data.astype(float)
        err_full = h[f"CHIP{chip}ERR.INT1"].data.astype(float)
    with fits.open(tw_path) as h:
        t = h[f"CHIP{chip}.INT1"].data
        r = np.where(t["Order"] == order)[0]
        if len(r) != 1:
            raise ValueError(f"order {order} not found on chip {chip}; "
                             f"available: {sorted(t['Order'])}")
        r = int(r[0])
        p_all = t["All"][r]
        p_lo, p_hi = t["Lower"][r], t["Upper"][r]
        p_a = t["SlitPolyA"][r]
        p_b, p_c = t["SlitPolyB"][r], t["SlitPolyC"][r]
        p_wl = t["Wavelength"][r]

    nfull = im_full.shape[0]
    if width is None:
        width = im_full.shape[1] - x0
    cols = np.arange(x0, x0 + width)
    x1 = cols + 1.0  # cr2res 1-based pixel coords

    yc = polyval_inc(p_all, x1)
    if nrows is None:
        height = np.min(polyval_inc(p_hi, x1) - polyval_inc(p_lo, x1))
        nrows = int(height) - 2 * trim
    jc = nrows // 2

    # sanity: tilt polynomial should pass through the trace
    xm = x1[len(x1) // 2]
    a_mid = polyval_inc(p_a, xm)
    b_mid = polyval_inc(p_b, xm)
    c_mid = polyval_inc(p_c, xm)
    yc_mid = polyval_inc(p_all, xm)
    xprime = a_mid + b_mid * yc_mid + c_mid * yc_mid ** 2
    if verbose:
        print(f"trace check at x={xm:.0f}: x'(yc) = {xprime:.2f} "
              f"(expect ~{xm:.0f}), "
              f"tilt = {b_mid + 2 * c_mid * yc_mid:+.4f} px/row")

    if width % xbin:
        raise ValueError("width must be divisible by xbin")
    wout = width // xbin
    im = np.full((nrows, wout), np.nan)
    unc = np.full((nrows, wout), np.nan)
    ycen = np.zeros(wout)
    slitcurve = np.zeros((wout, 3))

    for i in range(wout):
        grp = cols[i * xbin:(i + 1) * xbin]
        x1c = grp.mean() + 1.0  # 1-based pair centre
        yc_i = polyval_inc(p_all, x1c)
        # 1-based trace centre -> 0-based row index of pixel containing it
        y0 = yc_i - 1.0
        yint = int(np.floor(y0))
        ycen[i] = y0 - yint
        lo = yint - jc
        hi = lo + nrows
        src_lo, src_hi = max(lo, 0), min(hi, nfull)
        dst_lo, dst_hi = src_lo - lo, src_hi - lo
        im[dst_lo:dst_hi, i] = im_full[src_lo:src_hi, grp].sum(axis=1)
        unc[dst_lo:dst_hi, i] = np.sqrt(
            (err_full[src_lo:src_hi, grp] ** 2).sum(axis=1))

        b = polyval_inc(p_b, x1c)
        c = polyval_inc(p_c, x1c)
        slitcurve[i, 1] = (b + 2 * c * yc_i) / xbin
        slitcurve[i, 2] = c / xbin

    wl_centre = polyval_inc(p_wl, np.array([x1[0], xm, x1[-1]]))
    return dict(im=im, unc=unc, ycen=ycen, slitcurve=slitcurve,
                yc_abs=yc, nrows=nrows, wl_span=wl_centre,
                chip=chip, order=order, x0=x0, width=wout, xbin=xbin)


def gauss(x, amp, mu, sig, off):
    return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2) + off


def line_population(sw, half_rows=3, win=6):
    """Detect lines in the central rows and fit Gaussian width/amplitude."""
    jc = sw["nrows"] // 2
    band = sw["im"][jc - half_rows:jc + half_rows + 1]
    spec = np.nanmean(band, axis=0)
    spec = np.where(np.isfinite(spec), spec, np.nanmedian(spec))
    floor = np.nanpercentile(spec, 20)
    noise = np.nanstd(spec[spec < np.nanpercentile(spec, 50)])
    pk, _ = find_peaks(spec, height=floor + 8 * noise, distance=4)

    lines = []
    x = np.arange(spec.size)
    for p in pk:
        sel = slice(max(p - win, 0), min(p + win + 1, spec.size))
        xs, ys = x[sel], spec[sel]
        try:
            popt, _ = curve_fit(
                gauss, xs, ys,
                p0=[spec[p] - floor, p, 1.2, floor],
                maxfev=2000)
        except RuntimeError:
            continue
        amp, mu, sig, off = popt
        if amp <= 0 or not (0.3 < abs(sig) < win):
            continue
        fwhm = 2.3548 * abs(sig)
        lines.append((mu, amp, fwhm))
    return spec, np.array(lines)


def usable_lines(lines, min_amp=2000.0, fwhm_lo=2.0, fwhm_hi=3.5,
                 min_sep=10.0, edge=15.0, width=512):
    """Select lines that are bright, in the unresolved-FWHM cluster, away
    from swath edges, and isolated from *any* other detected line."""
    if len(lines) == 0:
        return np.empty((0, 3))
    keep = []
    for mu, amp, fwhm in lines:
        if amp < min_amp or not (fwhm_lo < fwhm < fwhm_hi):
            continue
        if mu < edge or mu > width - edge:
            continue
        sep = np.abs(lines[:, 0] - mu)
        if np.any((sep > 0) & (sep < min_sep)):
            continue
        keep.append((mu, amp, fwhm))
    return np.array(keep)


def plot_swath(sw, spec, lines, out_png, close=False):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [1, 1, 1.2]})
    ax = axes[0]
    vmax = np.nanpercentile(sw["im"], 99)
    ax.imshow(sw["im"], origin="lower", aspect="auto", cmap="viridis",
              vmin=0, vmax=vmax)
    tilt = sw["slitcurve"][sw["width"] // 2, 1]
    ax.set_title(f"chip {sw['chip']} order {sw['order']} x0={sw['x0']}  "
                 f"wl {sw['wl_span'][0]:.1f}-{sw['wl_span'][2]:.1f} nm  "
                 f"tilt {tilt:+.4f} px/row")
    ax.set_ylabel("row")

    ax = axes[1]
    ax.plot(spec, lw=0.7)
    if len(lines):
        ax.plot(lines[:, 0], np.interp(lines[:, 0], np.arange(spec.size),
                                       spec), "rx", ms=5)
    ax.set_yscale("log")
    ax.set_ylabel("central-rows mean")
    ax.set_xlabel("column")

    ax = axes[2]
    if len(lines):
        ax.scatter(lines[:, 2], lines[:, 1], s=18)
        for mu, amp, fwhm in lines:
            ax.annotate(f"{mu:.0f}", (fwhm, amp), fontsize=6, alpha=0.6,
                        xytext=(2, 2), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_xlabel("fitted FWHM [px]")
    ax.set_ylabel("fitted amplitude")
    ax.set_title("line population: look for the narrow (U) cluster")

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    if close:
        plt.close(fig)


def plot_all(args):
    outdir = os.path.join("scratch", "swath_plots")
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    n = 0
    for chip in (1, 2, 3):
        for order in range(1, 10):
            for x0 in range(0, 2048, args.width):
                try:
                    sw = cut_swath(args.image, args.tw, chip, order,
                                   nrows=args.nrows, x0=x0,
                                   width=args.width, verbose=False)
                except ValueError:
                    continue
                spec, lines = line_population(sw)
                out = os.path.join(
                    outdir, f"swath_{stem}_c{chip}o{order}x{x0:04d}.png")
                plot_swath(sw, spec, lines, out, close=True)
                n += 1
    print(f"saved {n} plots in {outdir}/")


def survey(args):
    print(f"{'chip':>4} {'ord':>3} {'x0':>5} {'tilt':>7} {'nlines':>6} "
          f"{'usable':>6}  usable lines (x: amp, FWHM)")
    rows = []
    for chip in (1, 2, 3):
        for order in range(1, 10):
            for x0 in range(0, 2048, args.width):
                try:
                    sw = cut_swath(args.image, args.tw, chip, order,
                                   nrows=args.nrows, x0=x0,
                                   width=args.width, verbose=False)
                except ValueError:
                    continue
                _, lines = line_population(sw)
                ok = usable_lines(lines, min_amp=args.min_amp,
                                  width=args.width)
                tilt = sw["slitcurve"][args.width // 2, 1]
                desc = "  ".join(f"{mu:.0f}: {amp:.0f}, {fw:.2f}"
                                 for mu, amp, fw in ok)
                rows.append((chip, order, x0, tilt, len(lines), len(ok),
                             desc))
    rows.sort(key=lambda r: (-r[5], -abs(r[3])))
    for chip, order, x0, tilt, nl, nu, desc in rows:
        if nu == 0 and not args.all:
            continue
        print(f"{chip:>4} {order:>3} {x0:>5} {tilt:>+7.4f} {nl:>6} "
              f"{nu:>6}  {desc}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="data/CRIRES_UNE_J.fits")
    p.add_argument("--tw", default="data/J1228_tw.fits")
    p.add_argument("--list", action="store_true")
    p.add_argument("--survey", action="store_true")
    p.add_argument("--plot-all", action="store_true",
                   help="save a diagnostic plot for every swath block")
    p.add_argument("--all", action="store_true",
                   help="in survey, also print swaths with 0 usable lines")
    p.add_argument("--chip", type=int, default=1)
    p.add_argument("--order", type=int, default=None)
    p.add_argument("--nrows", type=int, default=150)
    p.add_argument("--x0", type=int, default=0)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--min-amp", type=float, default=2000.0)
    args = p.parse_args()

    if args.list:
        list_orders(args.tw)
        return
    if args.survey:
        survey(args)
        return
    if args.plot_all:
        plot_all(args)
        return
    if args.order is None:
        list_orders(args.tw)
        return

    sw = cut_swath(args.image, args.tw, args.chip, args.order,
                   nrows=args.nrows, x0=args.x0, width=args.width)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    tag = f"{stem}_c{args.chip}o{args.order}x{args.x0}"

    spec, lines = line_population(sw)
    print(f"swath: {sw['nrows']} x {sw['im'].shape[1]}, "
          f"wl {sw['wl_span'][0]:.1f}-{sw['wl_span'][2]:.1f} nm, "
          f"{len(lines)} fitted lines")
    if len(lines):
        print(f"{'x':>8} {'amp':>10} {'FWHM':>6}")
        for mu, amp, fwhm in lines[np.argsort(lines[:, 2])]:
            print(f"{mu:>8.1f} {amp:>10.0f} {fwhm:>6.2f}")

    out_npz = os.path.join("data", f"swath_{tag}.npz")
    np.savez(out_npz, **{k: v for k, v in sw.items()
                         if isinstance(v, np.ndarray) or np.isscalar(v)},
             lines=lines)
    print(f"saved {out_npz}")

    os.makedirs("scratch", exist_ok=True)
    out_png = os.path.join("scratch", f"swath_{tag}.png")
    plot_swath(sw, spec, lines, out_png)
    print(f"saved {out_png}")
    if os.environ.get("SHOW", "1") != "0":
        plt.show()


if __name__ == "__main__":
    main()
