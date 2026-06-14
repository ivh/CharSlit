"""Bin-x2 LSF-recovery validation on real CRIRES+ U-Ne data (see OSAMP_EXPERIMENTS.md).

The native CRIRES swath (chip 2, order 4, x0=512, width=512, nrows=150) is
well sampled (FWHM ~2.9 px), so an unbinned extraction at osamp_spec=1 is
the ground truth. The same detector columns binned x2 in dispersion
(FWHM ~1.45 binned px, under-sampled) are extracted with osamp_spec=2,
whose fine grid lands exactly on the native pixel centres — index-by-index
comparison, no resampling.

Products compared per line and stacked:
  (a) truth      : xbin=1, s=1            on native px
  (b) staircase  : xbin=2, s=1            on 2-native-px bins
  (c) recovered  : xbin=2, s in {2, 3, 4}, lambda_fringe in {0, 1e-3, 1e-2}

Only s=2 lands exactly on the native pixel grid; s=3 (2/3 native px bins)
and s=4 (1/2 native px bins) are compared by evaluating their staircase at
the truth pixel centres (same convention as the synthetic experiment).

Run from repo root:  uv run scripts/experiment_binx2_crires.py
Outputs: scratch/experiment_binx2_crires_s{2,3,4}.png / .npz + metrics
on stdout.
"""

import argparse
import os

import matplotlib

if os.environ.get("SHOW", "1") == "0":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import charslit
from analyse_wide_bisector import bis_value, bisector
from make_swath import cut_swath

IMAGE = "data/CRIRES_UNE_J.fits"
TW = "data/J1228_tw.fits"
CHIP, ORDER, X0, WIDTH, NROWS = 2, 4, 512, 512, 150

# analysis lines (see OSAMP_EXPERIMENTS.md; native swath columns); 60.5/68.5 are 8 px
# apart so their windows are narrowed to avoid overlap
LINES = [37.8, 60.5, 68.5, 169.2, 310.7, 324.6, 444.6]
WIN_DEFAULT = 4.5
WIN_OVERRIDE = {60.5: 3.5, 68.5: 3.5}

FRINGES = [0.0, 1e-3, 1e-2]
OSAMPS = [2, 3, 4]
LEVELS = np.linspace(0.10, 0.95, 35)


def extract(sw, s, lambda_fringe=0.0):
    im, unc = sw["im"], sw["unc"]
    good = np.isfinite(im) & np.isfinite(unc) & (unc > 0)
    r = charslit.slitdec(
        np.where(good, im, 0.0),
        np.where(good, unc, 1.0),
        good.astype(np.uint8),
        sw["ycen"].copy(),
        sw["slitcurve"],
        np.zeros(im.shape[0]),
        osample=6, osamp_spec=s,
        lambda_sP=0.0, lambda_sL=1.0,
        lambda_fringe=lambda_fringe, kappa=10.0,
    )
    if r["return_code"] != 0:
        raise RuntimeError(
            f"slitdec failed: s={s} lf={lambda_fringe} "
            f"return_code={r['return_code']}")
    return r["spectrum"]


def fine_grid_native(ncols, s, xbin):
    """Fine-bin centres in native-pixel coordinates of the unbinned swath."""
    return (np.arange(ncols * s) + 0.5) / s * xbin


def norm_window(x, y, x0, win):
    """Area-normalised flux density in |x - x0| < win (own grid spacing)."""
    sel = np.abs(x - x0) < win
    xs, ys = x[sel], y[sel].astype(float)
    dx = xs[1] - xs[0]
    area = np.sum(ys) * dx
    return xs, ys / area if area != 0 else ys


def centroid(x, y):
    return float(np.sum(x * y) / np.sum(y))


def fwhm(x, y):
    pk = int(np.argmax(y))
    half = 0.5 * y[pk]
    left = right = np.nan
    for i in range(pk, 0, -1):
        if (y[i - 1] - half) * (y[i] - half) < 0:
            f = (half - y[i - 1]) / (y[i] - y[i - 1])
            left = x[i - 1] + f * (x[i] - x[i - 1])
            break
    for i in range(pk, len(x) - 1):
        if (y[i] - half) * (y[i + 1] - half) < 0:
            f = (half - y[i]) / (y[i + 1] - y[i])
            right = x[i] + f * (x[i + 1] - x[i])
            break
    return right - left


def gh_model(x, amp, mu, sig, h3, h4, off):
    """Gauss-Hermite line profile (van der Marel & Franx 1993):
    h3 ~ skewness, h4 ~ kurtosis."""
    w = (x - mu) / sig
    H3 = (2 * np.sqrt(2) * w ** 3 - 3 * np.sqrt(2) * w) / np.sqrt(6)
    H4 = (4 * w ** 4 - 12 * w ** 2 + 3) / np.sqrt(24)
    g = np.exp(-0.5 * w ** 2) / (sig * np.sqrt(2 * np.pi))
    return amp * g * (1 + h3 * H3 + h4 * H4) + off


def gh_binned(dx, nsub=8):
    """GH model averaged over bins of width dx (so under-sampled products
    are fitted with the correct forward model, not the continuous curve)."""
    offs = (np.arange(nsub) + 0.5) / nsub * dx - dx / 2

    def f(x, amp, mu, sig, h3, h4, off):
        return np.mean([gh_model(x + o, amp, mu, sig, h3, h4, off)
                        for o in offs], axis=0)
    return f


def fit_gh(x, y, x0):
    """Returns (popt, perr) or None. 6 free parameters: a 2-px staircase in
    a +-3.5 px window has fewer points than parameters and must fail."""
    dx = x[1] - x[0]
    # tight bounds: GH is only meaningful for |h3|,|h4| << 1, and the
    # pedestal/sigma/h4 degeneracy in a +-4.5 px window needs reining in
    p0 = [1.0, x0, 1.4, 0.0, 0.0, max(float(np.min(y)), 0.0)]
    bounds = ([0.3, x0 - 1.5, 0.6, -0.35, -0.35, -0.02],
              [3.0, x0 + 1.5, 2.8, 0.35, 0.35, 0.15])
    try:
        popt, pcov = curve_fit(gh_binned(dx), x, y, p0=p0, bounds=bounds,
                               maxfev=10000)
    except (RuntimeError, ValueError, TypeError):
        return None
    return popt, np.sqrt(np.diag(pcov))


def step_eval(xc, yc, dx, x):
    """Evaluate a staircase (bin centres xc, width dx) at positions x."""
    idx = np.floor((x - (xc[0] - dx / 2)) / dx).astype(int)
    idx = np.clip(idx, 0, yc.size - 1)
    return yc[idx]


def analyse_line(x0, win, truth, products):
    """truth/products: dict name -> (x_native, density). Returns metrics and
    bisector curves (relative to the truth centroid)."""
    xt, yt = norm_window(*truth, x0, win)
    cen_t = centroid(xt, yt)
    res = {}
    curves = {}
    bt = bisector(xt, yt, LEVELS) - cen_t
    curves["truth"] = bt
    res["truth"] = dict(fwhm=fwhm(xt, yt), dcent=0.0, rms=0.0,
                        bis=bis_value(LEVELS, bt))
    for name, (x, y) in products.items():
        xw, yw = norm_window(x, y, x0, win)
        b = bisector(xw, yw, LEVELS) - cen_t
        curves[name] = b
        if xw.size == xt.size and np.allclose(xw, xt):
            rms = float(np.sqrt(np.mean((yw - yt) ** 2)))
        else:  # staircase on a coarser grid: compare as a step function
            rms = float(np.sqrt(np.mean(
                (step_eval(xw, yw, xw[1] - xw[0], xt) - yt) ** 2)))
        res[name] = dict(fwhm=fwhm(xw, yw), dcent=centroid(xw, yw) - cen_t,
                         rms=rms, bis=bis_value(LEVELS, b))
    return res, curves, cen_t


def analyse_s(s, truth, products):
    """Per-line + stacked analysis for one set of products vs the truth.
    Returns (all_res, stacked bisector curves)."""
    names = ["truth"] + list(products)
    all_res, all_cents = [], []
    for x0 in LINES:
        win = WIN_OVERRIDE.get(x0, WIN_DEFAULT)
        res, _, cen_t = analyse_line(x0, win, truth, products)
        all_res.append(res)
        all_cents.append(cen_t)

    print(f"\n==== s={s}: per-line metrics (windows +-{WIN_DEFAULT} px, "
          f"+-{WIN_OVERRIDE[60.5]} for 60.5/68.5) ====")
    hdr = f"{'line':>6} {'product':<22} {'FWHM':>6} {'dcent':>8} " \
          f"{'rms':>9} {'BIS[mpx]':>9}"
    for x0, res in zip(LINES, all_res):
        print(hdr if x0 == LINES[0] else "")
        for name in names:
            m = res[name]
            print(f"{x0:>6.1f} {name:<22} {m['fwhm']:>6.3f} "
                  f"{m['dcent']:>+8.3f} {m['rms']:>9.5f} "
                  f"{m['bis']*1000:>+9.1f}")

    # mean per-line BIS restricted to lines where EVERY product has a
    # finite BIS (otherwise each product would average a different subset)
    finite = [all(np.isfinite(r[name]["bis"]) for name in names)
              for r in all_res]
    common = [x0 for x0, ok in zip(LINES, finite) if ok]
    print(f"\nlines with finite BIS in all products: {common}")

    # stacked-profile bisector: align each line's area-normalised profile
    # on its truth centroid, interpolate onto a common grid, average across
    # all lines, then take one bisector per product (robust at this S/N)
    xg = np.arange(-4.5, 4.5 + 1e-9, 0.25)
    stacked = {}
    print(f"\nstacked over {len(LINES)} lines:")
    print(f"{'product':<22} {'<BIS>common[mpx]':>17} {'BIS(stack)[mpx]':>16}")
    prods_all = {"truth": truth, **products}
    for name in names:
        profs = []
        for x0, cen_t in zip(LINES, all_cents):
            win = WIN_OVERRIDE.get(x0, WIN_DEFAULT)
            xw, yw = norm_window(*prods_all[name], x0, win)
            profs.append(np.interp(xg, xw - cen_t, yw,
                                   left=np.nan, right=np.nan))
        stack_prof = np.nanmean(profs, axis=0)
        b = bisector(xg[np.isfinite(stack_prof)],
                     stack_prof[np.isfinite(stack_prof)], LEVELS)
        stacked[name] = b
        mean_bis = np.nanmean([r[name]["bis"]
                               for r, ok in zip(all_res, finite) if ok])
        print(f"{name:<22} {mean_bis*1000:>+17.1f} "
              f"{bis_value(LEVELS, b)*1000:>+16.1f}")
    return all_res, stacked


def make_figure(s, xbin, truth, products, all_res, stacked, out_png):
    """7 line panels + stacked-bisector panel for one osamp_spec value."""
    names = ["truth"] + list(products)
    style = {
        "truth": dict(color="k", lw=2.0),
        "binned s=1": dict(color="0.6", lw=1.4, drawstyle="steps-mid"),
        f"binned s={s} lf=0": dict(color="C0", lw=1.0, alpha=0.6),
        f"binned s={s} lf=0.001": dict(color="C1", lw=1.4),
        f"binned s={s} lf=0.01": dict(color="C2", lw=1.2, ls="--"),
    }
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.ravel()
    for ax, x0, res in zip(axes, LINES, all_res):
        win = WIN_OVERRIDE.get(x0, WIN_DEFAULT)
        xt, yt = norm_window(*truth, x0, win)
        ax.plot(xt - x0, yt, label="truth", **style["truth"])
        for name, (x, y) in products.items():
            xw, yw = norm_window(x, y, x0, win)
            ax.plot(xw - x0, yw, label=name, **style[name])
        ax.set_title(f"line @ {x0:.1f}  "
                     f"truth FWHM {res['truth']['fwhm']:.2f} px", fontsize=9)
        ax.set_xlabel("native px from line")
        if x0 == LINES[0]:
            ax.legend(fontsize=7)

    ax = axes[-1]
    for name in names:
        st = style.get(name, {}).copy()
        st.pop("drawstyle", None)
        ax.plot(stacked[name] * 1000, LEVELS, label=name, **st)
    ax.set_xlabel("stacked-profile bisector [mpx from truth centroid]")
    ax.set_ylabel("fractional level of peak")
    ax.set_title("bisector of stacked profile (7 lines)", fontsize=9)
    ax.axvline(0, color="0.85", lw=0.5)
    ax.legend(fontsize=7)

    fig.suptitle(f"CRIRES+ U-Ne chip{CHIP} order{ORDER} x0={X0}: "
                 f"bin-x{xbin:g} self-validation, osamp_spec={s}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"saved {out_png}")
    return fig


def gauss_hermite_analysis(truth, products):
    """Bin-integrated Gauss-Hermite fit per line and product; table on
    stdout, returns {(line, product): (popt, perr)} with popt =
    (amp, mu, sigma, h3, h4, off)."""
    names = ["truth"] + list(products)
    prods_all = {"truth": truth, **products}
    results = {}
    print("\n==== Gauss-Hermite fits (bin-integrated; h3=skew, h4=kurt) ====")
    hdr = (f"{'line':>6} {'product':<22} {'FWHM_G':>7} {'dmu':>8} "
           f"{'h3':>7} {'+-':>6} {'h4':>7} {'+-':>6}")
    for x0 in LINES:
        win = WIN_OVERRIDE.get(x0, WIN_DEFAULT)
        print(hdr if x0 == LINES[0] else "")
        mu_t = np.nan
        for name in names:
            xw, yw = norm_window(*prods_all[name], x0, win)
            r = fit_gh(xw, yw, x0)
            results[(x0, name)] = r
            if r is None:
                print(f"{x0:>6.1f} {name:<22} fit failed "
                      f"({xw.size} points, 6 parameters)")
                continue
            (amp, mu, sig, h3, h4, off), perr = r
            if name == "truth":
                mu_t = mu
            print(f"{x0:>6.1f} {name:<22} {2.3548*sig:>7.3f} "
                  f"{mu - mu_t:>+8.3f} {h3:>+7.3f} {perr[3]:>6.3f} "
                  f"{h4:>+7.3f} {perr[4]:>6.3f}")
    return results


def gh_figure(results, products, out_png):
    names = ["truth"] + list(products)
    style = {"truth": dict(color="k", marker="s", ms=7, zorder=5),
             "binned s=1": dict(color="0.6", marker="o", ms=6)}
    cyc = ["C1", "C4", "C5", "C2", "C8", "C9"]
    for name in names:
        if name not in style:
            style[name] = dict(color=cyc[0], marker="D", ms=5)
            cyc.pop(0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    # fixed y-ranges so one broken blend fit cannot blow up the panels
    panels = [("FWHM_G [px]", lambda p: 2.3548 * p[2], None, None),
              ("mu - truth mu [px]", lambda p: p[1], 1, (-0.6, 0.6)),
              ("h3 (skewness)", lambda p: p[3], 3, (-0.45, 0.45)),
              ("h4 (kurtosis)", lambda p: p[4], 4, (-0.45, 0.45))]
    xpos = np.arange(len(LINES))
    for ax, (label, get, ierr, ylim) in zip(axes.ravel(), panels):
        for k, name in enumerate(names):
            vals, errs = [], []
            for x0 in LINES:
                r = results[(x0, name)]
                rt = results[(x0, "truth")]
                if r is None:
                    vals.append(np.nan)
                    errs.append(0.0)
                    continue
                v = get(r[0])
                if label.startswith("mu") and rt is not None:
                    v -= rt[0][1]
                vals.append(v)
                errs.append(r[1][ierr] if ierr is not None else 0.0)
            off = (k - len(names) / 2) * 0.06
            ax.errorbar(xpos + off, vals, yerr=errs, ls="none",
                        capsize=2, lw=1, label=name, **style[name])
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if label.startswith(("mu", "h")):
            ax.axhline(0, color="0.85", lw=0.5)
    for ax in axes[1]:
        ax.set_xticks(xpos, [f"{x0:.0f}" for x0 in LINES])
        ax.set_xlabel("line [native px]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Gauss-Hermite shape recovery (error bars: fit 1-sigma; "
                 "310.7/324.6 are blends)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"saved {out_png}")
    return fig


def parse_xbin(text):
    """'2', '1.5', or 'sqrt3' -> float bin factor."""
    if text.startswith("sqrt"):
        return float(np.sqrt(float(text[4:])))
    return float(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xbin", type=parse_xbin, default=2.0,
                   help="dispersion bin factor; accepts e.g. 2, 1.5, sqrt3")
    args = p.parse_args()
    xbin = args.xbin
    tag = f"_b{xbin:.4g}"
    os.makedirs("scratch", exist_ok=True)

    sw1 = cut_swath(IMAGE, TW, CHIP, ORDER, nrows=NROWS, x0=X0,
                    width=WIDTH, xbin=1)
    sw2 = cut_swath(IMAGE, TW, CHIP, ORDER, nrows=NROWS, x0=X0,
                    width=WIDTH, xbin=xbin)
    nbin = sw2["im"].shape[1]
    print(f"xbin={xbin:g}: {nbin} binned columns, "
          f"native FWHM ~2.9 px -> ~{2.9 / xbin:.2f} binned px")

    print("extracting truth (xbin=1, s=1) ...")
    sp_truth = extract(sw1, s=1)
    print(f"extracting staircase (xbin={xbin:g}, s=1) ...")
    sp_stair = extract(sw2, s=1)
    sp = {}
    for s in OSAMPS:
        for lf in FRINGES:
            print(f"extracting recovered (xbin={xbin:g}, s={s}, "
                  f"lambda_fringe={lf:g}) ...")
            sp[(s, lf)] = extract(sw2, s=s, lambda_fringe=lf)

    x_truth = fine_grid_native(WIDTH, 1, 1)
    x_stair = fine_grid_native(nbin, 1, xbin)

    # flux density per native px: a fine bin of s covers xbin/s native px
    truth = (x_truth, sp_truth)
    stair = (x_stair, sp_stair / xbin)

    npz = dict(x_truth=x_truth, sp_truth=sp_truth,
               x_stair=x_stair, sp_stair=sp_stair, xbin=xbin,
               lines=np.array(LINES), levels=LEVELS)
    for s in OSAMPS:
        x_s = fine_grid_native(nbin, s, xbin)
        products = {"binned s=1": stair}
        for lf in FRINGES:
            products[f"binned s={s} lf={lf:g}"] = (
                x_s, sp[(s, lf)] * s / xbin)
        all_res, stacked = analyse_s(s, truth, products)
        make_figure(s, xbin, truth, products, all_res, stacked,
                    f"scratch/experiment_binx2_crires{tag}_s{s}.png")
        npz[f"x_s{s}"] = x_s
        for lf in FRINGES:
            npz[f"sp_s{s}_lf{lf:g}"] = sp[(s, lf)]
        for name, b in stacked.items():
            npz[f"stacked_s{s}_{name.replace(' ', '_')}"] = b

    # Gauss-Hermite shape recovery: staircase + all s at the recommended
    # lambda_fringe=1e-3 against the truth
    gh_products = {"binned s=1": stair}
    for s in OSAMPS:
        x_s = fine_grid_native(nbin, s, xbin)
        gh_products[f"binned s={s} lf=0.001"] = (
            x_s, sp[(s, 1e-3)] * s / xbin)
    # diagnostic: with xbin=2 the s=2 fringe mode sits at exactly 1 native
    # px, where the GH fit is most easily fooled — compare suppressions
    gh_products["binned s=2 lf=0.01"] = (
        fine_grid_native(nbin, 2, xbin), sp[(2, 1e-2)] * 2 / xbin)
    gh_results = gauss_hermite_analysis(truth, gh_products)
    gh_figure(gh_results, gh_products,
              f"scratch/experiment_binx2_crires{tag}_gh.png")
    for (x0, name), r in gh_results.items():
        if r is not None:
            key = f"gh_{x0:g}_{name.replace(' ', '_').replace('=', '')}"
            npz[key] = np.vstack(r)

    out_npz = f"scratch/experiment_binx2_crires{tag}.npz"
    np.savez(out_npz, **npz)
    print(f"\nsaved {out_npz}")
    if os.environ.get("SHOW", "1") != "0":
        plt.show()


if __name__ == "__main__":
    main()
