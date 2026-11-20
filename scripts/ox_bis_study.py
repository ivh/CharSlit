"""Wide (ESPRESSO-like) or baseline LSF: BIS bias, rms_cont, FWHM vs (osample_x, penalty),
over sub-pixel line phases and noise seeds (synthetic line of experiment_synthetic_lsf.py).
Prints the table and saves scratch/ox_bis_study_<case>[_amp<amp>].png (k = 3 penalties).
Usage: ox_bis_study.py [wide|baseline] [amp]"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib; matplotlib.use("Agg")
import experiment_synthetic_lsf as E
import analyse_wide_bisector as B
import charslit
case = sys.argv[1] if len(sys.argv) > 1 else "wide"
amp = float(sys.argv[2]) if len(sys.argv) > 2 else 4000.0
tw, sg = dict(wide=(1.6, 0.5), baseline=(0.8, 0.25))[case]
nrows, ncols, OS, win = 80, 60, 6, 5.0
u, lsf = E.make_lsf(tophat_w=tw, sigma=sg)
c1, c2, jc = E.tilt_coeffs(nrows, 0.1, 0.001)
sc = np.zeros((ncols, 3)); sc[:, 1] = c1; sc[:, 2] = c2
ny = OS * (nrows + 1) + 1
levels = np.linspace(0.10, 0.95, 35)
PH = np.arange(8) / 8 + 0.045; SEEDS = range(6)
k6 = lambda m: [0.0] * 5 + [m]; k3 = lambda m: [0.0, 0.0, m]
MU3 = [1e-7, 1e-6, 8.2e-6, 3e-5, 1.64e-4, 1e-3]
PEN = [("none", [])] + [(f"mu3={m:g}", k3(m)) for m in MU3] + [("mu6=1e-10", k6(1e-10)), ("mu6=1.9e-8", k6(1.9e-8))]
CONF = [(1, "linear", [], dict())] + [(s, n, m, dict()) for s in (2, 3, 4, 8) for n, m in PEN]
images = {}
for ip, ph in enumerate(PH):
    x0 = ncols / 2 + ph
    xtw, tn = B.area_normalise(u + x0, lsf, x0, win)
    bt = B.bis_value(levels, B.bisector(xtw, tn, levels) - x0)
    xtw2, tn2 = E.norm_window(u + x0, lsf, x0, win)
    fwhm_true = E.lsf_metrics(xtw2, tn2, x0)["fwhm"]
    for sd in SEEDS:
        im, unc, _, _ = E.make_image(nrows, ncols, x0, 0.1, 0.001, u, lsf, amp=amp, noise=True, seed=100 * ip + sd)
        images[(ip, sd)] = (x0, im, unc, bt, xtw2, tn2)
print(f"{case} LSF, amp {amp:g}, truth BIS {bt*1e3:+.1f} mpx, {len(PH)} phases x {len(SEEDS)} seeds")
print(f"{'s':>2} {'penalty':13} {'BIS bias':>9} {'phase sd':>9} {'noise sd':>9} {'rms_cont':>9} {'FWHM':>6} {'cen sd':>7} {'iter':>5} {'maxed':>6}")
rows = {}
for s, name, mu, kw in CONF:
    bias = np.full((len(PH), len(SEEDS)), np.nan); rc = bias.copy(); fw = bias.copy(); cen = bias.copy(); it = bias.copy()
    for (ip, sd), (x0, im, unc, bt, xtw2, tn2) in images.items():
        r = charslit.slitdec(im, unc, np.ones(im.shape, np.uint8), np.full(ncols, 0.5), sc, np.zeros(ny),
                             osample=OS, osample_x=s, mu_x=mu, kappa=0.0, **kw)
        sp = r["spectrum_fine"]; xf = (np.arange(sp.size) + 0.5) / s
        xw, yw = B.area_normalise(xf, sp, x0, win)
        bias[ip, sd] = B.bis_value(levels, B.bisector(xw, yw, levels) - x0) - bt
        xw2, yw2 = E.norm_window(xf, sp, x0, win); m = E.lsf_metrics(xw2, yw2, x0)
        rc[ip, sd] = E.rms_vs_continuous(xw2, yw2, s, xtw2, tn2); fw[ip, sd] = m["fwhm"]; cen[ip, sd] = m["centroid_err"]; it[ip, sd] = r["info"][3]
    rows[(s, name)] = (np.nanstd(np.nanmean(bias, 1)) * 1e3, np.sqrt(np.nanmean(np.nanvar(bias, 1))) * 1e3,
                       np.nanmean(fw), np.nanmean(it > 20))
    print(f"{s:>2} {name:13} {np.nanmean(bias)*1e3:>+9.1f} {np.nanstd(np.nanmean(bias, 1))*1e3:>9.1f} "
          f"{np.sqrt(np.nanmean(np.nanvar(bias, 1)))*1e3:>9.1f} {np.nanmean(rc):>9.4f} {np.nanmean(fw):>6.3f} "
          f"{np.nanstd(np.nanmean(cen, 1))*1e3:>7.2f} {np.nanmean(it):>5.1f} {np.mean(it > 20):>6.2f}")

# ---- figure: one panel per quantity, x = mu3, one line per osample_x; master as reference
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SER = {2: ("#2a78d6", "o"), 3: ("#eb6834", "s"), 8: ("#1baf7a", "D")}
lin = rows[(1, "linear")]
panels = [(0, "Position-dependent BIS error [mpx]", True, None),
          (1, "BIS noise, one exposure [mpx]", True, None),
          (2, "Recovered FWHM [px]", False, fwhm_true)]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), facecolor=SURF)
for ax, (q, title, logy, truth) in zip(axes, panels):
    ax.set_facecolor(SURF)
    for sv, (col, mk) in SER.items():
        y = [rows[(sv, f"mu3={m:g}")][q] for m in MU3]
        ax.plot(MU3, y, color=col, lw=2, marker=mk, ms=6, mec=SURF, mew=1.2, label=f"osample_x = {sv}", zorder=3)
    ax.axhline(lin[q], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.annotate("master (linear split)", (MU3[0], lin[q]), xytext=(0, 4), textcoords="offset points",
                color=MUTED, fontsize=9, va="bottom")
    if truth is not None:
        ax.axhline(truth, color=INK, lw=1.0, zorder=2)
        ax.annotate("truth", (MU3[-1], truth), xytext=(0, 3), textcoords="offset points",
                    color=INK, fontsize=9, va="bottom", ha="right")
    ax.axvline(1.64e-4, color=GRID, lw=6, zorder=1)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("mu3 (penalty on the 3rd derivative)", color=MUTED, fontsize=9)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
for ax in axes[:2]:
    ax.annotate("default", (1.64e-4, 0), xycoords=("data", "axes fraction"), xytext=(-5, 3),
                textcoords="offset points", ha="right", va="bottom", fontsize=9, color=MUTED)
axes[1].legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left", bbox_to_anchor=(0.0, 0.08))
fig.suptitle(f"{case} LSF (true FWHM {fwhm_true:.2f} px), peak amplitude {amp:g}: "
             f"{len(PH)} line positions x {len(SEEDS)} noise seeds", x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = f"scratch/ox_bis_study_{case}" + ("" if amp == 4000 else f"_amp{amp:g}") + ".png"
fig.savefig(out, dpi=140, facecolor=SURF)
print("saved", out)
