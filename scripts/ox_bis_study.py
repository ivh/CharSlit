"""Wide (ESPRESSO-like) or baseline LSF: BIS bias, rms_cont, FWHM vs (osample_x, penalty),
over sub-pixel line phases and noise seeds (synthetic line of experiment_synthetic_lsf.py).
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
PEN = [("none", []), ("mu6=1e-11", k6(1e-11)), ("mu6=1e-10", k6(1e-10)), ("mu6=7.5e-10", k6(7.5e-10)),
       ("mu6=1.9e-8", k6(1.9e-8)), ("mu3=1e-6", k3(1e-6)), ("mu3=8.2e-6", k3(8.2e-6)), ("mu3=1.64e-4", k3(1.64e-4))]
CONF = [(1, "linear", [], dict(t_offset=0.0))] + [(s, n, m, {}) for s in (2, 3, 4, 8) for n, m in PEN]
images = {}
for ip, ph in enumerate(PH):
    x0 = ncols / 2 + ph
    xtw, tn = B.area_normalise(u + x0, lsf, x0, win)
    bt = B.bis_value(levels, B.bisector(xtw, tn, levels) - x0)
    xtw2, tn2 = E.norm_window(u + x0, lsf, x0, win)
    for sd in SEEDS:
        im, unc, _, _ = E.make_image(nrows, ncols, x0, 0.1, 0.001, u, lsf, amp=amp, noise=True, seed=100 * ip + sd)
        images[(ip, sd)] = (x0, im, unc, bt, xtw2, tn2)
print(f"{case} LSF, amp {amp:g}, truth BIS {bt*1e3:+.1f} mpx, {len(PH)} phases x {len(SEEDS)} seeds")
print(f"{'s':>2} {'penalty':13} {'BIS bias':>9} {'phase sd':>9} {'noise sd':>9} {'rms_cont':>9} {'FWHM':>6} {'cen sd':>7} {'iter':>5} {'maxed':>6}")
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
    print(f"{s:>2} {name:13} {np.nanmean(bias)*1e3:>+9.1f} {np.nanstd(np.nanmean(bias, 1))*1e3:>9.1f} "
          f"{np.sqrt(np.nanmean(np.nanvar(bias, 1)))*1e3:>9.1f} {np.nanmean(rc):>9.4f} {np.nanmean(fw):>6.3f} "
          f"{np.nanstd(np.nanmean(cen, 1))*1e3:>7.2f} {np.nanmean(it):>5.1f} {np.mean(it > 20):>6.2f}")
