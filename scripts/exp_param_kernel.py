"""Parametric (shift, width, skew[, kurtosis]) slit kernels vs nonparametric taps.

Same synthetic fibre pseudo-slit, scenarios and illuminations as exp_fibre_kernel.py.
Fixed calibrations are fitted on the uniformly illuminated frame and applied unchanged
to the others; per-frame variants refit on each frame (selfcal_param, starting from the
nominal geometry, or from the uniform calibration as prior). The change of extracted
line centroid / width against the uniform frame is the illumination-dependent bias.

Variants:
  kern_u          selfcal slitdeltas (k1) + selfcal_u taps, fixed (the current method)
  par3 / par4     selfcal_param, moments (1,2,3) / (1,2,3,4), fixed
  frame3_uni      selfcal_param (1,2,3) refitted on each frame; lit nodes, uniform gauge and
                  k2_offset from the uniform-frame calibration
  frame3_flux     the same with this frame's flux-weighted gauge (expected: plain-slitdec bias back)
  frame3_prior    uniform gauge, ridge towards the uniform-frame calibration (ridge 1e-2)

uv run --python 3.13 scripts/exp_param_kernel.py [--peak 1e5]
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fibreslit_sim as fs  # noqa: E402
import slitkernel as sk  # noqa: E402
from exp_fibre_kernel import KW, NCOLS, NFIB, NROWS, PX_MS, RN, TILT, YC, extract, illuminations, scenario  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)


def main(scenarios, peak=None, seed=0):
    rng = np.random.default_rng(seed)
    lines = fs.fp_lines(NCOLS, spacing=20.0, amp=3e6)
    sc = np.zeros((NCOLS, 2))
    sc[:, 1] = TILT
    yc = np.full(NCOLS, YC)
    print(f"ncols {NCOLS} nrows {NROWS} nfib {NFIB} tilt {TILT}, peak {peak or 'noiseless'}; m/s at {PX_MS:.0f} m/s/px")
    for scen in scenarios:
        per = scenario(scen)
        frames = {}
        for ill, flux in illuminations().items():
            fib = fs.make_fibres(NFIB, nrows=NROWS, **per)
            for f, fl in zip(fib, flux):
                f["flux"] = fl
            im = fs.render(NCOLS, NROWS, fib, lines, tilt=TILT, ycentre=NROWS / 2, continuum=2e3)
            if peak:
                im = im * peak / 1.3e5
                im = rng.poisson(im).astype(float) + rng.normal(0, RN, im.shape)
            frames[ill] = im
        uni = frames["uniform"]
        t0 = time.time()
        c1 = sk.selfcal(uni, yc, sc, np.zeros(NROWS), niter=6, **KW)
        sd = sk.rows_to_subpix(c1["slitdeltas"], KW["osample"], YC)
        cu = sk.selfcal_u(uni, yc, sc, sd, niter=6, rn=RN, **KW)
        t_u = time.time() - t0
        t0 = time.time()
        p3 = sk.selfcal_param(uni, yc, sc, np.zeros(NROWS), rn=RN, **KW)
        t_p = time.time() - t0
        p4 = sk.selfcal_param(uni, yc, sc, np.zeros(NROWS), rn=RN, moments=(1, 2, 3, 4), **KW)
        fixed = {
            "kern_u": (sd, cu["kernel"]),
            "par3": (p3["slitdeltas"], p3["kernel"]),
            "par4": (p4["slitdeltas"], p4["kernel"]),
        }
        perframe = {
            "frame3_uni": dict(lit=p3["lit"], gauge=p3["gauge"], k2_offset=p3["k2_offset"]),
            "frame3_flux": dict(lit=p3["lit"], gauge="flux", k2_offset=p3["k2_offset"]),
            "frame3_prior": dict(lit=p3["lit"], gauge=p3["gauge"], k2_offset=p3["k2_offset"], prior=p3["cumulants"],
                                 ridge=1e-2),
        }
        peaks, half = sk.line_windows(extract(uni, np.zeros(NROWS), None)[0]["spectrum"])
        print(f"\n=== {scen}   calibration chi2: kern_u {cu['chi2'][0]:.0f} -> {min(cu['chi2']):.1f} ({t_u:.1f} s), "
              f"par3 {min(p3['chi2']):.1f} ({t_p:.1f} s), par4 {min(p4['chi2']):.1f}")
        print(f"  {'variant':13s} {'illum.':11s} {'chi2/pix':>8s} {'dk1 [px]':>10s} {'[m/s]':>7s} {'dk2 [px2]':>10s}"
              f" {'dk3 [px3]':>10s}")
        for vname in ["plain", *fixed, *perframe]:
            ref = None
            for ill, im in frames.items():
                if vname == "plain":
                    r, chi2 = extract(im, np.zeros(NROWS), None)
                elif vname in fixed:
                    r, chi2 = extract(im, *fixed[vname])
                else:
                    c = sk.selfcal_param(im, yc, sc, np.zeros(NROWS), rn=RN, **{**KW, **perframe[vname]})
                    r, chi2 = extract(im, c["slitdeltas"], c["kernel"])
                m = sk.row_moments(r["spectrum"][None, :], peaks, half)[:, 0]
                if ref is None:
                    ref = m
                d = np.nanmean(m - ref, axis=1)
                print(f"  {vname:13s} {ill:11s} {chi2:8.1f} {d[0]:+10.5f} {d[0] * PX_MS:+7.2f} {d[1]:+10.4f}"
                      f" {d[2]:+10.4f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--peak", type=float, default=None, help="peak counts; adds Poisson + read noise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenarios", nargs="*", default=["geometry", "psf_width", "psf_skew", "psf_skew_mild"])
    a = ap.parse_args()
    main(a.scenarios, a.peak, a.seed)
