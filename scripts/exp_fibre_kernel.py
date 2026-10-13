"""Does a per-row slit kernel make slitdec's extracted LSF independent of slit illumination?

Synthetic fibre pseudo-slit (fibreslit_sim), FP-like unresolved lines on a weak
continuum. Calibrate on a uniformly illuminated frame, then extract frames with other
fibre illuminations using those fixed calibrations, and compare extracted line
centroids / widths with the uniform-illumination extraction. Any difference is an
extraction-induced, illumination-dependent bias, i.e. what separates a science
extraction from its calibration.

Calibration variants:
  nominal       true tilt only (slitcurve), slitdeltas = 0, no kernel
  deltas        + per-row slitdeltas self-calibrated from line centroids (slitkernel.selfcal, k1 only)
  deltas+kern   + nonparametric per-row kernels (slitkernel.selfcal_lsq)

uv run --python 3.13 scripts/exp_fibre_kernel.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import charslit  # noqa: E402
import fibreslit_sim as fs  # noqa: E402
import slitkernel as sk  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

NCOLS, NROWS, NFIB = 400, 120, 18
TILT = 0.05
RN = 10.0
KW = dict(osample=6, lambda_sP=0.0, lambda_sL=0.1, maxiter=30, kappa=0)
PX_MS = 750.0  # m/s per px, ANDES-R-like (R ~ 100k, ~4 px per resolution element)


def scenario(name):
    i = np.linspace(0, 1, NFIB)
    return {
        "geometry": {},
        "psf_width": dict(psf_sx=0.6 + 0.4 * i),
        "psf_skew": dict(psf_sx=0.6 + 0.4 * i, skew_eps=0.15 * i, skew_d=1.5),
        "psf_skew_mild": dict(skew_eps=0.03 * i, skew_d=1.0),
    }[name]


def illuminations():
    i = np.linspace(-1, 1, NFIB)
    return {
        "uniform": np.ones(NFIB),
        "gradient": 1 + 0.4 * i,
        "lower_half": np.where(i < 0, 1.0, 0.15),
        "ripple": 1 + 0.3 * np.cos(np.arange(NFIB) * np.pi),
    }


def extract(im, sd, kern):
    mask = np.ones(im.shape, np.uint8)
    unc = np.sqrt(np.abs(im) + RN**2)
    sc = np.zeros((NCOLS, 2))
    sc[:, 1] = TILT
    r = charslit.slitdec(im, unc, mask, np.zeros(NCOLS), sc, sd, slitkernel=kern, **KW)
    chi2 = (((im - r["model"]) / unc) ** 2)[:, 20:-20].mean()
    return r, chi2


def main(scenarios=("geometry", "psf_width", "psf_skew", "psf_skew_mild"), peak=None, seed=0):
    """peak: if given, scale frames to this peak count and add Poisson noise."""
    rng = np.random.default_rng(seed)
    lines = fs.fp_lines(NCOLS, spacing=20.0, amp=3e6)
    print(f"ncols {NCOLS} nrows {NROWS} nfib {NFIB} tilt {TILT} px/px, {len(lines)} lines, noiseless + RN {RN} floor")
    print(f"centroid shifts also in m/s at {PX_MS:.0f} m/s/px")
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
        sc = np.zeros((NCOLS, 2))
        sc[:, 1] = TILT
        uni = frames["uniform"]
        c1 = sk.selfcal(uni, np.zeros(NCOLS), sc, np.zeros(NROWS), niter=6, fit_k2=False, fit_k3=False, **KW)
        sd = sk.rows_to_subpix(c1["slitdeltas"], KW["osample"], 0.0)
        ck = sk.selfcal_lsq(uni, np.zeros(NCOLS), sc, sd, niter=6, rn=RN, **KW)
        variants = {
            "nominal": (np.zeros(NROWS), None),
            "deltas": (sd, None),
            "deltas+kern": (sd, ck["kernel"]),
        }
        peaks, half = sk.line_windows(extract(uni, *variants["nominal"])[0]["spectrum"])
        print(f"\n=== {scen}   (kernel selfcal chi2: {[round(x) for x in ck['chi2']]})")
        print(f"  {'variant':12s} {'illum.':11s} {'chi2/pix':>8s} {'dk1 [px]':>10s} {'[m/s]':>7s} {'dk2 [px2]':>10s} {'dk3 [px3]':>10s}")
        for vname, (sdv, kern) in variants.items():
            ref = None
            for ill, im in frames.items():
                r, chi2 = extract(im, sdv, kern)
                m = sk.row_moments(r["spectrum"][None, :], peaks, half)[:, 0]
                if ref is None:
                    ref = m
                d = np.nanmean(m - ref, axis=1)
                print(f"  {vname:12s} {ill:11s} {chi2:8.0f} {d[0]:+10.5f} {d[0] * PX_MS:+7.1f} {d[1]:+10.4f} {d[2]:+10.4f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--peak", type=float, default=None, help="peak counts; adds Poisson + read noise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenarios", nargs="*", default=["geometry", "psf_width", "psf_skew", "psf_skew_mild"])
    a = ap.parse_args()
    main(a.scenarios, a.peak, a.seed)
