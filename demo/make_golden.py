"""Generate golden reference data for the wasm demo correctness gate.

Builds synthetic swaths, runs charslit.slitdec with inputs constructed
exactly like demo_api.c constructs them (constant ycen, uniform
[0,c1,c2,0,0,0] slitcurve, zero slitdeltas, mask from isfinite, pix_unc
of ones), and dumps inputs + outputs to demo/golden.json.

Run with: uv run python demo/make_golden.py
"""

import json
import math
from pathlib import Path

import numpy as np

import charslit


def make_swath(ncols, nrows, c1, c2, seed, add_outliers=False, add_nans=False):
    rng = np.random.default_rng(seed)

    x = np.arange(ncols)
    spec = 200.0 + 50.0 * np.sin(x / 37.0)
    for center, amp, width in [(60, 900, 2.5), (150, 1500, 3.0), (230, 600, 2.0)]:
        spec = spec + amp * np.exp(-0.5 * ((x - center) / width) ** 2)

    y = np.arange(nrows)
    yc = nrows / 2.0
    profile = np.exp(-0.5 * ((y - yc) / (nrows / 6.0)) ** 2)
    profile /= profile.sum()

    # each row of the slit is shifted horizontally by the curvature polynomial
    im = np.empty((nrows, ncols))
    for iy in range(nrows):
        delta = c1 * iy + c2 * iy * iy
        im[iy] = np.interp(x - delta, x, spec) * profile[iy] * nrows

    im += rng.normal(0.0, np.sqrt(np.abs(im) + 1.0))

    if add_outliers:
        im[nrows // 3, ncols // 4] += 5000.0
        im[2 * nrows // 3, 3 * ncols // 4] -= 4000.0
    if add_nans:
        im[nrows // 2, 10] = np.nan
        im[5, ncols // 2] = np.nan

    return im


def run_case(name, im, ycen_frac, c1, c2, osample, lambda_sP, lambda_sL,
             maxiter, kappa):
    nrows, ncols = im.shape
    ny = osample * (nrows + 1) + 1

    ycen = np.full(ncols, ycen_frac)
    slitcurve = np.zeros((ncols, 6))
    slitcurve[:, 1] = c1
    slitcurve[:, 2] = c2
    slitdeltas = np.zeros(ny)
    mask = np.isfinite(im).astype(np.uint8)
    pix_unc = np.ones_like(im)

    res = charslit.slitdec(
        np.ascontiguousarray(im), pix_unc, mask, ycen, slitcurve, slitdeltas,
        osample=osample, lambda_sP=lambda_sP, lambda_sL=lambda_sL,
        maxiter=maxiter, kappa=kappa,
    )

    def tolist(a):
        # JSON has no NaN; encode as null, Node side maps null -> NaN
        return [None if not math.isfinite(v) else v for v in np.asarray(a).ravel()]

    print(f"{name}: status={res['info'][2]:.0f} iter={res['info'][3]:.0f} "
          f"return_code={res['return_code']}")

    return {
        "name": name,
        "ncols": ncols, "nrows": nrows,
        "ycen_frac": ycen_frac, "c1": c1, "c2": c2,
        "osample": osample, "lambda_sP": lambda_sP, "lambda_sL": lambda_sL,
        "maxiter": maxiter, "kappa": kappa,
        "im": tolist(im),
        "spectrum": tolist(res["spectrum"]),
        "slitfunction": tolist(res["slitfunction"]),
        "model": tolist(res["model"]),
        "uncertainty": tolist(res["uncertainty"]),
        "mask": np.asarray(res["mask"]).ravel().tolist(),
        "info": tolist(res["info"]),
        "return_code": res["return_code"],
    }


def main():
    cases = [
        run_case("flat",
                 make_swath(300, 40, 0.0, 0.0, seed=42),
                 ycen_frac=0.5, c1=0.0, c2=0.0, osample=6,
                 lambda_sP=0.0, lambda_sL=1.0, maxiter=20, kappa=10.0),
        run_case("curved",
                 make_swath(300, 40, -0.03, 0.0015, seed=43,
                            add_outliers=True, add_nans=True),
                 ycen_frac=0.5, c1=-0.03, c2=0.0015, osample=6,
                 lambda_sP=0.0, lambda_sL=1.0, maxiter=20, kappa=10.0),
        run_case("smoothed",
                 make_swath(200, 30, 0.01, -0.0005, seed=44),
                 ycen_frac=0.25, c1=0.01, c2=-0.0005, osample=4,
                 lambda_sP=1.0, lambda_sL=10.0, maxiter=20, kappa=0.0),
    ]

    out = Path(__file__).parent / "golden.json"
    with open(out, "w") as f:
        json.dump({"cases": cases}, f)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
