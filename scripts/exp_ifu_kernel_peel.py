"""IFU crosstalk: the DRL Fig 14 peel built from per-fibre slitdec models (slit curve + slit kernels).

Same frames and leak metric as exp_ifu_crosstalk.py. Per fibre, on a row crop centred on it:
  calibration  slit function from the fibre's masked flat; kernels from selfcal_u on an
               FP-like frame through that fibre alone; slit curve = the true tilt
  extraction   slitdec with that slit function, kernels and slit curve fixed (sP only)
  peel         pass 1 per fibre, subtract the neighbours' models, re-extract (pass 2);
               further passes iterate towards the joint solution

uv run --python 3.13 scripts/exp_ifu_kernel_peel.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import charslit  # noqa: E402
import exp_ifu_crosstalk as E  # noqa: E402
import slitkernel as sk  # noqa: E402

NFIB = 12
HALF = 4  # crop half-height in rows
KW = dict(osample=6, lambda_sP=0.0, lambda_sL=0.1, maxiter=20, kappa=0)
FP = np.zeros(E.NC)
FP[np.arange(10, E.NC - 10, 9)] = 3e5


def crop_of(yc, nrows):
    r0 = int(np.floor(yc)) - HALF
    return r0, yc - np.floor(yc)


def fibre_centres(geom, nrows):
    y0 = (nrows - (NFIB - 1) * geom["pitch"]) / 2
    return y0 + np.arange(NFIB) * geom["pitch"]


def calibrate(T, geom, nrows, use_kernels):
    sc = np.zeros((E.NC, 2))
    sc[:, 1] = geom["tilt"]
    cal = []
    for f, yc in enumerate(fibre_centres(geom, nrows)):
        r0, frac = crop_of(yc, nrows)
        rows = slice(r0, r0 + 2 * HALF)
        ycen = np.full(E.NC, frac)
        sd = np.zeros(2 * HALF)
        fp = E.image_of(T[f], FP + 2e3)[rows]
        if use_kernels:
            # the fibre wings are what overlaps the neighbours, so fit kernels there too
            c = sk.selfcal_u(fp, ycen, sc, sd, nk=9, niter=8, rn=E.RN, row_weight_min=0.02, **KW)
            kern, sL = c["kernel"], c["result"]["slitfunction"]
        else:
            kern = None
            sL = charslit.slitdec(fp, np.sqrt(np.abs(fp) + E.RN**2), np.ones(fp.shape, np.uint8), ycen.copy(), sc, sd,
                                  **KW)["slitfunction"]
        cal.append(dict(rows=rows, ycen=ycen, sc=sc, sd=sd, kern=kern, sL=sL))
    return cal


def extract(im, cal, sub=None):
    out, models = [], []
    for f, c in enumerate(cal):
        d = im[c["rows"]] - (0 if sub is None else sub[f][c["rows"]])
        r = charslit.slitdec(d, np.sqrt(np.abs(im[c["rows"]]) + E.RN**2), np.ones(d.shape, np.uint8), c["ycen"].copy(),
                             c["sc"], c["sd"], preset_slitfunc=c["sL"], slitkernel=c["kern"], **KW)
        out.append(r["spectrum"])
        m = np.zeros_like(im)
        m[c["rows"]] = r["model"]
        models.append(m)
    return np.array(out), np.array(models)


def peel(im, cal, npass):
    res = {}
    s, M = extract(im, cal)
    res[1] = s
    for p in range(2, npass + 1):
        tot = M.sum(0)
        s, M = extract(im, cal, sub=tot[None] - M)
        res[p] = s
    return res


def main():
    geom, s0 = dict(pitch=2.5, diam=2.5, tilt=0.05), 0.6
    print("YJH-like: fibres 2.5 px at 2.5 px pitch, PSF sigma 0.6, tilt 0.05; peak / flux leak in % of neighbour line")
    for sname, (truth, _) in E.scenarios(s0).items():
        T, nrows = E.templates(geom, truth, NFIB)
        S = E.spectra(NFIB)
        im = sum(E.image_of(T[f], S[f]) for f in range(NFIB))
        print(f"\n=== {sname}")
        for lab, kern in (("slit curve", False), ("slit curve + kernels", True)):
            cal = calibrate(T, geom, nrows, kern)
            res = peel(im, cal, 6)
            for p in (1, 2, 6):
                pk, fl = E.leak_metrics(res[p], NFIB)
                name = {1: "pass 1 (single)", 2: "pass 2 (peel)", 6: "pass 6 (~joint)"}[p]
                print(f"  {lab:22s} {name:16s} {pk * 100:7.3f}% {fl * 100:+7.3f}%")


if __name__ == "__main__":
    main()
