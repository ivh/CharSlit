"""Benchmark slitdec on the real data files. Usage: uv run python bench.py [nrepeat] [--osamp_spec S]

This times ONLY the charslit.slitdec C call (each case is loaded once, then the
call is repeated best-of-N) — no FITS IO or golden comparison in the timed loop.

Speedup analysis (baseline 8b97f33 "regression harness added", pre-speedup,
vs latest after Round 1 pixel-centric/band-layout + Round 2 dense-window fills):

  File           shape        baseline   latest   speedup
  CRIRES1        176x2048      5852 ms    414 ms   14.1x
  CRIRES2        176x2048      5861 ms    419 ms   14.0x
  ANDES_R_FP1    404x2556      4988 ms    424 ms   11.8x
  Hsim           90x53          8.3 ms    1.2 ms    7.0x
  Rsim           140x84        20.5 ms    3.1 ms    6.6x
  discontinuous  100x150       10.8 ms    3.7 ms    2.9x
  multislope     100x150       12.9 ms    4.4 ms    2.9x
  fixedslope     100x150       10.7 ms    3.9 ms    2.8x
  CRIRES_UNE_J   2048x2048    29178 ms   9277 ms    3.1x
  TOTAL                         45.9 s    10.6 s    4.35x

The full extraction reaches ~14x on representative curved spectrograph frames
(176x2048, matching cr2res swaths) — meeting/exceeding the ~10x seen in the
cr2res C pipeline. The aggregate looks like only ~4x for two reasons unrelated
to the C code:

  1. The TOTAL is dominated (~88%) by the atypical 2048x2048 CRIRES_UNE_J frame,
     which only speeds up 3.1x. Per-frame speedup tracks curvature / delta_x:
     frames with a real curvedelta solution (wide p_Aij band, many zeta entries
     per pixel) are exactly where the old O(mz^2) unique-key search + column-major
     striding hurt most, so the pixel-centric + dense-window rewrite wins ~14x.
     CRIRES_UNE_J has no curvature file (delta_x~0, near-diagonal spectrum system),
     so its cost is the large-nrows slit-function solve, which only gained from
     the band layout -> 3.1x.
  2. The pytest golden run (48s -> 12s) additionally includes Python FITS loading,
     fixture prep, and the rtol-1e-10 npz comparison, which do not speed up.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

import charslit

parser = argparse.ArgumentParser(description="Benchmark slitdec on the real data files.")
parser.add_argument("nrepeat", nargs="?", type=int, default=3,
                    help="best-of-N repeats per case (default: 3)")
parser.add_argument("--osamp_spec", type=int, default=1,
                    help="dispersion oversampling factor (default: 1)")
_args = parser.parse_args()
NREPEAT = _args.nrepeat
OSAMP_SPEC = _args.osamp_spec


def load_case(fits_path):
    fits_path = Path(fits_path)
    with fits.open(fits_path) as hdul:
        # Primary HDU if it holds a 2D image, else first 2D ImageHDU
        # (multi-extension ESO/CRIRES files store data in CHIPn.INT1).
        im = None
        if hdul[0].data is not None and np.ndim(hdul[0].data) == 2:
            im = hdul[0].data
        else:
            for h in hdul:
                if getattr(h, "data", None) is not None and np.ndim(h.data) == 2:
                    im = h.data
                    break
        if im is None:
            return None
        im = im.astype(np.float64)
    nrows, ncols = im.shape

    slitcurve = np.zeros((ncols, 3))
    slitdeltas = np.zeros(nrows)
    ycen = np.full(ncols, nrows / 2.0)
    lambda_sP, lambda_sL = 0.0, 1.0

    npz = fits_path.parent / f"curvedelta_{fits_path.stem}.npz"
    if npz.exists():
        with np.load(npz) as data:
            if "slitcurve" in data:
                slitcurve = data["slitcurve"].astype(np.float64)
            if "slitdeltas" in data:
                slitdeltas = data["slitdeltas"].astype(np.float64)
            if "ycen" in data:
                ycen = data["ycen"].astype(np.float64)
            if "lambda_sP" in data:
                lambda_sP = float(data["lambda_sP"])
            if "lambda_sL" in data:
                lambda_sL = float(data["lambda_sL"])

    nan_mask = np.isnan(im)
    mask = np.ones(im.shape, dtype=np.uint8)
    mask[nan_mask] = 0
    im[nan_mask] = 0.0
    pix_unc = np.sqrt(np.abs(im) + 1.0)
    pix_unc[nan_mask] = 1e10

    return dict(im=im, pix_unc=pix_unc, mask=mask, ycen=ycen,
                slitcurve=slitcurve, slitdeltas=slitdeltas,
                lambda_sP=lambda_sP, lambda_sL=lambda_sL)


if __name__ == "__main__":
    total = 0.0
    for f in sorted(Path("data").glob("*.fits")):
        case = load_case(f)
        if case is None:
            continue
        times = []
        for _ in range(NREPEAT):
            t0 = time.perf_counter()
            result = charslit.slitdec(
                case["im"], case["pix_unc"], case["mask"], case["ycen"],
                case["slitcurve"], case["slitdeltas"],
                osample=6, osamp_spec=OSAMP_SPEC,
                lambda_sP=case["lambda_sP"], lambda_sL=case["lambda_sL"],
            )
            times.append(time.perf_counter() - t0)
        best = min(times)
        total += best
        niter = int(result["info"][3])
        print(f"{f.name:25s} {case['im'].shape!s:12s} best={best*1000:9.2f} ms  "
              f"iters={niter:3d}  per-iter={best*1000/max(niter,1):8.2f} ms")
    print(f"{'TOTAL (best-of runs)':25s} {'':12s} {total*1000:14.2f} ms  (osamp_spec={OSAMP_SPEC})")
