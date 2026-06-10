"""Benchmark slitdec on the real data files. Usage: uv run python bench.py [nrepeat]"""

import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

import charslit

NREPEAT = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def load_case(fits_path):
    fits_path = Path(fits_path)
    with fits.open(fits_path) as hdul:
        im = hdul[0].data.astype(np.float64)
    if im.ndim != 2:
        return None
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
                osample=6, lambda_sP=case["lambda_sP"], lambda_sL=case["lambda_sL"],
            )
            times.append(time.perf_counter() - t0)
        best = min(times)
        total += best
        niter = int(result["info"][3])
        print(f"{f.name:25s} {case['im'].shape!s:12s} best={best*1000:9.2f} ms  "
              f"iters={niter:3d}  per-iter={best*1000/max(niter,1):8.2f} ms")
    print(f"{'TOTAL (best-of runs)':25s} {'':12s} {total*1000:14.2f} ms")
