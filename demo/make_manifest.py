"""Build demo/swaths.json: the sample-swath dropdown manifest.

Scans data/*.fits for 2D images and pairs them with representative
uniform-curvature defaults (median of the per-column curvedelta fit,
fractional part of ycen). Files with no curvedelta npz get zeros.
The demo/data symlink must point at ../data so the static server can
reach the files.

Run with: uv run python demo/make_manifest.py
"""

import json
from pathlib import Path

import numpy as np
from astropy.io import fits

SKIP = {"CRIRES_UNE_J"}  # 200 MB full frame, not a swath

def first_2d_image(path):
    with fits.open(path) as hdus:
        for hdu in hdus:
            if hdu.data is not None and getattr(hdu.data, "ndim", 0) == 2:
                return hdu.data.shape
    return None

def main():
    root = Path(__file__).parent.parent
    swaths = []
    for f in sorted(root.glob("data/*.fits")):
        if f.stem in SKIP:
            continue
        shape = first_2d_image(f)
        if shape is None:
            continue
        nrows, ncols = shape

        c1 = c2 = 0.0
        ycen_frac = 0.5
        npz = root / "data" / f"curvedelta_{f.stem}.npz"
        if npz.exists():
            d = np.load(npz, allow_pickle=True)
            sc = d["slitcurve"]
            c1 = float(np.median(sc[:, 1]))
            c2 = float(np.median(sc[:, 2])) if sc.shape[1] > 2 else 0.0
            ycen_frac = float(np.median(d["ycen"] % 1.0))

        swaths.append({
            "name": f.stem,
            "url": f"data/{f.name}",
            "ncols": ncols, "nrows": nrows,
            "c1": round(c1, 5), "c2": round(c2, 6),
            "ycen_frac": round(ycen_frac, 3),
            "fitted": npz.exists(),
        })
        print(f"{f.stem}: {ncols} x {nrows}  c1={c1:.5f} c2={c2:.6f} "
              f"ycen_frac={ycen_frac:.3f}{'' if npz.exists() else '  (no fit, zeros)'}")

    out = Path(__file__).parent / "swaths.json"
    json.dump({"swaths": swaths}, open(out, "w"), indent=1)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
