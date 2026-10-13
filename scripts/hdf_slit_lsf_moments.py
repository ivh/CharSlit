# /// script
# requires-python = ">=3.11"
# dependencies = ["h5py", "numpy", "scipy", "matplotlib"]
# ///
"""Per-row dispersion-direction LSF moments along the pseudo-slit, from a PyEchelle HDF.

Renders the monochromatic image of the whole slit (every fibre: uniform disk under its
affine transformation, convolved with its PSF) on a fine grid and measures, per fine
row, the centroid (k1), variance (k2) and third central moment (k3) along x.

k1(y) is what slitcurve + slitdeltas absorb. Variation of k2/k3 along the slit is what
slitdec's single shared LSF cannot represent.

Two PSF modes:
  perfibre  each fibre gets its own PSF from the HDF (the optical model)
  fibre1    every fibre gets fibre 1's PSF (what pyechelle <= 0.4.0 simulates: its PSF
            cache is keyed by (ccd, order) only, so the first fibre loaded wins)

The PSF axis convention is not settled: pyechelle's raytracer maps PSF array axis 0 to
detector x, while Zemax Huygens output is conventionally (row=y, col=x). --psf-x-axis
picks which array axis is treated as dispersion.

Usage:
  uv run scripts/hdf_slit_lsf_moments.py HDF.hdf --order 90 --wl-index 7 [--psf-x-axis 0]
"""

import argparse

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

FINE = 0.1  # px per fine-grid cell


def load(fn, order, wl_index):
    f = h5py.File(fn, "r")
    c = f["CCD_1"]
    pixelsize = float(c.attrs["pixelsize"])
    fibs = sorted(int(k.split("_")[1]) for k in c if k.startswith("fiber_"))
    t_ref = c[f"fiber_{fibs[0]}/order{order}"][:]
    wl = float(t_ref["wavelength"][wl_index])
    fibres = []
    for n in fibs:
        t = c[f"fiber_{n}/order{order}"][:]
        tf = {k: float(np.interp(wl, t["wavelength"], t[k])) for k in t.dtype.names if k != "wavelength"}
        g = c[f"fiber_{n}/psf_order_{order}"]
        wk = min(g.keys(), key=lambda k: abs(float(k.split("_")[1]) - wl))
        psf = np.array(g[wk], dtype=float)
        spacing = float(g[wk].attrs["dataSpacing"]) / pixelsize  # px per PSF sample
        fibres.append(dict(n=n, tf=tf, psf=psf / psf.sum(), spacing=spacing))
    f.close()
    return wl, fibres


def resample_psf(psf, spacing, x_axis):
    """PSF onto the FINE grid, returned as (y, x) array."""
    from scipy.ndimage import zoom

    p = psf if x_axis == 1 else psf.T
    z = zoom(p, spacing / FINE, order=1)
    z[z < 0] = 0
    return z / z.sum()


def disk_image(tf, ny_c, nx_c, y0, x0):
    """Uniform unit-diameter disk under the affine matrix, rasterised on the fine grid.

    Returns the image and the fine-grid origin (y0, x0) in detector px."""
    m0 = tf["scale_x"] * np.cos(tf["rotation"])
    m1 = -tf["scale_y"] * np.sin(tf["rotation"] + tf["shear"])
    m3 = tf["scale_x"] * np.sin(tf["rotation"])
    m4 = tf["scale_y"] * np.cos(tf["rotation"] + tf["shear"])
    M = np.array([[m0, m1], [m3, m4]])
    Minv = np.linalg.inv(M)
    sub = 4
    ys = y0 + (np.arange(ny_c * sub) + 0.5) * FINE / sub
    xs = x0 + (np.arange(nx_c * sub) + 0.5) * FINE / sub
    X, Y = np.meshgrid(xs - tf["translation_x"], ys - tf["translation_y"])
    u = Minv[0, 0] * X + Minv[0, 1] * Y
    v = Minv[1, 0] * X + Minv[1, 1] * Y
    # slit coordinates are in [0,1] with the disk centred on 0.5 (pyechelle 'circular')
    inside = ((u - 0.5) ** 2 + (v - 0.5) ** 2) <= 0.25
    img = inside.reshape(ny_c, sub, nx_c, sub).mean(axis=(1, 3))
    return img


def render(fibres, mode, x_axis):
    xc = np.array([f["tf"]["translation_x"] for f in fibres])
    yc = np.array([f["tf"]["translation_y"] for f in fibres])
    # translation is the image of slit coord (0,0); the disk centre is the image of (0.5,0.5)
    for f in fibres:
        t = f["tf"]
        m = np.array(
            [
                [t["scale_x"] * np.cos(t["rotation"]), -t["scale_y"] * np.sin(t["rotation"] + t["shear"])],
                [t["scale_x"] * np.sin(t["rotation"]), t["scale_y"] * np.cos(t["rotation"] + t["shear"])],
            ]
        )
        f["centre"] = m @ [0.5, 0.5] + [t["translation_x"], t["translation_y"]]
    cx = np.array([f["centre"][0] for f in fibres])
    cy = np.array([f["centre"][1] for f in fibres])
    pad = 30.0
    x0, x1 = cx.min() - pad, cx.max() + pad
    y0, y1 = cy.min() - pad, cy.max() + pad
    nx, ny = int((x1 - x0) / FINE), int((y1 - y0) / FINE)
    img = np.zeros((ny, nx))
    psf1 = resample_psf(fibres[0]["psf"], fibres[0]["spacing"], x_axis)
    half = 8.0  # px half-size of the disk stamp
    for f in fibres:
        psf = psf1 if mode == "fibre1" else resample_psf(f["psf"], f["spacing"], x_axis)
        sx0 = f["centre"][0] - half
        sy0 = f["centre"][1] - half
        # snap stamp origin to the global fine grid
        ix = int(round((sx0 - x0) / FINE))
        iy = int(round((sy0 - y0) / FINE))
        n = int(2 * half / FINE)
        d = disk_image(f["tf"], n, n, y0 + iy * FINE, x0 + ix * FINE)
        s = fftconvolve(d, psf, mode="full")
        oy = iy - psf.shape[0] // 2
        ox = ix - psf.shape[1] // 2
        img[oy : oy + s.shape[0], ox : ox + s.shape[1]] += s
    return img, x0, y0, (cy.min(), cy.max())


def row_moments(img, x0):
    x = x0 + (np.arange(img.shape[1]) + 0.5) * FINE
    w = img.sum(axis=1)
    ok = w > 1e-3 * w.max()
    k1 = np.full(len(w), np.nan)
    k2 = np.full(len(w), np.nan)
    k3 = np.full(len(w), np.nan)
    k1[ok] = (img[ok] * x).sum(1) / w[ok]
    d = x[None, :] - k1[ok, None]
    k2[ok] = (img[ok] * d**2).sum(1) / w[ok]
    k3[ok] = (img[ok] * d**3).sum(1) / w[ok]
    return w, k1, k2, k3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf")
    ap.add_argument("--order", type=int, default=90)
    ap.add_argument("--wl-index", type=int, default=7)
    ap.add_argument("--psf-x-axis", type=int, default=0, choices=[0, 1])
    ap.add_argument("--out", default="hdf_slit_lsf_moments")
    a = ap.parse_args()

    wl, fibres = load(a.hdf, a.order, a.wl_index)
    print(f"{a.hdf.split('/')[-1]}  order {a.order}  wl {wl:.5f} um  nfib {len(fibres)}  psf x-axis {a.psf_x_axis}")
    res = {}
    for mode in ["perfibre", "fibre1"]:
        img, x0, y0, (ylo, yhi) = render(fibres, mode, a.psf_x_axis)
        w, k1, k2, k3 = row_moments(img, x0)
        y = y0 + (np.arange(len(w)) + 0.5) * FINE
        inner = (y > ylo) & (y < yhi)  # between first and last fibre centre
        ww = np.where(inner, w, 0)
        m2 = np.nansum(ww * k2) / ww.sum()
        m3 = np.nansum(ww * k3) / ww.sum()
        d2 = k2[inner] - m2
        d3 = k3[inner] - m3
        res[mode] = dict(y=y, w=w, k1=k1, k2=k2, k3=k3, inner=inner, m2=m2, m3=m3)
        print(
            f"  {mode:8s}  <k2> = {m2:.4f} px^2 (sigma {np.sqrt(m2):.3f} px)  "
            f"dk2 rms {np.std(d2):.4f}  p2p {np.ptp(d2):.4f}  |  <k3> = {m3:+.4f}  dk3 rms {np.std(d3):.4f}  p2p {np.ptp(d3):.4f}"
        )
        # fibre-scale vs slit-scale: smooth over 2 fibre pitches
        pitch = np.median(np.diff(sorted(f["centre"][1] for f in fibres)))
        k = int(2 * pitch / FINE)
        sm = np.convolve(d2, np.ones(k) / k, mode="same")
        print(f"            dk2 slit-scale (smoothed over 2 pitches) p2p {np.ptp(sm[k:-k]):.4f}; fibre-scale rms {np.std((d2 - sm)[k:-k]):.4f}")

    fig, ax = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    for mode, ls in [("perfibre", "-"), ("fibre1", "--")]:
        r = res[mode]
        i = r["inner"]
        ax[0].plot(r["y"], r["w"], ls, label=mode)
        ax[1].plot(r["y"][i], r["k1"][i] - np.nanmedian(r["k1"][i]), ls, label=mode)
        ax[2].plot(r["y"][i], r["k2"][i] - r["m2"], ls, label=mode)
        ax[3].plot(r["y"][i], r["k3"][i] - r["m3"], ls, label=mode)
    ax[0].set_ylabel("row flux")
    ax[1].set_ylabel("k1 - median [px]")
    ax[2].set_ylabel("k2 - <k2> [px^2]")
    ax[3].set_ylabel("k3 - <k3> [px^3]")
    ax[3].set_xlabel("detector y [px]")
    ax[0].legend()
    ax[0].set_title(f"{a.hdf.split('/')[-1]}  order {a.order}  {wl:.5f} um  PSF x-axis={a.psf_x_axis}")
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=110)
    np.savez(
        a.out + ".npz",
        **{f"{m}_{k}": v for m in res for k, v in res[m].items() if isinstance(v, np.ndarray)},
        wl=wl,
    )
    print("wrote", a.out + ".png")


if __name__ == "__main__":
    main()
