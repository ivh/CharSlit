"""Synthetic pseudo-slit images: round fibres, per-fibre PSF, tilted slit, line spectrum.

Rendered on a fine grid and binned to pixels, independently of slitdec's own forward
model, so slitdec's separability assumption can be tested against it.
"""

import numpy as np
from scipy.signal import fftconvolve

FINE = 10  # fine cells per pixel


def _psf_x(xf, sig, skew_eps, skew_d):
    """1D dispersion-direction PSF on fine coords: Gaussian plus a small offset Gaussian."""
    g = (1 - skew_eps) * np.exp(-0.5 * (xf / sig) ** 2)
    if skew_eps:
        g = g + skew_eps * np.exp(-0.5 * ((xf - skew_d) / sig) ** 2)
    return g / g.sum()


def fibre_stamp(diam, xscale=1.0, shear=0.0, psf_sx=0.8, psf_sy=0.8, skew_eps=0.0, skew_d=0.0, half=10.0):
    """Monochromatic image of one fibre (uniform disk -> ellipse -> PSF), fine grid, unit sum.

    Returns (stamp, half) with the stamp centred on (half, half) px."""
    n = int(2 * half * FINE)
    c = (np.arange(n) + 0.5) / FINE - half
    X, Y = np.meshgrid(c, c)
    u = (X - shear * Y) / xscale
    disk = (u**2 + Y**2) <= (diam / 2) ** 2
    px = _psf_x(c, psf_sx, skew_eps, skew_d)
    py = np.exp(-0.5 * (c / psf_sy) ** 2)
    py /= py.sum()
    s = fftconvolve(disk.astype(float), px[None, :], mode="same")
    s = fftconvolve(s, py[:, None], mode="same")
    s[s < 0] = 0
    return s / s.sum()


def make_fibres(nfib=18, pitch=5.5, diam=5.4, y0=None, nrows=None, **per_fibre):
    """List of fibre dicts. per_fibre values may be scalars or arrays of length nfib."""
    if y0 is None:
        y0 = (nrows - (nfib - 1) * pitch) / 2
    fib = []
    for i in range(nfib):
        d = dict(yc=y0 + i * pitch, diam=diam, flux=1.0)
        for k, v in per_fibre.items():
            d[k] = v[i] if np.ndim(v) else v
        fib.append(d)
    return fib


def render(ncols, nrows, fibres, lines, tilt=0.0, ycentre=None, continuum=0.0, half=10.0):
    """Image (nrows, ncols): sum over fibres and lines of shifted fibre stamps.

    lines: iterable of (x_position_px, amplitude). tilt: x shift per px of y offset
    from ycentre, applied to the whole fine image (slit tilt). continuum: flux per px
    of dispersion, spread over the slit like the lines."""
    if ycentre is None:
        ycentre = nrows / 2
    H, W = nrows * FINE, ncols * FINE
    pad = int(half * FINE) + 2
    img = np.zeros((H + 2 * pad, W + 2 * pad))
    for f in fibres:
        kw = {k: f[k] for k in ("xscale", "shear", "psf_sx", "psf_sy", "skew_eps", "skew_d") if k in f}
        st = fibre_stamp(f["diam"], half=half, **kw) * f["flux"]
        n = st.shape[0]
        oy = int(round(f["yc"] * FINE)) - n // 2 + pad
        if continuum:
            prof = st.sum(1) * continuum
            img[oy : oy + n, pad : pad + W] += prof[:, None] / FINE
        for x, a in lines:
            ox = int(round(x * FINE)) - n // 2 + pad
            if ox < 0 or ox + n > img.shape[1]:
                continue
            img[oy : oy + n, ox : ox + n] += a * st
    img = img[pad : pad + H, pad : pad + W]
    if tilt:
        yf = (np.arange(H) + 0.5) / FINE - ycentre
        shifts = np.round(tilt * yf * FINE).astype(int)
        out = np.zeros_like(img)
        for r, s in enumerate(shifts):
            if s >= 0:
                out[r, s:] = img[r, : W - s]
            else:
                out[r, :s] = img[r, -s:]
        img = out
    return img.reshape(nrows, FINE, ncols, FINE).sum(axis=(1, 3))


def fp_lines(ncols, spacing=20.0, amp=1000.0, offset=7.3, margin=12):
    xs = np.arange(margin + offset, ncols - margin, spacing)
    return [(x, amp) for x in xs]
