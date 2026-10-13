"""Helpers for slitdec's per-row slit kernels.

slitdec models every slit row as the same spectrum sP, shifted along the curved slit.
For a fibre-fed pseudo-slit the dispersion-direction profile is not the same in every
row: rows through a fibre centre see the full disk chord, rows at fibre junctions see
short chords, and the PSF itself can change along the slit. `slitkernel` lets each
(sub)row convolve sP with its own small kernel; this module builds those kernels from
per-row cumulants and measures the cumulants a model is missing (data - model) on a
frame with isolated lines (FP/LFC), so they can be self-calibrated.
"""

import numpy as np

import charslit


def kernel_from_moments(k2, k3=None, nk=9):
    """Per-row kernels with unit sum, zero mean, variance k2 and third moment k3.

    k2, k3: arrays (nrows,) of per-row cumulants in px^2 / px^3 (detector-column grid),
    k2 >= 0. The symmetric part is Lindeberg's discrete Gaussian exp(-t) I_n(t), whose
    variance is exactly t (and a delta at t = 0); the skew part adds k3 * B3 with B3 the
    antisymmetric 5-tap vector of zero mean and unit third moment, which leaves sum and
    variance unchanged. Returns (nrows, nk).
    """
    from scipy.special import ive

    k2 = np.clip(np.atleast_1d(np.asarray(k2, float)), 0, None)
    k3 = np.zeros_like(k2) if k3 is None else np.atleast_1d(np.asarray(k3, float))
    o = np.arange(nk) - nk // 2
    K = ive(np.abs(o)[None, :], k2[:, None])
    K[k2 == 0] = (o == 0).astype(float)
    K /= K.sum(1, keepdims=True)
    B3 = np.zeros(nk)
    c = nk // 2
    B3[c - 2 : c + 3] = [-1 / 12, 1 / 6, 0, -1 / 6, 1 / 12]
    return K + k3[:, None] * B3[None, :]


def rows_to_subpix(v, osample, ycen_frac=0.0):
    """Per-detector-row values (nrows, ...) onto slitdec's ny subpixels.

    Uses slitdec's own subpixel geometry (zeta_tensors): subpixel i has its lower edge at
    detector y = i/osample - 1 + c - step (+ step when c is a multiple of step), c =
    ycen_frac, for a constant ycen fraction. The wrapper's own nrows -> ny stretch
    (i * (nrows-1)/(ny-1)) is up to ~1.5 rows off this, which matters for structure on
    the fibre scale.
    """
    v = np.asarray(v, float)
    nrows = v.shape[0]
    ny = osample * (nrows + 1) + 1
    step = 1.0 / osample
    c = float(ycen_frac)
    lower = np.arange(ny) / osample - 1 + c - step
    if np.isclose(np.fmod(c, step), 0.0):
        lower += step
    yc = lower + step / 2 - 0.5  # in units where row r has its centre at r
    flat = v.reshape(nrows, -1)
    out = np.stack([np.interp(yc, np.arange(nrows), flat[:, j]) for j in range(flat.shape[1])], axis=1)
    return out.reshape((ny,) + v.shape[1:])


def line_windows(profile, spacing=None, min_rel=0.2):
    """Peak positions and half-window of isolated lines in a 1D profile."""
    from scipy.signal import find_peaks

    p = np.asarray(profile, float)
    pk, _ = find_peaks(p, height=min_rel * np.nanmax(p), distance=3)
    if spacing is None:
        spacing = np.median(np.diff(pk)) if len(pk) > 1 else 10
    return pk, int(spacing // 2)


def row_moments(img, peaks, half, base=None):
    """Per-row k1, k2, k3 of each line window. Returns arrays (nrows, nlines), NaN if empty."""
    nrows, ncols = img.shape
    out = np.full((3, nrows, len(peaks)), np.nan)
    x = np.arange(ncols, dtype=float)
    for j, p in enumerate(peaks):
        lo, hi = p - half, p + half + 1
        if lo < 0 or hi > ncols:
            continue
        seg = img[:, lo:hi].astype(float)
        seg = seg - (np.minimum(seg[:, :1], seg[:, -1:]) if base is None else base)
        xs = x[lo:hi]
        w = seg.sum(1)
        ok = w > 0
        m1 = np.where(ok, (seg * xs).sum(1) / np.where(ok, w, 1), np.nan)
        d = xs[None, :] - m1[:, None]
        out[0, :, j] = m1
        out[1, :, j] = np.where(ok, (seg * d**2).sum(1) / np.where(ok, w, 1), np.nan)
        out[2, :, j] = np.where(ok, (seg * d**3).sum(1) / np.where(ok, w, 1), np.nan)
    return out


def missing_moments(im, model, peaks, half, mask=None):
    """Median over lines of (data - model) per-row k1, k2, k3. Returns three (nrows,) arrays.

    The model shares the window and baseline, so line-independent biases of the
    windowed moments cancel to first order.
    """
    if mask is not None:
        im = np.where(mask > 0, im, model)
    md = row_moments(im, peaks, half)
    mm = row_moments(model, peaks, half)
    d = md - mm
    with np.errstate(all="ignore"):
        return tuple(np.nanmedian(d[i], axis=1) for i in range(3))


def selfcal(
    im,
    ycen,
    slitcurve,
    slitdeltas,
    *,
    nk=9,
    niter=5,
    fit_k1=True,
    fit_k2=True,
    fit_k3=True,
    row_weight_min=0.2,
    max_step=(0.2, 0.5, 0.5),
    peaks=None,
    half=None,
    **slitdec_kw,
):
    """Iteratively calibrate per-row slitdeltas (k1) and kernels (k2, k3) on a line frame.

    Kernels are kept non-negative in k2 by referencing to the narrowest row: sP then
    carries the narrowest row's LSF and every row broadens it. Rows with less than
    row_weight_min of the peak row flux are not updated (slit edges); per-iteration
    updates of (k1, k2, k3) are clipped to max_step.
    Returns dict with slitdeltas, k2, k3, kernel, the final slitdec result and a
    history of chi2 per iteration.
    """
    nrows = im.shape[0]
    mask = np.isfinite(im).astype(np.uint8)
    im = np.nan_to_num(im)
    unc = np.sqrt(np.abs(im) + 1)
    sd = np.array(slitdeltas, float).copy()
    k2 = np.zeros(nrows)
    k3 = np.zeros(nrows)
    kw = dict(osample=6, lambda_sP=0.0, lambda_sL=0.1, maxiter=20, kappa=0)
    kw.update(slitdec_kw)
    hist = []
    res = None
    for it in range(niter + 1):
        kern = kernel_from_moments(k2, k3, nk) if it > 0 else None
        res = charslit.slitdec(im, unc, mask, np.array(ycen, float), slitcurve, sd, slitkernel=kern, **kw)
        r = (im - res["model"]) * mask
        hist.append(float(((r / unc) ** 2).mean()))
        if it == niter:
            break
        if peaks is None:
            peaks, half = line_windows(res["spectrum"])
        d1, d2, d3 = missing_moments(im, res["model"], peaks, half, mask)
        rowflux = (im * mask).sum(1)
        good = rowflux > row_weight_min * np.nanmax(rowflux)
        for d, lim in zip((d1, d2, d3), max_step):
            d[~good | ~np.isfinite(d)] = 0.0
            np.clip(d, -lim, lim, out=d)
        if fit_k1:
            sd = sd + d1
        if fit_k2:
            k2 = k2 + d2
        if fit_k3:
            k3 = k3 + d3
        # reference to the narrowest illuminated row; sP absorbs the common part
        k2 = k2 - np.min(k2[good])
        k2[~good] = 0.0
        k3[~good] = 0.0
    return dict(
        slitdeltas=sd,
        k2=k2,
        k3=k3,
        kernel=kernel_from_moments(k2, k3, nk),
        result=res,
        chi2=hist,
        peaks=peaks,
        half=half,
    )


def rows_to_subpix_nearest(v, osample, ycen_frac=0.0):
    """Like rows_to_subpix but piecewise constant: each subpixel takes its row's value."""
    v = np.asarray(v, float)
    nrows = v.shape[0]
    ny = osample * (nrows + 1) + 1
    step = 1.0 / osample
    c = float(ycen_frac)
    lower = np.arange(ny) / osample - 1 + c - step
    if np.isclose(np.fmod(c, step), 0.0):
        lower += step
    r = np.clip(np.floor(lower + step / 2).astype(int), 0, nrows - 1)
    return v[r]


def fit_row_kernels(im, model, nk=9, mask=None, edge=None, rows=None, unc=None, ridge=1e-3):
    """Per-row kernel D_r minimising |(im_r - D_r * model_r) / unc_r|^2 + ridge term, sum(D_r) = 1.

    The ridge pulls D_r towards a delta, scaled to the row's normal matrix
    (ridge * trace / nk), to suppress the alternating-tap solutions that smooth
    line profiles otherwise allow.

    A kernel constant within a detector row commutes with the per-subpixel shifts, so
    D_r * model_r is exactly the model with D_r composed onto that row's kernel (up to
    the subpixels shared with neighbouring rows). Returns (nrows, nk); rows not fitted
    get a delta.
    """
    nrows, ncols = im.shape
    h = nk // 2
    edge = h if edge is None else edge
    D = np.zeros((nrows, nk))
    D[:, h] = 1.0
    cols = slice(edge, ncols - edge)
    for r in range(nrows) if rows is None else rows:
        # columns of A: model shifted so that tap j (offset o = j - h) reads model[x - o]
        A = np.stack([np.roll(model[r], j - h)[cols] for j in range(nk)], axis=1)
        b = im[r, cols]
        wt = np.ones_like(b) if unc is None else 1.0 / unc[r, cols]
        if mask is not None:
            m = mask[r, cols] > 0
            A, b, wt = A[m], b[m], wt[m]
        A = A * wt[:, None]
        b = b * wt
        if len(b) < 2 * nk:
            continue
        # sum-to-one constraint via KKT system
        M = np.zeros((nk + 1, nk + 1))
        AtA = A.T @ A
        lam = ridge * np.trace(AtA) / nk
        delta = np.zeros(nk)
        delta[h] = 1.0
        M[:nk, :nk] = AtA + lam * np.eye(nk)
        M[:nk, nk] = M[nk, :nk] = 1.0
        rhs = np.append(A.T @ b + lam * delta, 1.0)
        try:
            D[r] = np.linalg.solve(M, rhs)[:nk]
        except np.linalg.LinAlgError:
            pass
    return D


def compose(K, D):
    """Row-wise convolution of kernels K (n, nk) with D (n, nk), cropped back to nk taps."""
    nk = K.shape[1]
    out = np.array([np.convolve(k, d, mode="full") for k, d in zip(K, D)])
    c = (out.shape[1] - nk) // 2
    return out[:, c : c + nk]


def selfcal_lsq(
    im, ycen, slitcurve, slitdeltas, *, nk=9, niter=6, row_weight_min=0.2, ycen_frac=0.0, rn=10.0, ridge=1e-3, stop_rel=0.02, **slitdec_kw
):
    """Nonparametric per-row kernels by repeated least-squares updates (fit_row_kernels).

    Each iteration: slitdec with the current kernels -> per-row correction kernel D_r
    against the current model -> K_r <- D_r * K_r. Kernels are per detector row,
    expanded piecewise-constant onto slitdec's subpixels. Stops once chi2 rises by
    more than stop_rel over the best iteration and returns the best kernels.
    """
    nrows = im.shape[0]
    mask = np.isfinite(im).astype(np.uint8)
    im = np.nan_to_num(im)
    unc = np.sqrt(np.abs(im) + rn**2)
    kw = dict(osample=6, lambda_sP=0.0, lambda_sL=0.1, maxiter=20, kappa=0)
    kw.update(slitdec_kw)
    os_ = kw["osample"]
    rowflux = (im * mask).sum(1)
    rows = np.nonzero(rowflux > row_weight_min * rowflux.max())[0]
    K = np.zeros((nrows, nk))
    K[:, nk // 2] = 1.0
    hist = []
    best = None
    for it in range(niter + 1):
        kern = None if it == 0 else rows_to_subpix_nearest(K, os_, ycen_frac)
        res = charslit.slitdec(im, unc, mask, np.array(ycen, float), slitcurve, slitdeltas, slitkernel=kern, **kw)
        chi2 = float((((im - res["model"]) / unc) ** 2 * mask).mean())
        hist.append(chi2)
        if best is None or chi2 < best[0]:
            best = (chi2, K.copy(), res)
        elif chi2 > (1 + stop_rel) * best[0]:
            # the composed kernels have started to overfit / drift: keep the best
            break
        if it == niter:
            break
        D = fit_row_kernels(im, res["model"], nk, mask, rows=rows, unc=unc, ridge=ridge)
        K = compose(K, D)
    _, K, res = best
    return dict(kernel_rows=K, kernel=rows_to_subpix_nearest(K, os_, ycen_frac), result=res, chi2=hist)
