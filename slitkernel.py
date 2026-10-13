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


def subpix_u(ny, osample):
    """Slit coordinate u = y_detector - ycen_frac of the centre of each slitdec subpixel.

    From slitdec's geometry (verified with single-subpixel preset slit functions): in a
    rectified column with ycen fraction c, subpixel i spans detector rows
    [i/os - 1 + c - 1/os, i/os - 1 + c). The special case of c an exact multiple of
    1/os (where slitdec shifts by one subpixel) is ignored.
    """
    step = 1.0 / osample
    return np.arange(ny) / osample - 1 - step / 2


def fit_kernels_u(im, model, ycen_frac, nodes, nk=9, mask=None, unc=None, ridge=1e-3, edge=None, smooth=0.0,
                  osample=6):
    """Correction kernels D(u) on slit-coordinate nodes, linear interpolation between nodes.

    Minimises sum over pixels of ((im - sum_j D_j model[x - o_j]) / unc)^2, where a
    pixel's kernel is the average of D(u) over its osample subpixels (u = y - ycen_frac,
    the pixel spans [row, row+1) in y), as in slitdec's model. Constraints/regularisers:
    sum_j D_j(node) = 1, a ridge towards a delta and an optional second-difference
    penalty across nodes (smooth), both scaled to the mean diagonal of the normal matrix.
    Returns (nnodes, nk).
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl

    nrows, ncols = im.shape
    h = nk // 2
    edge = h if edge is None else edge
    nn = len(nodes)
    cols = np.arange(edge, ncols - edge)
    rr, xx = np.meshgrid(np.arange(nrows), cols, indexing="ij")
    u0 = rr - np.asarray(ycen_frac)[xx]
    good = np.ones_like(u0, bool) if mask is None else mask[rr, xx] > 0
    good &= (u0 >= nodes[0]) & (u0 + 1 <= nodes[-1])
    rr, xx, u0 = rr[good], xx[good], u0[good]
    w = np.ones(len(rr)) if unc is None else 1.0 / unc[rr, xx]
    npix = len(rr)
    # subpixel-averaged hat-function weights: (npix, nn) sparse
    bw_r, bw_c, bw_v = [], [], []
    for sidx in range(osample):
        pos = np.interp(u0 + (sidx + 0.5) / osample, nodes, np.arange(nn))
        b0 = np.clip(np.floor(pos).astype(int), 0, nn - 2)
        f = pos - b0
        for bb, fw in ((b0, 1 - f), (b0 + 1, f)):
            bw_r.append(np.arange(npix))
            bw_c.append(bb)
            bw_v.append(fw / osample)
    B = sp.csr_matrix((np.concatenate(bw_v), (np.concatenate(bw_r), np.concatenate(bw_c))), shape=(npix, nn))
    # E = D - delta with sum_j E_j = 0, parametrised by the nk-1 taps other than the centre:
    # the model term for tap j != c is (M[x-o_j] - M[x]) * E_j
    others = [j for j in range(nk) if j != h]
    Mc = model[rr, xx]
    y = (im[rr, xx] - Mc) * w
    blocks = []
    for j in others:
        col = np.clip(xx - (j - h), 0, ncols - 1)
        v = (model[rr, col] - Mc) * w
        blocks.append(sp.diags(v) @ B)
    # unknown ordering: node-major (b * (nk-1) + k)
    A = sp.hstack(blocks).tocsc()
    perm = np.arange(nn * (nk - 1)).reshape(nk - 1, nn).T.ravel()
    A = A[:, perm]
    AtA = (A.T @ A).tocsc()
    lam = ridge * AtA.diagonal().mean()
    reg = lam * sp.identity(AtA.shape[0], format="csc")
    if smooth > 0:
        L = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(nn - 2, nn))
        L = sp.kron(L, sp.identity(nk - 1))
        reg = reg + smooth * AtA.diagonal().mean() * (L.T @ L).tocsc()
    E = spl.spsolve(AtA + reg, A.T @ y)
    E = E.reshape(nn, nk - 1)
    D = np.zeros((nn, nk))
    D[:, others] = E
    D[:, h] = 1.0 - E.sum(1)
    return D


def selfcal_u(im, ycen, slitcurve, slitdeltas, *, nk=9, niter=6, node_step=0.5, rn=10.0, ridge=1e-3, stop_rel=0.02,
              row_weight_min=0.2, smooth=0.0, **slitdec_kw):
    """Kernel self-calibration in slit coordinates, for swaths whose ycen fraction varies.

    ycen is passed to slitdec unchanged (absolute values are fine); its fractional part
    defines each column's slit coordinate. Kernels live on nodes in u (spacing node_step
    rows) and are interpolated onto slitdec's subpixels.
    """
    nrows, ncols = im.shape
    mask = np.isfinite(im).astype(np.uint8)
    im = np.nan_to_num(im)
    unc = np.sqrt(np.abs(im) + rn**2)
    kw = dict(osample=6, lambda_sP=0.0, lambda_sL=0.1, maxiter=20, kappa=0)
    kw.update(slitdec_kw)
    os_ = kw["osample"]
    ny = os_ * (nrows + 1) + 1
    ycen = np.asarray(ycen, float)
    frac = ycen - np.floor(ycen)
    nodes = np.arange(-1.0, nrows + 1.0 + 1e-9, node_step)
    uS = subpix_u(ny, os_)
    K = np.zeros((len(nodes), nk))
    K[:, nk // 2] = 1.0
    to_sub = lambda Kn: np.stack([np.interp(uS, nodes, Kn[:, j]) for j in range(nk)], axis=1)  # noqa: E731
    # nodes on weakly illuminated slit positions (edges, inter-slit gaps) stay a delta
    rowflux = (im * mask).sum(1)
    node_flux = np.interp(nodes, np.arange(nrows) + 0.5 - np.median(frac), rowflux, left=0, right=0)
    lit = node_flux > row_weight_min * rowflux.max()
    hist, best = [], None
    fitmask = mask.copy()
    for it in range(niter + 1):
        kern = None if it == 0 else to_sub(K)
        res = charslit.slitdec(im, unc, mask, ycen.copy(), slitcurve, slitdeltas, slitkernel=kern, **kw)
        if it == 0:
            # slitdec zeroes sP within delta_x of the swath ends, and the curved slit
            # spreads that over up to 2*delta_x columns of the model: exclude the wedge
            dx = int(res["info"][4])
            fitmask[:, : 2 * dx] = 0
            fitmask[:, ncols - 2 * dx :] = 0
        chi2 = float(((((im - res["model"]) / unc) ** 2)[fitmask > 0]).mean())
        hist.append(chi2)
        if best is None or chi2 < best[0]:
            best = (chi2, K.copy(), res)
        elif chi2 > (1 + stop_rel) * best[0]:
            break
        if it == niter:
            break
        D = fit_kernels_u(im, res["model"], frac, nodes, nk, fitmask, unc, ridge, smooth=smooth, osample=os_)
        D[~lit] = np.eye(nk)[nk // 2]
        K = compose(K, D)
    _, K, res = best
    return dict(nodes=nodes, kernel_nodes=K, kernel=to_sub(K), result=res, chi2=hist, fitmask=fitmask, lit=lit)
