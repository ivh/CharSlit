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


# 5-tap stencils on offsets -2..2 with zero sum and unit n-th moment; B1/B3 have no 3rd/1st
# moment and B2/B4 no 4th/2nd, so to first order around a delta, delta + sum a_n B_n has
# cumulants k_n = a_n.
STENCILS = {
    1: np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]),
    2: np.array([-1 / 24, 2 / 3, -5 / 4, 2 / 3, -1 / 24]),
    3: np.array([-1 / 12, 1 / 6, 0, -1 / 6, 1 / 12]),
    4: np.array([1 / 24, -1 / 6, 1 / 4, -1 / 6, 1 / 24]),
}


def kernel_from_moments(k2, k3=None, nk=9, k4=None):
    """Per-row kernels with unit sum, zero mean, variance k2 and third moment k3.

    k2, k3: arrays (nrows,) of per-row cumulants in px^2 / px^3 (detector-column grid),
    k2 >= 0. The symmetric part is Lindeberg's discrete Gaussian exp(-t) I_n(t), whose
    variance is exactly t (and a delta at t = 0); the skew part adds k3 * B3 with B3 the
    antisymmetric 5-tap vector of zero mean and unit third moment, which leaves sum and
    variance unchanged. k4 adds k4 * B4 (unit 4th moment, no 2nd): fourth cumulant in
    excess of the discrete Gaussian's own (= k2). k2 < 0 gives the first-order sharpening
    kernel delta + k2 * B2 (variance k2, but 4th moment 0 rather than Gaussian-like;
    crude below about -0.3 px^2). Returns (nrows, nk).
    """
    from scipy.special import ive

    k2 = np.atleast_1d(np.asarray(k2, float))
    k3 = np.zeros_like(k2) if k3 is None else np.atleast_1d(np.asarray(k3, float))
    o = np.arange(nk) - nk // 2
    kp = np.clip(k2, 0, None)
    K = ive(np.abs(o)[None, :], kp[:, None])
    K[kp == 0] = (o == 0).astype(float)
    K /= K.sum(1, keepdims=True)
    c = nk // 2
    if (k2 < 0).any():
        B2 = np.zeros(nk)
        B2[c - 2 : c + 3] = STENCILS[2]
        K = K + np.clip(k2, None, 0)[:, None] * B2[None, :]
    B3 = np.zeros(nk)
    B3[c - 2 : c + 3] = STENCILS[3]
    K = K + k3[:, None] * B3[None, :]
    if k4 is not None:
        B4 = np.zeros(nk)
        B4[c - 2 : c + 3] = STENCILS[4]
        K = K + np.atleast_1d(np.asarray(k4, float))[:, None] * B4[None, :]
    return K


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


def selfcal(im, ycen, slitcurve, slitdeltas, *, niter=5, row_weight_min=0.2, max_step=0.2, peaks=None, half=None,
            **slitdec_kw):
    """Iteratively calibrate per-row slitdeltas (k1) from data-vs-model line centroids.

    Rows with less than row_weight_min of the peak row flux are not updated (slit edges);
    per-iteration updates are clipped to max_step. Assumes a constant ycen fraction.
    The moment-matched k2/k3 kernels this used to fit are retired: they fit worse than
    least squares, see selfcal_param (whose moments=(1,) is the slit-coordinate k1 fit).
    Returns dict with slitdeltas, the final slitdec result and a history of chi2 per iteration.
    """
    mask = np.isfinite(im).astype(np.uint8)
    im = np.nan_to_num(im)
    unc = np.sqrt(np.abs(im) + 1)
    sd = np.array(slitdeltas, float).copy()
    kw = dict(osample=6, lambda_sP=0.0, lambda_sL=0.1, maxiter=20, kappa=0)
    kw.update(slitdec_kw)
    hist = []
    res = None
    for it in range(niter + 1):
        res = charslit.slitdec(im, unc, mask, np.array(ycen, float), slitcurve, sd, **kw)
        r = (im - res["model"]) * mask
        hist.append(float(((r / unc) ** 2).mean()))
        if it == niter:
            break
        if peaks is None:
            peaks, half = line_windows(res["spectrum"])
        d1 = missing_moments(im, res["model"], peaks, half, mask)[0]
        rowflux = (im * mask).sum(1)
        good = rowflux > row_weight_min * np.nanmax(rowflux)
        d1[~good | ~np.isfinite(d1)] = 0.0
        sd = sd + np.clip(d1, -max_step, max_step)
    return dict(slitdeltas=sd, result=res, chi2=hist, peaks=peaks, half=half)


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


def _node_weights(u0, nodes, osample):
    """Subpixel-averaged hat-function weights of pixels with lower edge u0 on the nodes: (npix, nn) sparse."""
    import scipy.sparse as sp

    nn, npix = len(nodes), len(u0)
    bw_r, bw_c, bw_v = [], [], []
    for sidx in range(osample):
        pos = np.interp(u0 + (sidx + 0.5) / osample, nodes, np.arange(nn))
        b0 = np.clip(np.floor(pos).astype(int), 0, nn - 2)
        f = pos - b0
        for bb, fw in ((b0, 1 - f), (b0 + 1, f)):
            bw_r.append(np.arange(npix))
            bw_c.append(bb)
            bw_v.append(fw / osample)
    return sp.csr_matrix((np.concatenate(bw_v), (np.concatenate(bw_r), np.concatenate(bw_c))), shape=(npix, nn))


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
    B = _node_weights(u0, nodes, osample)
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


def fit_cumulants_u(im, model, ycen_frac, nodes, moments=(1, 2, 3), mask=None, unc=None, ridge=1e-3, target=None,
                    smooth=0.0, current=None, edge=2, osample=6):
    """Cumulant corrections a_n(u) on slit-coordinate nodes, linear interpolation between nodes.

    Linearised around the current model: the correction kernel is delta + sum_n a_n B_n
    (STENCILS), averaged over each pixel's subpixels as in fit_kernels_u, so the fit is
    linear in a_n. Each moment's block gets a ridge towards target[n] (default 0, i.e. no
    change) and an optional second-difference penalty on the total current[n] + a_n
    across nodes, both scaled to that block's mean normal-matrix diagonal.
    Returns {n: (nnodes,)} for n in moments.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl

    nrows, ncols = im.shape
    nn = len(nodes)
    cols = np.arange(edge, ncols - edge)
    rr, xx = np.meshgrid(np.arange(nrows), cols, indexing="ij")
    u0 = rr - np.asarray(ycen_frac)[xx]
    good = np.ones_like(u0, bool) if mask is None else mask[rr, xx] > 0
    good &= (u0 >= nodes[0]) & (u0 + 1 <= nodes[-1])
    rr, xx, u0 = rr[good], xx[good], u0[good]
    w = np.ones(len(rr)) if unc is None else 1.0 / unc[rr, xx]
    B = _node_weights(u0, nodes, osample)
    y = (im[rr, xx] - model[rr, xx]) * w
    blocks = []
    for n in moments:
        v = sum(c * model[rr, np.clip(xx - o, 0, ncols - 1)] for o, c in zip(range(-2, 3), STENCILS[n]) if c)
        blocks.append(sp.diags(v * w) @ B)
    A = sp.hstack(blocks).tocsc()
    AtA = (A.T @ A).tocsc()
    rhs = A.T @ y
    d = AtA.diagonal()
    L = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(nn - 2, nn))
    LtL = (L.T @ L).tocsc()
    reg, extra = [], []
    for i, n in enumerate(moments):
        lam = ridge * d[i * nn : (i + 1) * nn].mean()
        R = lam * sp.identity(nn, format="csc")
        t = np.zeros(nn) if target is None or n not in target else np.asarray(target[n], float)
        e = lam * t
        if smooth > 0:
            mu = smooth * d[i * nn : (i + 1) * nn].mean()
            R = R + mu * LtL
            if current is not None and n in current:
                e = e - mu * (LtL @ np.asarray(current[n], float))
        reg.append(R)
        extra.append(e)
    a = spl.spsolve(AtA + sp.block_diag(reg, format="csc"), rhs + np.concatenate(extra))
    return {n: a[i * nn : (i + 1) * nn] for i, n in enumerate(moments)}


def stretch_rows(v, ny):
    """The wrapper's own nrows -> ny linear stretch of slitdeltas (pos = i (nrows-1)/(ny-1))."""
    v = np.asarray(v, float)
    return np.interp(np.arange(ny) * (len(v) - 1.0) / (ny - 1.0), np.arange(len(v)), v)


def shift_cumulants(slitcurve, slitdeltas, nrows, osample):
    """Cumulants (k2, k3, k4) per subpixel of slitdec's two-column split of a fractional shift.

    Subpixel i with shift delta deposits (1-f, f) on floor(delta), floor(delta)+1, f =
    delta - floor(delta): a Bernoulli offset with variance f(1-f), third cumulant
    f(1-f)(1-2f) and fourth f(1-f)(1-6f(1-f)). delta is slitdec's: the slit-curve
    polynomial at t = u - nrows//2 plus slitdeltas (ny,). Averaged over the swath's columns.
    """
    sc = np.atleast_2d(np.asarray(slitcurve, float))
    ny = len(slitdeltas)
    t = subpix_u(ny, osample) - nrows // 2
    delta = np.zeros((sc.shape[0], ny)) + np.asarray(slitdeltas, float)[None, :]
    for j in range(1, sc.shape[1]):
        delta += sc[:, j : j + 1] * t[None, :] ** j
    f = delta - np.floor(delta)
    v = f * (1 - f)
    return v.mean(0), (v * (1 - 2 * f)).mean(0), (v * (1 - 6 * v)).mean(0)


def selfcal_param(im, ycen, slitcurve, slitdeltas, *, moments=(1, 2, 3), nk=9, niter=12, node_step=0.5, rn=10.0,
                  ridge=1e-3, smooth=0.0, gauge="uniform", lit=None, prior=None, k2_offset=0.0, compensate=True, stop_rel=1e-3,
                  row_weight_min=0.2, **slitdec_kw):
    """Parametric alternative to selfcal_u: per-node cumulants instead of free kernel taps.

    Each slit-coordinate node carries a shift k1, which goes into slitdeltas, and width k2,
    skew k3 and optionally excess kurtosis k4, which become zero-mean kernels via
    kernel_from_moments. Iterations fit linearised corrections (fit_cumulants_u) against
    the current model and add them (cumulants add under convolution).

    slitdec splits a fractionally shifted subpixel over two columns, which adds width,
    skew and kurtosis that depend on the fractional shift (shift_cumulants; up to 0.25 px^2
    and +-0.096 px^3). With compensate=True the kernels subtract them, so the fitted
    cumulants describe the LSF and not slitdec's interpolation. Exact for a slit curve
    constant across the swath, column-averaged otherwise.

    Gauge: a cumulant common to all rows is degenerate with sP. Every cumulant is normalised
    to zero weighted mean over the lit nodes: gauge="uniform" (equal weights), "flux" (node
    flux of this frame; reproduces the plain-slitdec illumination dependence) or an array
    over nodes. Rows narrower than the gauge mean would need sharpening kernels, which fit
    worse; instead every kernel gets a common extra variance k2_offset (degenerate with
    sP, so sP is the gauge-mean row LSF narrowed by it). It grows within the run as far
    as needed and is returned; pass the calibration's value to per-frame fits so their sP
    has the same LSF. lit: fitted nodes, default those above
    row_weight_min of the peak row flux in this frame. The gauge is only frame-independent
    if lit is too: for per-frame fits pass lit and gauge from the calibration run.

    prior: {n: (nnodes,)} cumulants from a calibration run with the same nodes and gauge;
    the ridge then pulls towards it instead of towards no change, so a faint frame falls
    back to the calibration. slitdeltas: (nrows,) or (ny,) base geometry; k1 is added on top
    of it (an nrows input is stretched exactly as the wrapper would).

    A step that raises chi2 is halved and retried from the best point (down to 1/8); the
    fit stops when chi2 improves by less than stop_rel or after niter slitdec runs.
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
    sd0 = np.asarray(slitdeltas, float)
    sd0 = stretch_rows(sd0, ny) if len(sd0) == nrows else sd0.copy()
    nodes = np.arange(-1.0, nrows + 1.0 + 1e-9, node_step)
    nn = len(nodes)
    uS = subpix_u(ny, os_)
    rowflux = (im * mask).sum(1)
    node_flux = np.interp(nodes, np.arange(nrows) + 0.5 - np.median(frac), rowflux, left=0, right=0)
    lit = node_flux > row_weight_min * rowflux.max() if lit is None else np.asarray(lit, bool)
    if isinstance(gauge, str):
        g = {"uniform": np.ones(nn), "flux": node_flux}[gauge] * lit
    else:
        g = np.asarray(gauge, float) * lit
    g = g / g.sum()

    il = np.nonzero(lit)[0]
    nearest_lit = il[np.abs(np.arange(nn)[:, None] - il[None, :]).argmin(1)]

    def normalise(k):
        for n in k:
            # unlit nodes copy the nearest lit one: a zero there would be the gauge
            # reference (k2: the narrowest row), not a neutral value
            k[n] = k[n][nearest_lit]
            k[n] -= (g * k[n]).sum()
        return k

    off = [float(k2_offset)]

    def build(k):
        at = lambda n: np.interp(uS, nodes, k[n]) if n in k else np.zeros(ny)  # noqa: E731
        sd = sd0 + at(1)
        k2, k3, k4 = at(2), at(3), at(4) if 4 in moments else None
        if compensate:
            v2, v3, v4 = shift_cumulants(slitcurve, sd, nrows, os_)
            k2 = k2 - v2
            k3 = k3 - v3
            if k4 is not None:
                k4 = k4 - v4
        off[0] = max(off[0], -k2.min())
        return sd, kernel_from_moments(k2 + off[0], k3, nk, k4)

    k = {n: np.zeros(nn) for n in moments}
    if prior is not None:
        k = normalise({n: np.asarray(prior[n], float).copy() if n in prior else np.zeros(nn) for n in moments})
    hist, best = [], None
    fitmask = mask.copy()
    damp = 1.0
    for it in range(niter + 1):
        sd, kern = build(k)
        res = charslit.slitdec(im, unc, mask, ycen.copy(), slitcurve, sd, slitkernel=kern, **kw)
        if it == 0:
            # see selfcal_u: exclude the swath-end wedge
            dx = int(res["info"][4])
            fitmask[:, : 2 * dx] = 0
            fitmask[:, ncols - 2 * dx :] = 0
        chi2 = float(((((im - res["model"]) / unc) ** 2)[fitmask > 0]).mean())
        hist.append(chi2)
        if it == niter:
            break
        if best is None or chi2 < best[0]:
            if best is not None and chi2 > (1 - stop_rel) * best[0]:
                best = (chi2, {n: v.copy() for n, v in k.items()}, res, sd, kern)
                break
            best = (chi2, {n: v.copy() for n, v in k.items()}, res, sd, kern)
            target = None if prior is None else {n: prior[n] - k[n] for n in moments if n in prior}
            a = fit_cumulants_u(im, res["model"], frac, nodes, moments, fitmask, unc, ridge, target, smooth, k,
                                osample=os_)
            damp = min(1.0, 2 * damp)
        else:
            # the linearised step overshot (mostly k4): retry a shorter one from the best point
            damp /= 2
            if damp < 1 / 8:
                break
        k = normalise({n: best[1][n] + damp * np.where(lit, a[n], 0.0) for n in moments})
    if chi2 < best[0]:
        best = (chi2, k, res, sd, kern)
    _, k, res, sd, kern = best
    return dict(nodes=nodes, cumulants=k, slitdeltas=sd, kernel=kern, result=res, chi2=hist, fitmask=fitmask,
                lit=lit, gauge=g, k2_offset=off[0])
