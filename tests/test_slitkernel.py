"""Tests for the per-row slit kernel (slitkernel argument) and slitkernel.py helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import charslit  # noqa: E402
import fibreslit_sim as fs  # noqa: E402
import slitkernel as sk  # noqa: E402

NC, NR, TILT = 200, 60, 0.05
KW = dict(osample=6, lambda_sP=0.0, lambda_sL=0.1, maxiter=30, kappa=0)


@pytest.fixture(scope="module")
def fibre_frame():
    fib = fs.make_fibres(9, nrows=NR)
    im = fs.render(NC, NR, fib, fs.fp_lines(NC, amp=3e6), tilt=TILT, ycentre=NR / 2, continuum=2e3)
    sc = np.zeros((NC, 2))
    sc[:, 1] = TILT
    return im, sc


def _run(im, sc, kern=None, sd=None):
    unc = np.sqrt(np.abs(im) + 100)
    mask = np.ones(im.shape, np.uint8)
    sd = np.zeros(im.shape[0]) if sd is None else sd
    return charslit.slitdec(im, unc, mask, np.zeros(im.shape[1]), sc, sd, slitkernel=kern, **KW)


def test_delta_kernel_is_identity(fibre_frame):
    im, sc = fibre_frame
    r0 = _run(im, sc)
    for nk in (1, 3, 5):
        k = np.zeros((NR, nk))
        k[:, nk // 2] = 1.0
        r = _run(im, sc, k)
        np.testing.assert_allclose(r["model"], r0["model"], rtol=1e-10, atol=1e-8)


def test_common_kernel_absorbed_by_spectrum(fibre_frame):
    # the same kernel in every row is degenerate with sP: the model must not change
    im, sc = fibre_frame
    r0 = _run(im, sc)
    r = _run(im, sc, np.tile([0.1, 0.2, 0.4, 0.2, 0.1], (NR, 1)))
    np.testing.assert_allclose(r["model"][:, 10:-10], r0["model"][:, 10:-10], rtol=1e-6, atol=1e-3)


def test_shift_kernel_shifts_spectrum(fibre_frame):
    im, sc = fibre_frame
    r0 = _run(im, sc)
    r = _run(im, sc, np.tile([0.0, 0.0, 1.0], (NR, 1)))
    # tap at offset +1 moves the image right, so sP moves left by one column
    np.testing.assert_allclose(r["spectrum"][20:-20], r0["spectrum"][21:-19], rtol=1e-6)


def test_bad_kernel_shape_raises(fibre_frame):
    im, sc = fibre_frame
    with pytest.raises(RuntimeError):
        _run(im, sc, np.ones((NR, 4)) / 4)
    with pytest.raises(RuntimeError):
        _run(im, sc, np.ones((NR + 3, 3)) / 3)


def test_kernel_from_moments():
    k2 = np.array([0.0, 0.3, 1.0])
    k3 = np.array([0.0, 0.1, -0.2])
    K = sk.kernel_from_moments(k2, k3, 11)
    o = np.arange(11) - 5
    np.testing.assert_allclose(K.sum(1), 1, atol=1e-12)
    np.testing.assert_allclose((K * o).sum(1), 0, atol=1e-12)
    np.testing.assert_allclose((K * o**2).sum(1), k2, atol=2e-3)
    np.testing.assert_allclose((K * o**3).sum(1), k3, atol=2e-3)
    assert K[0, 5] == 1.0


def test_rows_to_subpix_matches_slitdec_geometry():
    # a delta in one row's slitdeltas mapped with rows_to_subpix_nearest must move only that row
    nr, os_ = 20, 6
    v = np.zeros(nr)
    v[7] = 1.0
    sub = sk.rows_to_subpix_nearest(v, os_, 0.0)
    assert sub.shape == (os_ * (nr + 1) + 1,)
    assert sub.sum() == os_


def test_selfcal_lsq_reduces_chi2(fibre_frame):
    im, sc = fibre_frame
    c = sk.selfcal_lsq(im, np.zeros(NC), sc, np.zeros(NR), niter=3, **KW)
    assert min(c["chi2"]) < 0.7 * c["chi2"][0]


def test_selfcal_u_reduces_chi2(fibre_frame):
    im, sc = fibre_frame
    c = sk.selfcal_u(im, np.full(NC, 1e-3), sc, np.zeros(NR), niter=3, **KW)
    assert min(c["chi2"]) < 0.7 * c["chi2"][0]
    assert c["kernel"].shape == (6 * (NR + 1) + 1, 9)


def test_subpix_u_matches_slitdec():
    # a single-subpixel preset slit function lands on the detector rows subpix_u predicts
    nr, nc, os_ = 20, 8, 6
    ny = os_ * (nr + 1) + 1
    u = sk.subpix_u(ny, os_)
    im = np.ones((nr, nc))
    for c in (0.1, 0.4, 0.9):
        for i in (31, 62, 75):
            sl = np.zeros(ny)
            sl[i] = 1.0
            r = charslit.slitdec(im, im, np.ones((nr, nc), np.uint8), np.full(nc, c), np.zeros((nc, 2)), np.zeros(nr),
                                 osample=os_, kappa=0, preset_slitfunc=sl)
            col = r["model"][:, 4] / r["model"][:, 4].sum()
            lo = u[i] + c - 0.5 / os_
            hi = lo + 1.0 / os_
            rows = np.arange(nr)
            expect = np.clip(np.minimum(hi, rows + 1) - np.maximum(lo, rows), 0, None) * os_
            np.testing.assert_allclose(col, expect, atol=1e-9)


def test_stencil_moments():
    o = np.arange(-2, 3)
    for n, B in sk.STENCILS.items():
        m = [(B * o**p).sum() for p in range(5)]
        expect = [0.0] * 5
        expect[n] = 1.0
        np.testing.assert_allclose(m, expect, atol=1e-12)


def test_kernel_from_moments_k4():
    K = sk.kernel_from_moments(np.array([0.5]), np.array([0.0]), 11, k4=np.array([0.2]))
    K0 = sk.kernel_from_moments(np.array([0.5]), np.array([0.0]), 11)
    o = np.arange(11) - 5
    np.testing.assert_allclose([(K - K0)[0] @ o**p for p in range(5)], [0, 0, 0, 0, 0.2], atol=1e-12)


def test_shift_cumulants():
    nr, os_ = 10, 6
    ny = os_ * (nr + 1) + 1
    v2, v3, v4 = sk.shift_cumulants(np.zeros((5, 2)), np.full(ny, -1.5), nr, os_)
    np.testing.assert_allclose([v2[0], v3[0], v4[0]], [0.25, 0.0, -0.125])
    v2, v3, _ = sk.shift_cumulants(np.zeros((5, 2)), np.full(ny, 0.2), nr, os_)
    np.testing.assert_allclose([v2[0], v3[0]], [0.16, 0.16 * 0.6])


def test_stretch_rows_matches_wrapper(fibre_frame):
    im, sc = fibre_frame
    sd = np.linspace(-0.3, 0.4, NR) ** 2
    ny = KW["osample"] * (NR + 1) + 1
    np.testing.assert_allclose(_run(im, sc, sd=sd)["model"], _run(im, sc, sd=sk.stretch_rows(sd, ny))["model"],
                               rtol=1e-10, atol=1e-8)


def _lsf_trend_frame(nr=24, nc=240, yc=0.3, seed=2):
    # rows with known per-row LSF: Gaussian width trend, shift, and a two-Gaussian skew
    r = np.arange(nr) + 0.5 - yc
    t = (r - nr / 2) / (nr / 2)
    slit = np.exp(-0.5 * ((r - nr / 2) / 6) ** 2)
    sig, shift, eps, dd = 0.9 + 0.15 * t, 0.15 * np.sin(2.5 * t), 0.25, 1.2 * (0.5 + 0.5 * t)
    fine = 20
    xf = (np.arange(nc * fine) + 0.5) / fine
    im = np.zeros((nr, nc))
    for i in range(nr):
        z = xf[None, :] - np.arange(15.3, nc - 10, 13.0)[:, None] - shift[i]
        p = ((1 - eps) * np.exp(-0.5 * (z / sig[i]) ** 2) + eps * np.exp(-0.5 * ((z - dd[i]) / sig[i]) ** 2)).sum(0)
        im[i] = slit[i] * p.reshape(nc, fine).mean(1) * 2e4 / sig[i]
    im = np.random.default_rng(seed).poisson(im + 50).astype(float)
    truth = {1: shift + eps * dd, 2: sig**2 + eps * (1 - eps) * dd**2, 3: eps * (1 - eps) * (1 - 2 * eps) * dd**3}
    return im, r, slit > 0.5, truth


def test_selfcal_param_recovers_cumulants():
    im, r, core, truth = _lsf_trend_frame()
    nc = im.shape[1]
    c = sk.selfcal_param(im, np.full(nc, 0.3), np.zeros((nc, 2)), np.zeros(im.shape[0]), rn=5.0, **KW)
    assert min(c["chi2"]) < 0.3 * c["chi2"][0]
    for n, tol in ((1, 0.01), (2, 0.04), (3, 0.02)):
        est = np.interp(r, c["nodes"], c["cumulants"][n])
        d = (est - est[core].mean()) - (truth[n] - truth[n][core].mean())
        assert np.sqrt((d[core] ** 2).mean()) < tol, n


def test_selfcal_param_reduces_chi2(fibre_frame):
    im, sc = fibre_frame
    c = sk.selfcal_param(im, np.full(NC, 1e-3), sc, np.zeros(NR), niter=4, **KW)
    assert min(c["chi2"]) < 0.7 * c["chi2"][0]
    ny = 6 * (NR + 1) + 1
    assert c["kernel"].shape == (ny, 9) and c["slitdeltas"].shape == (ny,)
    lit = c["lit"]
    for n in (1, 2, 3):
        assert abs((c["gauge"] * c["cumulants"][n]).sum()) < 1e-12
    assert lit.sum() > 0
