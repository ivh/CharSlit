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
