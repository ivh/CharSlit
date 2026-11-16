"""slitdec_ox-specific behaviour on synthetic swaths (tests/synth.py)."""

import numpy as np
import pytest

import charslit
from synth import box_int, fp_spec, make_swath

NC, NR = 400, 136
TR = slice(30, NC - 30)
C1 = 8 / NR
LINEAR = dict(osample_x=1, mu_x=[], t_offset=0.0)  # slitdec.c


@pytest.fixture(scope="module")
def fp_swath():
    fp = fp_spec(NC)
    im, ycen, sc, truth = make_swath(fp, NC, NR, c1=C1)
    return fp, im, ycen, sc, truth


def extract(im, ycen, sc, **kw):
    r = charslit.slitdec(im, np.ones_like(im), np.ones(im.shape, np.uint8), ycen, sc,
                         np.zeros(NR), kappa=0.0, **kw)
    assert r["return_code"] == 0 and r["info"][2] == 1
    return r


def line_shift(fp, spec):
    """Shift [px] of the pixel-integrated FP comb that best fits spec."""
    X = np.arange(NC)[TR]
    best = None
    for s in np.linspace(-0.012, 0.012, 481):
        m = 3000 * box_int(fp, X - 0.5 - s, X + 0.5 - s)
        a = (m @ spec[TR]) / (m @ m)
        c = ((spec[TR] - a * m) ** 2).sum()
        if best is None or c < best[0]:
            best = (c, s)
    return best[1]


@pytest.mark.parametrize("osample", [4, 8])
def test_tilt_reference_point(fp_swath, osample):
    """slitdec.c evaluates the tilt at the subpixel's lower edge: every line moves
    by c1/(2*osample). Evaluating at the centre (default t_offset) removes it."""
    fp, im, ycen, sc, _ = fp_swath
    expected = C1 / (2 * osample)
    old = line_shift(fp, extract(im, ycen, sc, osample=osample, **LINEAR)["spectrum"])
    new = line_shift(fp, extract(im, ycen, sc, osample=osample, osample_x=16)["spectrum"])
    assert abs(abs(old) - expected) < 0.25 * expected
    assert abs(new) < 0.25 * expected


def test_line_bias_vs_linear(fp_swath):
    """Fine bins remove most of the line-profile error of the two-column split."""
    _, im, ycen, sc, truth = fp_swath
    err = {}
    for name, kw in (("linear", LINEAR), ("ox8", {})):
        sp = extract(im, ycen, sc, osample=8, **kw)["spectrum"]
        err[name] = np.sqrt(((sp - truth)[TR] ** 2).mean()) / truth.max()
    assert err["ox8"] < 1e-3
    assert err["ox8"] < 0.1 * err["linear"]


def test_spectrum_is_mean_of_fine(fp_swath):
    _, im, ycen, sc, _ = fp_swath
    r = extract(im, ycen, sc, osample=8, osample_x=4)
    fine = r["spectrum_fine"].reshape(NC, 4).mean(axis=1)
    np.testing.assert_allclose(r["spectrum"], fine, rtol=1e-12, atol=1e-9)


def test_fine_spectrum_recovers_undersampled_lines():
    """Sampling recovery: with a weak penalty spectrum_fine follows lines of
    FWHM 1.5 px far better than the detector-grid staircase does."""
    fp = fp_spec(NC, fwhm=1.5, spacing=9.3)
    im, ycen, sc, _ = make_swath(fp, NC, NR, c1=C1)
    ox = 8
    xf = np.arange(NC * ox) / ox - 0.5
    truth = 3000 * box_int(fp, xf, xf + 1 / ox) * ox
    core = slice(30 * ox, (NC - 30) * ox)
    r = extract(im, ycen, sc, osample=8, osample_x=ox, mu_x=[0, 0, 1e-6])
    err = lambda v: np.sqrt(((v - truth)[core] ** 2).mean()) / truth[core].max()
    stair = np.repeat(extract(im, ycen, sc, osample=8, **LINEAR)["spectrum"], ox)
    assert err(r["spectrum_fine"]) < 0.02
    assert err(r["spectrum_fine"]) < 0.2 * err(stair)
