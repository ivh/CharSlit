"""slitdec_ox-specific behaviour on synthetic swaths (tests/synth.py)."""

import numpy as np
import pytest

import charslit
from synth import box_int, fp_spec, make_swath

NC, NR = 400, 136
TR = slice(30, NC - 30)
C1 = 8 / NR
LINEAR = dict(osample_x=1, mu_x=[], t_offset=0.0)  # the defaults: historic slitdec
OX8 = dict(osample_x=8, t_offset=0.5)              # recommended; mu_x defaults to the whitening penalty


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
    new = line_shift(fp, extract(im, ycen, sc, osample=osample, osample_x=16, t_offset=0.5)["spectrum"])
    assert abs(abs(old) - expected) < 0.25 * expected
    assert abs(new) < 0.25 * expected


def test_line_bias_vs_linear(fp_swath):
    """Fine bins remove most of the line-profile error of the two-column split."""
    _, im, ycen, sc, truth = fp_swath
    err = {}
    for name, kw in (("linear", LINEAR), ("ox8", OX8)):
        sp = extract(im, ycen, sc, osample=8, **kw)["spectrum"]
        err[name] = np.sqrt(((sp - truth)[TR] ** 2).mean()) / truth.max()
    assert err["ox8"] < 1e-3
    assert err["ox8"] < 0.1 * err["linear"]


def test_spectrum_is_mean_of_fine(fp_swath):
    _, im, ycen, sc, _ = fp_swath
    r = extract(im, ycen, sc, osample=8, osample_x=4, t_offset=0.5)
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
    r = extract(im, ycen, sc, osample=8, osample_x=ox, mu_x=[0, 0, 1e-6], t_offset=0.5)
    err = lambda v: np.sqrt(((v - truth)[core] ** 2).mean()) / truth[core].max()
    stair = np.repeat(extract(im, ycen, sc, osample=8, **LINEAR)["spectrum"], ox)
    assert err(r["spectrum_fine"]) < 0.02
    assert err(r["spectrum_fine"]) < 0.2 * err(stair)


def test_out_x_is_mean_of_fine_bins(fp_swath):
    _, im, ycen, sc, _ = fp_swath
    r = extract(im, ycen, sc, osample=8, out_x=2, **OX8)
    np.testing.assert_allclose(r["spectrum_out"], r["spectrum_fine"].reshape(NC * 2, 4).mean(axis=1),
                               rtol=1e-13, atol=1e-9)
    assert "spectrum_out" not in extract(im, ycen, sc, osample=8, **OX8)
    np.testing.assert_allclose(extract(im, ycen, sc, osample=8, out_x=1, **OX8)["spectrum_out"], r["spectrum"],
                               rtol=1e-12, atol=1e-9)
    with pytest.raises(RuntimeError):
        extract(im, ycen, sc, osample=8, out_x=3, **OX8)


class TestPropagatedUncertainty:
    @pytest.fixture(scope="class")
    def flat_runs(self):
        nc, nr = 160, 60
        m, ycen, sc, _ = make_swath(dict(kind="flat"), nc, nr, c1=4 / nr, nsub=8)
        rng = np.random.default_rng(7)
        out = {}
        for name, kw in (("linear", LINEAR), ("weak", dict(osample_x=8, mu_x=[0, 0, 1e-6], t_offset=0.5))):
            runs = []
            for _ in range(40):
                d = m + np.sqrt(m) * rng.standard_normal(m.shape)
                a = (d, np.sqrt(m), np.ones(d.shape, np.uint8), ycen, sc, np.zeros(nr))
                runs.append((charslit.slitdec(*a, osample=6, kappa=0.0, **kw),
                             charslit.slitdec(*a, osample=6, kappa=0.0, uncertainty="propagated", **kw)))
            out[name] = runs
        return out, slice(20, nc - 20)

    @pytest.mark.parametrize("name", ["linear", "weak"])
    def test_matches_empirical_scatter(self, flat_runs, name):
        """On a tilted slit the residual-based sigma is ~25 % low for the linear split
        and for a weak penalty; the propagated one follows the actual scatter."""
        runs, tr = flat_runs
        spec = np.array([p["spectrum"][tr] for _, p in runs[name]])
        emp = spec.std(0, ddof=1).mean()
        resid = np.mean([r["uncertainty"][tr] for r, _ in runs[name]])
        prop = np.mean([p["uncertainty"][tr] for _, p in runs[name]])
        assert resid / emp < 0.88
        assert 0.93 < prop / emp < 1.07

    @pytest.mark.parametrize("name", ["linear", "weak"])
    def test_lag1_covariance(self, flat_runs, name):
        runs, tr = flat_runs
        spec = np.array([p["spectrum"][tr] for _, p in runs[name]])
        d = spec - spec.mean(0)
        emp = (d[:, 1:] * d[:, :-1]).mean() / d.var()
        cov = np.mean([p["covariance"][tr] for _, p in runs[name]], axis=0)
        assert emp < -0.1
        assert abs(cov[:, 1].mean() / cov[:, 0].mean() - emp) < 0.05

    def test_outputs(self, fp_swath):
        _, im, ycen, sc, _ = fp_swath
        unc = np.sqrt(np.abs(im) + 1)
        a = (im, unc, np.ones(im.shape, np.uint8), ycen, sc, np.zeros(NR))
        r0 = charslit.slitdec(*a, kappa=0.0, **OX8)
        r1 = charslit.slitdec(*a, kappa=0.0, uncertainty="propagated", out_x=2, nlag=3, **OX8)
        r2 = charslit.slitdec(*a, kappa=0.0, uncertainty="scaled", out_x=2, nlag=3, **OX8)
        assert "covariance" not in r0 and "chi2" not in r0
        np.testing.assert_array_equal(r0["spectrum"], r1["spectrum"])
        assert r1["covariance"].shape == (NC * 2, 4)
        assert np.isfinite(r1["covariance"]).all() and (r1["covariance"][2 * 30:-2 * 30, 0] > 0).all()
        np.testing.assert_allclose(r2["uncertainty"], r1["uncertainty"] * np.sqrt(r1["chi2"]), rtol=1e-12)
        np.testing.assert_allclose(r2["covariance"], r1["covariance"] * r1["chi2"], rtol=1e-12)
        # detector-grid variance = variance of the mean of the two half-pixel bins
        c = r1["covariance"]
        var = (c[0::2, 0] + c[1::2, 0] + 2 * c[0::2, 1]) / 4
        np.testing.assert_allclose(np.sqrt(var), r1["uncertainty"], rtol=1e-9, atol=1e-12)
        with pytest.raises(RuntimeError):
            charslit.slitdec(*a, uncertainty="horne")


def test_defaults_are_historic(fp_swath):
    """A plain call is the linear split; osample_x > 1 alone gets the whitening penalty."""
    _, im, ycen, sc, _ = fp_swath
    a = (im, np.ones_like(im), np.ones(im.shape, np.uint8), ycen, sc, np.zeros(NR))
    np.testing.assert_array_equal(charslit.slitdec(*a)["spectrum"], charslit.slitdec(*a, **LINEAR)["spectrum"])
    np.testing.assert_array_equal(charslit.slitdec(*a, osample_x=8)["spectrum"],
                                  charslit.slitdec(*a, osample_x=8, mu_x=[0, 0, 1.64e-4])["spectrum"])
