"""The IDL REDUCE entry point slit_func_ox (idl/slitdec_ox_idl.c), called through ctypes
with IDL's CALL_EXTERNAL convention (argc, argv of pointers) on synthetic swaths."""
import ctypes as ct
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import charslit
from synth import fp_spec, make_swath

ROOT = Path(__file__).resolve().parents[1]
NC, NR, OS, OX = 150, 30, 6, 4
YLL = NR // 2
MU_X, MU_Y = (0.0, 0.0, 1.64e-4), (1e-2, 0.0)


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    if shutil.which("cc") is None:
        pytest.skip("no C compiler")
    so = tmp_path_factory.mktemp("idl") / "libslitdec_ox_idl.so"
    subprocess.run(["cc", "-O2", "-fPIC", "-Wall", "-Werror", "-shared", f"-I{ROOT / 'slitdec'}",
                    "-o", str(so), str(ROOT / "idl/slitdec_ox_idl.c"),
                    str(ROOT / "slitdec/slitdec.c"), "-lm"], check=True)
    L = ct.CDLL(str(so))
    L.slit_func_ox.restype = ct.c_int
    return L


def swath(flux, c1=1.5 / NR, gain=2.0, rn=10.0, seed=1):
    img, ycen, sc, _ = make_swath(fp_spec(NC), NC, NR, c1=c1, flux=flux,
                                  profile=lambda t: (np.abs(t) < 12) / 24.0)
    rng = np.random.default_rng(seed)
    im = rng.poisson(img * gain) / gain + rng.normal(0, rn / gain, img.shape)
    return im, ycen, sc


def call(L, im, ycen, sc, weights=0, gain=1.0, read_noise=0.0, mux=MU_X, muy=MU_Y, nargs=27):
    psf = np.zeros((NC, 3))
    psf[:, 1:] = sc[:, 1:3]
    a = [np.array(NC, np.int32), np.array(NR, np.int32), np.array(OS, np.int32),
         np.array(OX, np.int32), np.ascontiguousarray(im, float), (ycen % 1).copy(),
         np.floor(ycen).astype(np.int32), np.array(YLL, np.int32), psf]
    a += [np.array(v, float) for v in tuple(mux) + tuple(muy)]
    a += [np.array(100, np.int32), np.array(10.0), np.array(weights, np.int32),
          np.array(gain, float), np.array(read_noise, float), np.array(0, np.int32),
          np.ones((NR, NC), np.uint8), np.zeros((NR, NC)), np.zeros(NC), np.zeros(NC * OX),
          np.zeros((NR + 1) * OS + 1), np.zeros(NC), np.zeros(11)]
    a = a[:nargs]
    rc = L.slit_func_ox(len(a), (ct.c_void_p * len(a))(*[x.ctypes.data for x in a]))
    if nargs != 27:
        return dict(rc=rc)
    return dict(rc=rc, model=a[21], sp=a[22], sp_fine=a[23], unc=a[25], info=a[26])


def python(im, ycen, sc, **kw):
    return charslit.slitdec(im, kw.pop("pix_unc", np.ones_like(im)), np.ones(im.shape, np.uint8),
                            ycen, sc, np.zeros(NR), osample=OS, osample_x=OX, mu_x=list(MU_X),
                            mu_y=list(MU_Y), lambda_sL=0.0, maxiter=100, kappa=10.0,
                            y_lower_lim=YLL, **kw)


@pytest.mark.parametrize("bad", [dict(nargs=26), dict(weights=1, gain=2.0, read_noise=0.0),
                                 dict(weights=2), dict(mux=(0.0, -1.0, 0.0)),
                                 dict(muy=(0.0, 0.0))])
def test_rejects_bad_arguments(lib, bad):
    im, ycen, sc = swath(2e3)
    assert call(lib, im, ycen, sc, **bad)["rc"] == -1


def test_same_as_python_with_reduce_edges(lib):
    im, ycen, sc = swath(2e5)
    r = call(lib, im, ycen, sc)
    p = python(im, ycen, sc)
    assert r["rc"] == 0
    m = p["margin"]
    edge = int(r["info"][4])
    assert edge == m - 2 == int(np.ceil(1.5 / NR * (NR - YLL + 1)))
    assert np.all(r["sp"][:edge] == 0) and np.all(r["sp"][NC - edge:] == 0)
    assert np.all(r["sp_fine"][:edge * OX] == 0) and np.all(r["unc"][NC - edge:] == 0)
    s = slice(edge, NC - edge)
    # different initial sP (column sums here, 1 in the Python wrapper): agreement to the
    # convergence tolerance
    np.testing.assert_allclose(r["sp"][s], p["spectrum"][s], rtol=1e-4)
    np.testing.assert_array_equal(r["sp_fine"].reshape(-1, OX).mean(1)[s], r["sp"][s])


def test_weights_depend_on_read_noise_squared_over_gain_only(lib):
    im, ycen, sc = swath(2e5)
    r1 = call(lib, im, ycen, sc, weights=1, gain=2.0, read_noise=10.0)
    r2 = call(lib, im, ycen, sc, weights=1, gain=8.0, read_noise=20.0)
    np.testing.assert_array_equal(r1["sp"], r2["sp"])


@pytest.mark.parametrize("flux, tol", [(2e3, 3e-5), (2e5, 1e-6)])
def test_weights_are_model_over_gain_plus_read_noise(lib, flux, tol):
    """weight_mode 1 must give 1/(model/gain + (RN/gain)^2). A zero pix_unc would make
    slitdec fit a = 0, i.e. run unweighted without saying so."""
    gain, rn = 2.0, 10.0
    b = (rn / gain) ** 2
    im, ycen, sc = swath(flux, gain=gain, rn=rn)
    rw = call(lib, im, ycen, sc, weights=1, gain=gain, read_noise=rn)
    r0 = call(lib, im, ycen, sc)
    # reference: variances from the final model, so the fitted slope is 1/gain
    pe = python(im, ycen, sc, pix_unc=np.sqrt(np.maximum(rw["model"], 0) / gain + b),
                weights="model", noise_floor=b)
    s = slice(6, NC - 6)
    effect = np.std(rw["sp"][s] / r0["sp"][s] - 1)
    err = np.std(rw["sp"][s] / pe["spectrum"][s] - 1)
    assert effect > 50 * err
    assert err < tol
