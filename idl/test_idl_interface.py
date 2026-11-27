"""Call slit_func_ox in libslitdec_ox_idl.so the way IDL CALL_EXTERNAL does, on the
swath saved by mkslitf.pro (test.sav), and compare with REDUCE libraries (slit_func_2d).

usage: python test_idl_interface.py [test.sav] [reference_lib.so ...]
"""
import ctypes as ct
import os
import sys
import numpy as np
from scipy.io import readsav

here = os.path.dirname(os.path.abspath(__file__))
sav = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "test.sav")
refs = sys.argv[2:]
d = readsav(sav)


def native(a, dt):
    return np.ascontiguousarray(np.asarray(a), dtype=dt).copy()


def call_reduce(lib):
    """slit_func_2d of the REDUCE library, with mkslitf.pro's 25 arguments."""
    L = ct.CDLL(lib)
    nc, nysf, osamp = int(d["nc"]), int(d["nysf"]), int(d["osamp"])
    ny = (nysf + 1) * osamp + 1
    a = [np.array(nc, np.int32), np.array(nysf, np.int32), np.array(osamp, np.int32),
         native(d["sf"], np.float64), native(d["ycen_fract"], np.float64),
         native(d["ycen_int"], np.int32), np.array(int(d["y_lower_lim"]), np.int32),
         native(d["psf_curve"], np.float64), np.array(int(d["idelta_x"]), np.int32),
         np.array(float(d["lambda_sp"]), np.float64), np.array(float(d["lambda_sf"]), np.float64),
         native(d["msk"], np.uint8), np.zeros((nysf, nc)), np.zeros(nc), np.zeros(nc),
         np.zeros(ny), np.zeros(nc)]
    # scratch arrays, sized as mkslitf.pro allocates them (the original code uses them)
    a += [native(d[k], d[k].dtype.newbyteorder("=")) * 0
          for k in ("xi", "zeta", "m_zeta", "l_aij", "l_bj", "p_aij", "p_bj")]
    a += [np.array(0, np.int32)]
    argv = (ct.c_void_p * len(a))(*[x.ctypes.data for x in a])
    L.slit_func_2d.restype = ct.c_int
    rc = L.slit_func_2d(ct.c_int(len(a)), argv)
    return dict(rc=rc, sp=a[13], sL=a[15], model=a[12], unc=a[16], mask=a[11])


def legacy_mu1_y(lib):
    """mu1_y equivalent to REDUCE's lambda_sf on this swath, from slitdec() (exported
    by the same library) run with the legacy lambda_sL penalty."""
    L = ct.CDLL(lib)
    nc, nr, os_ = int(d["nc"]), int(d["nysf"]), int(d["osamp"])
    ny = (nr + 1) * os_ + 1
    im = native(d["sf"], np.float64)
    ycen = native(d["ycen_fract"], np.float64) + native(d["ycen_int"], np.int32)
    sc = np.zeros((nc, 6)); sc[:, 1:3] = native(d["psf_curve"], np.float64)[:, 1:3]
    mux = np.zeros(3); info = np.zeros(11); info[4] = -11
    sp, sl, model, unc = np.zeros(nc), np.zeros(ny), np.zeros((nr, nc)), np.zeros(nc)
    sp[:] = np.abs(im * native(d["msk"], np.uint8)).sum(0)
    P, I, D = ct.POINTER(ct.c_double), ct.c_int, ct.c_double
    ptr = lambda a: a.ctypes.data_as(P)
    L.slitdec.restype = I
    L.slitdec(I(nc), I(nr), ptr(im), ptr(np.zeros((nr, nc))),
              native(d["msk"], np.uint8).ctypes.data_as(ct.POINTER(ct.c_ubyte)),
              ptr(ycen), ptr(sc), ptr(np.zeros(ny)), I(os_), D(0.0), D(float(d["lambda_sf"])),
              I(100), D(10.0), I(0), I(0), D(-1.0), ptr(sp), ptr(sl), ptr(model), ptr(unc),
              ptr(info), I(1), ptr(mux), I(3), None, I(0), I(int(d["y_lower_lim"])), None,
              I(0), I(1), I(0), None, None)
    return -info[9]


def call(lib, ox=8, mux=(0.0, 0.0, 1.64e-4), muy=(1e-2, 0.0), maxiter=100, kappa=10.0,
         weights=0, gain=1.0, read_noise=0.0, nargs=27):
    """slit_func_ox with its 27 arguments."""
    L = ct.CDLL(lib)
    nc, nysf, osamp = int(d["nc"]), int(d["nysf"]), int(d["osamp"])
    ny = (nysf + 1) * osamp + 1
    a = [np.array(nc, np.int32), np.array(nysf, np.int32), np.array(osamp, np.int32),
         np.array(ox, np.int32), native(d["sf"], np.float64), native(d["ycen_fract"], np.float64),
         native(d["ycen_int"], np.int32), np.array(int(d["y_lower_lim"]), np.int32),
         native(d["psf_curve"], np.float64)]
    a += [np.array(v, np.float64) for v in tuple(mux) + tuple(muy)]
    a += [np.array(maxiter, np.int32), np.array(kappa, np.float64), np.array(weights, np.int32),
          np.array(gain, np.float64), np.array(read_noise, np.float64), np.array(0, np.int32),
          native(d["msk"], np.uint8), np.zeros((nysf, nc)), np.zeros(nc), np.zeros(nc * ox),
          np.zeros(ny), np.zeros(nc), np.zeros(11)]
    a = a[:nargs]
    argv = (ct.c_void_p * len(a))(*[x.ctypes.data for x in a])
    L.slit_func_ox.restype = ct.c_int
    rc = L.slit_func_ox(ct.c_int(len(a)), argv)
    if nargs != 27:
        return dict(rc=rc)
    return dict(rc=rc, mask=a[20], model=a[21], sp=a[22], sp_fine=a[23], sL=a[24], unc=a[25],
                info=a[26])


def report(name, r, ref=None):
    dx = int(d["idelta_x"])
    sl = slice(dx, int(d["nc"]) - dx)
    resid = (r["model"] - d["sf"])[:, sl][r["mask"][:, sl] > 0]
    s = (f"{name:34s} rc={r['rc']:2d} median sP={np.median(r['sp'][sl]):10.1f} "
         f"rms(model-im)={resid.std():8.2f} masked={np.sum(r['mask'] == 0):4d}")
    if ref is not None:
        s += f"  rms(sP/ref-1)={np.std(r['sp'][sl] / ref['sp'][sl] - 1):.2e}"
    print(s)


lib = os.path.join(here, "libslitdec_ox_idl.so")
print(f"{sav}: nc={int(d['nc'])} nysf={int(d['nysf'])} osamp={int(d['osamp'])} "
      f"y_lower_lim={int(d['y_lower_lim'])} shear={np.median(d['psf_curve'][:, 1]):+.4f}")
assert call(lib, nargs=26)["rc"] == -1, "wrong argument count accepted"
assert call(lib, weights=1, gain=2.0, read_noise=0.0)["rc"] == -1, "weights without read noise accepted"
assert call(lib, weights=2)["rc"] == -1, "unknown weight_mode accepted"
assert call(lib, mux=(0.0, -1.0, 0.0))["rc"] == -1, "negative mu accepted"
assert call(lib, muy=(0.0, 0.0))["rc"] == -1, "no slit-function penalty accepted"
r8 = call(lib, 8)
r16 = call(lib, 16)
mu1_eq = legacy_mu1_y(lib)
print(f"lambda_sf = {float(d['lambda_sf'])} is mu1_y = {mu1_eq:.4g} on this swath")
r1 = call(lib, 1, (0.0, 0.0, 0.0), muy=(mu1_eq, 0.0))
assert r8["rc"] == 0
assert np.allclose(r8["sp_fine"].reshape(-1, 8).mean(1)[5:-5], r8["sp"][5:-5]), "sP != mean(sP_fine)"
report("slitdec_ox ox=8 mu3_x=1.64e-4 mu1_y=1e-2", r8)
report("slitdec_ox ox=16", r16, r8)
i = r8["info"]
print(f"  info: delta_x {i[4]:.0f}, damping x {i[7]:.3g}, resolution x {i[8]:.3f} px")
for m in (1e-3, 1e-1):
    ry = call(lib, 8, muy=(m, 0.0))
    report(f"slitdec_ox ox=8, mu1_y={m:g}", ry, r8)
    print(f"  info: damping y {ry['info'][9]:.3g}, resolution y {ry['info'][10]:.3f} px")
report("slitdec_ox ox=1 mu_x=0 mu1_y=equiv", r1, r8)

# weights: variance = model/gain + (RN/gain)^2, invariant to gain when RN scales with it
# (only the ratio RN^2/gain matters). Same as charslit's model weights with the
# pix_unc the wrapper builds and noise_floor = (RN/gain)^2.
sys.path.insert(0, os.path.join(here, ".."))
import charslit
rw = call(lib, 8, weights=1, gain=2.0, read_noise=10.0)
rw2 = call(lib, 8, weights=1, gain=8.0, read_noise=20.0)
report("slitdec_ox ox=8 weights gain 2 RN 10", rw, r8)
assert rw["rc"] == 0 and np.array_equal(rw["sp"], rw2["sp"]), "weights depend on more than RN^2/gain"
im = native(d["sf"], np.float64)
ycen = native(d["ycen_fract"], np.float64) + native(d["ycen_int"], np.int32)
sc = np.zeros((int(d["nc"]), 6)); sc[:, 1:3] = native(d["psf_curve"], np.float64)[:, 1:3]
rp = charslit.slitdec(im, np.sqrt(np.maximum(im, 0) / 2.0 + 25.0), native(d["msk"], np.uint8),
                      ycen, sc, np.zeros(int(d["nysf"])), osample=int(d["osamp"]), osample_x=8,
                      mu_x=[0.0, 0.0, 1.64e-4], mu_y=[1e-2, 0.0], lambda_sL=0.0, maxiter=100,
                      kappa=10.0, weights="model", noise_floor=25.0,
                      y_lower_lim=int(d["y_lower_lim"]))
print(f"  weighted slit_func_ox vs charslit model weights: max |dsP|/sP "
      f"{np.max(np.abs(rw['sp'] - rp['spectrum'])[10:-10] / np.abs(rp['spectrum'][10:-10])):.2e}")
for ref in refs:
    # the REDUCE library uses lambda_sf; slitdec_ox's equivalent penalty is mu1_y below.
    # Not bit-identical any more: slitdec evaluates the tilt at the subpixel centre, REDUCE at
    # its lower edge, so lines differ by a shift of c1/(2 osample) (docs/slitdec_ox.md).
    rr = call_reduce(ref)
    report(os.path.basename(ref), rr, r8)
    report("  slitdec_ox ox=1 mu_x=0 mu1_y=equiv", r1, rr)
    print(f"    max |dsP| {np.max(np.abs(r1['sp'] - rr['sp'])):.3g}  max |dsL| "
          f"{np.max(np.abs(r1['sL'] - rr['sL'])):.3g}  mask diffs {np.sum(r1['mask'] != rr['mask'])}")
