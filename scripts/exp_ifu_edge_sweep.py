"""IFU crosstalk without odd/even masks: fibre profiles from the slit edge (AP-517, Q9 figure).

Same fibres, spectra and crosstalk metric as exp_ifu_crosstalk.py, but the traces drift through one
pixel WRAPS times across the swath, as along a real order; with a constant pixel phase no one-sided
profile can be measured below the pixel scale. Per-column rectified crops: row start
floor(yc + d(x)) - HALF, ycen = the fractional part.

Edge sweep, from both slit edges inwards: for the fibre at the edge, fit its profile to the pixels on
the outward side only, model the whole fibre, subtract, and the next fibre becomes the edge.
  flat sweep     on an all-fibre flat; profile symmetric about the trace centre by construction,
                 linear between nodes 0.1 px apart in |u - c| out to one pitch, flat spectrum a
                 low-order polynomial along the order; the profiles then feed the Fig 14 column peel, iterated to convergence
  science sweep  on the science frame with slitdec: sL and sP fitted on the outward half, sL mirrored
                 about the centre, sP refitted; optional refinement passes with free sL and sP

Lessons that shaped it:
  - the outward half must stop at one pitch: beyond it lies the previous fibre's subtraction residual,
    which a mirror copies onto the next fibre's centre and the sweep diverges
  - a free spectrum per column absorbs profile errors that vary with pixel phase, so a one-sided
    profile is only determined with a smooth spectrum: fine on a flat, not on a science frame
  - slitdec zeroes sP within delta_x of the swath ends; with a tilted slit that biases sL
    asymmetrically, so the end columns are masked
  - the centre cannot be measured from one side; it has to come from the trace

uv run --python 3.13 scripts/exp_ifu_edge_sweep.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import charslit  # noqa: E402
import exp_ifu_crosstalk as E  # noqa: E402
import exp_ifu_kernel_peel as KP  # noqa: E402
import fibreslit_sim as fs  # noqa: E402
import slitkernel as sk  # noqa: E402

N, NC = 12, E.NC
NPH = 10  # fibreslit_sim renders on a 0.1 px grid: phases in 0.1 px steps
WRAPS = 4
EDGE = 4  # swath-end columns left out of every fit
STRIP = 12  # template columns rendered either side of the line
NSUB = 200  # samples per pixel for the pixel averages of the profile basis


def spectra():
    """Lines only in fibres 0, 4, 8: each odd fibre 1-9 has exactly one lit neighbour."""
    S = np.full((N, NC), E.CONT)
    for f in (0, 4, 8):
        S[f, E.LINE_X] += E.LINE_AMP
    return S


# (fibre, lit fibres it is measured against): the odd fibres sit next to one lit fibre, 2, 6 and 10 two places from
# lit ones, where a peel stopped after two passes leaves its second-order residual
TARGETS = [(1, [0]), (3, [4]), (5, [4]), (7, [8]), (9, [8]), (2, [0, 4]), (6, [4, 8]), (10, [8])]
HW = 10


def crosstalk(out):
    """Share of each target fibre's flux from one lit neighbour (DRL Fig 13), in %, all fibres equally bright."""
    x = []
    for k, lit in TARGETS:
        leak = sum((out[k, x0 - HW : x0 + HW + 1] - np.median(out[k])).sum() for x0 in E.LINE_X)
        line = np.mean([sum((out[n, x0 - HW : x0 + HW + 1] - np.median(out[n])).sum() for x0 in E.LINE_X) for n in lit])
        c = leak / line / len(lit)
        x.append(c / (1 + 2 * c) * 100)
    return np.array(x)


def templates(geom, psf, dy=0.0):
    pitch, diam = geom["pitch"], geom["diam"]
    nrows = int(np.ceil((N - 1) * pitch + 2 * diam + 14))
    y0 = (nrows - (N - 1) * pitch) / 2 + 0.002 + dy
    T = np.zeros((N, nrows, NC))
    for k in range(N):
        fib = fs.make_fibres(N, pitch=pitch, diam=diam, nrows=nrows, y0=y0, **psf(N))
        for j, f in enumerate(fib):
            f["flux"] = 1.0 if j == k else 0.0
        T[k][:, E.X0 - STRIP : E.X0 + STRIP + 1] = fs.render(2 * STRIP + 1, nrows, fib, [(float(STRIP), 1.0)],
                                                              tilt=geom["tilt"], ycentre=nrows / 2 + 1e-3)
    return T, y0 - dy


class Frames:
    """Per-fibre images; column x is taken from the template rendered at trace offset ph(x) / NPH."""

    def __init__(self, geom, psf):
        self.geom = geom
        rendered = [templates(geom, psf, dy=k / NPH) for k in range(NPH)]
        self.Tk = np.array([T for T, _ in rendered])
        self.nrows = self.Tk.shape[2]
        self.ph = (np.arange(NC) * NPH * WRAPS // NC) % NPH
        self.delta = self.ph / NPH
        self.centres = rendered[0][1] + np.arange(N) * geom["pitch"]

    def image(self, f, spec):
        imgs = [E.image_of(self.Tk[k][f], spec) for k in range(NPH)]
        return np.stack([imgs[self.ph[x]][:, x] for x in range(NC)], axis=1)


class Crops:
    def __init__(self, fr, weights):
        self.fr = fr
        self.half = int(np.ceil(1.6 * fr.geom["pitch"]))
        self.kw = dict(KP.KW, **({"weights": "unc"} if weights == "variance" else {}))
        self.sc = np.zeros((NC, 2))
        self.sc[:, 1] = fr.geom["tilt"]
        self.rr = np.arange(2 * self.half)
        self.c = []
        for f in range(N):
            y = fr.centres[f] + fr.delta
            self.c.append(dict(r0=np.floor(y).astype(int) - self.half, frac=y - np.floor(y)))

    def idx(self, f):
        return self.c[f]["r0"][None, :] + self.rr[:, None], np.arange(NC)[None, :]

    def cut(self, img, f):
        return img[self.idx(f)]

    def paste(self, crop, f, shape):
        out = np.zeros(shape)
        out[self.idx(f)] = crop
        return out

    def urow(self, f):
        """Slit coordinate u of each crop pixel centre; the fibre centre is at u = half."""
        return (self.rr + 0.5)[:, None] - self.c[f]["frac"][None, :]

    def dec(self, d, mask, f, ref, **extra):
        mask = mask.copy()
        mask[:, :EDGE] = 0
        mask[:, -EDGE:] = 0
        return charslit.slitdec(d, np.sqrt(np.abs(ref) + E.RN**2), mask, self.c[f]["frac"].copy(), self.sc,
                                np.zeros(d.shape[0]), **self.kw, **extra)


def trace_centres(fr, cr):
    """Fibre centres in u from each fibre's own flat: a perfect trace."""
    uc = []
    for f in range(N):
        fl = cr.cut(fr.image(f, np.full(NC, 2e3)), f)
        sL = cr.dec(fl, np.ones(fl.shape, np.uint8), f, fl)["slitfunction"]
        u = sk.subpix_u(len(sL), cr.kw["osample"])
        uc.append(float((u * sL).sum() / sL.sum()))
    return np.array(uc), u


def sweep_order():
    return [(f, True) for f in range(N // 2)] + [(f, False) for f in range(N - 1, N // 2 - 1, -1)]


def outward(cr, f, c0, low, inner, L):
    u = cr.urow(f)
    return (u <= c0 + inner) & (u >= c0 - L) if low else (u >= c0 - inner) & (u <= c0 + L)


# ---------------------------------------------------------------- flat sweep, symmetric profile


def sym_basis(cr, f, c0, L, dt):
    """B[row, col, node]: pixel average of the hat function of each node in t = |u - c0|.
    Depends on the column only through ycen's fractional part, so it is built once per phase."""
    nodes = np.arange(0, L + dt / 2, dt)
    s = (np.arange(NSUB) + 0.5) / NSUB
    ph, inv = np.unique(np.round(cr.c[f]["frac"], 9), return_inverse=True)
    Bp = np.empty((len(cr.rr), len(ph), len(nodes)))
    for i, p in enumerate(ph):
        t = np.abs(cr.rr[:, None] + s[None, :] - p - c0)
        hat = np.clip(1 - np.abs(t[..., None] - nodes) / dt, 0, None)
        hat[t > L] = 0
        Bp[:, i] = hat.mean(1)
    return Bp[:, inv]


def sym_fit(d, w, B, sdeg=2, niter=15):
    """d ~ s(x) * (B p): s a polynomial of degree sdeg in x, p the symmetric profile; alternating LS."""
    nr, nc, nk = B.shape
    p = np.exp(-0.5 * (np.arange(nk) / (nk / 2)) ** 2)
    xx = np.linspace(-1, 1, nc)
    P = np.polynomial.polynomial
    for _ in range(niter):
        prof = B @ p
        den = (w * prof**2).sum(0)
        ok = den > 0
        s = (w * d * prof).sum(0) / np.where(ok, den, 1)
        # a free s per column would absorb profile errors that vary with the pixel phase
        s = P.polyval(xx, P.polyfit(xx[ok], s[ok], sdeg, w=np.sqrt(den[ok])))
        A = (B * s[None, :, None]).reshape(-1, nk) * np.sqrt(w).reshape(-1, 1)
        H = A.T @ A
        p = np.linalg.solve(H + 1e-6 * np.trace(H) / nk * np.eye(nk), A.T @ (d * np.sqrt(w)).reshape(-1))
    return s, B @ p


def flat_sweep(flat, cr, uc, duc=0.0, support=1.0, inner=0.5, dt=0.1):
    """Per-fibre flat images from an all-fibre flat; uniform weights (the flat is bright everywhere)."""
    duc = np.broadcast_to(duc, (N,))
    L = support * cr.fr.geom["pitch"]
    resid, models = flat.copy(), np.zeros((N,) + flat.shape)
    for f, low in sweep_order():
        c0 = uc[f] + duc[f]
        u = cr.urow(f)
        # whole pixels on the outward side, up to `inner` px past the centre
        keep = ((u + 0.5 <= c0 + inner) & (u - 0.5 >= c0 - L)) if low else ((u - 0.5 >= c0 - inner) & (u + 0.5 <= c0 + L))
        keep = keep.astype(float)
        keep[:, :EDGE] = 0
        keep[:, -EDGE:] = 0
        s, prof = sym_fit(cr.cut(resid, f), keep, sym_basis(cr, f, c0, L, dt))
        models[f] = cr.paste(prof * s[None, :], f, flat.shape)
        resid -= models[f]
    models[:, :, :EDGE] = models[:, :, EDGE : EDGE + 1]
    models[:, :, -EDGE:] = models[:, :, -EDGE - 1 : -EDGE]
    return models


# ---------------------------------------------------------------- science sweep, slitdec + mirror


def mirror(sL, u, c0, keep_low):
    src = u <= c0 if keep_low else u >= c0
    out = sL.copy()
    out[~src] = np.interp(2 * c0 - u[~src], u, sL, left=0, right=0)  # mirrored points lie on the kept side
    return out


def science_sweep(im, cr, uc, u, refine=0, support=1.0):
    L = support * cr.fr.geom["pitch"]
    resid, models, spec = im.copy(), np.zeros((N,) + im.shape), np.zeros((N, NC))
    for f, low in sweep_order():
        d, ref = cr.cut(resid, f), cr.cut(im, f)
        mask = outward(cr, f, uc[f], low, 0.5, L).astype(np.uint8)
        r = cr.dec(d, mask, f, ref)
        sLm = mirror(r["slitfunction"], u, uc[f], low)
        sLm[np.abs(u - uc[f]) > L] = 0
        r = cr.dec(d, mask, f, ref, preset_slitfunc=sLm)
        models[f] = cr.paste(r["model"], f, im.shape)
        spec[f] = r["spectrum"]
        resid -= models[f]
    for _ in range(refine):
        tot, new = models.sum(0), np.zeros_like(models)
        for f in range(N):
            d = cr.cut(im - (tot - models[f]), f)
            r = cr.dec(d, np.ones(d.shape, np.uint8), f, cr.cut(im, f))
            new[f] = cr.paste(r["model"], f, im.shape)
            spec[f] = r["spectrum"]
        models = new
    return spec


# ---------------------------------------------------------------- experiment


def round_psf(s0):
    return lambda n: dict(psf_sx=s0, psf_sy=s0)


def yskew_psf(s0):
    """Core with 6 % of its light in a satellite 1.5 sigma along the slit: an asymmetric profile."""
    return lambda n: dict(psf_sx=s0, psf_sy=s0, psf_rot=np.pi / 2, skew_eps=0.06, skew_d=1.5 * s0)


def run(geom, psf, nrand=4, seed=3):
    """{weights: {case: mean |crosstalk| in %}} for one arm and PSF."""
    fr = Frames(geom, psf)
    Sp = spectra()
    im = sum(fr.image(f, Sp[f]) for f in range(N))
    flat = sum(fr.image(f, np.full(NC, 2e3)) for f in range(N))
    Tmask = np.array([fr.image(f, np.ones(NC)) for f in range(N)])
    cr0 = Crops(fr, "uniform")
    uc, u = trace_centres(fr, cr0)
    rng = np.random.default_rng(seed)
    prof = {"flat sweep": flat_sweep(flat, cr0, uc)}
    for dc in (0.005, 0.01, 0.02):
        prof[f"centres off by {dc} px"] = flat_sweep(flat, cr0, uc, duc=dc)
    prand = [flat_sweep(flat, cr0, uc, duc=rng.normal(0, 0.01, N)) for _ in range(nrand)]
    out = {}
    for w in ("uniform", "variance"):
        ww = np.ones_like(im) if w == "uniform" else 1 / (np.abs(im) + E.RN**2)
        mx = lambda s: float(np.abs(crosstalk(s)).mean())  # noqa: E731
        r = {"single": mx(E.extract_single(im, Tmask, ww)), "masked flats": mx(E.extract_peel(im, Tmask, ww, npass=None))}
        r.update({k: mx(E.extract_peel(im, P, ww, npass=None)) for k, P in prof.items()})
        r["centres off by 0.01 px rms"] = float(np.mean([mx(E.extract_peel(im, P, ww, npass=None)) for P in prand]))
        cr = Crops(fr, w)
        r["science sweep"] = mx(science_sweep(im, cr, uc, u))
        r["science sweep + 2 passes"] = mx(science_sweep(im, cr, uc, u, refine=2))
        out[w] = r
    return out


GEOMS = {"R": (dict(pitch=5.5, diam=5.4, tilt=0.05), 0.8), "YJH": (dict(pitch=2.5, diam=2.5, tilt=0.05), 0.6)}


def main():
    for g, (geom, s0) in GEOMS.items():
        for pname, psf in (("round PSF", round_psf(s0)), ("y-skewed core", yskew_psf(s0))):
            for w, r in run(geom, psf).items():
                print(f"\n=== {g} | {pname} | {w} weights: crosstalk after extraction")
                for k, v in r.items():
                    print(f"  {k:32s} {v:7.3f} %")


if __name__ == "__main__":
    main()
