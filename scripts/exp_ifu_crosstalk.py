"""IFU crosstalk: per-column fixed-profile extraction vs spectro-perfectionism (SP).

Overlapping fibres carry different spectra: even fibres have emission lines, odd fibres
continuum only. Whatever line structure shows up in an odd fibre's extracted spectrum is
leaked from its neighbours. Everything is built from per-fibre templates T_f (the 2D image
of a unit line through fibre f alone, rendered by fibreslit_sim), so data, masked flats
and SP's A matrix come from one consistent forward model.

Methods
  single    each fibre alone: profile-weighted optimal extraction, neighbours ignored.
  peel      DRL v2.0 Fig 14: single, then subtract the neighbours' models (their
            single-pass spectra times their masked-flat profiles) and re-extract.
  column    per-column weighted LS for all fibre amplitudes with fixed profiles
            P_f(y; x) from per-fibre (masked) flats. Output = spectrum * fibre LSF.
  SP exact  A built from the true T_f; f-hat smoothed with one common Gaussian.
  SP <err>  A built from a PSF with a realistic error (named per scenario).

Leak metrics, per odd fibre and averaged over its neighbours' line positions:
  peak  max |leak| / peak of the neighbours' lines in the same method's output
  flux  integrated leak / integrated line flux (signed)

uv run --python 3.13 scripts/exp_ifu_crosstalk.py
"""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fibreslit_sim as fs  # noqa: E402

NC = 240
RN = 10.0
X0 = NC // 2
LINE_X = np.arange(20, NC - 20, 25)  # emission lines in even fibres (integer columns)
LINE_AMP = 3e5
CONT = 2e3


def templates(geom, psf, nfib):
    pitch, diam = geom["pitch"], geom["diam"]
    nrows = int(np.ceil((nfib - 1) * pitch + 2 * diam + 12))
    T = []
    for k in range(nfib):
        fib = fs.make_fibres(nfib, pitch=pitch, diam=diam, nrows=nrows, **psf(nfib))
        for j, f in enumerate(fib):
            f["flux"] = 1.0 if j == k else 0.0
        T.append(fs.render(NC, nrows, fib, [(float(X0), 1.0)], tilt=geom["tilt"], ycentre=nrows / 2 + 1e-3))
    return np.array(T), nrows


def image_of(Tf, spec):
    """Image of spectrum `spec` (per integer column) through fibre template Tf."""
    return fftconvolve(Tf, spec[None, :], mode="full")[:, X0 : X0 + NC]


def spectra(nfib):
    S = np.full((nfib, NC), CONT)
    for k in range(0, nfib, 2):
        S[k, LINE_X] += LINE_AMP
    return S


def extract_column(im, Tflat, w):
    """Per-column LS for all fibre amplitudes with fixed profiles from the flats."""
    nfib = len(Tflat)
    P = Tflat / Tflat.sum(1, keepdims=True)  # (nfib, nrows, ncols) normalised per column
    out = np.zeros((nfib, NC))
    for x in range(NC):
        A = P[:, :, x].T * np.sqrt(w[:, x])[:, None]
        b = im[:, x] * np.sqrt(w[:, x])
        out[:, x] = np.linalg.lstsq(A, b, rcond=None)[0]
    return out


def extract_single(im, Tflat, w, sub=None):
    P = Tflat / Tflat.sum(1, keepdims=True)
    if sub is None:
        sub = np.zeros((len(P),) + im.shape)
    return np.array([(P[f] * w * (im - sub[f])).sum(0) / (P[f] ** 2 * w).sum(0) for f in range(len(P))])


def extract_peel(im, Tflat, w):
    P = Tflat / Tflat.sum(1, keepdims=True)
    s1 = extract_single(im, Tflat, w)
    model = s1[:, None, :] * P
    return extract_single(im, Tflat, w, sub=model.sum(0)[None] - model)


def build_A(T):
    nfib, nrows, _ = T.shape
    rows, cols, vals = [], [], []
    for f in range(nfib):
        r_, c_ = np.nonzero(T[f] > 1e-9 * T[f].max())
        v = T[f][r_, c_]
        for k in range(NC):
            c = c_ + (k - X0)
            ok = (c >= 0) & (c < NC)
            rows.append(r_[ok] * NC + c[ok])
            cols.append(np.full(ok.sum(), f * NC + k))
            vals.append(v[ok])
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nrows * NC, nfib * NC))


def extract_sp(im, T, w):
    A = build_A(T)
    W = sp.diags(w.ravel())
    AtWA = (A.T @ W @ A).toarray()
    fhat = np.linalg.solve(AtWA + 1e-12 * np.trace(AtWA) / len(AtWA) * np.eye(len(AtWA)), A.T @ (w.ravel() * im.ravel()))
    return fhat.reshape(len(T), NC)


def leak_metrics(out, nfib, smooth=None):
    if smooth is not None:
        out = np.array([np.convolve(o, smooth, mode="same") for o in out])
    peak, flux = [], []
    for k in range(1, nfib - 1, 2):  # odd fibres with two line neighbours
        base = np.median(out[k])
        for x in LINE_X:
            win = slice(x - 8, x + 9)
            leak = out[k, win] - base
            line = 0.5 * (out[k - 1, win] + out[k + 1, win]) - 0.5 * (np.median(out[k - 1]) + np.median(out[k + 1]))
            peak.append(np.max(np.abs(leak)) / np.max(line))
            flux.append(leak.sum() / line.sum())
    return float(np.median(peak)), float(np.median(flux))


def crosstalk_fraction(Tflat, geom, nrows, nfib):
    """Share of fibre k's flux that lands within +-pitch/2 of fibre k+1's centre row."""
    k = nfib // 2
    prof = Tflat[k][:, X0]
    prof = prof / prof.sum()
    y = np.arange(nrows) + 0.5
    y0 = (nrows - (nfib - 1) * geom["pitch"]) / 2
    ck1 = y0 + (k + 1) * geom["pitch"]
    return float(prof[np.abs(y - ck1) < geom["pitch"] / 2].sum())


def run(geom, truth, errors, nfib=12, label=""):
    T, nrows = templates(geom, truth, nfib)
    S = spectra(nfib)
    im = sum(image_of(T[f], S[f]) for f in range(nfib))
    w = 1.0 / (np.abs(im) + RN**2)
    Tflat = np.array([image_of(T[f], np.ones(NC)) for f in range(nfib)])
    G = np.exp(-0.5 * (np.arange(-6, 7) / 1.2) ** 2)
    G /= G.sum()
    res = {"single fibre (neighbours ignored)": leak_metrics(extract_single(im, Tflat, w), nfib)}
    res["peel (DRL Fig 14)"] = leak_metrics(extract_peel(im, Tflat, w), nfib)
    res["column (masked-flat profiles)"] = leak_metrics(extract_column(im, Tflat, w), nfib)
    res["SP, exact PSF"] = leak_metrics(extract_sp(im, T, w), nfib, G)
    for name, psf in errors.items():
        Te, _ = templates(geom, psf, nfib)
        res[f"SP, {name}"] = leak_metrics(extract_sp(im, Te, w), nfib, G)
    xt = crosstalk_fraction(Tflat, geom, nrows, nfib)
    print(f"\n=== {label}   (crosstalk into the neighbour's core rows: {xt * 100:.1f} %)")
    print(f"  {'method':34s} {'peak leak':>10s} {'flux leak':>10s}")
    for k, (pk, fl) in res.items():
        print(f"  {k:34s} {pk * 100:9.3f}% {fl * 100:+9.3f}%")
    return xt, res


def scenarios(s0):
    """(truth, {error name: assumed PSF}) per scenario; s0 = core PSF sigma of the geometry."""
    lin = lambda n: np.linspace(0, 1, n)  # noqa: E731
    avg = lambda n: dict(psf_sx=s0, psf_sy=s0)  # noqa: E731
    return {
        "disks only (round PSF)": (avg, {}),
        "PSF x-width + skew trend": (
            lambda n: dict(psf_sx=s0 * (0.75 + 0.5 * lin(n)), psf_sy=s0, skew_eps=0.06 * lin(n), skew_d=1.5 * s0),
            {"slit-average PSF": avg},
        ),
        "rotated elliptical core (30 deg)": (
            lambda n: dict(psf_sx=1.25 * s0, psf_sy=0.75 * s0, psf_rot=np.radians(30)),
            {
                "rotation 25 deg": lambda n: dict(psf_sx=1.25 * s0, psf_sy=0.75 * s0, psf_rot=np.radians(25)),
                "slit-average PSF": avg,
            },
        ),
        "3% halo, elongated in x": (
            lambda n: dict(psf_sx=s0, psf_sy=s0, halo_frac=0.03, halo_sx=4.0, halo_sy=1.5),
            {
                "halo amplitude -30%": lambda n: dict(psf_sx=s0, psf_sy=s0, halo_frac=0.021, halo_sx=4.0, halo_sy=1.5),
                "halo ignored": avg,
            },
        ),
    }


def main():
    geoms = {
        "R-like: fibres 5.4 px, PSF sigma 0.8": (dict(pitch=5.5, diam=5.4, tilt=0.05), 0.8),
        "YJH-like: fibres 2.5 px, PSF sigma 0.6": (dict(pitch=2.5, diam=2.5, tilt=0.05), 0.6),
    }
    for gname, (geom, s0) in geoms.items():
        for sname, (truth, errs) in scenarios(s0).items():
            run(geom, truth, errs, label=f"{gname} | {sname}")


if __name__ == "__main__":
    main()
