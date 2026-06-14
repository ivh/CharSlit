"""Self-contained demo: does dispersion-direction oversampling (osamp_spec)
recover sub-pixel line structure that osamp_spec=1 throws away?

Builds a synthetic, *tilted*, *under-sampled* emission line and extracts it at
osamp_spec = 1, 2, 3 directly through charslit. The slit tilt is what makes
super-resolution possible: each detector row samples the line at a slightly
different sub-pixel phase, so the fine grid is constrained where the osamp=1
block-mean staircase would alias it away.

Run:
    uv run compare_osamp.py                 # default tilted under-sampled line
    uv run compare_osamp.py --sigma 0.4 --slope 0.15
    SHOW=0 uv run compare_osamp.py          # save PNG only, no window

Outputs compare_osamp.png and prints a recovered-FWHM table. If the FWHM at
s>1 is meaningfully narrower than at s=1, the tilt carried real super-resolution
content. On data with little tilt / well-sampled lines the rows collapse onto
the same FWHM, matching the OSAMP_SPEC.md finding.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import charslit


def make_tilted_line(nrows, ncols, osample, sigma_disp, slope, line_x0,
                     amp=2000.0, read_noise=2.0, seed=0):
    """A single narrow emission line tilted across the slit.

    Row j has its line centre at line_x0 + slope*(j - jc): the tilt is centred
    on the slit-centre row jc, which is where slitdec references the extracted
    spectrum, so the recovered peak lands at line_x0. The linear tilt
    coefficient is c1 = slope. The dispersion profile is a Gaussian of width
    sigma_disp, integrated over each detector pixel by rendering on a Q-times
    finer grid and binning.
    """
    Q = 50
    sub = (np.arange(ncols * Q) + 0.5) / Q  # sub-pixel column centres
    jc = (nrows - 1) / 2.0
    slit = np.exp(-0.5 * ((np.arange(nrows) - jc) / (nrows / 5.0)) ** 2)

    im = np.zeros((nrows, ncols))
    for j in range(nrows):
        centre = line_x0 + slope * (j - jc)
        g = np.exp(-0.5 * ((sub - centre) / sigma_disp) ** 2)
        pix = g.reshape(ncols, Q).mean(axis=1)  # integrate over each pixel
        im[j] = amp * slit[j] * pix

    rng = np.random.RandomState(seed)
    im += rng.normal(0.0, read_noise, im.shape)

    ny = osample * (nrows + 1) + 1
    data = {
        "im": im,
        "pix_unc": np.sqrt(np.abs(im) + read_noise ** 2),
        "mask": np.ones(im.shape, dtype=np.uint8),
        "ycen": np.full(ncols, 0.5),
        "slitcurve": np.zeros((ncols, 3)),
        "slitdeltas": np.zeros(ny),
        "osample": osample,
        "ncols": ncols,
    }
    data["slitcurve"][:, 1] = slope  # tell slitdec about the tilt
    return data


def extract(data, s, lambda_sP, lambda_fringe):
    r = charslit.slitdec(
        data["im"], data["pix_unc"], data["mask"].copy(), data["ycen"],
        data["slitcurve"], data["slitdeltas"],
        osample=data["osample"], osamp_spec=s,
        lambda_sP=lambda_sP, lambda_fringe=lambda_fringe,
    )
    assert r["return_code"] == 0, f"slitdec failed at osamp_spec={s}"
    sp = r["spectrum"]
    xf = (np.arange(sp.size) + 0.5) / s  # fine-bin centres in detector pixels
    return xf, sp


def fwhm(xf, sp, x0, win=6.0):
    """FWHM (in detector pixels) of the peak near x0, by half-max crossings."""
    sel = np.abs(xf - x0) < win
    x, y = xf[sel], sp[sel].astype(float)
    if y.max() <= 0:
        return np.nan
    y = y - np.median(y)  # drop baseline
    pk = y.argmax()
    half = y[pk] / 2.0

    def cross(idx_range):
        for a, b in idx_range:
            if (y[a] - half) * (y[b] - half) < 0:
                t = (half - y[a]) / (y[b] - y[a])
                return x[a] + t * (x[b] - x[a])
        return np.nan

    left = cross([(i - 1, i) for i in range(pk, 0, -1)])
    right = cross([(i, i + 1) for i in range(pk, len(x) - 1)])
    return right - left if np.isfinite(left) and np.isfinite(right) else np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nrows", type=int, default=30)
    p.add_argument("--ncols", type=int, default=60)
    p.add_argument("--osample", type=int, default=6)
    p.add_argument("--sigma", type=float, default=0.45,
                   help="line sigma in detector pixels (FWHM=2.355*sigma)")
    p.add_argument("--slope", type=float, default=0.15,
                   help="tilt c1 in pixels/row; larger = more phase diversity")
    p.add_argument("--lambda-sP", type=float, default=0.0)
    p.add_argument("--lambda-fringe", type=float, default=0.0)
    p.add_argument("--osamps", type=int, nargs="+", default=[1, 2, 3])
    args = p.parse_args()

    line_x0 = args.ncols / 2 + 0.37  # deliberately sub-pixel
    data = make_tilted_line(args.nrows, args.ncols, args.osample,
                            args.sigma, args.slope, line_x0)
    true_fwhm = 2.3548 * args.sigma

    print(f"tilted line: sigma={args.sigma} (true FWHM={true_fwhm:.3f} px), "
          f"slope={args.slope} px/row, tilt span={args.slope*(args.nrows-1):.2f} px")
    print(f"lambda_sP={args.lambda_sP}, lambda_fringe={args.lambda_fringe}\n")
    print(f"{'osamp_spec':>10}  {'recovered FWHM [px]':>20}")

    curves = {}
    for s in args.osamps:
        xf, sp = extract(data, s, args.lambda_sP, args.lambda_fringe)
        fw = fwhm(xf, sp, line_x0)
        curves[s] = (xf, sp, fw)
        print(f"{s:>10}  {fw:>20.3f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    truth_x = np.linspace(line_x0 - 6, line_x0 + 6, 400)
    truth_y = np.exp(-0.5 * ((truth_x - line_x0) / args.sigma) ** 2)
    ax.plot(truth_x, truth_y, color="0.6", lw=2, ls="--", label="true line profile")
    for s in args.osamps:
        xf, sp, fw = curves[s]
        sel = np.abs(xf - line_x0) < 6
        y = sp[sel] / np.nanmax(sp[sel])
        ax.plot(xf[sel], y, marker="o", ms=3, lw=1.2,
                label=f"osamp_spec={s} (FWHM {fw:.2f} px)")
    ax.axvline(line_x0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("dispersion [detector pixel]")
    ax.set_ylabel("extracted spectrum (peak-normalised)")
    ax.set_title(f"osamp_spec super-resolution on a tilted under-sampled line "
                 f"(true FWHM={true_fwhm:.2f} px)")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_osamp.png")
    fig.savefig(out, dpi=120)
    print(f"\nSaved {out}")
    if os.environ.get("SHOW", "1") != "0":
        plt.show()


if __name__ == "__main__":
    main()
