# Pixel weights in slitdec (`weights=`)

`slitdec` decomposes a swath into a spectrum sP (one value per column) and a slit function sL
(one value per subpixel row) by alternating two linear least-squares solves. Since commit
`328e699` both solves can weight the detector pixels:

```python
charslit.slitdec(im, pix_unc, mask, ycen, slitcurve, slitdeltas,
                 weights="none",      # "none" | "unc" | "model" | "flux"
                 noise_floor=None)    # only for "model"
```

The same options exist in the ports: `extract_weights` in the cr2res recipes and
`extraction_weights` / `extraction_noise_floor` in PyReduce. All weights are scaled to mean 1
over the usable pixels, so `lambda_sL` keeps its meaning, and they act in both solves, on the flat
fast path as well as on the general (curved) path.

## Why the default is not optimal at high S/N

`weights="none"` is plain unweighted least squares, the historic solution, and it stays
bit-identical to it. It ignores `pix_unc` completely.

Look at one column with a fixed slit profile P_y (the share of the column's light in row y) and
data D_y = f P_y + noise. The estimate of the column's flux f with pixel weights w_y is

    f_hat = sum_y w_y P_y D_y / sum_y w_y P_y^2

This has two consequences.

**Noise.** At high S/N the noise is photon noise, Var(D_y) ∝ f P_y. The optimal weights are
then w_y ∝ 1/(f P_y), and f_hat becomes sum D_y / sum P_y: a plain column sum. With w_y = 1
instead, the variance is larger by

    (sum P^3)(sum P) / (sum P^2)^2  >=  1     (Cauchy-Schwarz)

That is 2/√3 ≈ 1.15 for a Gaussian profile, a 15 % variance penalty. On bright ESPRESSO LFC
frames it was 8 %. Unweighted least squares is only optimal when all pixels have the same variance,
i.e. when read noise or background dominates, which is the low-S/N regime.

**Line profile.** If the dispersion-direction profile L_y differs from row to row, the extracted
line profile is a weighted mix of the rows. The data contribute with weight w_y P_y, and D_y
itself scales with P_y, so

    none:            LSF ∝ sum_y P_y^2 L_y     (rows count with their flux squared)
    column sum:      LSF ∝ sum_y P_y   L_y     (rows count with their flux)

The default therefore favours the bright rows. On a fibre-fed slit the bright rows are the fibre
or slice centres, where the fibre chord is longest and the line widest. On ESPRESSO 2x1 LFC frames
(A·a, order 55) the brightest row has a Gaussian σ of 1.92 px, the edge rows 1.39–1.56 px. The
extracted line widths were:

| extraction | row mixing | σ |
|---|---|---|
| column sum | ∝ flux | 1.773 px |
| espdr (Horne, bright lines) | ≈ ∝ flux | 1.764 px |
| slitdec `"none"` | ∝ sL × flux | 1.807 px |

So R came out about 4–5 % too low with the default (order 55: 125.3k against 131.1k with
`"flux"` and 131.2k from espdr). Synthetic frames with identical rows cannot show this effect.

## The options

| `weights` | pixel weight | refreshed | what it gives |
|---|---|---|---|
| `"none"` (default) | 1 | – | historic solution; noise-optimal only at low S/N; LSF ∝ Σ P² L |
| `"unc"` | 1 / pix_unc² | static | true inverse variance, *if* pix_unc does not come from the noisy data |
| `"model"` | 1 / (a·max(model,0) + b) | every iteration | inverse variance from the smooth model; noise-optimal if pix_unc describes the noise |
| `"flux"` | 1 / v (slit-function value of the pixel) | every iteration | column-sum (bright-source) mixing at any S/N; LSF ∝ Σ P L |

`"model"` and `"flux"` need a model, so their first iteration is unweighted. With a preset slit
function and `kappa=0`, which otherwise needs only one pass, they run one extra pass.

### `"unc"`: static inverse variance

The weights are 1/pix_unc², fixed for the whole fit. This is right if pix_unc is the true
variance, for example from a model of everything in the swath. It is wrong if pix_unc was computed
from the data themselves, such as √|counts|. Pixels that fluctuate low then get too much weight
and those that fluctuate high too little. The weights correlate with the noise and distort the
line profiles: on real ESPRESSO LFC swaths R looked 1 % better, but the χ² of a 1D LSF fit
went from 29 (unweighted) to 169. Variances from the model instead gave χ² 24.5.

### `"model"`: model-variance weights

The weights come from slitdec's own model, which is smooth and does not carry the noise:

    w = 1 / (a · max(model, 0) + b)

a and b are refitted to pix_unc² at every refresh, as a straight line pix_unc² ≈ a·model + b over
the usable pixels (`variance_fit` in `slitdec.c`):

- **Why affine.** The decomposed image is usually background subtracted (nodding A−B, sky or
  scattered-light subtraction). The model then holds only the object, while the variance still
  contains the read noise and the shot noise of what was subtracted. b collects those, and
  a = 1/(gain · frames combined) scales the object's shot noise. a does *not* drop out next to b:
  it sets the point where the object's photon noise takes over from the floor. The first version
  used 1/(model + noise_floor), i.e. a = 1. On 100 averaged CRIRES+ A−B pairs that was about
  200× too steep and gave more noise and single-pixel spikes than `"none"`.
- **Why a robust fit.** Unmasked hot pixels have huge pix_unc and dominate a plain least-squares
  fit. In 17 % of the CRIRES+ swaths that fit collapsed to a = 0. The fit is instead a line through
  the medians of model and pix_unc² in up to 32 bins of equal pixel count along the model. Ties in
  the sort are broken by pix_unc², so the numpy and numba ports reproduce the bins exactly.
- **Limits.** a ≥ 0, and b is at least the 5th percentile of pix_unc². `noise_floor >= 0` fixes
  b (in the units of `im`) and only a is fitted. With constant or missing pix_unc the fit gives
  a = 0, i.e. uniform weights, the same as `"none"`.

In the photon-limited limit (a·model ≫ b) the weights become ∝ 1/model and the solution moves to
the column-sum behaviour above. In the read- or background-limited limit (a·model ≪ b) they are
uniform and the result is `"none"`. This is the inverse-variance (Horne-like) optimum, and it
carries two caveats:

1. **The LSF moves with S/N**, from Σ P L on bright data to Σ P² L on faint data. On a series
   of 19 ESPRESSO LFC frames, with fainter data simulated through the floor (faintness k relative
   to the real frames, peak ≈ 3·10⁴ e⁻/px), R in order 55 was 131.0k at k = 1, 130.3k at 10⁻²,
   128.0k at 10⁻³ and 125.8k at 10⁻⁴, ending at the unweighted 125.3k. The frame-to-frame scatter
   was ±0.1k. These frames were single exposures in electrons without a subtracted background, so
   a = 1 was the right scale there, and the old formula's results there stand.
2. **It is only as good as pix_unc.** The weights assume that pix_unc² is the real variance,
   including subtracted backgrounds and frame averaging. If the real scatter has other components
   (flat field, fringing, profile mismatch), the weights are wrong. MODEL also knows only the
   object it decomposes: light from anything else in the swath, such as neighbour fibres in an
   IFU crop, looks faint to it and gets large weights. In synthetic IFU peel tests that made the
   cross-talk leak worse (1.74 → 2.74 %), while true-variance `"unc"` reduced it to 0.58–0.89 %.

### `"flux"`: S/N-independent column-sum mixing

The weight is 1/v, where v is the pixel's slit-function value (model / spectrum), floored at 1e-3
of its maximum. Rows then enter sP in proportion to their flux, as in a column sum, at any S/N.
This gives the sharp bright-source LSF Σ P L independent of S/N, and needs no units and no
pix_unc. The price is noise when the data are not photon limited. Variance relative to the optimal
weights in the ESPRESSO series:

| k | 1 | 0.1 | 10⁻² | 10⁻³ | 10⁻⁴ |
|---|---|---|---|---|---|
| `"none"` | 1.08 | 1.08 | 1.05 | 1.02 | 1.00 |
| `"flux"` | 1.00 | 1.01 | 1.10 | 1.54 | 2.09 |

For a typical RV target (extracted S/N ~100 per pixel, k ≈ 0.05–0.1) the choice hardly matters:
`"model"` moves R by ~0.2 % against the bright calibration, `"flux"` costs ~1 % in variance. For
faint work (k ≈ 10⁻³–10⁻²) `"model"` changes R by 1–2.5 % between calibration and science, while
`"flux"` keeps it fixed at a 10–50 % variance cost. Only widths were measured; whether asymmetric
rows also shift line centroids with S/N under `"model"` is open. On
background-limited CRIRES+ nodding the A/B noise was 1.2–1.5× that of `"none"`, and the faintest
orders became unusable: wing pixels with v ≈ 0 get the largest weights.

## What real data said (CRIRES+ L/M nodding, cr2res, 2026-10)

Two data sets were reduced with each mode and compared by the A/B white noise (MAD of the first
difference of A/B) and by single-pixel spikes.

- **TWA 5, M band, 100 A−B pairs, background limited.** The fitted a·model/b at the profile peak
  is only 0.07–0.37 even in the brightest orders. MODEL is therefore within ~1 % of NONE, as
  expected. FLUX is clearly worse.
- **ω Car, L band, 11 pairs, much brighter relative to the background.** a·model/b at the peak
  reaches a median of ~6, so MODEL does lean on the core. Its A/B noise was still 0–10 % *above*
  NONE. The real pixel scatter there is ~16× what ERR (pix_unc) predicts, so the noise is not the
  photon noise that MODEL's weights assume.

So on real data with an incomplete error model, the expected high-S/N gain of MODEL did not
appear. The fitted a/b is physically right (a matched 1/(gain · NDIT · pairs) to ~10 % on both
data sets), so this is not a fitting problem.

## Recommendations

- Keep `"none"` as the default. It is robust, its LSF does not depend on S/N, and at low S/N it
  is the optimum anyway.
- Use `"model"` where pix_unc is a trustworthy variance and the swath contains only the object:
  bright, photon-limited data with a correct error budget. Check that it actually lowers the
  noise on your data.
- Use `"flux"` when an S/N-independent, sharp LSF matters more than noise, e.g. for LSF or
  resolution studies on bright sources.
- Use `"unc"` only with variances that do not come from the noisy data, e.g. from a model of
  everything in a crowded swath.
- A residual-based floor (variance of the faint pixels after an unweighted pass) does not work on
  real data: the faint pixels are gaps and slit edges, dominated by unmodelled scattered light.

## Sources

- ESPRESSO LSF and noise studies (2026-10): `~/pipes/espresso_lfc_20230124`, series script
  `analysis/snr_lsf.py`; 2022-11-24 LFC,LFC frames, fibre A slices a/b.
- CRIRES+ L/M comparisons (2026-10): cr2res branch `speed`, commits `ba5b9c2` (variance fit) and
  `598f5cf` (swath merge); data sets TWA 5 (M4368, 2025-03-20) and ω Car (L3340, 2024-03-18) in
  `~/CRIRES_LM`.
