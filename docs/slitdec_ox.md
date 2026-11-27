# slitdec_ox: slitdec with oversampling in x and y

`slitdec/slitdec.c` is N. Piskunov's slitdec_ox (https://github.com/nikolai-piskunov/slitdec_ox,
GPL-3.0 like the rest of SlitDec; its history is merged in), which he derived from our
`slitdec.c` (c878d00). It came back in November 2026 as the one engine, under the old file
and function names: the IDL REDUCE interface moved to `idl/` (see idl/README.md), the
ctypes wrapper was removed,
`lambda_sP` made an argument, `weights='model'` brought up to 5e38be4
(a·model + b), and `out_x` and the propagated uncertainties added. The C function
`slitdec()` keeps its 21 historic arguments in the old order; the new ones follow.

**The defaults are the historic slitdec**: `osample_x=1`, no `mu_x`. A plain call gives the
old result except that the tilt is evaluated at the y-subpixel centre (see "Tilt reference
point" below): untilted swaths are bit-identical, tilted ones shift by −c1/(2·osample).
Everything below is opt-in.

With `osample_x > 1` the spectrum is solved on that many fine bins per column. Each
y-subpixel of the slit function, shifted by the curvature, contributes to a detector pixel
by the overlap of each shifted fine bin with that pixel (box integration). The spectrum
penalty is a k-th difference in physical units, `mu_x * F * ∫ (d^k sP/dx^k)^2 dx`, with F
the median data curvature per pixel, so `mu_x` does not depend on `osample_x`, flux or
weights. `spectrum` stays on the detector grid (mean of the fine bins = what a vertical
slit records); `spectrum_fine` is the fine solution.

Recommended when the slit is tilted or lines are under-sampled:
`slitdec.slitdec(..., osample_x=8)`.

New arguments of `slitdec.slitdec`:
- `osample_x=1`: fine bins per column.
- `mu_x=None`: penalties on the 1st, 2nd, ... derivative of the spectrum. None means no
  penalty at `osample_x=1` and `(0, 0, 1.64e-4)` otherwise.
- `mu_y=None`: the same for the slit function; None = `lambda_sL`.
- `y_lower_lim=None`: trace row in the swath; None = nrows // 2.
- `out_x=None`: n (dividing `osample_x`) adds `spectrum_out`, the spectrum on n bins per
  column (mean of adjacent fine bins, nested in the detector pixels).
- `uncertainty='residual'`, `nlag=2`: see "Uncertainties".

`lambda_sP` is the historic unscaled first-difference penalty, applied between fine bins.
Extra results: `spectrum_fine` (ncols·osample_x) and `info_ox` = [dropped zeta entries, sP
band half-width, damping x, resolution x [px], damping y, resolution y [px]]; with `out_x`
`spectrum_out`; with `uncertainty` other than `'residual'` `covariance` and `chi2`.

Measurements behind the recommendations here, and the results of November 2026:
[ox_experiments.md](ox_experiments.md).

## Choosing osample_x and mu_x (ox_geometry.py)

- **mu_x needs no geometry.** When the tilt spreads the row phases (|c_n| = |Σ q e^{2πi n s}|/Σq
  small for n = 1..3, true for shear ≳ 1 px), the aliases decouple and noise and
  transfer are independent of the tilt. Then var/mean = (1/2π)∫R(ν)² dν with
  R = B²/(B² + μν^{2k}); var/mean = 1 gives **μ = 1.64e-4 for k = 3** (1.6e-3 for k = 2,
  1.67e-5 for k = 4). Transfer at that μ: 0.997 at π/2, 0.956 at 3π/4, 0.72 at Nyquist
  (unavoidable, shared by ±π). For shear ≲ 0.5 px the output stays ~Poisson (0.94).
- **osample_x is set by the narrowest line, not by the tilt**: rms line error ≈
  0.025 (h/σ)², h = 1/osample_x. FWHM 4.5 px: 8 → 1e-4, 16 → 2.5e-5.
- **The penalty strength is a noise/resolution trade.** The default whitens the noise and is
  right for lines of FWHM ≳ 1.8 px and at low S/N. For lines narrower than a pixel at high
  S/N use mu3 ≈ 1e-6 (`mu_x=(0, 0, 1e-6)`) and read `spectrum_fine`; the extraction noise is
  then as for the linear split (1.7× in variance on a tilted slit). k = 3 is enough.
- **Never `mu_x=[]` with `osample_x > 1`**: patterns that repeat every pixel with zero mean
  are exact null modes, the solve does not converge.
- **`osample_x = 8` whenever it is more than 1**; smaller values are not better for anything
  measured. `mu_x=None` then gives the whitening penalty.
- **`out_x`**: 2 is enough for FWHM ≈ 1.8 px, lines narrower than a pixel need 4. The saved
  values are bin averages.
- **osample (y)**: no new requirement; the shift change over one subpixel, c1/osample,
  only adds c1²/(12 osample²) px² to the line variance (1e-6 relative for ANDES).

## Results (synthetic 400 × 136 swath, flat-topped slit, joint sL/sP fit, D3, μ = 1.64e-4)

Noise relative to a vertical slit; line bias = rms(sP − truth)/peak on a noise-free FP
frame (FWHM 4.5 px).

| shear 8 px, weights `none` | var/mean | lag-1 | line bias | peak | s/call |
|---|---|---|---|---|---|
| slitdec.c (linear) | 1.72 | −0.26 | 8.0e-3 | 1.0184 | 0.05 |
| osample_x = 4 | 1.07 | −0.04 | 4.8e-4 | 1.0011 | 0.14 |
| osample_x = 8 | 1.02 | −0.01 | 1.2e-4 | 1.0002 | 0.26 |
| osample_x = 16 | 1.01 | 0.00 | 2.7e-5 | 1.0000 | 0.52 |

Shear 2 and 16 px and `model` weights give the same picture (osample_x = 8: var/mean
1.02–1.04, bias 1e-4). The reported SIG (residual based) is ≈ √S in all cases, which
is now correct for osample_x ≥ 8 and 1.3× too small for slitdec.c.

Re-measured for shears of 0.5 to 16 px, and for a weak penalty, in ox_experiments.md,
section 2: the penalty whitens the noise, the fine bins remove the line bias.

## Uncertainties (`uncertainty=`)

`'residual'` (default, as slitdec): per column from data − model. It is ≈ √S whatever the
extraction noise is, so 1.3× too small on a tilted slit with the linear split or a weak
penalty. The cr2res PIPE-12658 formula (Horne, per column from pix_unc) has
the same blind spot (0.73-0.77 in the same test) and was not ported.

`'propagated'`: the pixel variances carried through the last spectrum solve,
C = N⁻¹ M N⁻¹ with N = AᵀWA + penalties and M = AᵀW V W A (`propagate_unc`). One banded
solve per output bin (y = N⁻¹b, var = yᵀMy, covariances yᵀMy′), each cut off where y has
decayed to 1e-10. V is `pix_unc²`, or the fitted a·model + b with `weights='model'`.
Returns `uncertainty` (detector grid), `covariance` (ncols·out_x, nlag+1) on the `out_x`
grid (lags 0..`nlag`, default 2) and `chi2`, the reduced χ² of the residuals against V.
`'scaled'` multiplies the variances by `chi2`, for pix_unc of unknown scale.

Validation (ox_experiments.md, section 9): reported σ / empirical scatter is 1.02-1.03 in
all cases, including those where `'residual'` gives 0.73-0.77, and the predicted lag-1
correlations match the measured ones to 0.01. Cost +2-3 % of the run time.

Real data (November 2026, `osample_x=1`, ported to cr2res and used by PyReduce): two
independent halves of a nodding sequence extracted separately, the scatter of their
difference against the reported σ (1 = right, > 1 = σ too small), median over orders:

| data | residual / Horne | propagated |
|---|---|---|
| CRIRES+ M4368, 200 × 20 s, thermal background dominates | Horne 1.04, flux-free orders 1.2-1.9 | 0.97 (flux-free 0.85-0.99) |
| CRIRES+ Y1029, 32 × 120 s, source dominates | Horne 1.36 | 1.13 |
| SOXS NIR nod, two A-B pairs (PyReduce) | | 1.27 with pix_unc from the A-B image, 1.00 with (A+B)/g + 2 RN²/g² |

The propagation itself holds on real data; the misses are in V. Per-column formulas (Horne,
residual) fail where the star dominates the variance and on tilted orders (Horne/propagated
1.11 untilted to 1.26 at the largest tilt in Y). In CRIRES+ Y the remaining 13 % is the
pipeline's photon-noise term; on the SOXS A-B frames pix_unc built from the difference image
lacked the sky and the second read noise. So pix_unc must be the variance of the data
passed in, not rebuilt from it. Details: cr2re-pipe `PIPE-12658_more_errorprop`, PyReduce
7851f7b.

Limits: the slit function is taken as known; the number is the noise of the regularised
estimate and says nothing about smoothing bias; and it is only as good as V. Use `'scaled'`
or check `chi2` when pix_unc is not a real variance. Input-masked pixels that the outlier
rejection re-enables enter with their pix_unc, see [../LATER.md](../LATER.md). A column
that no unmasked pixel contributes to gets NaN in `uncertainty` and `covariance` (it would
otherwise come out as the leakage through the penalty, near 0).

## Regression and tests

`tests/test_regression.py` and `tests/test_slitdec.py` are the files of the old engine (one
line added for the new result keys): the golden suite passes at the defaults (rtol 1e-10,
exact mask), and a one-off comparison against the old C was bit-identical for all four
weight modes. `tests/test_ox.py` covers what is new: tilt reference point, line bias,
`spectrum` = mean of `spectrum_fine`, sampling recovery, `out_x`, propagated uncertainties.

Swath edges: the outermost columns of the solution are unreliable, up to many times the
flux on real data. They are returned as solved, not zeroed. `result['margin']` (=
`info[4]`, also `slitdec.edge_margin()` before the solve) is the number of columns per
end not to use: the largest slit-image shift, rounded up, plus `SLITDEC_EDGE` = 2. Give
them zero weight when merging swaths and drop them at the order ends. The constant covers
the linear split; the untilted ox ripple reaches further (ox_experiments.md, section 3).

## Tilt reference point (fixed November 2026)

The historic slitdec.c, and cr2res (`cr2res_extract.c`, same loop), evaluate the curvature
at t = (lower edge of the y-subpixel) − ycen instead of the subpixel centre. Every line
then shifts by c1/(2·osample): 3.7 mpx for c1 = 8/136 at osample 8, scaling with
1/osample. It is smooth in x and calibrates out if science and calibration share osample
and tilt, but it changes with osample and with any tilt difference (~5 m/s-equivalent for
c1 ≈ 0.12 at osample 8 on ANDES). The port first kept it as `t_offset=0` (default) with
0.5 opt-in; now the centre is hard-wired and `t_offset` is gone, for every `osample_x`. `tests/test_ox.py::test_tilt_reference_point` checks that no shift is left.

Regression goldens before and after (osample 6 for the real frames; spectrum shift fitted
outside the margin, max differences relative to the spectrum peak):

| case | median c1 | −c1/(2·osample) [mpx] | fitted shift [mpx] | max diff | after shift |
|---|---|---|---|---|---|
| ANDES_R_FP1 | +0.079 | −6.6 | −6.8 | 1.6e-3 | 9e-5 |
| Rsim | +0.109 | −9.1 | −9.6 | 2.5e-3 | 4e-4 |
| Hsim (c2 ≠ 0) | +0.060 | −5.0 | −6.8 | 3.7e-3 | 1.1e-3 |
| CRIRES1 | −0.052 | +4.4 | +4.3 | 2.7e-3 | 3.1e-3 |
| CRIRES2 (c2 ≠ 0) | −0.056 | +4.7 | +5.3 | 3.2e-3 | 1.8e-3 |
| synthetic curved_slitdeltas (osample 4, 15 rows) | +0.010 | −1.3 | −2.1 | 1e-4 | 9e-5 |

The untilted cases (simple, masked, CRIRES_UNE_J, discontinuous, fixedslope, multislope)
are bit-identical. The remaining differences are the c2 term (the half-step also moves
c2·t²) and, on CRIRES1/2, 2 and 6 pixels that kappa clipping now treats differently.

## Sampling recovery (the former `osamp_spec` branch)

`osamp_spec` solved the spectrum on s fine bins per column to recover line shape below the
pixel scale (bisector/RV work on under-sampled lines). It used the same forward model, so
here that purpose is served by `spectrum_fine` (or `spectrum_out`) with a weak penalty; its
C code and `lambda_fringe` are not ported. The experiment scripts came along with
`osamp_spec=s` → `osample_x=s` and `mu3` grids in place of the `lambda_fringe` grids:
`scripts/compare_osamp.py`, `experiment_synthetic_lsf.py`, `experiment_binx2_crires.py`,
`analyse_wide_bisector.py`, `make_swath.py`, and new `ox_bis_study.py`. Background and
experiment design: [bisector_experiments.md](bisector_experiments.md); results:
ox_experiments.md, sections 4-8.

## Penalties: mu_k in x and y (penalty_design.py, sl_penalty_study.py)

`slitdec()` (`mu_x`, `mu_y` in Python) takes `mu_x[k-1]`, `mu_y[k-1]` (k = 1..15): penalty
sum_k mu_k F ∫ (d^k s)^2 in physical units, i.e. the filter 1/(1 + sum_k mu_k nu^2k),
nu in rad/px, independent of osample, osample_x, flux, weights (F = median diagonal
x os^2; coefficient on k-th differences mu_k · diag_med · os^(2k+1)). No mu_y: the
legacy lambda_sL (mean diagonal, 1st differences of subpixels). `info_ox[2..5]` holds
the damping ratio ζ = mu1/(2√mu2) and nominal resolution π/ν½ [px] for x and y.
ζ ≥ 1: the penalty kernel is a convolution of two exponentials, so it cannot ring.
For the legacy lambda_sL, info_ox[4] = −(equivalent mu1_y): REDUCE's lambda_sf = 2 at
osamp 10 is mu1_y = 1.9e-3, resolution 0.14 px, i.e. hardly any smoothing.

**x (penalty_design.py):** at the strength that whitens the noise (var/mean = 1),
all shapes (k = 1, 2, 3, critically damped 1+2) give the extracted pixel spectrum
the same ~2-2.5 % negative kernel lobes: in x the ringing comes from the pixel
band limit, not the penalty. k = 3 has the flattest transfer (0.997 at π/2 vs
0.956 for k = 1) and stays the default: mu3_x = 1.64e-4.

**y (sl_penalty_study.py, C code, 40 x 200 synthetic swaths, osample 10):**
penalties compared at equal nominal resolution r.
- Edges blurred to ~1 px (realistic): no penalty rings (≤ 0.5 %).
- Edges sharper than a pixel (0.25 px): every penalty rings with ~1 px period,
  4 % with good phase coverage, 10 % with none (|c1| = 1); the cause is information
  the pixels never recorded, and only smoothing (larger r) reduces it.
- Pure k = 2 is worse in both respects: still rings 4-7 % at r = 1 px where
  k = 1 drops to 0.4-3 %, and amplifies noise 2.5x at small r. This matches
  REDUCE experience that the 1st-derivative term must be kept.
- Critically damped 1+2 ≈ k = 1 within a few %: no measurable gain from mu2_y.
- Realistic 1 px edges: r = 0.3 px (mu1_y = (0.3/π)^2 = 9e-3) keeps the edge width of
  legacy (r ≈ 0.1-0.14) with 1.6x lower noise; r = 0.6 widens edges by 0.2 px.
So use k = 1 in y; mu1_y ≈ 1e-2 is a reasonable start. Not default yet (mu_y=None keeps
lambda_sL). Choosing r per swath from the data (GCV or discrepancy principle) is the
objective next step.

## Not done

- Real frames: only the golden-suite FITS files at the default (historic) settings; no
  assessment of `osample_x=8` on real data yet.
- Memory: the zeta list holds ≈ (osample+1)(osample_x+1) entries per pixel
  (≈ 90 MB for a 500 × 136 swath at 8/8; the list length is now counted exactly).
- Speed: no flat fast path for osample_x > 1; 5× (ox 8) to 10× (ox 16) slower than
  slitdec.c on a tilted swath.
- The default uncertainty is still the residual-based one. `'propagated'` has been checked
  on real CRIRES+ and SOXS nodding data at `osample_x=1` (see "Uncertainties"), not yet
  at `osample_x > 1` nor with a slit function that varies along the swath.
