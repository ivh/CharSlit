# slitdec_ox: slitdec with oversampling in x and y

On this branch `charslit.slitdec` runs `slitdec/slitdec_ox.c` instead of `slitdec.c`.
The C code is N. Piskunov's (https://github.com/nikolai-piskunov/slitdec_ox, GPL-3.0,
see `slitdec/LICENSE.slitdec_ox`; its history is merged into this branch), with the
IDL REDUCE interface and the ctypes wrapper removed, `lambda_sP` and `t_offset` made
arguments, and `weights='model'` brought up to master (5e38be4, a·model + b).

`slitdec_ox.c` is CharSlit `slitdec.c` (c878d00, as vendored in PyReduce) with the
spectrum solved on `osample_x` fine bins per column. Each y-subpixel of the slit
function, shifted by the curvature, contributes to a detector pixel by the overlap
of each shifted fine bin with that pixel (box integration). The spectrum penalty is a
k-th difference in physical units, `mu_x * F * ∫ (d^k sP/dx^k)^2 dx`, with F the median
data curvature per pixel, so `mu_x` does not depend on `osample_x`, flux or weights.
`sP` comes back on the detector grid (mean of the fine bins = what a vertical slit
records); `sP_fine` optionally returns the fine solution.

Python: `charslit.slitdec(im, pix_unc, mask, ycen, slitcurve, slitdeltas, ...)` as on
master, plus `osample_x=8`, `mu_x=(0, 0, 1.64e-4)` (penalties on the 1st, 2nd, ...
derivative of the spectrum), `mu_y=None` (slit function; None = `lambda_sL`),
`t_offset=0.5` and `y_lower_lim=None` (trace row in the swath; None = nrows // 2).
Extra outputs: `spectrum_fine` (ncols·osample_x) and `info_ox` = [dropped zeta entries,
sP band half-width, damping x, resolution x [px], damping y, resolution y [px]].
`lambda_sP` is master's unscaled first-difference penalty, applied between fine bins.

## Choosing osample_x and mu_x (ox_geometry.py)

- **mu_x needs no geometry.** When the tilt spreads the row phases (|c_n| = |Σ q e^{2πi n s}|/Σq
  small for n = 1..3, true for shear ≳ 1 px), the aliases decouple and noise and
  transfer are independent of the tilt. Then var/mean = (1/2π)∫R(ν)² dν with
  R = B²/(B² + μν^{2k}); var/mean = 1 gives **μ = 1.64e-4 for k = 3** (1.6e-3 for k = 2,
  1.67e-5 for k = 4). Transfer at that μ: 0.997 at π/2, 0.956 at 3π/4, 0.72 at Nyquist
  (unavoidable, shared by ±π). For shear ≲ 0.5 px the output stays ~Poisson (0.94).
- **osample_x is set by the narrowest line, not by the tilt**: rms line error ≈
  0.025 (h/σ)², h = 1/osample_x. FWHM 4.5 px: 8 → 1e-4, 16 → 2.5e-5.
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

Re-measured 2026-11-16 (`scripts/ox_noise.py <shear> <weights> 60`), var/mean relative to
a vertical slit, identical for `none` and `model` weights to ±0.01:

| shear [px] | linear | ox 4 | ox 8 | ox 16 | ox 8, mu3 = 1e-6 |
|---|---|---|---|---|---|
| 0.5 | 1.26 | 1.00 | 0.96 | 0.96 | 1.11 |
| 2 | 1.91 | 1.08 | 1.03 | 1.02 | 1.84 |
| 8 | 1.74 | 1.08 | 1.03 | 1.02 | 1.68 |
| 16 | 1.77 | 1.09 | 1.04 | 1.03 | 1.71 |

The noise is whitened by the penalty, not by the fine bins: at mu3 = 1e-6 (the
sampling-recovery setting) the excess is back, while the line bias stays at 1e-4. The excess
is anticorrelated (lag-1 −0.16 to −0.31) and mostly averages out: variance of 8-pixel sums
1.09-1.12 for both the linear split and mu3 = 1e-6, 1.02-1.04 for the default. The reported
uncertainty is ≈ 0.99 √S in every case, i.e. 1.3× too small wherever var/mean ≈ 1.7.

## Uncertainties (`uncertainty=`)

`'residual'` (default, as slitdec): per column from data − model. It is ≈ √S whatever the
extraction noise is, so 1.3× too small on a tilted slit with the linear split or a weak
penalty (table above). The cr2res PIPE-12658 formula (Horne, per column from pix_unc) has
the same blind spot (0.73-0.77 in the same test) and was not ported.

`'propagated'`: the pixel variances carried through the last spectrum solve,
C = N⁻¹ M N⁻¹ with N = AᵀWA + penalties and M = AᵀW V W A (`propagate_unc`). One banded
solve per output bin (y = N⁻¹b, var = yᵀMy, covariances yᵀMy′), each cut off where y has
decayed to 1e-10. V is `pix_unc²`, or the fitted a·model + b with `weights='model'`.
Returns `uncertainty` (detector grid), `covariance` (ncols·out_x, nlag+1) on the `out_x`
grid (lags 0..`nlag`, default 2) and `chi2`, the reduced χ² of the residuals against V.
`'scaled'` multiplies the variances by `chi2`, for pix_unc of unknown scale.

Reported σ / empirical scatter (`scripts/ox_unc_check.py`, flat spectrum, 40 realisations,
photon noise; all four weight modes agree to ±0.01):

| shear | setting | residual | propagated | on out_x = 2 | lag-1 corr. emp / predicted |
|---|---|---|---|---|---|
| 0 | linear | 1.01 | 1.03 | | 0.00 / 0.00 |
| 2 px | linear | 0.73 | 1.03 | | −0.32 / −0.32 |
| 8 px | linear | 0.77 | 1.02 | | −0.26 / −0.27 |
| 2-8 px | ox 8, default | 0.99-1.00 | 1.02-1.03 | 1.02 | −0.02 / −0.02 (out_x 2: +0.49 / +0.49) |
| 2-8 px | ox 8, mu3 = 1e-6 | 0.74-0.77 | 1.02-1.03 | 1.03 | −0.17 / −0.17 (out_x 2: −0.36 / −0.37) |

With 30 e⁻ read noise: propagated 1.03-1.06 (20 realisations), residual 0.79-0.83 and
1.07-1.12. The 2-3 % excess is partly the small-sample bias of the empirical σ.
Cost on real frames: +2-3 % of the run time (ANDES_R_FP1 2.00 → 2.06 s at ox 8).

Limits: the slit function is taken as known; the number is the noise of the regularised
estimate and says nothing about smoothing bias; and it is only as good as V. On the test
frames, whose fixture pix_unc is √(|im|+1) rather than a real variance, chi2 is 3-95, so
use `'scaled'` or check `chi2` before trusting `'propagated'` on real data.

**Outlier rejection re-enables masked pixels.** With `kappa > 0` the mask is rewritten each
iteration from |model − im| < kappa·dev, ignoring the input mask (slitdec.c on master does
the same): on CRIRES2 10089 of 12366 input-masked pixels come back as good. They enter the
fit with whatever `im` holds there, and with the fixture's pix_unc = 1e10 for them the
propagated variance explodes (correctly, for the solve as done). Not changed here: fixing
it alters the goldens.

## Regression

`osample_x=1, mu_x=[], t_offset=0` reproduces master's slitdec.c: `tests/test_regression.py`
runs the golden suite with these settings (all 12 cases pass at rtol 1e-10, exact mask), and
a one-off ctypes comparison against master's C (2026-11-06; tilted and sloped traces, noise,
kappa 10) was bit-identical for all four weight modes. The default-run tests in
`tests/test_slitdec.py` use the ox defaults; `tests/test_tilt_reference.py` checks the tilt
reference shift, the line bias against the linear split and `spectrum` = mean of
`spectrum_fine`. `scripts/ox_noise.py` reproduces the table above.

## Swath edges

The first `delta_x` columns are zeroed as in slitdec.c, but the transient reaches further.
Flat spectrum, 200 × 136 swath, osample 8, relative error in the first kept columns:

| shear | linear (slitdec.c) | osample_x = 8 |
|---|---|---|
| 0 (c1 = 1e-12) | 7e-6 | +5e-3, −4e-3, +2e-3, … (decays over ~8 columns) |
| 2 px | +0.12, −0.04, +0.01, … | +4e-3, −5e-3, +2e-3, … |
| 8 px | +8e-3, −2e-3, … | −3e-4, +3e-4, … |

With a real tilt ox is better than the linear split; only an untilted swath gets a new edge
ripple (the k = 3 penalty couples the edge bins, from which off-swath parts are dropped, to
the interior). Overlapping swaths hide both.

## Finding in slitdec.c: tilt evaluated half a subpixel off

slitdec.c evaluates the curvature at t = (lower edge of the y-subpixel) − ycen instead
of the subpixel centre. Every line then shifts by c1/(2·osample): 3.7 mpx for c1 = 8/136
at osample 8, scaling exactly with 1/osample (`test_tilt_reference.py`). It is smooth in
x and calibrates out if science and calibration share osample and tilt, but it changes
with osample and with any tilt difference (~5 m/s-equivalent for c1 ≈ 0.12 at osample 8 on
ANDES). slitdec_ox evaluates at the centre (`slitdec_ox_t_offset = 0.5`, a global for
now). The same one-line fix applies to slitdec.c.

## Sampling recovery (the former `osamp_spec` branch)

`osamp_spec` (fine spectrum bins to recover structure below the pixel scale, for
bisector/RV work on under-sampled lines) used the same forward model, so its purpose is
served here by reading `spectrum_fine`; its C code and `lambda_fringe` are not ported. The
experiment scripts and docs came along (`scripts/compare_osamp.py`,
`experiment_synthetic_lsf.py`, `experiment_binx2_crires.py`, `analyse_wide_bisector.py`,
`make_swath.py`; `docs/OSAMP_*.md`, which describe the old implementation), with
`osamp_spec=s` → `osample_x=s` and the `lambda_fringe` grids replaced by `mu3` grids.
All checks below 2026-11-16.

**Equivalence.** One sP solve with a common preset slit function, s = 2, 3, same
`lambda_sP > 0` in both codes, `t_offset=0`: fine spectra agree to 6e-10 and models to
7e-10 away from the swath ends (the 1e-9 ridge). They differ in the first ~`delta_x`+6
columns only (bins hanging off the swath are dropped here, deposited partly there).

**Null space.** A pattern that repeats every pixel and has zero mean integrates to zero over
any unit-wide pixel, whatever the shift: s−1 exact null modes that no tilt can constrain (the
"fringe at the osamp frequency" of OSAMP_SPEC.md; the tilt constrains everything else).
Without a penalty the fine solution is therefore not defined: both codes hit `maxiter` on
the synthetic LSF cases and the result keeps drifting (osamp_spec, s = 3: rms_cont 0.077 at
20 iterations, 0.090 at 100), and `osample_x=8, mu_x=[]` returns garbage. Do not use
`mu_x=[]` with `osample_x > 1`. The baseline numbers in OSAMP_EXPERIMENTS.md (rms_cont
0.081/0.053 at s = 2/3, "lambda_fringe = 0") are not reproduced by any version of the
osamp_spec C code with today's script (0.174/0.077, June, October and current code alike);
a weak penalty gives about those values, so they were probably made with one.

**Transfer of `spectrum_fine`.** Measured on a noise-free random line list (s = 3, shear
8 px) it follows B²/(B² + P/1.5), B = sinc(ν/2) the pixel box, P = Σ mu_k ν^2k, not the
nominal 1/(1 + P) that `info_ox` reports: exactly 0 at 2π, and at the default mu3 0.65 at
1.1π, 0.30 at 1.4π, 0.06 at 1.6π. mu3 = 8.2e-6 keeps 1.0 to 1.4π and 0.58 at 1.6π.
The bin values behave like point samples (a bin-averaged truth is exceeded by 1/sinc²(ν/2s)).

**Which settings** (`scripts/ox_bis_study.py`: synthetic asymmetric line on a slit with 4 px of
tilt sweep, 8 sub-pixel line positions × 6 noise seeds, peak ~2000-3000 counts; BIS bias =
mean over all, phase sd = scatter of the per-position means, which is what fakes an RV
signal; mpx):

| LSF | setting | BIS bias | phase sd | noise sd | FWHM (truth) | rms_cont |
|---|---|---|---|---|---|---|
| wide, 1.79 px | linear (master) | −6.6 | 60.1 | 3.0 | 1.855 (1.785) | 0.057 |
| | ox 3, mu3 = 1e-6 | −0.6 | 3.3 | 6.1 | 1.785 | 0.018 |
| | ox 8, mu3 = 1e-6 | −0.5 | 3.2 | 6.3 | 1.784 | 0.0070 |
| | ox 8, default 1.64e-4 | −2.4 | 2.5 | 3.5 | 1.816 | 0.0074 |
| narrow, 0.89 px | linear (master) | −10.5 | 144.3 | 1.2 | 1.316 (0.894) | 0.136 |
| | ox 3, mu3 = 1e-6 | −4.5 | 4.2 | 2.0 | 0.960 | 0.050 |
| | ox 8, mu3 = 1e-6 | −5.5 | 0.5 | 1.6 | 0.967 | 0.025 |
| | ox 8, default 1.64e-4 | −6.4 | 0.4 | 0.4 | 1.235 | 0.062 |

- The position-dependent bisector error of the linear split (60-144 mpx) is gone for any
  `osample_x ≥ 2` with any penalty tried (phase sd at the level the noise allows).
- `osample_x = 8` is as good as 2-4 on the wide LSF and clearly better on the narrow one
  (phase sd 0.3-1 mpx against 1-6 at s = 3 and 7-19 at s = 2), and better on rms_cont
  throughout, so there is no reason for a separate small-s mode; the two purposes differ
  only in `mu_x`. Figures: `scratch/ox_bis_study_<case>.png` (mu3 from 1e-7 to 1e-3).
- The penalty strength is a noise/resolution trade. The default is right for lines of
  FWHM ≳ 1.8 px (2 % broader, lowest noise) and at low S/N (wide LSF, peak ~100 counts: BIS
  noise sd 14 vs 29 mpx for 1e-6). For lines narrower than a pixel at high S/N use
  mu3 ≈ 1e-6 (FWHM 0.97 vs 1.24 at the default, truth 0.89).
- k = 6 instead of k = 3 at matched strength made no measurable difference; one knob (mu3)
  is enough.
- Real data, CRIRES+ U-Ne binned by √3 (`experiment_binx2_crires.py --xbin sqrt3`), BIS of the
  7-line stack, truth −162 mpx: staircase −397; ox s = 2/3/4 at mu3 = 1e-6: −157/−182/−194,
  8.2e-6: −178/−166/−168, default: −175/−148/−146. (`lambda_fringe = 1e-3` gave −330..−365.)
  Not analysed further (per-line metrics, s = 8).

**The other cases of OSAMP_EXPERIMENTS.md, re-run** (single realisation each, as there;
`experiment_synthetic_lsf.py`, tags `_ox`, `_notilt`, `_lowsn`, `_ssweep`, `_narrow`, `_wide`):

| case (true FWHM) | s = 1 | s = 2 | s = 3 | at mu3 |
|---|---|---|---|---|
| baseline (0.894): FWHM | 1.114 | 0.956 | 0.968 | 1e-6 |
| baseline: rms_cont | 0.108 | 0.080 | 0.050 | 1e-6 |
| narrow (0.444): FWHM | 0.991 | 0.722 | 0.713 | 1e-6 |
| narrow: rms_cont | 0.240 | 0.194 (8.2e-6) | 0.157 | 1e-6 |
| wide (1.785): FWHM | 1.772 | 1.786 | 1.822 | 1e-6 |
| wide: rms_cont | 0.058 | 0.027 | 0.018 | any |
| low S/N (peak 87): rms_cont | 0.112 | 0.077 (8.2e-6) | 0.054 | 1e-6 |
| no tilt (0.001 px/row): rms_cont | 0.098 | 0.083 | 0.073 | 1e-6 |

- Baseline s sweep at mu3 = 1e-6: rms_cont 0.108, 0.080, 0.050, 0.039, 0.032, 0.029, 0.024
  for s = 1, 2, 3, 4, 5, 6, 8, with FWHM 0.95-0.97 throughout: no saturation at s = 5 and no
  undershoot, unlike the unregularised sweep.
- Without tilt nothing is gained (FWHM 1.18-1.28 against 1.18 at s = 1) but nothing breaks
  either; the unregularised s = 2 gave rms_cont 0.27 there.
- CRIRES+ at xbin = 2 (commensurate), BIS of the stack, truth −162 mpx: staircase −260;
  s = 2/3/4 at mu3 = 1e-6: −228/−201/−140, 8.2e-6: −169/−195/−156, default: −157/−165/−143.
- Hsim at s = 3: max|fine − staircase| stays at 19-29 % of the peak for mu3 from 0 to 1e-2
  (29 % unpenalised). A derivative penalty pulls towards a smooth curve, not the staircase,
  and a smooth curve through an emission line differs from its staircase by about that
  much. So this number is not a measure of fringe noise, and the conclusion in
  OSAMP_SPEC.md that Hsim's sub-pixel content is all fringe does not follow from it.
- June outputs of the old engine that the re-runs would have overwritten are in
  `scratch/old_engine/` (the wide s-sweep, bisector and √3 plots were already overwritten).

Cost on real frames, ms per iteration at `osample_x` = 1/2/3/8 (`scripts/bench.py 2 <ox> 1.2e6`):
ANDES_R_FP1 404 × 2556: 73/113/143/325; CRIRES2 176 × 2048: 21/32/40/91 (osamp_spec:
62/112/293 and 18/35/93 at s = 1/2/3).

## Penalties: mu_k in x and y (penalty_design.py, sl_penalty_study.py)

`slitdec_ox2` (`mu_x`, `mu_y` in Python) takes `mu_x[k-1]`, `mu_y[k-1]` (k = 1..15): penalty
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

- Real frames: only the golden-suite FITS files at compatibility settings; no assessment of the
  ox defaults on real data yet.
- Memory: the zeta list holds ≈ (osample+1)(osample_x+1) entries per pixel
  (≈ 90 MB for a 500 × 136 swath at 8/8; the list length is now counted exactly).
- Speed: no flat fast path for osample_x > 1; 5× (ox 8) to 10× (ox 16) slower than
  slitdec.c on a tilted swath.
- The default uncertainty is still the residual-based one; `'propagated'` has not been
  checked on real frames with real variances, nor with a varying slit function.
