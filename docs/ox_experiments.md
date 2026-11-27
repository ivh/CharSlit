# slitdec_ox: experiments and results, November 2026

What was measured while making N. Piskunov's `slitdec_ox` the engine of `slitdec.slitdec`
(2026-11-06 to 2026-11-16), and what follows from it. Reference for the engine and its
arguments: [slitdec_ox.md](slitdec_ox.md). Bisector primer and the design of the
sampling-recovery experiments: [bisector_experiments.md](bisector_experiments.md).

Unless stated otherwise: synthetic swaths (`tests/synth.py`, flat-topped slit, photon noise,
flux 3000 per column), `osample=8`, `kappa=0`, joint slit-function/spectrum fit. "Linear"
means `osample_x=1, mu_x=[], t_offset=0`, the historic `slitdec.c`. "Default" (or "ox 8,
default") means `osample_x=8, t_offset=0.5` with the default penalty `mu_x=(0, 0, 1.64e-4)`;
these were the function defaults while the experiments ran and were made opt-in afterwards.
`t_offset` was later removed and the centre (0.5) hard-wired for all modes, so a plain call
now is "linear" with `t_offset=0.5`; the measurements below were made with the stated
value. Nothing here is from real frames with real variances
except where a CRIRES+ or ANDES file is named.

## Summary

1. **The engine reproduces master** at the compatibility settings, bit for bit, in all
   four weight modes.
2. **The linear split is 1.7-1.9× too noisy on a tilted slit** (variance relative to a
   vertical slit) and biases line shapes by 8e-3 of the peak. Fine bins remove the line
   bias; the **penalty**, not the fine bins, removes the noise excess.
3. **The `osamp_spec` branch used the same forward model** and is folded in. Its
   `lambda_fringe` is dropped; without any penalty the fine spectrum is not defined.
4. **One oversampling serves both purposes.** `osample_x=8` is as good as or better than
   2-4 everywhere; noise optimisation and sampling recovery differ only in `mu3`
   (1.64e-4 against about 1e-6), and they trade against each other.
5. **The position-dependent bisector error of the linear split (60-144 mpx) disappears**
   for any `osample_x ≥ 2` with any penalty tried; on real CRIRES+ data the stacked
   bisector is recovered to 15-30 mpx where the staircase is off by 235.
6. **A coarser saved grid (`out_x`) is limited by the grid itself**, not by the solve.
7. **Residual-based uncertainties, and the Horne formula of cr2res PIPE-12658, are 25 % low**
   wherever the noise excess of point 2 is present. Propagating the pixel variances
   through the solve gives 1.02-1.03 everywhere and the right neighbour covariances.
8. **Found along the way:** master evaluated the tilt half a subpixel off (now fixed); outlier
   rejection re-enables zero-filled NaN pixels ([../LATER.md](../LATER.md)).

## 1. Integration and regression

- Golden suite (3 synthetic cases, 9 real frames; spectrum, slit function, model,
  uncertainty at rtol 1e-10, mask exact) passes at the compatibility settings, including the
  case with `lambda_sP > 0`. `lambda_sP` and the a·model + b variance fit of
  `weights='model'` were added to the ox engine for this.
- Direct comparison with master's C compiled separately (tilted and sloped traces, noise,
  kappa 10): bit-identical for `none`, `unc`, `model`, `flux`.
- **Tilt reference point.** master evaluates the curvature at the lower edge of each
  y-subpixel. Measured line shift at osample 4 / 8 / 16: +7.85 / +4.20 / +2.40 mpx for the
  linear path against c1/(2·osample) = 7.35 / 3.68 / 1.84 (c1 = 8/136); the remaining
  ~0.5 mpx is the linear split itself. With `t_offset=0.5` and fine bins: 0.00.
- **Line bias** on a noise-free FP comb (FWHM 4.5 px), rms/peak: linear 8.0e-3,
  `osample_x` 4 / 8 / 16: 4.8e-4 / 1.2e-4 / 2.7e-5.

## 2. Extraction noise against shear

Variance relative to a vertical slit (`scripts/ox_noise.py <shear> <weights> 60`); `none`
and `model` weights agree to ±0.01.

| shear [px] | linear | ox 4 | ox 8 | ox 16 | ox 8, mu3 = 1e-6 |
|---|---|---|---|---|---|
| 0.5 | 1.26 | 1.00 | 0.96 | 0.96 | 1.11 |
| 2 | 1.91 | 1.08 | 1.03 | 1.02 | 1.84 |
| 8 | 1.74 | 1.08 | 1.03 | 1.02 | 1.68 |
| 16 | 1.77 | 1.09 | 1.04 | 1.03 | 1.71 |

- At the weak penalty the excess is back while the line bias stays at 1e-4: the noise is
  whitened by the penalty.
- The excess is anticorrelated (lag-1 −0.16 to −0.31) and mostly averages out: variance of
  8-pixel sums 1.09-1.12 for the linear split and for mu3 = 1e-6, 1.02-1.04 at the default.

## 3. Swath edges

Measured when the solver still zeroed the first `delta_x` columns (the largest slit-image
shift; it now returns them and reports a margin of that shift + 2 instead). The transient
reaches further. Flat spectrum, 200 × 136, relative error in the columns from `delta_x` on:

| shear | linear | ox 8, default |
|---|---|---|
| 0 (c1 = 1e-12) | 7e-6 | +5e-3, −4e-3, +2e-3, … (decays over ~8 columns) |
| 2 px | +0.12, −0.04, +0.01, … | +4e-3, −5e-3, +2e-3, … |
| 8 px | +8e-3, −2e-3, … | −3e-4, +3e-4, … |

With a real tilt ox is better; only an untilted swath gets a new ripple (the penalty couples
the edge bins, from which off-swath parts are dropped, to the interior). Overlapping swaths
hide both.

## 4. Relation to the `osamp_spec` branch

- **Equivalence.** One spectrum solve with a common preset slit function, s = 2, 3, the same
  `lambda_sP > 0` in both codes, `t_offset=0`: fine spectra agree to 6e-10 and models to
  7e-10 away from the swath ends (the level of the ox ridge). They differ in the first
  ~`delta_x`+6 columns only (bins hanging off the swath are dropped here, partly deposited
  there).
- **Null space.** A pattern that repeats every pixel with zero mean integrates to zero over
  any unit-wide pixel, whatever the shift: s−1 exact null modes that no tilt can constrain.
  Without a penalty both codes hit `maxiter` on the synthetic LSF cases and keep drifting
  (osamp_spec, s = 3: rms_cont 0.077 at 20 iterations, 0.090 at 100); `osample_x=8,
  mu_x=[]` returns garbage. **Never use `mu_x=[]` with `osample_x > 1`.**
- **`lambda_fringe`** (deviation of the fine bins from their pixel mean) acts roughly as
  λ(1 − B²), B = sinc(ν/2): only 1.3× stronger at the fringe (2π) than at 1.25π, where a
  k = 3 derivative penalty is 17× stronger, and it is tied to the pixel grid. Its strong
  limit is the staircase. Not ported. (The 1 − B² form is a calculation, not measured.)
- **Not reproduced.** The June baseline (rms_cont 0.081 / 0.053 at s = 2 / 3 with
  "lambda_fringe = 0") is not obtained with any version of the osamp_spec C code and
  today's script: 0.174 / 0.077 for the June, October and current code alike. A weak
  penalty gives about the documented values (`lambda_sP=1e-6`: 0.104 / 0.053), so those runs
  probably had one; not established.
- **Cost**, ms per iteration at `osample_x` = 1 / 2 / 3 / 8 (`scripts/bench.py 2 <ox> 1.2e6`):
  ANDES_R_FP1 404 × 2556: 73 / 113 / 143 / 325; CRIRES2 176 × 2048: 21 / 32 / 40 / 91.
  osamp_spec's own doc gave 62 / 112 / 293 and 18 / 35 / 93 at s = 1 / 2 / 3.

## 5. Transfer of the fine spectrum

Noise-free random line list (FWHM 1.2 px), s = 3, shear 8 px, cross-spectrum of
`spectrum_fine` with the bin-averaged truth, in bands of ν:

| ν/π | 0.35 | 0.62 | 0.88 | 1.12 | 1.37 | 1.63 | 1.85 | 2.00 | 2.27 | 2.75 |
|---|---|---|---|---|---|---|---|---|---|---|
| mu3 = 1.64e-4 | 1.01 | 1.02 | 0.97 | 0.65 | 0.30 | 0.06 | 0.005 | 0.000 | 0.002 | 0.01 |
| mu3 = 8.2e-6 | 1.01 | 1.03 | 1.07 | 1.08 | 1.00 | 0.58 | 0.10 | 0.002 | 0.02 | 0.21 |
| mu6 = 1.9e-8 | 1.01 | 1.03 | 1.07 | 0.99 | 0.56 | 0.07 | 0.004 | 0.000 | 0.000 | 0.003 |
| mu6 = 7.5e-10 | 1.01 | 1.03 | 1.07 | 1.12 | 1.11 | 0.70 | 0.09 | 0.001 | 0.01 | 0.06 |

- The shape is B²/(B² + P/~1.5) with P = Σ mu_k ν^2k: an exact zero at 2π and a much steeper
  roll-off above Nyquist than the nominal 1/(1 + P) that `info_ox` reports as resolution.
- Values above 1 are a matter of definition: the fine-bin values behave like point samples,
  and a bin-averaged truth is exceeded by 1/sinc²(ν/2s) (7 % at 0.9π for s = 3).
- A weak k = 3 penalty leaks at 2.75π (0.21), k = 6 does not (0.06); in the line-shape
  studies below this made no measurable difference.
- The script for this measurement was a one-off and is not in the repo.

## 6. Line shape: bisector study

`scripts/ox_bis_study.py [wide|baseline] [amp]`: the asymmetric line of
`experiment_synthetic_lsf.py` on a slit with 4 px of tilt sweep, 8 sub-pixel line positions
× 6 noise seeds, mu3 from 1e-7 to 1e-3. BIS bias = mean over all; phase sd = scatter of the
per-position means (what fakes an RV signal); noise sd = scatter between seeds; mpx.
Figures: `scratch/ox_bis_study_{wide,baseline,wide_amp200}.png`.

| LSF | setting | BIS bias | phase sd | noise sd | FWHM (truth) | rms_cont |
|---|---|---|---|---|---|---|
| wide, 1.79 px | linear | −6.6 | 60.1 | 3.0 | 1.855 (1.785) | 0.057 |
| | ox 2, mu3 = 1e-6 | −1.2 | 4.1 | 6.0 | 1.790 | 0.028 |
| | ox 3, mu3 = 1e-6 | −0.6 | 3.3 | 6.1 | 1.785 | 0.018 |
| | ox 8, mu3 = 1e-6 | −0.5 | 3.2 | 6.3 | 1.784 | 0.0070 |
| | ox 8, default | −2.4 | 2.5 | 3.5 | 1.816 | 0.0074 |
| | ox 8, mu3 = 1e-3 | −5.3 | 1.9 | 1.8 | 1.915 | 0.012 |
| narrow, 0.89 px | linear | −10.5 | 144.3 | 1.2 | 1.316 (0.894) | 0.136 |
| | ox 2, mu3 = 1e-6 | −1.3 | 8.7 | 3.0 | 0.961 | 0.081 |
| | ox 3, mu3 = 1e-6 | −4.5 | 4.2 | 2.0 | 0.960 | 0.050 |
| | ox 8, mu3 = 1e-6 | −5.5 | 0.5 | 1.6 | 0.967 | 0.025 |
| | ox 8, default | −6.4 | 0.4 | 0.4 | 1.235 | 0.062 |
| wide, peak ~100 counts | linear | −6.9 | 60.1 | 12.9 | 1.857 | 0.058 |
| | ox 8, mu3 = 1e-6 | −3.2 | 17.0 | 29.2 | 1.773 | 0.011 |
| | ox 8, default | −4.1 | 8.9 | 14.2 | 1.814 | 0.0083 |

- The position-dependent error of the linear split is gone for any `osample_x ≥ 2` with any
  penalty; what remains is at the level the noise allows (noise sd / √6).
- `osample_x=8` equals 2-4 on the wide line and is clearly better on the narrow one
  (phase sd 0.3-1 mpx against 1-6 at 3 and 7-19 at 2).
- The penalty is a noise/resolution trade: default for FWHM ≳ 1.8 px (2 % broader, lowest
  noise) and at low S/N; mu3 ≈ 1e-6 for lines narrower than a pixel at high S/N (FWHM 0.97
  against 1.24, truth 0.89).
- Unpenalised runs do not converge and are noisier; at `osample_x=8` they are garbage
  (wide: BIS noise 88 mpx, FWHM 1.57).
- k = 6 at matched strength (mu6 = 1e-10 against mu3 = 1e-6) gives the same numbers.

## 7. The June experiments, re-run on this engine

`scripts/experiment_synthetic_lsf.py` (one realisation each, as originally), values at
mu3 = 1e-6 unless noted; outputs `scratch/experiment_synthetic_lsf_{ox,notilt,lowsn,ssweep,narrow,wide}.png`.

| case (true FWHM) | quantity | s = 1 | s = 2 | s = 3 |
|---|---|---|---|---|
| baseline (0.894) | FWHM | 1.114 | 0.956 | 0.968 |
| | rms_cont | 0.108 | 0.080 | 0.050 |
| narrow (0.444) | FWHM | 0.991 | 0.722 | 0.713 |
| | rms_cont | 0.240 | 0.194 (mu3 8.2e-6) | 0.157 |
| wide (1.785) | FWHM | 1.772 | 1.786 | 1.822 |
| | rms_cont | 0.058 | 0.027 | 0.018 |
| low S/N (peak 87) | rms_cont | 0.112 | 0.077 (mu3 8.2e-6) | 0.054 |
| no tilt (0.001 px/row) | rms_cont | 0.098 | 0.083 | 0.073 |

- **Baseline s sweep:** rms_cont 0.108, 0.080, 0.050, 0.039, 0.032, 0.029, 0.024 for
  s = 1, 2, 3, 4, 5, 6, 8, FWHM 0.95-0.97 throughout. No saturation at s = 5 and no
  undershoot, unlike the unregularised June sweep.
- **Wide s sweep, bisector** (`analyse_wide_bisector.py`, truth BIS +24.9 mpx): bias +81.3
  at s = 1 (June: +80.2), −6.1 / −2.3 / −4.7 / −5.4 at s = 2 / 3 / 4 / 8.
- **No tilt:** nothing gained (FWHM 1.19-1.28 against 1.18 at s = 1), nothing broken; the
  unregularised s = 2 gave rms_cont 0.27 there.
- **CRIRES+ U-Ne, binned by √3** (`experiment_binx2_crires.py --xbin sqrt3`), BIS of the
  7-line stack, truth −162 mpx: staircase −397 (June: −400); s = 2 / 3 / 4 at mu3 = 1e-6:
  −157 / −182 / −194; 8.2e-6: −178 / −166 / −168; default: −175 / −148 / −146. June,
  `lambda_fringe=1e-3`: −330 to −365. Per-line metrics and s = 8 not analysed.
- **CRIRES+ binned by 2** (commensurate): staircase −260; mu3 = 1e-6: −228 / −201 / −140;
  8.2e-6: −169 / −195 / −156; default: −157 / −165 / −143.
- **Hsim, s = 3:** max|fine − staircase| stays at 19-29 % of the peak for mu3 from 0 to 1e-2.
  A smooth curve through an emission line differs from its staircase by about that much, so
  the number does not measure fringe noise; the June conclusion that Hsim's sub-pixel
  content is all fringe does not follow from it.
- `compare_osamp.py` (σ = 0.45 px line, true FWHM 1.06): recovered FWHM 1.15 / 1.14 / 1.09 at
  s = 1 / 2 / 3 with mu3 = 1e-6.
- June outputs that the re-runs would have overwritten are in `scratch/old_engine/`; the
  June wide s-sweep, bisector and √3 plots had already been overwritten.

## 8. Saving on a coarser sub-pixel grid (`out_x`)

Same study as section 6, measured on the saved grid, mu3 = 1e-6 unless noted:

| LSF | solved | saved | phase sd | noise sd | FWHM (truth) |
|---|---|---|---|---|---|
| wide | 2 | 2 | 4.1 | 6.0 | 1.790 (1.785) |
| | 8 | 2 | 3.5 | 4.7 | 1.842 |
| | 8 | 2 (default mu3) | 3.3 | 3.0 | 1.873 |
| | 8 | 8 | 3.2 | 6.3 | 1.784 |
| narrow | 2 | 2 | 8.7 | 3.0 | 0.961 (0.894) |
| | 8 | 2 | 10.0 | 1.9 | 1.062 |
| | 9 | 3 | 2.9 | 1.6 | 1.009 |
| | 8 | 4 | 1.0 | 1.6 | 0.986 |
| | 8 | 8 | 0.5 | 1.6 | 0.967 |

- For lines of FWHM ≈ 1.8 px a 2× grid is enough; for lines narrower than a pixel the saved
  grid limits (use 4×).
- Saved values are bin averages; a metric that interpolates between bin centres reads them
  3 % (wide) to 10 % (narrow) too broad at 2×. Values solved directly at 2 behave like point
  samples and do not show this.
- The grouping phase is a free choice; phase 0 nests the bins in detector pixels and is
  kept. Averaging phases only smooths. Two diagnostics that use the phase are in
  [../LATER.md](../LATER.md).

## 9. Uncertainties

Reported σ / empirical scatter (`scripts/ox_unc_check.py`, flat spectrum, 40 realisations,
photon noise; the four weight modes agree to ±0.01). "Horne" is the per-column formula of
cr2res PIPE-12658, Σ M P / Σ (M P²/V), evaluated on the same runs by a one-off script
(its column was rescaled by hand after a units slip).

| shear | setting | residual | Horne | propagated | on out_x = 2 | lag-1 corr. emp / predicted |
|---|---|---|---|---|---|---|
| 0 | linear | 1.01 | 1.01 | 1.03 | | 0.00 / 0.00 |
| 2 px | linear | 0.73 | 0.73 | 1.03 | | −0.32 / −0.32 |
| 8 px | linear | 0.77 | 0.77 | 1.02 | | −0.26 / −0.27 |
| 2-8 px | ox 8, default | 0.99-1.00 | 1.00 | 1.02-1.03 | 1.02 | −0.02 / −0.02 (out_x 2: +0.49 / +0.49) |
| 2-8 px | ox 8, mu3 = 1e-6 | 0.74-0.77 | 0.75-0.78 | 1.02-1.03 | 1.03 | −0.17 / −0.17 (out_x 2: −0.36 / −0.37) |

- With 30 e⁻ read noise (20 realisations): propagated 1.03-1.06, residual 0.79-0.83 where the
  excess is present and 1.07-1.12 elsewhere. Part of the 2-3 % excess of the propagated value
  is the small-sample bias of the empirical σ.
- cr2res uses the linear split, so its Horne errors should be low wherever the shear across
  the extraction height is appreciable (12 % at 0.5 px, from var 1.26). Inferred from this
  test, not checked on CRIRES+ data. The bulk of PIPE-12658 is upstream variance bookkeeping
  (flat/dark errors cancelling in nodding and polarimetry), which is not our concern here.
- Cost on real frames: +2-3 % of the run time (ANDES_R_FP1 at ox 8: 2.00 → 2.06 s).
- On the test frames the fixture's pix_unc is √(|im|+1), not a real variance: chi2 = 3-95.
  `'propagated'` is only as good as pix_unc; use `'scaled'` or look at `chi2`.
- Not covered: slit-function error (taken as known), smoothing bias of the penalty, the
  selection effect of outlier clipping.

## 10. Earlier statements that turned out wrong

Recorded because some of them were written into docs or commit messages on the way.

- "Slit tilt breaks the fringe degeneracy" (June spec): the s−1 modes are exact nulls for any
  tilt; the tilt constrains everything else.
- "Hsim's sub-pixel content is all fringe noise" (June spec): the metric does not show that.
- "Agreement of the two engines to 1e-9 without a penalty" as an equivalence test: the
  unpenalised fine spectra differ at order one; they agree to 6e-10 with a common penalty.
- "k = 6 at mu6 = 1.9e-8 for sampling recovery": derived from the penalty-only filter; the
  transfer is B²/(B² + P/1.5), and k = 6 brings nothing measurable over k = 3.
- "`osample_x=8` is as good as 2-4 on every metric": better than 2-3 for narrow lines.
- "Solved at 8 and saved at 2 is better than solved at 2": about equal; the saved grid limits.
- "The oversampling fixes the tilt noise": the penalty does.
- "Outlier rejection un-masks pixels" as a C bug: masked pixels holding a value may come
  back by design; the fault is zero-filled NaNs in the callers.

## Open

- Real frames with real variances: none of `osample_x=8`, the weak penalty or the
  propagated uncertainties have been assessed on them.
- A slit function that varies along the dispersion, or line shapes that vary along the slit
  (the ESPRESSO case that motivated `weights=`): all synthetic tests have identical rows.
- The transfer and bisector results assume a tilt sweep of several pixels; dependence on the
  sweep (1, 2, 5 px) and on `ncols` is not mapped (see LATER.md).
- PyReduce master does not pass `osample_x` / `mu_x` (only its `osamp_spec` branch hacks the old
  arguments in through environment variables); it needs them before it can use this for
  sampling recovery.
- [../LATER.md](../LATER.md): validation on real frames, the NaN issue, phase diagnostics.
