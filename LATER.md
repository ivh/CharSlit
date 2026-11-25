# LATER

Deferred work. Do not start on these unless asked.

## Validate the opt-in modes on real frames (after the slitdec_ox merge, 2026-11-16)

Everything measured so far (docs/ox_experiments.md) is on synthetic swaths plus two CRIRES+
binning experiments; no real frame with real variances has gone through the new modes. The
defaults are the historic slitdec and are covered by the golden suite. To do, on frames
where science and calibration are extracted identically:

- **`osample_x=8` with the default penalty.** Does the extraction noise drop
  as on the synthetics (variance 1.7-1.9× → 1.03 relative to a vertical slit for shears of
  2 px and more)? Line shapes and wavelength solution against the historic extraction
  (which also differs by the tilt-reference shift c1/(2·osample), now fixed in every mode).
  Run time and memory on full orders (4.5× per iteration on ANDES; no fast path for
  untilted frames above `osample_x=1`, the 2048 × 2048 no-curvature frame was never run).
- **The weak penalty (`mu_x=(0, 0, 1e-6)`), reading `spectrum_fine` / `spectrum_out`.** Is
  sub-pixel line shape recovered on genuinely under-sampled data (ESPRESSO-class), and how
  much does the bisector depend on the line's sub-pixel position? Choice of `out_x`
  (2 for FWHM ≈ 1.8 px, 4 below a pixel on the synthetics).
- **`uncertainty='propagated'` / `'scaled'`.** Reported σ and neighbour covariances against
  the scatter between repeated exposures; how far `chi2` is from 1 with real pix_unc
  (3-95 with the test fixture's √(|im|+1)); frames where pix_unc is known not to describe
  the noise (CRIRES+ L/M nodding).
- **What the synthetics cannot show:** a slit function that varies along the dispersion,
  line shapes that vary along the slit (fibre-chord rows, the case behind `weights=`), and
  the slit-function error that the propagated uncertainty ignores.
- PyReduce reads `PYREDUCE_CHARSLIT_OSAMP_SPEC` / `LAMBDA_FRINGE`; rename to `osample_x` /
  `mu_x` (and vendor the new `slitdec.c`) before it can run any of this.

## Phase diagnostics for `out_x < osample_x` (2026-11-16)

With e.g. `osample_x=8, out_x=2` the grouping of fine bins into output bins can start at
any of `osample_x / out_x` offsets. `out_x` itself stays as it is (phase 0, bins nested in
detector pixels; science and calibration frames must be extracted identically, the target
rebinning is only known later). Two diagnostics come out of the phase freedom:

1. **Is the output grid fine enough?** Rebin `spectrum_fine` at every phase and measure the
   same line quantity (centroid, width, bisector) on each. Data and solve are identical, so
   the spread is the sampling error of the output grid alone. Seen before as 9-10 mpx of
   position-dependent BIS error for a 0.89 px line at 2x, ~3 at 3x, ~1 at 4x. A few lines of
   numpy on existing output; use it to choose `out_x` for a given line width.

2. **Does the tilt supply the sub-pixel information?** Call with
   `uncertainty='propagated', out_x=osample_x` and fold `covariance[:, 0]` modulo
   `osample_x`: the fine-bin variance as a function of position inside the pixel. Depends
   only on geometry, weights and variances, so it could run once per swath. Flat spectrum,
   photon noise, min-max relative to the mean:

   | shear across slit | default mu3 | mu3 = 1e-6 |
   |---|---|---|
   | 0 | 0.91-1.09 | 0.89-1.11 |
   | 0.25 px | 1.00-1.00 | 0.37-1.68 |
   | 0.5 px | 0.93-1.07 | 0.62-1.47 |
   | 1 px | 0.98-1.02 | 0.95-1.08 |
   | 2 px | 0.98-1.02 | 0.94-1.07 |
   | 8 px | 1.00-1.00 | 1.00-1.00 |

   Flat = the solution does not know where the pixel boundaries are. Strong variation
   flags partial phase coverage (0.25-0.5 px shear with a weak penalty; half-pixel bins
   then differ 1.80/0.86/0.48/0.86 in variance between the four phases). Catch: at zero
   shear the pattern is mild although the data hold no sub-pixel information at all (the
   penalty fills it in), so a flat pattern is necessary, not sufficient. A test for the
   no-information case is still missing (idea: sensitivity of the covariance to mu3).

Not to do: averaging the phases for the output. It convolves with the bin box a second
time (triangular kernel, 1 px base): more smoothing, more correlation, no information.

## Carried over from the osamp_spec branch (June 2026)

- **Dependence of sampling recovery on the geometry.** All bisector results so far use one
  slit (80 rows, 4 px tilt sweep, 60 columns). Repeat `scripts/ox_bis_study.py` for tilt
  sweeps of 1, 2 and 5 px and other `ncols`, to see where the recovery stops working. The
  multi-seed part of the original item is done (8 line positions × 6 seeds).
- **CRIRES+ bin experiment:** another incommensurate factor (`--xbin 1.5` or `sqrt2`), more
  lines (other swaths/orders) to beat the ~100 mpx noise floor of the stacked BIS, and the
  residual h4 bias common to all recovered products.

