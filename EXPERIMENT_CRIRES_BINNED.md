# Experiment: bin-x2 / bin-sqrt(3) LSF recovery on real CRIRES+ U-Ne data

Self-calibration test of `osamp_spec` on real data (see plan.md for the
original design). The native CRIRES+ swath (chip 2, order 4, x0=512,
width=512, nrows=150; tilt -0.053 px/row, ~8 px sweep; S/N ~100 per
column) is well sampled (half-max FWHM ~3.0-3.4 px), so the unbinned
extraction at `osamp_spec=1` serves as ground truth. The same columns,
rebinned in dispersion, are extracted with `osamp_spec=s` and compared
to the truth line by line.

Script: `scripts/experiment_binx2_crires.py` (`--xbin 2` default,
accepts fractional factors like `sqrt3`). Outputs to gitignored
`scratch/experiment_binx2_crires_b{xbin}_{s2,s3,s4,gh}.png` + `.npz`.
`cut_swath` in `scripts/make_swath.py` now supports fractional `xbin`
(flux-conserving splitting of boundary columns, single shared integer
trace shift per bin; integer paths verified bit-identical to before).

Products per line (7 lamp lines; 310.7/324.6 are blends at full-slit
level) and stacked: truth (xbin=1, s=1), staircase (binned, s=1),
recovered (binned, s in {2,3,4} x lambda_fringe in {0, 1e-3, 1e-2}).
Metrics: half-max FWHM, flux centroid, profile rms vs truth, bisector +
BIS (HARPS-style), and bin-integrated Gauss-Hermite fits (h3 = skew,
h4 = kurtosis; |h3|,|h4| <= 0.35, free pedestal).

## Findings, xbin=2 (FWHM ~1.45 binned px, but commensurate!)

1. **Centroids**: staircase has phase-dependent centroid errors up to
   0.31 px (mean 0.19); s=2 lf=1e-3 reduces this to 0.025 px (~8x).
   s=3/4 are worse (~0.07) — s=2 fine bins land exactly on native
   pixel centres, flattering this comparison (see finding 7).
2. **Profile rms**: s=2 lf=1e-3 beats the staircase at all 7 lines by
   ~25-35%. Genuine sub-pixel content, not an Hsim-like collapse.
3. **Bisector**: per-line BIS at S/N~100 is noise (+-300-500 mpx; the
   fine grid amplifies noise). Stacked-profile BIS: truth -162 mpx,
   staircase -278, s=2/3/4 lf=1e-3 all -260..-305 — no recovery, and
   the ~110 mpx offset is ~1 sigma of the 7-line stack noise floor.
4. **s>2 adds nothing**: the binning factor is 2, so s=2 already
   reaches the information ceiling (the native grid); s=3/4 spread the
   same information over noisier bins (lf=0 collapses to ~1 px spikes).
5. **Gauss-Hermite, s=2 pathological**: h4 pegs at the bound (+0.35)
   with FWHM_G collapsing to ~2.0 at nearly every line, and stronger
   lambda_fringe does NOT cure it — ringing phase-locked at the
   native-pixel period (= exactly the s=2 fringe mode). s=3/4 recover
   FWHM_G within ~0.15 px and h3 sign/magnitude at most lines; h4 is
   systematically biased positive everywhere.
6. **Staircase forward-fitting caveat**: bin-integrated GH fitting of
   the 2-px staircase recovers centroids to +-0.07 px — much better
   than its naive flux centroid (0.19 px). The model-free advantage of
   `osamp_spec` is the resampled product itself, not something a
   careful per-line forward fit could not partially match. Per-line
   h3/h4 from the staircase are meaningless (fewer points than
   parameters).

## Findings, xbin=sqrt(3) (FWHM ~1.67 binned px, incommensurate)

7. **The commensurability was distorting the test in both directions.**
   With an irrational bin factor (no integer relation between binned
   and native grids — the realistic case for an undersampled
   spectrograph):
   - The s=2 GH pathology vanishes: s=2/3/4 give nearly identical,
     well-behaved fits; FWHM_G within ~0.1 px and dmu < 0.05 px of
     truth at the clean lines; h3 recovered in sign but washed toward
     0; h4 still biased (now toward boxy at some lines).
   - The s=2 centroid advantage of finding 1 was partly grid-alignment
     flattery: with sqrt(3), clean-line dcent is ~0.06 px for all s.
8. **lambda_fringe flips role.** The detector-box Fourier null no
   longer coincides with the osamp-period mode, so the fringe mode is
   not degenerate: lf=0 at s=2 becomes the best per-line product
   (rms 0.023-0.035, beating lf=1e-3 and the staircase), while lf>0
   over-smooths. Stacked-profile BIS: **lf=0 tracks the truth across
   all three s values (-137/-174/-159 mpx vs truth -162)** while
   lf=1e-3 (-330..-365) and the staircase (-400) are far off. Three
   independent extractions agreeing within ~20 mpx suggests this is
   real — the first configuration in this experiment that recovers the
   truth bisector. Practical recipe for realistic sampling: little or
   no fringe regularization; the lambda_fringe~1e-3 guidance from the
   synthetics was partly compensating for commensurate test setups.
9. Remaining lf=0 caveats: at s>=3 individual faint lines still
   collapse to spikes (FWHM 0.3-1.3 px); per-line BIS remains noise;
   the bisector statement holds only for the 7-line stack.

## Methodology notes / caveats

- plan.md's line FWHMs (2.5-2.9 px) were Gaussian fits to flat-topped
  profiles; the half-max width of the same data is 3.0-3.4 px. The
  truth extraction adds no measurable smear.
- Truth GH shape: FWHM_G 2.66-2.78 px, h3 ~ +0.02..+0.07,
  h4 ~ -0.05..-0.09 (slightly boxy, as expected for a slit-image LSF).
- Fractional rebinning of pixel-integrated data carries the native
  1-px box along (a real coarse detector would be box(xbin) only) and
  edge-shared pixels mildly correlate adjacent bin noise (slitdec
  assumes independence). Both affect all products equally.
- The truth is itself an extraction with ~1% noise; comparisons are
  "how much do we recover", not few-mpx claims.

## Open items

- Counter-check finding 8 with another incommensurate factor
  (e.g. `--xbin 1.5` or `sqrt2`).
- More lines (other swaths/orders, survey mode in make_swath) to beat
  the ~100 mpx stacked-BIS noise floor.
- Understand the residual h4 bias common to all recovered products.
