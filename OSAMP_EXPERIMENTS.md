# osamp_spec experiments

Empirical validation of `osamp_spec` super-resolution. Feature reference and
design decisions: `OSAMP_SPEC.md`. This document collects the bisector primer,
the synthetic asymmetric-LSF recovery study, and the real-CRIRES bin-x2
self-calibration test.

Top-level conclusion across all of it: dispersion-direction oversampling
recovers genuine sub-pixel line structure **only** when the line is truly
under-sampled and the slit carries enough tilt-phase diversity; otherwise the
fine spectrum collapses to the block-mean staircase. The robust product even in
the marginal case is the sub-pixel centroid / wavelength zero-point, not a
narrower profile.

---

# 1. Bisector primer

A short introduction for someone who has not used line bisectors before, in the
specific context of how we use them here (`scripts/analyse_wide_bisector.py`,
the wide-LSF experiment, RV-style shape diagnostics).

## What it is

Take any line profile `y(x)` with a single peak. Pick a fractional level `h`
between 0 and 1, draw the horizontal line at `y = h * max(y)`, and find where
that line crosses the profile on the left and on the right (`x_L(h)`,
`x_R(h)`). The **bisector** is the midpoint:

```
x_B(h) = ( x_L(h) + x_R(h) ) / 2
```

Sweep `h` from near the continuum (e.g. 0.1) up to near the peak (e.g. 0.95)
and you get a curve `x_B(h)` -- the bisector. It tells you where the "middle"
of the line is at every depth.

For a perfectly symmetric line (Gaussian, top-hat, sinc, anything mirror-
symmetric about its centre) the bisector is a vertical line: `x_B(h)` is the
same number for every `h`.

For an asymmetric line the bisector tilts or curves. A line with a blue wing
(extra flux on the low-x side) gets *wider on the blue side at low levels*, so
`x_L` is further left near the continuum than near the peak, and `x_B` shifts
blueward as `h` decreases. The shape of the bisector encodes the asymmetry of
the profile in a way that is largely independent of overall width or amplitude.

## Why we care: the bisector is the RV-shape diagnostic

Radial-velocity pipelines (HARPS, ESPRESSO, EXPRES, ...) measure a star's
velocity by cross-correlating its spectrum against a template. The output is a
CCF, which is essentially an average line profile. A *shift* in that profile is
interpreted as a Doppler shift. A *shape change* in that profile (stellar
activity, granulation, spots rotating in and out of view) also shifts the
centroid in a way that looks identical to a Doppler shift in the CCF, but is
not a real velocity.

The bisector is what lets you tell them apart:

- A **real Doppler shift** translates the whole profile rigidly: every
  `x_B(h)` moves by the same amount. The bisector keeps its shape and just
  slides horizontally.
- A **shape change** (asymmetric LSF distortion, stellar activity signal,
  instrumental imperfection in the line wings) tilts or warps the bisector.
  The centroid moves but `x_B(h)` is no longer a vertical line.

HARPS et al. summarise this with a single number, the **bisector inverse
slope** (BIS):

```
BIS = mean( x_B for h in 0.60..0.85 )  -  mean( x_B for h in 0.10..0.40 )
```

i.e. the average bisector position near the line top minus the average near the
line bottom. BIS = 0 means a vertically straight bisector (symmetric line).
BIS != 0 means there is real asymmetry. If a star's *radial velocity* and its
*BIS* are correlated across many exposures, the "RV signal" is almost certainly
a shape distortion -- stellar activity mimicking a planet, or an instrumental
shape artefact -- not an orbiting companion. This is a routine vetting step for
every published planet detection from these instruments.

The unit of BIS is whatever pixel/wavelength unit the bisector is in. We report
it in **mpx** (milli-pixels). At ESPRESSO-class dispersion that is roughly
0.5 - 1 m/s per mpx, so a BIS error of 10 mpx is already at the m/s level.

## How we use it here

In `scripts/experiment_synthetic_lsf.py` we generate a known asymmetric LSF (a
0.8 px top-hat core convolved with a 0.25 px Gaussian, plus a 15% blue
exponential wing) and try to recover it with `slitdec` at different
`osamp_spec` values. The recovered fine spectrum near the single emission line
*is* the recovered LSF, in detector-pixel units. `scripts/analyse_wide_bisector.py`:

1. Computes the bisector of the **truth** LSF on its dense 1/50-px grid.
2. Computes the bisector of each **recovered** LSF (one per `osamp_spec`) on
   its own fine grid.
3. Computes BIS for each. The relevant quantity is the **bias**: recovered BIS
   minus truth BIS. A bias of 0 means the pipeline reproduces the line's
   asymmetry correctly; a non-zero bias means the pipeline will spit out a
   systematic that correlates with the line's sub-pixel position -- exactly
   what mimics a planet in CCF analysis.

## Pitfalls

- Bisectors require a profile with a clear single peak. Noisy spectra with
  multiple local maxima will give nonsense crossings.
- The choice of level windows for BIS is conventional but not unique; HARPS
  uses 0.10..0.40 and 0.60..0.90 on the continuum-normalised CCF. We follow
  that here.
- A bisector measured on a too-coarsely-sampled profile (e.g. `osamp_spec=1`
  on an under-sampled LSF) is dominated by the bin edges -- it is not a
  measurement of the underlying LSF shape so much as of how the LSF happened to
  fall on the detector grid.

## References

- Queloz et al. 2001, "No planet for HD 166435", A&A 379, 279
  ([ADS](https://ui.adsabs.harvard.edu/abs/2001A&A...379..279Q/abstract)) --
  classic example of using bisector / RV correlation to retract a candidate
  detection in favour of a star-spot explanation.
- Toner & Gray 1988, "The Star Patch on the G8 Dwarf chi Bootis A", ApJ 334,
  1008 ([ADS](https://ui.adsabs.harvard.edu/abs/1988ApJ...334.1008T/abstract))
  -- introduced the bisector *velocity span*, the conceptual ancestor of
  HARPS-style BIS.
- ESPRESSO pipeline description (DRS): the CCF and its BIS are standard
  pipeline products precisely for the reasons above.

---

# 2. Synthetic asymmetric-LSF recovery on a variable-tilt slit

Script: `scripts/experiment_synthetic_lsf.py` (standalone, run with
`uv run scripts/experiment_synthetic_lsf.py`; `SHOW=0` for batch).

## Setup

- 80 rows x 60 columns, single **unresolved** emission line at column 30.37
  (deliberately off pixel centre).
- **Asymmetric LSF**: top-hat core (0.8 px full width) convolved with a
  0.25 px-sigma Gaussian, plus a blue (low-x) exponential wing (15% of flux,
  0.8 px scale). True FWHM = 0.894 px, i.e. under-sampled at the detector.
- **Variable slit tilt**: 0.1 px/row at the bottom row, decreasing linearly to
  0.001 px/row at the top row. Integrated, that is a quadratic horizontal
  displacement, exactly representable by the degree-2 slitcurve in the frame
  slitdec uses (t = y - nrows/2): c1 = 0.049873, c2 = -6.266e-4. Total
  horizontal shift across the slit: ~4 px.
- The image generation integrates 8 sub-row positions per detector row, so the
  within-row shear of the varying tilt is in the data.
- Photon (Poisson) noise + 2 e- read noise; peak pixel ~3200 counts. A
  noiseless realisation with identical weighting is extracted alongside.
- Extraction: `osamp_spec` s = 1, 2, 3; `lambda_fringe` in
  {0, 0.001, 0.01, 0.1, 1}; `lambda_sP = 0`, `kappa = 0`.

## Metrics

All profiles are area-normalised over a +/-5 px window around the line.

- `rms`: residual vs the true LSF binned onto the extraction's own fine grid.
- `rms_cont`: residual of the recovered staircase vs the **continuous** true
  LSF, evaluated on a common 1/50-px grid -- comparable across s, includes
  each grid's intrinsic binning error.
- `FWHM`, blue/red wing flux (integral of the normalised profile over 1-4 px on
  each side of the centre), centroid error.

## Results (noisy case, best lambda_fringe per s = 0 everywhere)

| quantity            | truth  | s=1    | s=2    | s=3    |
|---------------------|--------|--------|--------|--------|
| FWHM [px]           | 0.894  | 1.112  | 0.945  | 0.905  |
| blue wing flux      | 0.046  | -0.004 | 0.044  | 0.042  |
| red wing flux       | 0.001  | -0.046 | 0.003  | -0.000 |
| centroid error [px] | —      | +0.000 | -0.003 | -0.008 |
| rms_cont            | —      | 0.107  | 0.081  | 0.053  |

(centroid error quoted relative to the truth's own -0.107 px centroid offset,
which comes from the blue wing.)

Noiseless numbers are nearly identical (rms_cont 0.108 / 0.081 / 0.051), i.e.
at this S/N the recovery is **geometry-limited, not noise-limited**.

## Findings

1. **The LSF is substantially recoverable.** At s=3 the recovered FWHM is
   within 1.2% of truth (vs 24% too broad at s=1), and the blue/red wing
   asymmetry is recovered with the correct sign and ~90% of the correct
   amplitude. Shape error vs the continuous LSF halves from s=1 to s=3
   (rms_cont 0.107 -> 0.053). The flat-top character of the core is visible in
   the s=2/3 reconstructions.
2. **lambda_fringe = 0 is optimal here.** With ~4 px of total tilt sweep (full
   sub-pixel phase coverage several times over), the fine modes are well
   constrained and no fringe ripple appears. Any lambda_fringe > 0 (even 0.001)
   erodes genuine super-resolution content: it drags the recovered profile
   toward the block-mean staircase and flips the apparent asymmetry. This is
   the constructive counterpart of the Hsim finding in OSAMP_SPEC.md: there the
   content was pure fringe noise and lambda_fringe correctly collapsed it; here
   the content is real and lambda_fringe should be off (or very small).
3. **Control (constant 0.001 px/row tilt, `--tag _notilt`)**: without
   meaningful tilt, s>1 at lambda_fringe=0 is dominated by the fringe mode
   (rms_cont 0.27 at s=2) and lambda_fringe=0.01 collapses it back to exactly
   the s=1 staircase quality (rms_cont 0.096 vs 0.098). Confirms the
   super-resolution signal comes from the tilt-induced phase diversity, not
   from the solver inventing structure.
4. **The varying tilt is fine for the forward model.** The 0.1 -> 0.001 px/row
   linear variation is exactly a degree-2 slitcurve; nothing special is needed.
   The information content is dominated by the high-tilt (bottom) half of the
   slit; the top half contributes mostly flux, not phase diversity.
5. **Sub-pixel centroid** is recovered to better than 0.01 px at every s,
   consistent with OSAMP_SPEC.md's note that the wavelength zero-point is the
   robust product even when shape recovery is marginal.
6. **Gains continue past s=3 and saturate near s=5**
   (`--osamps 1 2 3 4 5 6 --fringes 0 --tag _ssweep`): rms_cont = 0.107, 0.081,
   0.053, 0.039, 0.0345, 0.0355 for s = 1..6. Beyond s~5 the geometry/noise
   floor is reached and the recovered FWHM starts to undershoot slightly
   (0.85-0.87 px vs 0.894 truth).
7. **At low S/N the optimal lambda_fringe becomes small-but-nonzero**
   (`--amp 100 --tag _lowsn`, peak ~90 counts): lambda_fringe=0 lets fringe
   noise leak in (rms_cont 0.12-0.14 at s=2/3, centroid off by up to 0.2 px),
   while lambda_fringe=0.001 restores a clean recovery (rms_cont ~0.091, still
   beating s=1 at 0.112). Practical rule: lambda_fringe should scale with the
   noise level -- 0 when tilt-phase coverage is strong and S/N is high, ~1e-3
   otherwise.

## Width sweep: how recoverability depends on LSF width

Re-run with `--tophat-w` and `--sigma` to scan the core width (the blue wing
convolution kernel is left unchanged at 15% / tau=0.8 px). Three configurations
bracket the baseline:

| case      | top-hat / sigma (px) | true FWHM (px) | true blue / red wing |
|-----------|----------------------|----------------|----------------------|
| narrow    | 0.4 / 0.125          | 0.444          | 0.042 / 0.000        |
| baseline  | 0.8 / 0.25           | 0.894          | 0.046 / 0.001        |
| wide      | 1.6 / 0.5            | 1.785          | 0.115 / 0.064        |

Recovered FWHM and rms_cont, noisy case, best `lambda_fringe` per s:

| case     | quantity   | truth | s=1   | s=2   | s=3   |
|----------|------------|-------|-------|-------|-------|
| narrow   | FWHM [px]  | 0.444 | 0.991 | 0.890 (l_fr=1e-3) | 0.597 |
| narrow   | rms_cont   | -     | 0.240 | 0.209 | 0.156 |
| baseline | FWHM       | 0.894 | 1.112 | 0.945 | 0.905 |
| baseline | rms_cont   | -     | 0.107 | 0.081 | 0.053 |
| wide     | FWHM       | 1.785 | 1.767 | 1.789 | 1.818 |
| wide     | rms_cont   | -     | 0.058 | 0.028 | 0.019 |

8. **The narrower the LSF, the bigger the absolute win from `osamp_spec > 1`
   -- but also the higher the residual floor.** At FWHM=0.44 px (severely
   under-sampled) s=1 is hopeless (FWHM 2.2x too broad), s=3 closes most of the
   gap but the truth is still only ~1.3 fine bins wide. Around s=2 the optimal
   `lambda_fringe` already shifts to small-but-nonzero (1e-3 wins), consistent
   with the high-noise regime in finding 7.
9. **At realistic ESPRESSO-like sampling (~2 px FWHM), s=1 nails the FWHM to
   <2%.** Width is not the diagnostic that exposes the staircase artefact here
   -- shape is. See the next section.

## Wide case (ESPRESSO-like): bisector analysis

For a spectrograph with FWHM slightly under 2 px (the regime where
`osamp_spec=1` already gets the width right), the question is whether shape is
also right. The relevant metric is the **bisector** (section 1). s-sweep at
`lambda_fringe = 0`, wide LSF (FWHM 1.785 px), noisy case, generated by
`experiment_synthetic_lsf.py --tophat-w 1.6 --sigma 0.5 --osamps 1 2 3 4 5 6
--fringes 0 --tag _wide_ssweep` and analysed by
`scripts/analyse_wide_bisector.py`:

| s  | recovered BIS [mpx] | **BIS bias vs truth [mpx]** | blue wing | red wing |
|----|---------------------:|----------------------------:|----------:|---------:|
| truth | +24.9            | 0                           | 0.115     | 0.064    |
| 1  | +105.1               | **+80.2**                   | 0.022     | 0.134    |
| 2  | +19.6                | -5.3                        | 0.137     | 0.036    |
| 3  | +30.8                | +5.9                        | 0.099     | 0.069    |
| 4  | +29.2                | +4.3                        | 0.085     | 0.089    |
| 5  | +21.4                | -3.5                        | 0.116     | 0.059    |
| 6  | +13.3                | -11.6                       | 0.102     | 0.070    |

10. **s=1 inverts the asymmetry sign and biases BIS by +80 mpx.** Block-mean
    staircase puts the wing flux on whichever side of the sub-pixel centroid
    the line happens to sit (blue wing comes out as red, etc.). At ESPRESSO-
    class dispersion this is roughly tens of m/s of fake RV signal correlated
    with the sub-pixel centroid -- exactly what mimics a planet in CCF
    analysis. **FWHM-only validation is not enough for RV work**; the bisector
    is.
11. **s = 2..5 collapse the BIS bias to single-digit mpx.** This is the regime
    where shape is recovered well enough that the pipeline does not introduce
    its own systematic. s=4-5 look slightly better than s=3 here but are
    outside the explicitly validated range -- see "Deferred work" in
    OSAMP_SPEC.md before recommending them.
12. **s=6 starts under-correcting** (BIS -12 mpx vs truth), consistent with
    OSAMP_SPEC.md's prediction that gains saturate near s=5 and the fringe-mode
    null begins to dominate. At s >= 6 `lambda_fringe` would need to come back
    on.

## Outputs

All gitignored (under `scratch/` for batch runs); regenerate with the commands
above. `_notilt` (control), `_narrow`, `_wide`, `_wide_ssweep` tags as
referenced; `scripts/analyse_wide_bisector.py` produces the bisector diagnostic.

---

# 3. Real CRIRES+ U-Ne bin-x2 self-calibration

Self-calibration test of `osamp_spec` on real data. The native CRIRES+ swath is
well sampled, so the unbinned extraction at `osamp_spec=1` is ground truth; the
same columns, rebinned in dispersion (under-sampled), are extracted with
`osamp_spec=s` and compared to the truth line by line.

Script: `scripts/experiment_binx2_crires.py` (`--xbin 2` default, accepts
fractional factors like `sqrt3`). Outputs to gitignored
`scratch/experiment_binx2_crires_b{xbin}_{s2,s3,s4,gh}.png` + `.npz`.

## Design

The native data (instrumental FWHM ~2.9 px) is well sampled, so binning x2 in
dispersion (FWHM ~1.45 binned px, ESPRESSO-like) and extracting with
`osamp_spec=2` lands on **exactly the native pixel grid** -- a direct,
resampling-free comparison with real noise, real LSF, real tilt.

Data (in `data/`, gitignored):

- `data/CRIRES_UNE_J.fits` -- CRIRES+ U-Ne lamp, reduced (UTIL_CALIB), 3 chips,
  each with `CHIPn.INT1` (image, ADU) + `CHIPnERR.INT1` (errors).
- `data/J1228_tw.fits` -- cr2res trace-wave table per chip: `All` = trace y(x)
  polynomial (increasing powers, **1-based** pixel coords), `SlitPolyA/B/C` =
  a(x), b(x), c(x) of the tilt polynomial x'(y) = a + b*y + c*y^2 in absolute
  detector coords.

`scripts/make_swath.py` cuts the swath: integer-aligns each column on the
trace, passes only frac(ycen), and converts tilt to slitdec's t-relative-to-
trace-centre frame (`c1 = b + 2*c*yc`, `c2 = c`). With `xbin=k`, pairs/groups
of native columns are cut with a single shared integer shift (from the
group-centre trace) and summed; uncertainties in quadrature; `c1, c2` divided
by the bin factor; frac-ycen from the group centre. Fractional `xbin` is
supported (flux-conserving splitting of boundary columns; integer paths
verified bit-identical to before).

Chosen swath: `--chip 2 --order 4 --x0 512 --width 512 --nrows 150` (wl
1253.7-1255.7 nm, tilt -0.053 px/row => ~8 px sweep over the slit, S/N ~100 per
column). Seven analysis lines at swath columns 37.8, 60.5, 68.5, 169.2, 310.7,
324.6, 444.6; 310.7/324.6 are blends at full-slit level; 60.5/68.5 are only 8 px
apart (overlapping windows).

Products per line and stacked: truth (xbin=1, s=1), staircase (binned, s=1),
recovered (binned, s in {2,3,4} x lambda_fringe in {0, 1e-3, 1e-2}). Metrics:
half-max FWHM, flux centroid, profile rms vs truth, bisector + BIS
(HARPS-style), and bin-integrated Gauss-Hermite fits (h3 = skew, h4 = kurtosis;
|h3|,|h4| <= 0.35, free pedestal).

## Findings, xbin=2 (FWHM ~1.45 binned px, but commensurate!)

1. **Centroids**: staircase has phase-dependent centroid errors up to 0.31 px
   (mean 0.19); s=2 lf=1e-3 reduces this to 0.025 px (~8x). s=3/4 are worse
   (~0.07) -- s=2 fine bins land exactly on native pixel centres, flattering
   this comparison (see finding 7).
2. **Profile rms**: s=2 lf=1e-3 beats the staircase at all 7 lines by ~25-35%.
   Genuine sub-pixel content, not an Hsim-like collapse.
3. **Bisector**: per-line BIS at S/N~100 is noise (+-300-500 mpx; the fine grid
   amplifies noise). Stacked-profile BIS: truth -162 mpx, staircase -278,
   s=2/3/4 lf=1e-3 all -260..-305 -- no recovery, and the ~110 mpx offset is
   ~1 sigma of the 7-line stack noise floor.
4. **s>2 adds nothing**: the binning factor is 2, so s=2 already reaches the
   information ceiling (the native grid); s=3/4 spread the same information over
   noisier bins (lf=0 collapses to ~1 px spikes).
5. **Gauss-Hermite, s=2 pathological**: h4 pegs at the bound (+0.35) with
   FWHM_G collapsing to ~2.0 at nearly every line, and stronger lambda_fringe
   does NOT cure it -- ringing phase-locked at the native-pixel period (=
   exactly the s=2 fringe mode). s=3/4 recover FWHM_G within ~0.15 px and h3
   sign/magnitude at most lines; h4 is systematically biased positive
   everywhere.
6. **Staircase forward-fitting caveat**: bin-integrated GH fitting of the 2-px
   staircase recovers centroids to +-0.07 px -- much better than its naive flux
   centroid (0.19 px). The model-free advantage of `osamp_spec` is the
   resampled product itself, not something a careful per-line forward fit could
   not partially match. Per-line h3/h4 from the staircase are meaningless
   (fewer points than parameters).

## Findings, xbin=sqrt(3) (FWHM ~1.67 binned px, incommensurate)

7. **The commensurability was distorting the test in both directions.** With an
   irrational bin factor (no integer relation between binned and native grids --
   the realistic case for an undersampled spectrograph):
   - The s=2 GH pathology vanishes: s=2/3/4 give nearly identical, well-behaved
     fits; FWHM_G within ~0.1 px and dmu < 0.05 px of truth at the clean lines;
     h3 recovered in sign but washed toward 0; h4 still biased (now toward boxy
     at some lines).
   - The s=2 centroid advantage of finding 1 was partly grid-alignment
     flattery: with sqrt(3), clean-line dcent is ~0.06 px for all s.
8. **lambda_fringe flips role.** The detector-box Fourier null no longer
   coincides with the osamp-period mode, so the fringe mode is not degenerate:
   lf=0 at s=2 becomes the best per-line product (rms 0.023-0.035, beating
   lf=1e-3 and the staircase), while lf>0 over-smooths. Stacked-profile BIS:
   **lf=0 tracks the truth across all three s values (-137/-174/-159 mpx vs
   truth -162)** while lf=1e-3 (-330..-365) and the staircase (-400) are far
   off. Three independent extractions agreeing within ~20 mpx suggests this is
   real -- the first configuration in this experiment that recovers the truth
   bisector. Practical recipe for realistic sampling: little or no fringe
   regularization; the lambda_fringe~1e-3 guidance from the synthetics was
   partly compensating for commensurate test setups.
9. Remaining lf=0 caveats: at s>=3 individual faint lines still collapse to
   spikes (FWHM 0.3-1.3 px); per-line BIS remains noise; the bisector statement
   holds only for the 7-line stack.

## Methodology notes / caveats

- The plan's line FWHMs (2.5-2.9 px) were Gaussian fits to flat-topped
  profiles; the half-max width of the same data is 3.0-3.4 px. The truth
  extraction adds no measurable smear.
- Truth GH shape: FWHM_G 2.66-2.78 px, h3 ~ +0.02..+0.07, h4 ~ -0.05..-0.09
  (slightly boxy, as expected for a slit-image LSF).
- Fractional rebinning of pixel-integrated data carries the native 1-px box
  along (a real coarse detector would be box(xbin) only) and edge-shared pixels
  mildly correlate adjacent bin noise (slitdec assumes independence). Both
  affect all products equally.
- The truth is itself an extraction with ~1% noise; comparisons are "how much
  do we recover", not few-mpx claims.
- slitdec evaluates tilt at t relative to the trace centre row
  (`y_lower_lim = nrows/2`); `cut_swath` returns slitcurve in that frame. The C
  code zeroes the first/last `delta_x * osamp_spec` fine bins; with ~8 px sweep
  delta_x ~ 3 (binned), so ~6 fine bins at each edge are dead -- all 7 lines
  are well inside.

## Open items

- Counter-check finding 8 with another incommensurate factor (e.g. `--xbin 1.5`
  or `sqrt2`).
- More lines (other swaths/orders, survey mode in `make_swath`) to beat the
  ~100 mpx stacked-BIS noise floor.
- Understand the residual h4 bias common to all recovered products.
