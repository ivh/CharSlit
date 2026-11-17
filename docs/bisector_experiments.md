# Bisector primer and the sampling-recovery experiments

What the bisector is and why it is the metric here, and the design of the two experiments
that test whether the fine spectrum (`spectrum_fine`, `osample_x > 1`) recovers line shape
below the pixel scale. Written in June 2026 for the `osamp_spec` branch and trimmed when
that branch was folded into `slitdec_ox`: the results obtained with the old engine are
removed (they are in the history of this file and on the `osamp_spec` branch); current
results are in [ox_experiments.md](ox_experiments.md).

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
`osample_x` values. The recovered fine spectrum near the single emission line
*is* the recovered LSF, in detector-pixel units. `scripts/analyse_wide_bisector.py`:

1. Computes the bisector of the **truth** LSF on its dense 1/50-px grid.
2. Computes the bisector of each **recovered** LSF (one per `osample_x`) on
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
- A bisector measured on a too-coarsely-sampled profile (e.g. `osample_x=1` (linear split)
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
- Extraction: `osample_x` s = 1, 2, 3 (s = 1 is the linear split of slitdec.c); penalty
  `mu3` in {1e-6, 8.2e-6, 1.64e-4} (`--fringes`); `lambda_sP = 0`, `kappa = 0`.

## Metrics

All profiles are area-normalised over a +/-5 px window around the line.

- `rms`: residual vs the true LSF binned onto the extraction's own fine grid.
- `rms_cont`: residual of the recovered staircase vs the **continuous** true
  LSF, evaluated on a common 1/50-px grid -- comparable across s, includes
  each grid's intrinsic binning error.
- `FWHM`, blue/red wing flux (integral of the normalised profile over 1-4 px on
  each side of the centre), centroid error.


## Outputs

All gitignored (under `scratch/` for batch runs); regenerate with the commands
above. Tags used: `_ox` (baseline), `_notilt` (control), `_lowsn`, `_ssweep`, `_narrow`,
`_wide`, `_wide_ssweep`; `scripts/analyse_wide_bisector.py` produces the bisector diagnostic.

---

# 3. Real CRIRES+ U-Ne bin-x2 self-calibration

Self-calibration test of the fine spectrum on real data. The native CRIRES+ swath is
well sampled, so the unbinned extraction at `osample_x=1` (linear split) is ground truth; the
same columns, rebinned in dispersion (under-sampled), are extracted with
`osample_x=s` and compared to the truth line by line.

Script: `scripts/experiment_binx2_crires.py` (`--xbin 2` default, accepts
fractional factors like `sqrt3`). Outputs to gitignored
`scratch/experiment_binx2_crires_b{xbin}_{s2,s3,s4,gh}.png` + `.npz`.

## Design

The native data (instrumental FWHM ~2.9 px) is well sampled, so binning x2 in
dispersion (FWHM ~1.45 binned px, ESPRESSO-like) and extracting with
`osample_x=2` lands on **exactly the native pixel grid** -- a direct,
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
recovered (binned, s in {2,3,4} x mu3 in {1e-6, 8.2e-6, 1.64e-4}). Metrics:
half-max FWHM, flux centroid, profile rms vs truth, bisector + BIS
(HARPS-style), and bin-integrated Gauss-Hermite fits (h3 = skew, h4 = kurtosis;
|h3|,|h4| <= 0.35, free pedestal).


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
  code then zeroed the first/last `delta_x * osample_x` fine bins; with ~8 px sweep
  delta_x ~ 3 (binned), so ~6 fine bins at each edge are dead -- all 7 lines
  are well inside.

## Open items

- Counter-check the sqrt(3) result with another incommensurate factor (e.g. `--xbin 1.5`
  or `sqrt2`).
- More lines (other swaths/orders, survey mode in `make_swath`) to beat the
  ~100 mpx stacked-BIS noise floor.
- Understand the residual h4 bias common to all recovered products.
