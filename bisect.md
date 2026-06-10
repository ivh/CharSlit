# Bisector primer

A short introduction for someone who has not used line bisectors before, in
the specific context of how we use them in this repo
(`analyse_wide_bisector.py`, the wide-LSF experiment, RV-style shape
diagnostics).

## What it is

Take any line profile `y(x)` with a single peak. Pick a fractional level
`h` between 0 and 1, draw the horizontal line at `y = h * max(y)`, and find
where that line crosses the profile on the left and on the right
(`x_L(h)`, `x_R(h)`). The **bisector** is the midpoint:

```
x_B(h) = ( x_L(h) + x_R(h) ) / 2
```

Sweep `h` from near the continuum (e.g. 0.1) up to near the peak (e.g.
0.95) and you get a curve `x_B(h)` -- the bisector. It tells you where the
"middle" of the line is at every depth.

For a perfectly symmetric line (Gaussian, top-hat, sinc, anything mirror-
symmetric about its centre) the bisector is a vertical line: `x_B(h)` is
the same number for every `h`.

For an asymmetric line the bisector tilts or curves. A line with a blue
wing (extra flux on the low-x side) gets *wider on the blue side at low
levels*, so `x_L` is further left near the continuum than near the peak,
and `x_B` shifts blueward as `h` decreases. The shape of the bisector
encodes the asymmetry of the profile in a way that is largely independent
of overall width or amplitude.

## Why we care: the bisector is the RV-shape diagnostic

Radial-velocity pipelines (HARPS, ESPRESSO, EXPRES, ...) measure a star's
velocity by cross-correlating its spectrum against a template. The output
is a CCF, which is essentially an average line profile. A *shift* in that
profile is interpreted as a Doppler shift. A *shape change* in that
profile (stellar activity, granulation, spots rotating in and out of view)
also shifts the centroid in a way that looks identical to a Doppler shift
in the CCF, but is not a real velocity.

The bisector is what lets you tell them apart:

- A **real Doppler shift** translates the whole profile rigidly: every
  `x_B(h)` moves by the same amount. The bisector keeps its shape and
  just slides horizontally.
- A **shape change** (asymmetric LSF distortion, stellar activity signal,
  instrumental imperfection in the line wings) tilts or warps the
  bisector. The centroid moves but `x_B(h)` is no longer a vertical line.

HARPS et al. summarise this with a single number, the **bisector inverse
slope** (BIS):

```
BIS = mean( x_B for h in 0.60..0.85 )  -  mean( x_B for h in 0.10..0.40 )
```

i.e. the average bisector position near the line top minus the average
near the line bottom. BIS = 0 means a vertically straight bisector
(symmetric line). BIS != 0 means there is real asymmetry. If a star's
*radial velocity* and its *BIS* are correlated across many exposures, the
"RV signal" is almost certainly a shape distortion -- stellar activity
mimicking a planet, or an instrumental shape artefact -- not an orbiting
companion. This is a routine vetting step for every published planet
detection from these instruments.

The unit of BIS is whatever pixel/wavelength unit the bisector is in. We
report it in **mpx** (milli-pixels). At ESPRESSO-class dispersion that is
roughly 0.5 - 1 m/s per mpx, so a BIS error of 10 mpx is already at the
m/s level.

## How we use it in this repo

In `experiment_synthetic_lsf.py` we generate a known asymmetric LSF (a
0.8 px top-hat core convolved with a 0.25 px Gaussian, plus a 15% blue
exponential wing) and try to recover it with `slitdec` at different
`osamp_spec` values. The recovered fine spectrum near the single emission
line *is* the recovered LSF, in detector-pixel units.

`analyse_wide_bisector.py`:

1. Computes the bisector of the **truth** LSF on its dense 1/50-px grid.
2. Computes the bisector of each **recovered** LSF (one per `osamp_spec`
   value) on its own fine grid.
3. Computes BIS for each, both for the truth and for each recovered
   profile. The relevant quantity is the **bias**: recovered BIS minus
   truth BIS. A bias of 0 means the pipeline reproduces the line's
   asymmetry correctly; a non-zero bias means the pipeline will spit out
   a systematic that correlates with the line's sub-pixel position --
   exactly what mimics a planet in CCF analysis.

What we saw on the wide-LSF case (true FWHM 1.785 px, ESPRESSO-like
under-sampling):

- Truth BIS: about +25 mpx (blue wing tilts the bisector blueward at
  lower levels)
- `osamp_spec=1`: recovered BIS about +105 mpx, **bias = +80 mpx**.
  Block-mean staircase puts the asymmetry on whichever side of the
  centred sub-pixel the line happens to sit -- a bias of about +80 mpx
  on the bisector is roughly tens of m/s of fake RV signal, correlated
  with the sub-pixel centroid. Catastrophic for high-precision RV work.
- `osamp_spec >= 2`: bias collapses to single-digit mpx and stays there
  for s=2..5. This is the regime where `slitdec` *recovers the LSF
  shape* well enough that the bisector does not introduce its own
  systematic.

That is the whole reason BIS is the metric we care about for the wide
case, even though FWHM was already fine at `osamp_spec=1`: **the
pipeline's BIS bias is what would show up as a fake planet, FWHM is
not**.

## Reading bisector plots in this repo

`analyse_wide_bisector.png`, left panel:

- y-axis: fractional level `h` of the line (0.1 near continuum, 0.95
  near peak).
- x-axis: bisector position relative to line centre, in mpx.
- Dashed grey: truth bisector. A modestly blue-leaning curve.
- Coloured lines: bisectors recovered at each `osamp_spec`.
- The s=1 curve is wildly off, displaced and warped. The s>=2 curves
  cluster around the truth.

`analyse_wide_bisector.png`, middle panel: BIS as a function of
`osamp_spec`. Dashed grey is the truth BIS; the recovered BIS should
land on that line. The s=1 point is far above it; s>=2 hover within
a handful of mpx.

## Pitfalls

- Bisectors require a profile with a clear single peak. Noisy spectra
  with multiple local maxima will give nonsense crossings.
- The choice of level windows for BIS is conventional but not unique;
  HARPS uses 0.10..0.40 and 0.60..0.90 on the continuum-normalised
  CCF. We follow that here.
- A bisector measured on a too-coarsely-sampled profile (e.g.
  `osamp_spec=1` on an under-sampled LSF) is dominated by the bin
  edges, which is exactly why the s=1 case looks so bad in our
  analysis -- it is not a measurement of the underlying LSF shape so
  much as a measurement of how the LSF happened to fall on the
  detector grid.

## References

- Queloz et al. 2001, "No planet for HD 166435", A&A 379, 279
  ([ADS](https://ui.adsabs.harvard.edu/abs/2001A&A...379..279Q/abstract))
  -- classic example of using bisector / RV correlation to retract a
  candidate detection in favour of a star-spot explanation.
- Toner & Gray 1988, "The Star Patch on the G8 Dwarf chi Bootis A",
  ApJ 334, 1008
  ([ADS](https://ui.adsabs.harvard.edu/abs/1988ApJ...334.1008T/abstract))
  -- introduced the bisector *velocity span* (the conceptual ancestor
  of HARPS-style BIS) as a stellar-surface diagnostic.
- ESPRESSO pipeline description (DRS): the CCF and its BIS are
  standard pipeline products precisely for the reasons above.
