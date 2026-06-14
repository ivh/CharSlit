# Plan: bin-x2 LSF-recovery validation on real CRIRES+ U-Ne data

Goal: validate `osamp_spec` super-resolution on real data by a
self-calibration trick. The native CRIRES data is well sampled
(instrumental FWHM ~2.9 px), so an unbinned extraction at `osamp_spec=1`
serves as ground truth. We then bin the image x2 in the dispersion
direction (FWHM ~1.45 binned px, under-sampled, ESPRESSO-like) and
extract with `osamp_spec=2`, which lands on **exactly the native pixel
grid** — direct comparison, real noise, real LSF, real tilt.

Background reading (all in repo root): `OSAMP_SPEC.md` (the feature),
`EXPERIMENT_SYNTHETIC_LSF.md` (synthetic precursor; findings 10-12 are
the bisector results this experiment mirrors), `bisect.md` (bisector/BIS
primer), `LATER.md` (deferred work — do not pick up).

## Data (already in place)

- `data/CRIRES_UNE_J.fits` — CRIRES+ U-Ne lamp, reduced (UTIL_CALIB),
  3 chips, each with `CHIPn.INT1` (image, ADU) + `CHIPnERR.INT1` (errors)
- `data/J1228_tw.fits` — cr2res trace-wave table per chip: `All` =
  trace y(x) polynomial (increasing powers, **1-based** pixel coords),
  `SlitPolyA/B/C` = a(x), b(x), c(x) of the tilt polynomial
  x'(y) = a + b*y + c*y^2 in absolute detector coords
- `scripts/make_swath.py` — swath cutter, already supports everything needed:
  - integer-aligns each column on the trace, passes only frac(ycen)
  - converts tilt to slitdec's t-relative-to-trace-centre frame:
    `c1 = b + 2*c*yc`, `c2 = c`
  - `xbin=2`: pairs of native columns cut with a **single shared
    integer shift** (from the pair-centre trace), summed; unc in
    quadrature; `c1, c2` divided by xbin (delta is in binned px, rows
    unchanged); frac-ycen from the pair centre
  - verified: trace check prints x'(yc) ~ x at swath centre

## Chosen swath

`--chip 2 --order 4 --x0 512 --width 512 --nrows 150`
(wl 1253.7-1255.7 nm, tilt -0.053 px/row => ~8 px sweep over the slit).

Analysis lines (fitted on central rows; FWHM < 3 px cluster = unresolved):

| x (swath col) | amp (ADU/px) | FWHM |
|------|-----|------|
| 37.8 | 38 | 2.58 |
| 60.5 | 96 | 2.89 |
| 68.5 | 126 | 2.52 |
| 169.2 | 60 | 2.75 |
| 310.7 | 21 | 2.79 |
| 324.6 | 17 | 2.93 |
| 444.6 | 111 | 2.83 |

Caveats: 60.5/68.5 are only 8 px apart — windows overlap, flag or use
narrower windows; 310.7 has a rejected broad line at 302.8 nearby.
Amplitudes are per-pixel; the full-slit extraction (~150 rows) gives
~10^4 counts per column at line peaks, S/N ~100.

## Steps

1. **Cut both swaths** with `make_swath.cut_swath(img, tw, 2, 4,
   nrows=150, x0=512, width=512)`, once with `xbin=1` (512 native cols)
   and once with `xbin=2` (256 binned cols).

2. **Extract truth**: unbinned swath, `osamp_spec=1`. Mask NaN pixels
   (mask=0, im NaN -> 0, keep unc finite). Use the ERR extension as
   `pix_unc`. Suggested: `osample=6, lambda_sP=0, lambda_sL=1,
   kappa=10` (real data has outliers). Truth spectrum: 512 native px.

3. **Extract binned**: `osamp_spec=2` (and `osamp_spec=1` for the
   staircase reference). Fine grid of s=2 on 256 binned cols = 512 bins
   of 1 native px, centres coinciding with native pixel centres —
   compare index-by-index, no resampling. Run `lambda_fringe` in
   {0, 1e-3, 1e-2}: synthetic finding 7 says low S/N wants ~1e-3.

4. **Per-line comparison** (the 7 lines above): area-normalise truth
   and recovered profiles over a +-4.5 native-px window; compute per
   line: FWHM, centroid difference, rms of (recovered - truth), and
   bisector/BIS following `scripts/analyse_wide_bisector.py` (bisector(),
   bis_value() — top 0.6-0.85 minus bottom 0.1-0.4 of peak). BIS per
   single line will be noisy at S/N~100; also report the **average
   bisector across all 7 lines** (align each on its truth centroid
   first) — that's the headline number.

5. **Compare three products** per line and stacked: (a) truth
   (unbinned s=1), (b) binned s=1 staircase (2-native-px bins), (c)
   binned s=2. Question: does (c) recover the truth's line shape and
   bisector where (b) cannot? Expected from synthetics: binned-s2 BIS
   should track truth to ~10 mpx; binned-s1 should show a large
   centroid-phase-dependent bisector error.

6. **Figure**: per-line panel grid (truth/recovered overlays) + one
   summary panel (stacked bisectors) + metrics table printed to
   stdout. Save PNG + NPZ (both gitignored).

## Gotchas / conventions

- Build first: `uv pip install -e . --force-reinstall --no-deps`
  (osamp_spec branch). Always run python via `uv run`.
- slitdec evaluates tilt at t relative to the **trace centre row**
  (y_lower_lim = nrows/2, slitdec.c:880); cut_swath already returns
  slitcurve in that frame. ycen passed to slitdec is the frac part
  only (array length ncols).
- The C code zeroes the first/last `delta_x * osamp_spec` fine bins;
  with ~8 px sweep delta_x ~ 3 (binned), so ~6 fine bins at each edge
  are dead — all 7 lines are well inside.
- slitdeltas: zeros (length nrows or ny both accepted by the wrapper).
- `return_code != 0` => extraction failed; print info and stop.
- Don't trust the truth blindly: it is itself an extraction with its
  own noise (~1%); the comparison floor is set by it. Good enough for
  "how much do we recover", not for few-mpx claims.
- Note for results: edge rows were already cut (nrows=150 of ~176),
  and the two brightest detector lines (10-60 kADU, possibly
  nonlinear) are NOT in this swath — by design.

## Definition of success

Binned+s=2 recovers per-line FWHM and stacked bisector close to the
unbinned truth, clearly beating the binned s=1 staircase. If instead
s=2 collapses to the staircase (like Hsim in OSAMP_SPEC.md), report
that honestly — with ~8 px of tilt sweep and S/N~100 the synthetic
results (finding 7: lambda_fringe ~1e-3 regime) suggest recovery
should be partial but real.
