# Experiment: asymmetric-LSF recovery with osamp_spec on a variable-tilt slit

Script: `experiment_synthetic_lsf.py` (standalone, run with
`uv run experiment_synthetic_lsf.py`; `SHOW=0` for batch). Requires the
`osamp_spec` code (merged into this branch from `origin/osamp_spec`).

## Setup

- 80 rows x 60 columns, single **unresolved** emission line at column 30.37
  (deliberately off pixel centre)
- **Asymmetric LSF**: top-hat core (0.8 px full width) convolved with a
  0.25 px-sigma Gaussian, plus a blue (low-x) exponential wing (15% of flux,
  0.8 px scale). True FWHM = 0.894 px, i.e. under-sampled at the detector.
- **Variable slit tilt**: 0.1 px/row at the bottom row, decreasing linearly
  to 0.001 px/row at the top row. Integrated, that is a quadratic horizontal
  displacement, exactly representable by the degree-2 slitcurve in the frame
  slitdec uses (t = y - nrows/2): c1 = 0.049873, c2 = -6.266e-4. Total
  horizontal shift across the slit: ~4 px.
- The image generation integrates 8 sub-row positions per detector row, so
  the within-row shear of the varying tilt is in the data.
- Photon (Poisson) noise + 2 e- read noise; peak pixel ~3200 counts. A
  noiseless realisation with identical weighting is extracted alongside.
- Extraction: `osamp_spec` s = 1, 2, 3; `lambda_fringe` in
  {0, 0.001, 0.01, 0.1, 1}; `lambda_sP = 0`, `kappa = 0`.

## Metrics

All profiles are area-normalised over a +/-5 px window around the line.

- `rms`: residual vs the true LSF binned onto the extraction's own fine grid
- `rms_cont`: residual of the recovered staircase vs the **continuous** true
  LSF, evaluated on a common 1/50-px grid — comparable across s, includes
  each grid's intrinsic binning error
- `FWHM`, blue/red wing flux (integral of the normalised profile over
  1-4 px on each side of the centre), centroid error

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

Noiseless numbers are nearly identical (rms_cont 0.108 / 0.081 / 0.051),
i.e. at this S/N the recovery is **geometry-limited, not noise-limited**.

## Findings

1. **The LSF is substantially recoverable.** At s=3 the recovered FWHM is
   within 1.2% of truth (vs 24% too broad at s=1), and the blue/red wing
   asymmetry is recovered with the correct sign and ~90% of the correct
   amplitude. Shape error vs the continuous LSF halves from s=1 to s=3
   (rms_cont 0.107 -> 0.053). The flat-top character of the core is visible
   in the s=2/3 reconstructions.

2. **lambda_fringe = 0 is optimal here.** With ~4 px of total tilt sweep
   (full sub-pixel phase coverage several times over), the fine modes are
   well constrained and no fringe ripple appears. Any lambda_fringe > 0
   (even 0.001) erodes genuine super-resolution content: it drags the
   recovered profile toward the block-mean staircase and flips the apparent
   asymmetry. This is the constructive counterpart of the Hsim finding in
   OSAMP_SPEC.md: there the content was pure fringe noise and lambda_fringe
   correctly collapsed it; here the content is real and lambda_fringe should
   be off (or very small).

3. **Control (constant 0.001 px/row tilt, `--tag _notilt`)**: without
   meaningful tilt, s>1 at lambda_fringe=0 is dominated by the fringe mode
   (rms_cont 0.27 at s=2) and lambda_fringe=0.01 collapses it back to
   exactly the s=1 staircase quality (rms_cont 0.096 vs 0.098). Confirms the
   super-resolution signal comes from the tilt-induced phase diversity, not
   from the solver inventing structure.

4. **The varying tilt is fine for the forward model.** The 0.1 -> 0.001
   px/row linear variation is exactly a degree-2 slitcurve; nothing special
   is needed. The information content is dominated by the high-tilt
   (bottom) half of the slit; the top half contributes mostly flux, not
   phase diversity.

5. **Sub-pixel centroid** is recovered to better than 0.01 px at every s,
   consistent with OSAMP_SPEC.md's note that the wavelength zero-point is
   the robust product even when shape recovery is marginal.

## Outputs

- `experiment_synthetic_lsf.png` / `.npz` — main run (gitignored; regenerate
  with the command above)
- `experiment_synthetic_lsf_notilt.png` / `.npz` — control run
