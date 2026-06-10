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

6. **Gains continue past s=3 and saturate near s=5**
   (`--osamps 1 2 3 4 5 6 --fringes 0 --tag _ssweep`): rms_cont = 0.107,
   0.081, 0.053, 0.039, 0.0345, 0.0355 for s = 1..6. Beyond s~5 the
   geometry/noise floor is reached and the recovered FWHM starts to
   undershoot slightly (0.85-0.87 px vs 0.894 truth).

7. **At low S/N the optimal lambda_fringe becomes small-but-nonzero**
   (`--amp 100 --tag _lowsn`, peak ~90 counts): lambda_fringe=0 lets fringe
   noise leak in (rms_cont 0.12-0.14 at s=2/3, centroid off by up to
   0.2 px), while lambda_fringe=0.001 restores a clean recovery (rms_cont
   ~0.091, still beating s=1 at 0.112). Practical rule: lambda_fringe
   should scale with the noise level — 0 when the tilt-phase coverage is
   strong and S/N is high, ~1e-3 otherwise.

## Width sweep: how recoverability depends on LSF width

Re-run with `--tophat-w` and `--sigma` to scan the core width (the blue
wing convolution kernel is left unchanged at 15% / tau=0.8 px). Three
configurations bracket the baseline:

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
   under-sampled) s=1 is hopeless (FWHM 2.2x too broad), s=3 closes most
   of the gap but the truth is still only ~1.3 fine bins wide. Around s=2
   the optimal `lambda_fringe` already shifts to small-but-nonzero
   (1e-3 wins), consistent with the high-noise regime in finding 7.
9. **At realistic ESPRESSO-like sampling (~2 px FWHM), s=1 nails the FWHM
   to <2%.** Width is not the diagnostic that exposes the staircase
   artefact here -- shape is. See the next section.

## Wide case (ESPRESSO-like): bisector analysis

For a spectrograph with FWHM slightly under 2 px (the regime where
`osamp_spec=1` already gets the width right), the question is whether
shape is also right. The relevant metric is the **bisector**, which is
how RV pipelines (HARPS, ESPRESSO) distinguish real Doppler shifts from
asymmetric line-shape distortions. See `bisect.md` for a primer.

s-sweep at `lambda_fringe = 0`, wide LSF (FWHM 1.785 px), noisy case,
generated by `experiment_synthetic_lsf.py --tophat-w 1.6 --sigma 0.5
--osamps 1 2 3 4 5 6 --fringes 0 --tag _wide_ssweep` and analysed by
`analyse_wide_bisector.py`:

| s  | recovered BIS [mpx] | **BIS bias vs truth [mpx]** | blue wing | red wing |
|----|---------------------:|----------------------------:|----------:|---------:|
| truth | +24.9            | 0                           | 0.115     | 0.064    |
| 1  | +105.1               | **+80.2**                   | 0.022     | 0.134    |
| 2  | +19.6                | -5.3                        | 0.137     | 0.036    |
| 3  | +30.8                | +5.9                        | 0.099     | 0.069    |
| 4  | +29.2                | +4.3                        | 0.085     | 0.089    |
| 5  | +21.4                | -3.5                        | 0.116     | 0.059    |
| 6  | +13.3                | -11.6                       | 0.102     | 0.070    |

10. **s=1 inverts the asymmetry sign and biases BIS by +80 mpx.** Block-
    mean staircase puts the wing flux on whichever side of the sub-pixel
    centroid the line happens to sit (blue wing comes out as red, etc.).
    At ESPRESSO-class dispersion this is roughly tens of m/s of fake RV
    signal correlated with the sub-pixel centroid -- exactly what mimics
    a planet in CCF analysis. **FWHM-only validation is not enough for
    RV work**; the bisector is.
11. **s = 2..5 collapse the BIS bias to single-digit mpx.** This is the
    regime where shape is recovered well enough that the pipeline does
    not introduce its own systematic. s=4-5 look slightly better than
    s=3 here, but are outside the explicitly tested range in
    OSAMP_SPEC.md -- see `LATER.md` for the validation work needed
    before recommending them.
12. **s=6 starts under-correcting** (BIS -12 mpx vs truth), consistent
    with OSAMP_SPEC.md's prediction that gains saturate near s=5 and
    the fringe-mode null in the forward model begins to dominate. At
    s >= 6 `lambda_fringe` would need to come back on.

## Outputs

- `experiment_synthetic_lsf.png` / `.npz` -- main run (gitignored; regenerate
  with the command above)
- `experiment_synthetic_lsf_notilt.png` / `.npz` -- control run (no tilt)
- `experiment_synthetic_lsf_narrow.png` / `.npz` -- narrow LSF
- `experiment_synthetic_lsf_wide.png` / `.npz` -- wide LSF (ESPRESSO-like)
- `experiment_synthetic_lsf_wide_ssweep.png` / `.npz` -- wide LSF, s=1..6
- `analyse_wide_bisector.py` / `analyse_wide_bisector.png` -- bisector
  diagnostics on the wide s-sweep
