# Later (slit kernels)

Deferred items from the `slitkernel` branch. Context in CLAUDE.md, "Slit kernels".

- **chi2 floor.** With kernels the synthetic calibration frame still sits at chi2 ~50
  (noisy) / ~210 (noiseless). Find what is left: within-row fibre structure finer than the
  0.5-row nodes, the renderer's 0.1-px tilt quantisation, sP sampling on the 1-px grid.
  Try node_step 0.25 and the osamp_spec branch (sP on a finer grid).
- **Speed.** Kernels expand every zeta entry nk-fold (x9 runtime at nk = 9). Applying the
  kernel to the merged per-pixel window in the fills/model instead would keep zeta small.
- **Wrapper nrows -> ny mapping** for slitdeltas/preset_slitfunc/slitkernel is a linear
  stretch up to ~1.5 rows off slitdec's geometry. Fixing it needs ycen and changes results
  (goldens).
- **ycen fraction = k/osample** shifts slitdec's subpixel grid by one subpixel (d1 = step
  convention). Intended?
- **Slitdeltas in slit coordinates.** `selfcal` (k1 -> slitdeltas) assumes a constant ycen
  fraction; on real swaths the kernels' shift component absorbs per-u centroid errors
  instead. A `selfcal_u`-style k1 fit would separate geometry from LSF.
- **x-dependence.** Kernels are constant across a swath; on ANDES_R_FP1 the residual
  geometry varies with x within 256 columns. Overlapping swaths + interpolation, as PyReduce
  does for sL.
- **ANDES_R_FP1 kernel structure.** Variance structure of +-0.5 px^2 on 20-40-row scales
  along the slit, consistent between swaths: identify (fibre transmission pattern? beat
  between fibre pitch and rows?).
- **Realistic illumination test.** Needs E2E frames with per-fibre PSFs: patch pyechelle's
  PSF cache (keyed by ccd, order only) in the E2E env, then repeat exp_fibre_kernel's
  illumination test on simulated FP frames with different fibre illuminations.
- **PSF axis convention.** pyechelle's raytracer maps PSF array axis 0 to detector x;
  exopl_refl/phase6 took sum(axis=0) as the LSF, i.e. the cross-dispersion profile if
  pyechelle is right. Check which, then recheck phase6.
- **PyReduce integration** (env-var hook like osamp_spec's).
