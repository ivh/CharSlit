# Later (slit kernels)

Deferred items from the `slitkernel` branch. Context in CLAUDE.md, "Slit kernels".

- **chi2 floor.** With kernels the synthetic calibration frame still sits at chi2 ~50
  (noisy) / ~210 (noiseless, from ~440). Probed (2026-10-13): the floor does not move with
  node_step (1.0/0.5/0.25: 236/215/212), nk (9/13) or slit tilt (0: 197). The residual sits
  in the line cores and alternates at the fibre pitch (junction rows modelled too broad,
  centre rows too narrow). The alternating scheme (fit D against the current model, slitdec
  re-solves sP/sL) stalls after 2-3 iterations and then degrades; a common initial
  broadening (so rows need broadening, not sharpening), weaker sL smoothing and an
  unweighted fit (slitdec's own solves ignore pix_unc) did not help. Next: solve the
  kernels as a third block inside slitdec's iteration (same objective as sP/sL), and/or
  sP on a finer grid (osamp_spec branch).
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
