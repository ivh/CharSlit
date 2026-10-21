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
  kernels as a third block inside slitdec's iteration (same objective as sP/sL). (sP on a
  finer grid was tried, see the next item: not the cause.)
- **Shift-interpolation broadening in slitdec's model.** The fractional slit-curve shift
  splits each sP value over two columns with weights (1-f, f), which broadens the model
  profile by variance f(1-f): 0 at integer shift, 0.25 px^2 at half a pixel. On the
  synthetic slit (tilt 0.05) the plain model's row variance oscillates with period 1/tilt =
  20 rows and amplitude exactly f(1-f) (corr 1.00); at tilt 0 it is flat. That is ~13 % of
  the data's row-to-row variance swing (0.23 of 1.73 px^2). Kernels absorb it when the
  tilt is constant across the swath (phase is a function of y only), but not where
  tilt/curvature vary along x, so it may feed the chi2 plateau on real swaths.
  Tried the osamp_spec branch (sP on a 1/s-pixel grid, exact pixel-overlap deposit) on the
  same frames: s = 2 cuts the artefact to 0.054 px^2 (= 1/s^2) but chi2 only drops 3 % and
  the illumination biases do not improve; s >= 3 brings the artefact back through the
  fine-grid fringe mode and makes the LSF unstable against slit illumination (up to 48 m/s,
  2.75 px^2 with slitdeltas); lambda_fringe = 0.01 collapses to s = 1, lambda_sP 0.1-10 is
  far too strong at this flux (no scaling). Not worth a merge with slitkernel on this
  evidence; would need a real swath with x-varying tilt to show a gain.
- **Per-frame kernel fit on the science frame** (agreed plan, 2026-10-15; implemented as
  `selfcal_param` 2026-10-21, synthetic results in CLAUDE.md: the gauge test came out as
  predicted, <= 0.8 m/s with the calibration's gauge, 12/33 m/s with the frame's flux
  gauge. Still open: ESPRESSO real swaths, a transfer test, and the items below). Fit the kernels
  on the frame being extracted, from its own residuals, then re-extract, instead of only
  taking them from an FP calibration. Design:
  - **Parametrise** each node as shift, width and skew (3 parameters) instead of 9 free
    taps. This limits overfitting: 9 taps x ~60 nodes is ~500 parameters per ~7700-pixel
    swath. It also matches the planned starting point for LSF determination once an LFC
    is available.
  - **FP-calibrated kernels** serve as the starting point and as a ridge prior. A
    low-S/N frame then falls back to them, and a bright one refines them.
  - **Gauge must not depend on the flux.** Normalise the kernels to a delta over a fixed
    set of rows with fixed weights (for example uniform over the nominal slit), not
    weighted by the frame's own flux. With a flux-weighted gauge, sP becomes the frame's
    flux-weighted mean LSF, which is what plain slitdec already gives, so the illumination
    bias would come back. Calibration frames must use the same gauge.
  - **Expected noise cost:** about sigma_RV x the rms departure of the illumination from
    the gauge weights, so ~0.1-0.2 sigma_RV in quadrature at 10-20 % non-uniformity.
    Line-shape errors scale with the spectral gradients, as does the information to fit
    them. Centroid errors do not shrink on line-poor spectra; only their noise grows.
  - **Valid only where all rows see the same spectrum.** That holds for a star on the R
    slit. It fails for the YJH IFU and for resolved targets, where the kernels would absorb
    real spatial-spectral structure, so those keep the calibration kernels.
  - **Test:** run the illumination test with per-frame kernels twice, once with a
    flux-weighted gauge and once with a uniform one. Expected: the flux-weighted run gives
    the plain-slitdec biases back (+7.6/-13.4 m/s in plot_slitkernel_overview panel e); the
    uniform one reaches the calibrated-kernel level (<=0.2 m/s).
  - **Relation to the "third block inside slitdec" item above:** this is that item,
    applied per frame.
- **Speed.** Kernels expand every zeta entry nk-fold (x9 runtime at nk = 9). Applying the
  kernel to the merged per-pixel window in the fills/model instead would keep zeta small.
- **Wrapper nrows -> ny mapping** for slitdeltas/preset_slitfunc/slitkernel is a linear
  stretch up to ~1.5 rows off slitdec's geometry. Fixing it needs ycen and changes results
  (goldens).
- **ycen fraction = k/osample** shifts slitdec's subpixel grid by one subpixel (d1 = step
  convention). Intended?
- **Slitdeltas in slit coordinates.** Done by `selfcal_param(moments=(1,))` (or with k2, k3).
  `selfcal_u`'s taps still have a free first moment, so they keep absorbing per-u centroid
  errors; switching the scripts' `selfcal` k1 step to selfcal_param would separate them.
- **selfcal_param: k2 gauge per frame.** `k2_offset` grows when a frame needs more
  broadening than the calibration (dk2 up to -0.13 px^2 on lower_half), which changes sP's
  LSF width. Centroids are unaffected; the LSF of the extracted spectrum is not. Options:
  fix the offset with a margin from the calibration, or report sP's LSF as gauge-mean
  minus k2_offset per frame.
- **selfcal_param: k4.** Lower chi2 than the taps but trades against k2/k3 and needs step
  halving (the linearised B4 step overshoots); biases not better than par3 in the
  illumination test. Undersampled lines (sigma ~0.9 px) make k2 and k4 nearly degenerate.
- **selfcal_param: real data.** First testbed: the ESPRESSO LFC swaths
  (~/pipes/espresso_lfc_20230124), where the free taps needed 1-row nodes and ridge 1e-2 at
  ~7 rows per slice. Compare chi2 out of sample (fibre A -> B, LFC -> FP) and the
  illumination test there. Compensation is column-averaged where the tilt varies along x.
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
