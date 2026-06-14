# osamp_spec / super-resolution status

State of the dispersion-direction oversampling (super-resolution) work, as of
2026-06-05. Work lives on the `osamp_spec` branch.

## Branch / commit layout

`origin/master` and the base of `osamp_spec` both already carry the two feature
commits:

- `6bdeada` Add osamp_spec for dispersion-direction oversampling
- `47867ea` Add lambda_fringe: selective regularizer for osamp-period fringe
  (kind=0 only; also adds `OSAMP_SPEC.md`)

Local `master` is intentionally parked 2 behind at `e3c21a1` (pre-feature).

This session added, on top of `47867ea`:

- **maxiter / non-finite convergence fix** (slitdec.c, own commit). The
  do/while previously let a non-finite cost bypass `maxiter` and hang forever
  (e.g. `kappa=0` leaves NaN cells un-masked). This is an independent bugfix —
  **cherry-pick it to `master` later** (per the plan).
- **s=1 regression test** (`tests/test_osamp_spec.py` + committed golden
  `tests/golden/osamp_spec1.npz`, with a `.gitignore` negation so the `.npz`
  is tracked). Pins the osamp_spec=1 solver outputs so future refactors can't
  silently perturb the bit-identical path.
- **`compare_osamp.py`** — self-contained s=1/2/3 demo (replaces the old
  PyReduce-coupled sweep scripts, which were deleted).

Nothing pushed this session.

## What the feature does

`osamp_spec = s` oversamples the **dispersion** (wavelength) direction, so the
extracted spectrum comes back on a fine grid of length `ncols * s` instead of
`ncols`. Super-resolution relies on slit-tilt / `slitdeltas` phase diversity:
different detector rows sample the line profile at different sub-pixel phases,
so the fine grid can recover sub-pixel structure the `osamp=1` block-mean
staircase throws away. `osamp_spec=1` is bit-identical to the pre-change code.

The committed regularizer for the osamp-period fringe null space is
`lambda_fringe` (kind=0): penalty `lf * ||L sP||^2` with `L = I - (1/s)J` per
block. Row-sum zero, so the coarse-averaged spectrum is untouched. Default 0.

## Decisions made this session

- **kind=1 dropped entirely.** The WIP frequency-weighted variant
  (`CHARSLIT_LAMBDA_FRINGE_KIND` env flag) and the whole flag are gone. Reason:
  at s=2 the within-block non-DC subspace is 1-D (pure Nyquist), so kind=0 and
  kind=1 are scalar multiples — identical. At s=3 it is 2-D but still dominated
  by `k=2π/3`; kind=1 buys only marginal discrimination. Real room needs
  **s ≥ 4**. Sweeps confirmed zero meaningful improvement at s≤3, so the branch
  was a dead end.
- **Mechanism validated** with `compare_osamp.py`: on a synthetic tilted,
  under-sampled line (true FWHM 1.06 px, 4.35 px tilt span), s=1 recovers
  FWHM 1.18 px (coarse-grid broadening) while s=2/3 recover 1.06 px — the true
  width. This is the slit-tilt super-resolution payoff, reproducible with no
  external data.

## Preserved experimental findings (ANDES YJH, H channel)

From the (now-deleted) PyReduce sweeps. Worth keeping as lessons:

- **Super-resolution is real only when the line is genuinely undersampled.**
  psf30 (FWHM 0.3 px) clearly recovers super-resolved peaks above the osamp=1
  staircase at s=2/3 with small `lambda_fringe` and small `lambda_sP`. psf70
  (FWHM 0.7 px, near Nyquist) shows marginal-to-no gain — Nyquist already
  captured the content. On Hsim (no intentional dither) even tiny
  `lambda_fringe` collapses the fine spectrum to the staircase, because the
  sub-pixel content is essentially all fringe-mode noise.
- **`lambda_sP` is far more sensitive on the fine grid than expected.** Even
  `lsp = 1e-8` suppresses the sharpest sub-pixel peaks ~80%; broader peaks
  survive to ~1e-4. The effect is peak-width dependent. `lf = 1e-3` kills
  super-resolution regardless of `lsp`.

Net: super-resolution works with **small lambda_fringe and small lambda_sP**
on genuinely undersampled lines; it cannot manufacture detail beyond Nyquist.

## PyReduce integration (separate repo, `/Users/tom/PyReduce`)

Env-var overrides drive the experiments and live in the PyReduce working tree,
not here: `PYREDUCE_CHARSLIT_OSAMP_SPEC`, `PYREDUCE_CHARSLIT_LAMBDA_FRINGE`,
`PYREDUCE_CHARSLIT_LAMBDA_SP`, `PYREDUCE_CHARSLIT_DUMP_DIR`; plus
`PYREDUCE_OUTPUT_SUBDIR`, `ANDES_SCIENCE_FILE`, `ANDES_TRACE_RANGE` in
`examples/andes_yjh.py`.

**Gotcha**: PyReduce's `.venv` installs `charslit` from git, NOT this dev tree.
After editing C here, reinstall into PyReduce:

```
cd /Users/tom/PyReduce && uv pip install -e /Users/tom/CharSlit.git --force-reinstall --no-deps
```

## Open questions / next steps

- **Real slightly-undersampled science data is expected**, and slit tilt is the
  most promising lever to recover it. `compare_osamp.py` is the harness to vet
  that case before committing more solver machinery.
- Only revisit **s ≥ 4** (where frequency-weighted regularization would have
  real discriminating room) or a **PSF-shape prior** (TV / Gaussian on
  `sp_fine`) once there is concrete undersampled data that needs it. Do not
  tune regularizer variants against data that lacks the signal.
- `OSAMP_SPEC.md` "Pending": the s=1 regression test is now done; a synthetic
  super-resolution test confirming the regularizer preserves genuine content
  could be added (compare_osamp.py already demonstrates this informally).
