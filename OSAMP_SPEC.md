# osamp_spec: dispersion-direction oversampling

Status: landed in `slitdec/`, re-implemented on top of the zeta-only /
dense-window solver (the speed rewrite that removed the `xi` tensor).
`osamp_spec=1` is bit-identical to master: the full golden regression
suite (`tests/test_regression.py`, rtol 1e-10) passes unchanged, and the
committed `tests/golden/osamp_spec1.npz` matches at rtol 1e-9.

## What it does

New integer argument to `slitdec(...)`, typical values 1..4 (default 1).
With `osamp_spec=s` the extracted spectrum `sP` is stored on a grid of
length `ncols * s`, each fine bin covering `1/s` of a detector pixel in
the dispersion direction.

The super-resolution comes from slit tilt (`slitcurve`) and any per-row
offsets (`slitdeltas`): different detector rows sample slightly different
sub-pixel wavelengths at the same detector column, so the fine bins are
distinguishable provided the phase `frac(delta(iy))` varies across the
slit.

## Forward-model change

The geometry now lives in a single `zeta` tensor (subpixels contributing
to each detector pixel; the inverse `xi` tensor is gone). `zeta_tensors`
loops over `x_fine = 0..ncols_fine-1`, with `x = x_fine / osamp_spec` the
detector column driving `slitcurve[x]`, `ycen[x]`, `ycen_offset[x]`
(unchanged semantics). Each `zeta` entry stores the fine bin index
`x_fine` (which is just `x` when `s == 1`). For each fine subpixel at slit
row `iy`:

- horizontal extent before tilt: `[x_off_frac, x_off_frac + 1/s)` with
  `x_off_frac = (x_fine - x*s)/s`
- after tilt: `[x_off_frac + delta, x_off_frac + delta + 1/s)`
- deposited into detector columns `floor(L)` and `floor(R)` with overlap
  weights summing to `1/s`
- vertical weighting (the Case A / B / C slit-row sharing) is unchanged

The deposit branches on `osamp_spec`: the `s == 1` path is the original
coarse deposit kept verbatim (`ix1 = (int)delta`, `ix2 = ix1 + signum`),
so the per-entry float values and the `zeta` append order are unchanged
and the s=1 result stays bit-identical to master. The `s > 1` path uses
the `floor(L)/floor(R)` split above. The downstream `sP` normal-matrix
fill (master's dense-window merge, row-major band layout) is generalized
by parameter only — the band centre/threshold `2*delta_x` becomes
`delta_x_fine = (nx-1)/2`, which equals `2*delta_x` at `s == 1`.

## Matrix / solver

- `sP` normal matrix is `(ncols_fine, nx)` with
  `nx = 4*delta_x*osamp_spec + 2*osamp_spec - 1`
- first-difference smoothing stencil `(-1, 2, -1)` applied between
  adjacent fine bins, weight `lambda_sP`
- `bandsol(p_Aij, p_bj, ncols_fine, nx)` on the fine grid
- boundary zeroing: first / last `delta_x * osamp_spec` fine bins

Sanity guard: `if (nx > ncols_fine) return -1` (was `ncols` — the old
check misfired for small images at `osamp_spec >= 4`; fixed).

## Tested

- `tests/test_regression.py` (s=1 vs master golden, rtol 1e-10): all
  synthetic + real-FITS cases pass — `osamp_spec=1` is bit-identical to
  the pre-osamp_spec master code across spectrum / slitfunction / model /
  uncertainty / mask
- `tests/test_osamp_spec.py`: default == explicit s=1, fine-grid shapes
  `ncols*s` for s=2,3, and the committed golden match at rtol 1e-9
- Full default `pytest`: green
- `osamp_spec = 1..5` converge / run on synthetic tilted data
  (return_code=0)

## Known behavior / caveats

**1. Fringing at the osamp frequency.** With `lambda_sP=0` the extracted
fine spectrum shows high-frequency ripple at `osamp_spec` cycles per
detector pixel. Root cause: detector-pixel integration is a box of width
1, whose Fourier transform has a zero exactly at that frequency — so the
mode is a near-null of the forward model. Slit tilt breaks the
degeneracy only as strongly as the sub-pixel phase `frac(delta(iy))`
sweeps across the slit; for Hsim (5 px tilt across 90 rows) the mode is
formally full-rank but weakly constrained, so noise leaks in.

Mitigation options:

1. Post-hoc boxcar of width `osamp_spec` — exact null at fringe
   frequency, but kills all super-resolution
2. Post-hoc Fourier notch at `k = ncols` — selective but can ring
3. `lambda_sP > 0` — the `(-1, 2, -1)` stencil has peak penalty at
   Nyquist, so it does hit the fringe mode hardest, but it also
   broadens lines
4. **Selective in-solver regularizer** `L = I - mean_over_osamp`
   (IMPLEMENTED as `lambda_fringe`). Penalty
   `lambda_fringe * Σ_blocks Σ_i (sP_i - mean_i)²`. Surgical on the
   fringe mode; row-sum is zero so the coarse-averaged spectrum is
   untouched. Added per-block as `+lambda_fringe*(1-1/s)` on the
   diagonal and `-lambda_fringe/s` on within-block off-diagonals.
   Only active when `osamp_spec > 1`. Default `lambda_fringe = 0.0`.

## Finding: lambda_fringe collapses Hsim to a staircase

A sweep of `lambda_fringe` on Hsim at `osamp_spec=3`, `lambda_sP=0`:

| lambda_fringe | max &#124;sP − staircase&#124; | fraction of peak |
|---------------|------------------------------:|-----------------:|
| 0             | 301k                         | 28%              |
| 0.01          | 6571                         | 0.6%             |
| 0.1           | 700                          | 0.07%            |
| 1             | 70                           | 0.007%           |
| 10            | 7                            | 0.0001%          |

Even tiny `lambda_fringe` collapses the fine spectrum to the
block-mean staircase. This is diagnostic: on Hsim the unregularised
sub-pixel detail is essentially all fringe-mode noise. There is no
value of `lambda_fringe` that kills the fringe while preserving real
super-resolution content, because the data does not contain any.

Mechanistically: Hsim has ~5 px of tilt across 90 slit rows, enough
for the forward-model normal matrix to be formally full-rank at the
osamp-period mode, but not enough phase-coverage S/N for the
non-block-mean content to be determined above noise. The selective
regulariser correctly reports this by collapsing to the staircase.

For this kind of data the useful product from `osamp_spec > 1` is the
sub-pixel *centroid / wavelength-zero-point* accuracy (the block-mean
staircase is still positioned correctly on the fine grid), not
narrower line profiles. To get real super-resolution above the
staircase you need more phase coverage across the slit — stronger
tilt, taller slit, or a designed non-uniform `slitdeltas` dither
(difficult on a pupil-sliced stable spectrograph without moving
parts).

**2. `lambda_sP` is not auto-scaled with `osamp_spec`.** The same
`lambda_sP=1.0` is a weaker prior per unit wavelength on a finer grid.
Caller is responsible for picking a value appropriate to their
`osamp_spec`. This is an intentional design choice from the plan.

**3. Band-solver cost grows as `ncols_fine * nx² ≈ osamp_spec³`.**
Acceptable for `osamp_spec ≤ 4`. Memory grows as `osamp_spec²`.

## Design decisions

- **`lambda_fringe` is the only fringe regularizer (kind=0).** An earlier WIP
  frequency-weighted variant (`kind=1`, behind a `CHARSLIT_LAMBDA_FRINGE_KIND`
  env flag) was dropped: at s=2 the within-block non-DC subspace is 1-D (pure
  Nyquist) so the two variants are scalar multiples (identical), and at s=3 it
  is 2-D but still dominated by `k=2π/3`, so kind=1 buys only marginal
  discrimination. Real room for frequency weighting needs s>=4; sweeps showed
  no meaningful gain at s<=3, so the flag and the variant were removed.
- **Mechanism validated** on synthetic and real data — see
  `OSAMP_EXPERIMENTS.md`. Headline: super-resolution is real only when the line
  is genuinely under-sampled *and* the slit carries enough tilt-phase
  diversity; on well-sampled or no-tilt data the fine spectrum collapses to the
  block-mean staircase. `lambda_sP` is far more aggressive on the fine grid
  than on the coarse one (even `1e-8` erodes the sharpest sub-pixel peaks), so
  the practical recipe is small `lambda_sP` and small-or-zero `lambda_fringe`.

## PyReduce integration

The super-resolution experiments on real survey data are driven from PyReduce
(separate repo, `/Users/tom/PyReduce`) via env-var overrides that live there,
not here: `PYREDUCE_CHARSLIT_OSAMP_SPEC`, `PYREDUCE_CHARSLIT_LAMBDA_FRINGE`,
`PYREDUCE_CHARSLIT_LAMBDA_SP`, `PYREDUCE_CHARSLIT_DUMP_DIR`.

**Gotcha**: PyReduce's `.venv` installs `charslit` from git, not this dev tree.
After editing C here, reinstall into PyReduce:

```
cd /Users/tom/PyReduce && uv pip install -e /Users/tom/CharSlit.git --force-reinstall --no-deps
```

## Files touched

- `slitdec/slitdec.c` — `MAX_ZETA_Z`/`MAX_SP`/`MAX_PAIJ_X`/`MAX_PBJ`
  macros (`_osamp_spec`, `_ncols_fine`), `zeta_add` and `zeta_tensors`
  (`osamp_spec` arg, fine-grid deposit branch), `slitdec` (`ncols_fine`,
  `nx`/`delta_x_fine`, `nx > ncols_fine` guard, sP-fill band offsets,
  fine-grid smoothing / diagonal regularization / convergence /
  uncertainty / boundary zeroing), `lambda_fringe` block operator
- `slitdec/slitdec.h` — `osamp_spec` in `slitdec` and `zeta_tensors`;
  `lambda_fringe` in `slitdec`
- `slitdec/slitdec_wrapper.cpp` — `osamp_spec` (default 1),
  `lambda_fringe` (default 0.0), output shapes `ncols * osamp_spec`,
  docstring
- `bench.py` — `--osamp_spec` flag for scaling measurements

No changes to `make_curvedelta.py`, `plotting.py`, or the slit-function
(`sL`) path.

## Deferred work

**Do not start on these until explicitly asked** — a parking lot so the work
record is captured without committing to it.

- **Validate `osamp_spec` at s>3 before pinning regression tests beyond s=1.**
  The wide-LSF s-sweep showed s=4-5 beating s=3 on BIS bias with s=6 starting
  to overshoot (`OSAMP_EXPERIMENTS.md`, synthetic findings 6 and 10-12), but
  that is one realisation. Before pinning s=2..5 to a golden, establish:
  - **Geometry robustness** — repeat the sweep at different total tilt sweeps
    (1, 2, 5 px), different `ncols`, different line positions; the optimal s
    must be stable, else the s=4-5 gain was a one-realisation accident.
  - **Multi-seed scatter** — 20-50 noise seeds, report mean±std of BIS bias
    per s, to tell bias from a single bad draw and to compare s=3/4/5 properly.
  - **Mechanism for the s=6 over-correction** — confirm it is the
    forward-model fringe-mode null saturating (caveat 1 above), not the
    boundary zeroing (`first/last delta_x*osamp_spec` fine bins) eating a
    larger fraction of the recoverable spectrum (vary the boundary width and
    watch whether s=6 tracks it).
- **Synthetic super-resolution regression test** (tilted slit, narrow line,
  `osamp_spec=3`) confirming the regularizer preserves genuine content when it
  exists; `scripts/compare_osamp.py` already demonstrates this informally.
