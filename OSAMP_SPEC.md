# osamp_spec: dispersion-direction oversampling

Status: landed in `slitdec/`. `osamp_spec=1` is bit-identical to the
pre-change implementation.

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

`xi_zeta_tensors` now loops over `x_fine = 0..ncols_fine-1`. For each
fine subpixel at slit row `iy`:

- detector-column lookup `x = x_fine / osamp_spec` drives
  `slitcurve[x]`, `ycen[x]`, `ycen_offset[x]` (unchanged semantics)
- horizontal extent before tilt: `[x_off_frac, x_off_frac + 1/s)` with
  `x_off_frac = (x_fine - x*s)/s`
- after tilt: `[x_off_frac + delta, x_off_frac + delta + 1/s)`
- deposited into detector columns `floor(L)` and `floor(R)` with overlap
  weights summing to `1/s`
- vertical weighting (the Case A / B / C slit-row sharing) is unchanged

The `osamp_spec == 1` branch is a verbatim copy of the original code so
that float accumulation order in the matrix assembly is preserved.
Removing that branch caused a ~21% model divergence on the ill-conditioned
`curved` regression case purely from summation-order float noise.

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

- `verify_identity.py` (regression against pre-change baseline):
  `simple`, `curved`, `Hsim` all `max|diff| = 0.000e+00` across
  `spectrum`, `slitfunction`, `model`, `uncertainty`
- Full `pytest`: 24/24 pass
- `pytest -m save_output`: all visual plots render; no new anomalies
- `osamp_spec = 1..5` converge on Hsim.fits (return_code=0, status=1)

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

## Files touched

- `slitdec/slitdec.c` — `xi_zeta_tensors`, `slitdec`, index macros,
  allocations, matrix-assembly loops, boundary zeroing, `nx > ncols_fine`
  guard
- `slitdec/slitdec.h` — `osamp_spec`, `lambda_fringe` in signatures
- `slitdec/slitdec_wrapper.cpp` — `osamp_spec` (default 1),
  `lambda_fringe` (default 0.0), output shapes `ncols * osamp_spec`,
  docstring

No changes to `make_curvedelta.py`, `plotting.py`, tests, or fixtures.

## Pending

- Add regression test pinning `osamp_spec=1` to a saved reference
- Add synthetic super-resolution test (tilted slit, narrow line,
  `osamp_spec=3`) — should show a non-trivial departure from
  staircase at moderate `lambda_fringe`, confirming the regulariser
  preserves genuine super-resolution content when it exists
