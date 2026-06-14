## Results

Benchmark: `uv run python bench.py` (best-of-N wall time over the real data
files in `data/`, Release build, Apple Silicon).

| Stage | Total | CRIRES per-iteration |
|---|---|---|
| Baseline (d7f0d89) | 2430 ms | ~80 ms |
| Pixel-centric fills + band layout (d096a31) | ~780 ms | ~24-26 ms |
| xi tensor removed (fc1602f) | 634 ms | ~21 ms |

Overall ~3.8x. Peak geometry memory roughly halved (the xi allocation alone
was ~139 MB for a 176x2048 swath at osample=6: ncols * ny * 4 * 16 B).

### Round 2: dense-window fills + sequential model pass (2f71957)

A further ~20% on top of Round 1, ported from the cr2res pipeline
(`~/pipes/cr2rep.git` commit `d4e9c1f`). Same-machine before/after
(best-of-N per-iteration):

| File | Before (a34af8b) | After (2f71957) |
|---|---|---|
| CRIRES1 (176x2048) | 25.5 ms | 20.0 ms |
| ANDES_R_FP1 (404x2556) | 89.6 ms | 71.4 ms |

Output is preserved within the regression tolerance (golden references
unchanged; the only differences are at the rounding level from reordered
accumulation). See "What was changed", Round 2 below.

## Correctness guarantees

Before touching the C code, golden-reference regression tests were added
(`tests/test_regression.py`): they pin spectrum, slit function, model,
uncertainty (rtol 1e-10) and the output mask (exact) for three frozen
synthetic cases and all real FITS files. Regenerate with
`uv run pytest tests/test_regression.py --update-golden`.

The optimized code is not bitwise identical to the baseline — reordered
accumulation and FMA contraction change results at the rounding level — but
the maximum scaled deviation from the baseline references is ~1e-13, far
inside the test tolerance. The iteration count and rejected-pixel masks are
unchanged on all test data.

## What was changed and why

### 1. Band-matrix storage layout (row-major)

`laij_index` / `paij_index` / `a_index` were column-major
(`(y)*MAX_X + (x)`), so both the SLE fill loops and `bandsol` strided through
memory with stride ny (or ncols). Transposed to row-major
(`(x)*band_width + (y)`): the band entries of one matrix row are now
contiguous, matching the access pattern of both the fill and the solver.
`bandsol` itself is unchanged apart from the index macro.

### 2. Pixel-centric SLE fills using zeta only

The original fills iterated over subpixels via the xi tensor and, for each xi
entry, scanned the target pixel's zeta list — an O(mz) inner search per
contribution, with scattered writes across the band matrix.

Key observation: both SLE matrices are sums, over detector pixels, of
products of *pairs of entries in that pixel's zeta list* (weighted by sP for
the sL system and by sL for the sP system). So the fills were rewritten to
iterate over detector pixels directly:

- read each pixel's zeta list once, sequentially;
- skip masked pixels with a single test (the old code paid the full loop);
- merge duplicate keys first (several zeta entries of one pixel can share the
  same iy, or the same x), shrinking the pair loop;
- accumulate only the upper band, exploiting symmetry of both matrices, then
  mirror to the lower band in one cheap pass at the end.

This removed the O(mz^2)-ish scattered access pattern that dominated the
profile, and it made the xi tensor unused in the iteration loop.

### 3. xi tensor removed entirely

With the fills pixel-centric, xi (the subpixel -> detector-pixel mapping,
4 corners per subpixel) was written every call but never read. Removing it
also simplified `xi_zeta_tensors` dramatically: the three corner-bookkeeping
cases (A/B/C, distinguishing which xi corner a contribution belongs to)
turned out to insert *identical* zeta entries, in the same order, in all
three cases. The function is now `zeta_tensors` (~210 lines instead of ~460)
with a small `zeta_add` helper; the per-subpixel geometry (Horner evaluation
of the curvature polynomial + slitdeltas, weight split between the two
columns ix1/ix2) is untouched.

Also deleted: `create_spectral_model` (dead code — no callers anywhere, not
exposed by the Python wrapper) and the `xi_ref` typedef.

### Round 2 details

Three changes, all output-preserving (each band element gets at most one
contribution per pixel, so reordering the per-pixel accumulation cannot
change the sums beyond rounding):

1. **Dense-window SLE fills.** Round 1 merged a pixel's zeta entries by
   linear-searching a list of unique keys (`zk`/`zw`). The zeta build now
   records per pixel the key ranges (`zeta_rng`: min/max of `iy` and of
   `x`, maintained in `zeta_add`), so the merge scatters straight into a
   dense window `zw[key - k0]`. The `iy` range is at most `2*osample`
   (the band width the matrix assumes) and the `x` range at most
   `2*delta_x`, so band accumulation walks the window row-wise and the
   inner loop is contiguous in both operands. An over-wide list (extreme
   geometry beyond the band width) falls back to the old unique-key
   search. The `zw`/`zk` scratch buffers are sized `max(MAX_ZETA_Z, nx)`
   so the spectrum window fits.
2. **Sequential model pass.** The model loop now iterates `x` outermost so
   the zeta tensor (the largest array) is read sequentially instead of
   with a ~ny*3*(osample+1) stride; per-pixel sums accumulate into a local
   then write once (the separate model-zeroing pass is gone).
3. (cr2res-only, not ported: the `extract_traces` per-trace merge via
   direct buffer access and the OpenMP trace loop are pipeline-specific.)

## Scaling notes

Cost is linear in the number of detector pixels with a flat per-pixel cost;
the former quadratic-in-mz inner work is gone. Larger swaths benefit at least
as much as the test data. `bandsol` is linear in matrix size and is no longer
a bottleneck.

## Reproducing

```bash
uv pip install -e . --force-reinstall --no-deps
uv run pytest                 # includes golden regression tests
uv run python bench.py 5      # benchmark, best of 5
```
