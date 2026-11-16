# Zero-filled NaN pixels re-enter the fit (parked 2026-11-16)

Found while validating `uncertainty='propagated'` on the CRIRES test frames. Not fixed.

## What happens

1. The callers replace non-finite pixels by 0 and mask them:
   - `tests/conftest.py` (`real_data_files`, `im[nan_mask] = 0.0`) and `scripts/bench.py`
   - PyReduce `pyreduce/extract.py`: `data[~np.isfinite(data)] = 0`, then added to the
     mask. Its saved `input_mask` is only used for plotting.
2. With `kappa > 0` the C code rewrites the whole mask every iteration,
   `mask = |model - im| < kappa * dev` for the columns inside `delta_x`, ignoring the
   input mask. Same on master (`slitdec/slitdec.c`, the kappa block) and on `slitdec_ox`
   (`slitdec/slitdec_ox.c`).
3. A zero-filled pixel passes that test wherever the model is faint, comes back as good and
   is fitted as a measurement of zero flux.

If the NaN is left in the image the pixel cannot come back: a comparison with NaN is false.

## Measured (settings that reproduce master: `osample_x=1, mu_x=[], t_offset=0`, osample 6)

| frame | masked on input | back as good, zero-filled | back as good, NaN kept | status zero-filled / NaN kept |
|---|---|---|---|---|
| CRIRES2 | 12366 (3.4 %) | 10092 | 0 | -1 / -1 |
| CRIRES1 | 16710 (4.6 %) | 8734 | 0 | -1 / 1 |

Spectrum from the zero-filled input relative to the NaN-kept one:

| frame | median | rms | worst column | columns off by > 1 % |
|---|---|---|---|---|
| CRIRES2 | -2.5 % | 8.5 % | 335 % | 1630 of 2019 |
| CRIRES1 | -0.8 % | 4.4 % | 74 % | 984 of 2038 |

So the CRIRES goldens pin this behaviour, and the bench timings of those frames (21
iterations = `maxiter`) measure it.

How it surfaced: the fixture gives those pixels `pix_unc = 1e10`; once they are unmasked
the propagated variance explodes (median propagated/residual 2.5e8 on CRIRES2), which is
the right answer for the solve as actually done.

## Not checked

- Whether a NaN in a masked pixel is harmless on every path (weights, `variance_fit`, the
  flat fast path, the residual uncertainty). The NaN-kept runs above returned finite
  spectra, on the general path with `weights='none'` only.
- How often non-finite pixels reach PyReduce's extraction in practice, and what cr2res
  passes for bad pixels.

## Decision taken

Masked pixels that hold a value may be unmasked again by the kappa test: wanted, the BPM
can be overzealous. Only pixels without a value must never come back.

## Options

- **C code (preferred):** treat non-finite pixels as permanently bad, so that leaving NaN
  in is the supported way to say "no data"; pixels flagged by a BPM but holding a value
  keep their second chance. Callers stop zero-filling.
- **Callers only:** stop zero-filling. Works for the mask today, relies on the unchecked
  points above.

Either way the CRIRES goldens change (regenerate with `--update-golden`) and PyReduce's
zero-fill line should go. Master has the same behaviour, so the fix belongs there too.
