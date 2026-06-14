# Deferred todos

**Do not start on any of these until explicitly asked.** This file is a parking
lot for follow-ups that came up in conversation but were intentionally not
pursued, so the work record is captured without committing to it.

## Validate osamp_spec at s > 3 before pinning regression tests

Context: `tests/test_osamp_spec.py` pins `osamp_spec=1` against a saved
reference. The wide-LSF s-sweep (`experiment_synthetic_lsf_wide_ssweep`)
showed s=4-5 beating s=3 on BIS bias, with s=6 starting to overshoot. That
is interesting but not validated. OSAMP_SPEC.md explicitly deferred
characterising s>3.

Pin only after the following are understood:

- **Geometry robustness.** Repeat the wide-LSF s-sweep at different total
  tilt sweeps (e.g. 1, 2, 5 px), different `ncols`, different line
  positions. If the s with minimum BIS bias is stable across these, the
  sweet spot is real. If it moves around, the apparent gain at s=4-5 was a
  one-realisation accident.
- **Multi-seed scatter.** Run the same experiment with 20-50 noise seeds
  and report mean +- std of BIS bias per s. Tells us whether the s=6
  undershoot is bias or just one bad draw, and gives a uncertainty bar
  to compare s=3 vs s=4 vs s=5 properly.
- **Mechanism for s=6 over-correction.** OSAMP_SPEC.md predicts gains
  saturate near s=5 because of the fringe-mode null in the forward model;
  verify that's what's happening at s=6 rather than e.g. the boundary
  zeroing (`first/last delta_x * osamp_spec` fine bins set to zero) eating
  a larger fraction of the recoverable spectrum. A controlled diagnostic
  would be to vary the boundary-zero width and watch whether the s=6
  behaviour tracks it.

Only after all three of the above land should a regression test be added
that pins s=2..5 (or whatever the validated range turns out to be) to a
saved reference.
