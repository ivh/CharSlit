# slitdec for IDL REDUCE

`slit_func_ox` is the IDL `CALL_EXTERNAL` entry point to slitdec, built from the same
`slitdec/slitdec.c` as the Python package. The algorithm, the oversampled spectrum
(`osample_x`) and the penalties in physical units (`mu_x`, `mu_y`) are described in
[docs/slitdec_ox.md](../docs/slitdec_ox.md); Nik Piskunov's working notes, including the
REDUCE and SALT-HRS tests, are in [docs/ox_notes.md](../docs/ox_notes.md).

## Building

```sh
sh idl/build.sh
```

builds `idl/libslitdec_ox_idl.so` with the system C compiler.

## Calling

All 27 arguments are required and there are no defaults, so every call states the
parameters it uses (list in the header of `slitdec_ox_idl.c`):

```idl
iret = CALL_EXTERNAL(lib, 'slit_func_ox', ncols, nrows, osample, osample_x, im, $
         ycen_frac, ycen_offset, y_lower_lim, PSF_curve, mu1_x, mu2_x, mu3_x, $
         mu1_y, mu2_y, maxiter, kappa, weight_mode, gain, read_noise, sL_preset, $
         mask, model, sP, sP_fine, sL, unc, info)
```

Compared with REDUCE's `slit_func_2d`, `lambda_sP` and `lambda_sF` are replaced by the
`mu` penalties, and `delta_x` (returned in `info[4]`), `sP_old` and the scratch arrays
are gone. As with `slit_func_2d`, the `delta_x` columns at each swath end (the largest
slit-image shift) come back as 0. `weight_mode = 1` gives optimal-extraction weights, with
the variance model/gain + (read_noise/gain)² taken from the current model. In
`mkslitf.pro` the call is selected with a structure that must hold every parameter (gain
and read noise come from its `GAIN` and `READN` keywords):

```idl
mkslitf, ..., SLITDEC_OX={LIBRARY:'/path/to/libslitdec_ox_idl.so', OSAMPLE_X:8L, $
                          MU1_X:0d, MU2_X:0d, MU3_X:1.64d-4, MU1_Y:1d-2, MU2_Y:0d, $
                          MAXITER:100L, KAPPA:10d, WEIGHTS:0L}
```

With `osample_x = 1` and no spectrum penalty the result is REDUCE's except for the tilt
reference point: slitdec evaluates the slit curvature at the centre of each y-subpixel,
`slit_func_2d` at its lower edge, which shifts every line by `c1 / (2 osample)` pixels.

## Tests

- `tests/test_idl.py` (part of `uv run pytest`): builds the library and calls it the way
  `CALL_EXTERNAL` does on synthetic swaths: argument checks, agreement with the Python
  package, REDUCE edge convention, weights.
- `idl/test_idl_interface.py [test.sav] [reference_lib.so ...]`: the same on the swath that
  `mkslitf.pro` saves, compared with REDUCE libraries.
- `idl_test_ox.pro`: the same from inside IDL (`idl -e "idl_test_ox"`).

## Status

Tested on synthetic swaths, on one CRIRES+ swath saved from `mkslitf.pro`, and in full
SALT-HRS red-arm reductions with IDL REDUCE (white extraction noise where REDUCE's is blue).
About 3× slower than REDUCE per SALT-HRS frame.
