#!/bin/sh
# IDL REDUCE: CALL_EXTERNAL entry point slit_func_ox (mkslitf.pro, SLITDEC_OX={...}),
# built from the same slitdec.c as the Python package.
set -e
cd "$(dirname "$0")"
cc -O2 -fPIC -Wall -shared -I../slitdec -o libslitdec_ox_idl.so slitdec_ox_idl.c ../slitdec/slitdec.c -lm
