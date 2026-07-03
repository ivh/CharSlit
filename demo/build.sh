#!/bin/sh
# Build the wasm demo module. Requires emsdk (brew install emscripten).
set -e
cd "$(dirname "$0")/.."

emcc slitdec/slitdec.c demo/demo_api.c -O2 \
  -sMODULARIZE -sEXPORT_ES6 -sALLOW_MEMORY_GROWTH \
  -sEXPORTED_FUNCTIONS=_demo_extract,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=HEAPF64,HEAPU8 \
  -o demo/slitdec.mjs

ls -la demo/slitdec.mjs demo/slitdec.wasm
