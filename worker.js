// Solver worker: owns the wasm module and the current image so slider
// changes only ship parameters, not the pixel data.
import createModule from "./slitdec.mjs";

// onmessage must be assigned synchronously: with top-level await, messages
// posted while the module is still loading get dispatched into the void.
let M = null;
const backlog = [];
onmessage = (e) => { if (M) handle(e); else backlog.push(e); };

createModule().then((mod) => {
  M = mod;
  postMessage({ type: "ready" });
  for (const e of backlog) handle(e);
  backlog.length = 0;
});

let ncols = 0, nrows = 0;
let buf = null; // { osample, ny, p_im, p_sP, p_sL, p_model, p_unc, p_info, p_mask }

function freeBuffers() {
  if (!buf) return;
  for (const k of ["p_im", "p_sP", "p_sL", "p_model", "p_unc", "p_info", "p_mask"])
    M._free(buf[k]);
  buf = null;
}

function allocBuffers(osample) {
  const ny = osample * (nrows + 1) + 1;
  const npix = ncols * nrows;
  buf = {
    osample, ny,
    p_im: M._malloc(npix * 8),
    p_sP: M._malloc(ncols * 8),
    p_sL: M._malloc(ny * 8),
    p_model: M._malloc(npix * 8),
    p_unc: M._malloc(ncols * 8),
    p_info: M._malloc(5 * 8),
    p_mask: M._malloc(npix),
  };
}

let image = null; // Float64Array master copy, re-uploaded before each run
                  // because slitdec reads im in place and NaNs must survive

function handle(e) {
  const msg = e.data;

  if (msg.type === "setImage") {
    freeBuffers();
    ncols = msg.ncols;
    nrows = msg.nrows;
    image = msg.im;
    return;
  }

  if (msg.type !== "run" || !image) return;
  const p = msg.params;

  if (!buf || buf.osample !== p.osample) {
    freeBuffers();
    allocBuffers(p.osample);
  }
  const { ny, p_im, p_sP, p_sL, p_model, p_unc, p_info, p_mask } = buf;
  const npix = ncols * nrows;

  M.HEAPF64.set(image, p_im / 8);

  const t0 = performance.now();
  const ret = M._demo_extract(
    ncols, nrows, p_im,
    p.ycen_frac, p.c1, p.c2,
    p.osample, p.lambda_sP, p.lambda_sL, p.maxiter, p.kappa,
    p_sP, p_sL, p_model, p_unc, p_info, p_mask);
  const ms = performance.now() - t0;

  // slice() after the call: memory growth invalidates earlier heap views
  const out = {
    type: "result",
    seq: msg.seq,
    ret, ms, ny, ncols, nrows,
    spectrum: M.HEAPF64.slice(p_sP / 8, p_sP / 8 + ncols),
    slitfunction: M.HEAPF64.slice(p_sL / 8, p_sL / 8 + ny),
    model: M.HEAPF64.slice(p_model / 8, p_model / 8 + npix),
    unc: M.HEAPF64.slice(p_unc / 8, p_unc / 8 + ncols),
    info: M.HEAPF64.slice(p_info / 8, p_info / 8 + 5),
    mask: M.HEAPU8.slice(p_mask, p_mask + npix),
    osample: p.osample,
  };
  postMessage(out, [
    out.spectrum.buffer, out.slitfunction.buffer, out.model.buffer,
    out.unc.buffer, out.info.buffer, out.mask.buffer,
  ]);
}
