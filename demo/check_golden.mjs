// Correctness gate: run the wasm module on the golden inputs and compare
// against charslit's outputs. Usage: node demo/check_golden.mjs
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import createModule from "./slitdec.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const golden = JSON.parse(readFileSync(join(here, "golden.json"), "utf8"));
const M = await createModule();

const RTOL = 1e-8;

function fromJson(arr) {
  return Float64Array.from(arr, (v) => (v === null ? NaN : v));
}

function compare(name, got, want, rtol) {
  let worst = 0;
  let nbad = 0;
  const atol = 1e-10 * Math.max(...Array.from(want, Math.abs).filter(Number.isFinite), 1);
  for (let i = 0; i < want.length; i++) {
    if (!Number.isFinite(want[i]) && !Number.isFinite(got[i])) continue;
    const diff = Math.abs(got[i] - want[i]);
    const rel = diff / (Math.max(Math.abs(want[i]), Math.abs(got[i])) + atol);
    worst = Math.max(worst, rel);
    if (diff > rtol * Math.max(Math.abs(want[i]), Math.abs(got[i])) + atol) nbad++;
  }
  const ok = nbad === 0;
  console.log(`    ${name.padEnd(14)} ${ok ? "OK " : "FAIL"} worst rel diff ${worst.toExponential(2)}`);
  return ok;
}

let allOk = true;

for (const c of golden.cases) {
  const { ncols, nrows, osample } = c;
  const ny = osample * (nrows + 1) + 1;
  const npix = ncols * nrows;

  const p_im = M._malloc(npix * 8);
  const p_sP = M._malloc(ncols * 8);
  const p_sL = M._malloc(ny * 8);
  const p_model = M._malloc(npix * 8);
  const p_unc = M._malloc(ncols * 8);
  const p_info = M._malloc(5 * 8);
  const p_mask = M._malloc(npix);

  M.HEAPF64.set(fromJson(c.im), p_im / 8);

  const t0 = performance.now();
  const ret = M._demo_extract(
    ncols, nrows, p_im,
    c.ycen_frac, c.c1, c.c2,
    osample, c.lambda_sP, c.lambda_sL, c.maxiter, c.kappa,
    p_sP, p_sL, p_model, p_unc, p_info, p_mask);
  const ms = performance.now() - t0;

  // re-read heap views after the call: memory growth invalidates old ones
  const sP = M.HEAPF64.slice(p_sP / 8, p_sP / 8 + ncols);
  const sL = M.HEAPF64.slice(p_sL / 8, p_sL / 8 + ny);
  const model = M.HEAPF64.slice(p_model / 8, p_model / 8 + npix);
  const unc = M.HEAPF64.slice(p_unc / 8, p_unc / 8 + ncols);
  const info = M.HEAPF64.slice(p_info / 8, p_info / 8 + 5);
  const mask = M.HEAPU8.slice(p_mask, p_mask + npix);

  console.log(`case ${c.name}: ret=${ret} status=${info[2]} iter=${info[3]} (${ms.toFixed(0)} ms)`);

  let ok = ret === c.return_code && info[2] === c.info[2] && info[3] === c.info[3];
  if (!ok) {
    console.log(`    return/info   FAIL want ret=${c.return_code} status=${c.info[2]} iter=${c.info[3]}`);
  }
  ok = compare("spectrum", sP, fromJson(c.spectrum), RTOL) && ok;
  ok = compare("slitfunction", sL, fromJson(c.slitfunction), RTOL) && ok;
  ok = compare("model", model, fromJson(c.model), RTOL) && ok;
  ok = compare("uncertainty", unc, fromJson(c.uncertainty), RTOL) && ok;

  const wantMask = Uint8Array.from(c.mask);
  let maskDiff = 0;
  for (let i = 0; i < npix; i++) if (mask[i] !== wantMask[i]) maskDiff++;
  console.log(`    ${"mask".padEnd(14)} ${maskDiff === 0 ? "OK " : "FAIL"} ${maskDiff} differing pixels`);
  ok = maskDiff === 0 && ok;

  allOk = allOk && ok;
  for (const p of [p_im, p_sP, p_sL, p_model, p_unc, p_info]) M._free(p);
  M._free(p_mask);
}

console.log(allOk ? "\nALL CASES MATCH" : "\nMISMATCH DETECTED");
process.exit(allOk ? 0 : 1);
