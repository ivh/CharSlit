// slitdec web demo: UI, plotting and file loading. The solver itself runs
// in worker.js on the wasm build of slitdec.c.

import { parseFits, parseNpy } from "./io.js";

// ---------------------------------------------------------------- state

let image = null;      // Float64Array, row-major, row 0 = bottom
let ncols = 0, nrows = 0;
let nBadInput = 0;     // non-finite input pixels, to report kappa rejections
let lastResult = null;

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- sliders

const SYN_C1 = -0.03, SYN_C2 = 0.0015;

const SLIDERS = [
  { id: "osample", label: "osample", min: 2, max: 10, step: 1, value: 6,
    fmt: (v) => v.toFixed(0) },
  { id: "c1", label: "curvature c1", min: -0.15, max: 0.15, step: 0.0005,
    value: SYN_C1, fmt: (v) => v.toFixed(4) },
  { id: "c2", label: "curvature c2", min: -0.002, max: 0.002, step: 0.00002,
    value: SYN_C2, fmt: (v) => v.toFixed(5) },
  { id: "ycen", label: "ycen fraction", min: 0, max: 1, step: 0.01, value: 0.5,
    fmt: (v) => v.toFixed(2) },
  { id: "lsL", label: "lambda_sL (log)", min: -3, max: 3, step: 0.05, value: 0,
    map: (v) => Math.pow(10, v), fmt: (v) => Math.pow(10, v).toPrecision(3) },
  { id: "lsP", label: "lambda_sP (log, min = off)", min: -4, max: 3, step: 0.05,
    value: -4, map: (v) => (v <= -4 ? 0 : Math.pow(10, v)),
    fmt: (v) => (v <= -4 ? "off" : Math.pow(10, v).toPrecision(3)) },
  { id: "kappa", label: "kappa (0 = off)", min: 0, max: 20, step: 0.5, value: 10,
    fmt: (v) => (v === 0 ? "off" : v.toFixed(1)) },
  { id: "maxiter", label: "maxiter", min: 1, max: 50, step: 1, value: 20,
    fmt: (v) => v.toFixed(0) },
];

const sliderEls = {};

function buildSliders() {
  const host = $("sliders");
  for (const s of SLIDERS) {
    const row = document.createElement("div");
    row.className = "slider-row";
    row.innerHTML =
      `<div class="lbl"><span>${s.label}</span><span class="val"></span></div>` +
      `<input type="range" min="${s.min}" max="${s.max}" step="${s.step}" value="${s.value}">`;
    const input = row.querySelector("input");
    const val = row.querySelector(".val");
    const update = () => { val.textContent = s.fmt(parseFloat(input.value)); };
    input.addEventListener("input", () => { update(); scheduleRun(); });
    update();
    host.appendChild(row);
    sliderEls[s.id] = input;
  }
}

function sliderValue(id) {
  const s = SLIDERS.find((x) => x.id === id);
  const raw = parseFloat(sliderEls[id].value);
  return s.map ? s.map(raw) : raw;
}

function setSlider(id, v) {
  sliderEls[id].value = v;
  sliderEls[id].dispatchEvent(new Event("input"));
}

function currentParams() {
  return {
    osample: Math.round(sliderValue("osample")),
    c1: sliderValue("c1"),
    c2: sliderValue("c2"),
    ycen_frac: sliderValue("ycen"),
    lambda_sL: sliderValue("lsL"),
    lambda_sP: sliderValue("lsP"),
    kappa: sliderValue("kappa"),
    maxiter: Math.round(sliderValue("maxiter")),
  };
}

// ------------------------------------------------------- synthetic swath

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeSynthetic() {
  const nc = 400, nr = 48;
  const rand = mulberry32(42);
  const gauss = () => {
    const u = Math.max(rand(), 1e-12), v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  const spec = new Float64Array(nc);
  const lines = [[55, 900, 2.5], [140, 1600, 3.0], [210, 500, 1.8], [320, 1100, 2.2]];
  for (let x = 0; x < nc; x++) {
    let s = 200 + 50 * Math.sin(x / 37);
    for (const [c, a, w] of lines) s += a * Math.exp(-0.5 * ((x - c) / w) ** 2);
    spec[x] = s;
  }
  const specAt = (x) => {
    if (x <= 0) return spec[0];
    if (x >= nc - 1) return spec[nc - 1];
    const i = Math.floor(x), f = x - i;
    return (1 - f) * spec[i] + f * spec[i + 1];
  };

  const yc = nr / 2, sigma = nr / 6;
  const profile = new Float64Array(nr);
  let psum = 0;
  for (let y = 0; y < nr; y++) { profile[y] = Math.exp(-0.5 * ((y - yc) / sigma) ** 2); psum += profile[y]; }

  const im = new Float64Array(nc * nr);
  for (let y = 0; y < nr; y++) {
    const delta = SYN_C1 * y + SYN_C2 * y * y;
    const p = profile[y] / psum * nr;
    for (let x = 0; x < nc; x++) {
      const v = specAt(x - delta) * p;
      im[y * nc + x] = v + gauss() * Math.sqrt(Math.abs(v) + 1);
    }
  }

  // hot/cold pixels for the kappa slider to eat, and a few dead (NaN) ones
  for (let k = 0; k < 8; k++) {
    const x = Math.floor(rand() * nc);
    const y = Math.floor(8 + rand() * (nr - 16));
    im[y * nc + x] += (rand() < 0.5 ? -1 : 1) * (2000 + rand() * 3000);
  }
  im[Math.floor(nr / 2) * nc + 30] = NaN;
  im[10 * nc + 260] = NaN;

  return { im, ncols: nc, nrows: nr };
}

// ------------------------------------------------------- worker plumbing

const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
let workerReady = false;
let seq = 0;
let inFlight = false;
let runQueued = false;
let debounceTimer = null;

function scheduleRun() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => { runQueued = true; maybeRun(); }, 60);
}

function maybeRun() {
  if (!workerReady || inFlight || !runQueued || !image) return;
  runQueued = false;
  inFlight = true;
  worker.postMessage({ type: "run", seq: ++seq, params: currentParams() });
}

worker.onmessage = (e) => {
  if (e.data.type === "ready") { workerReady = true; maybeRun(); return; }
  inFlight = false;
  if (e.data.seq === seq) {      // stale replies are dropped
    lastResult = e.data;
    render(e.data);
  }
  maybeRun();
};

function setImage(imArr, nc, nr, source) {
  image = imArr;
  ncols = nc;
  nrows = nr;
  nBadInput = 0;
  for (let i = 0; i < imArr.length; i++) if (!Number.isFinite(imArr[i])) nBadInput++;
  lastResult = null;
  const copy = new Float64Array(imArr);
  worker.postMessage({ type: "setImage", ncols: nc, nrows: nr, im: copy },
                     [copy.buffer]);
  $("filemsg").textContent = source;
  runQueued = true;
  maybeRun();
}

// ---------------------------------------------------------------- colormaps

function buildLut(anchors) {
  const lut = new Uint8Array(256 * 3);
  const n = anchors.length - 1;
  for (let i = 0; i < 256; i++) {
    const t = i / 255 * n;
    const k = Math.min(Math.floor(t), n - 1);
    const f = t - k;
    for (let c = 0; c < 3; c++)
      lut[i * 3 + c] = Math.round((1 - f) * anchors[k][c] + f * anchors[k + 1][c]);
  }
  return lut;
}

const VIRIDIS = buildLut([
  [68, 1, 84], [71, 44, 122], [59, 81, 139], [44, 113, 142], [33, 144, 141],
  [39, 173, 129], [92, 200, 99], [170, 220, 50], [253, 231, 37],
]);
const DIVERGING = buildLut([
  [59, 76, 192], [124, 159, 249], [222, 220, 218], [245, 152, 105], [180, 4, 38],
]);

function percentiles(arr, lo, hi) {
  const finite = [];
  const stride = Math.max(1, Math.floor(arr.length / 100000));
  for (let i = 0; i < arr.length; i += stride)
    if (Number.isFinite(arr[i])) finite.push(arr[i]);
  if (finite.length === 0) return [0, 1];
  finite.sort((a, b) => a - b);
  const at = (p) => finite[Math.min(finite.length - 1, Math.floor(p / 100 * finite.length))];
  return [at(lo), at(hi)];
}

// ------------------------------------------------------------------ plots

// row 0 of the data is drawn at the BOTTOM (astronomy convention)
function drawImage(canvas, data, vmin, vmax, lut, mask) {
  canvas.width = ncols;
  canvas.height = nrows;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(ncols, nrows);
  const px = img.data;
  const scale = vmax > vmin ? 255 / (vmax - vmin) : 0;
  for (let y = 0; y < nrows; y++) {
    const srcRow = (nrows - 1 - y) * ncols;
    for (let x = 0; x < ncols; x++) {
      const v = data[srcRow + x];
      const o = (y * ncols + x) * 4;
      if (!Number.isFinite(v)) {
        px[o] = 90; px[o + 1] = 90; px[o + 2] = 96;
      } else {
        let i = Math.round((v - vmin) * scale);
        i = i < 0 ? 0 : i > 255 ? 255 : i;
        px[o] = lut[i * 3]; px[o + 1] = lut[i * 3 + 1]; px[o + 2] = lut[i * 3 + 2];
      }
      if (mask && mask[srcRow + x] === 0) {   // masked/rejected: red tint
        px[o] = Math.min(255, px[o] * 0.4 + 200);
        px[o + 1] *= 0.35; px[o + 2] *= 0.35;
      }
      px[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function preparePlot(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function linePlot(canvas, xs, ys, { band = null, color = "#7ec8e3", x0 = null, x1 = null } = {}) {
  const { ctx, w, h } = preparePlot(canvas);
  const ml = 54, mr = 8, mt = 8, mb = 22;
  const pw = w - ml - mr, ph = h - mt - mb;

  let ymin = Infinity, ymax = -Infinity;
  const consider = (v) => { if (Number.isFinite(v)) { if (v < ymin) ymin = v; if (v > ymax) ymax = v; } };
  for (let i = 0; i < ys.length; i++) {
    consider(ys[i]);
    if (band) { consider(band[0][i]); consider(band[1][i]); }
  }
  if (!(ymin < ymax)) { ymin = 0; ymax = 1; }
  const pad = 0.05 * (ymax - ymin || 1);
  ymin -= pad; ymax += pad;

  const xmin = x0 !== null ? x0 : xs[0];
  const xmax = x1 !== null ? x1 : xs[xs.length - 1];
  const X = (x) => ml + (x - xmin) / (xmax - xmin) * pw;
  const Y = (y) => mt + (ymax - y) / (ymax - ymin) * ph;

  // frame and a few ticks
  ctx.strokeStyle = "#2a2e36";
  ctx.strokeRect(ml, mt, pw, ph);
  ctx.fillStyle = "#9aa0aa";
  ctx.font = "11px system-ui";
  ctx.textAlign = "right";
  for (const fy of [0, 0.5, 1]) {
    const yv = ymin + fy * (ymax - ymin);
    ctx.fillText(yv.toPrecision(3), ml - 5, Y(yv) + 4);
    ctx.strokeStyle = "#22262d";
    ctx.beginPath(); ctx.moveTo(ml, Y(yv)); ctx.lineTo(ml + pw, Y(yv)); ctx.stroke();
  }
  ctx.textAlign = "center";
  for (const fx of [0, 0.25, 0.5, 0.75, 1]) {
    const xv = xmin + fx * (xmax - xmin);
    ctx.fillText(xv.toFixed(0), X(xv), h - 6);
  }

  if (band) {
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < xs.length; i++) {
      if (!Number.isFinite(band[0][i])) { started = false; continue; }
      if (!started) { ctx.moveTo(X(xs[i]), Y(band[0][i])); started = true; }
      else ctx.lineTo(X(xs[i]), Y(band[0][i]));
    }
    for (let i = xs.length - 1; i >= 0; i--) {
      if (!Number.isFinite(band[1][i])) continue;
      ctx.lineTo(X(xs[i]), Y(band[1][i]));
    }
    ctx.closePath();
    ctx.fillStyle = "rgba(126,200,227,0.18)";
    ctx.fill();
  }

  ctx.beginPath();
  let started = false;
  for (let i = 0; i < xs.length; i++) {
    if (!Number.isFinite(ys[i])) { started = false; continue; }
    if (!started) { ctx.moveTo(X(xs[i]), Y(ys[i])); started = true; }
    else ctx.lineTo(X(xs[i]), Y(ys[i]));
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.3;
  ctx.stroke();
}

const STATUS_TEXT = {
  1: ["converged", "ok"],
  [-1]: ["out of iterations", "bad"],
  [-2]: ["curvature too large", "bad"],
  [-3]: ["non-finite result", "bad"],
  [-99]: ["allocation failed", "bad"],
};

function render(r) {
  const [vmin, vmax] = percentiles(image, 1, 99);
  drawImage($("c-data"), image, vmin, vmax, VIRIDIS, r.mask);
  drawImage($("c-model"), r.model, vmin, vmax, VIRIDIS, null);

  const resid = new Float64Array(ncols * nrows);
  for (let i = 0; i < resid.length; i++) resid[i] = image[i] - r.model[i];
  const [, rhi] = percentiles(resid.map(Math.abs), 0, 99);
  drawImage($("c-resid"), resid, -rhi, rhi, DIVERGING, null);

  const xs = Float64Array.from({ length: ncols }, (_, i) => i);
  const lo = new Float64Array(ncols), hi = new Float64Array(ncols);
  for (let i = 0; i < ncols; i++) {
    lo[i] = r.spectrum[i] - r.unc[i];
    hi[i] = r.spectrum[i] + r.unc[i];
  }
  linePlot($("c-spec"), xs, r.spectrum, { band: [lo, hi] });

  const sx = Float64Array.from({ length: r.ny }, (_, i) => i / r.osample);
  linePlot($("c-slit"), sx, r.slitfunction, { color: "#a5d68f" });

  const [txt, cls] = STATUS_TEXT[r.info[2]] || [`status ${r.info[2]}`, "bad"];
  const st = $("st-status");
  st.textContent = txt;
  st.className = cls;
  $("st-iter").textContent = r.info[3].toFixed(0);
  $("st-dx").textContent = r.info[4].toFixed(0);
  $("st-ms").textContent = r.ms.toFixed(1) + " ms";
  let nMasked = 0;
  for (let i = 0; i < r.mask.length; i++) if (r.mask[i] === 0) nMasked++;
  $("st-rej").textContent = `${nMasked - nBadInput} (+${nBadInput} NaN)`;
}

// ----------------------------------------------------------- hover readout

function attachHover(canvas) {
  canvas.addEventListener("mousemove", (e) => {
    if (!image) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / rect.width * ncols);
    const yTop = Math.floor((e.clientY - rect.top) / rect.height * nrows);
    const y = nrows - 1 - yTop;
    if (x < 0 || x >= ncols || y < 0 || y >= nrows) return;
    const i = y * ncols + x;
    const f = (v) => (Number.isFinite(v) ? v.toFixed(1) : "NaN");
    let txt = `x=${x} y=${y}  data=${f(image[i])}`;
    if (lastResult) {
      txt += `  model=${f(lastResult.model[i])}` +
             `  resid=${f(image[i] - lastResult.model[i])}` +
             (lastResult.mask[i] === 0 ? "  [masked]" : "");
    }
    $("hoverbox").textContent = txt;
  });
  canvas.addEventListener("mouseleave", () => { $("hoverbox").innerHTML = "&nbsp;"; });
}

// ------------------------------------------------------------ file parsing


// sample-swath dropdown, fed by swaths.json (generated by make_manifest.py;
// absent in a deployment without bundled data, so it hides itself)
async function initSwathDropdown() {
  const sel = $("swathsel");
  let manifest;
  try {
    const resp = await fetch("swaths.json");
    if (!resp.ok) return;
    manifest = await resp.json();
  } catch {
    return;
  }
  for (const s of manifest.swaths) {
    const opt = document.createElement("option");
    opt.value = s.name;
    opt.textContent = `${s.name} (${s.ncols} x ${s.nrows})${s.fitted ? "" : " - no curvature fit"}`;
    sel.appendChild(opt);
  }
  sel.hidden = false;
  sel.addEventListener("change", async () => {
    const s = manifest.swaths.find((x) => x.name === sel.value);
    if (!s) return;
    $("filemsg").textContent = `loading ${s.name}...`;
    try {
      const buffer = await (await fetch(s.url)).arrayBuffer();
      const { data, ncols: nc, nrows: nr } = parseFits(buffer);
      setSlider("c1", s.c1);
      setSlider("c2", s.c2);
      setSlider("ycen", s.ycen_frac);
      setImage(data, nc, nr,
               `${s.name}: ${nc} x ${nr}${s.fitted ? ", fitted c1/c2/ycen" : ""}`);
    } catch (err) {
      $("filemsg").textContent = `${s.name}: ${err.message}`;
    }
  });
}

async function loadFile(file) {
  try {
    const buffer = await file.arrayBuffer();
    const isNpy = file.name.toLowerCase().endsWith(".npy");
    const { data, ncols: nc, nrows: nr } = isNpy ? parseNpy(buffer) : parseFits(buffer);
    // uploads are pitched as flattish swaths: reset curvature to zero
    setSlider("c1", 0);
    setSlider("c2", 0);
    $("swathsel").value = "";
    setImage(data, nc, nr, `${file.name}: ${nc} x ${nr}`);
  } catch (err) {
    $("filemsg").textContent = `${file.name}: ${err.message}`;
  }
}

// ------------------------------------------------------------------- init

buildSliders();
for (const id of ["c-data", "c-model", "c-resid"]) attachHover($(id));

$("fileinput").addEventListener("change", (e) => {
  if (e.target.files.length) loadFile(e.target.files[0]);
  e.target.value = "";
});

initSwathDropdown();

$("resetbtn").addEventListener("click", () => {
  $("swathsel").value = "";
  setSlider("c1", SYN_C1);
  setSlider("c2", SYN_C2);
  const syn = makeSynthetic();
  setImage(syn.im, syn.ncols, syn.nrows, "built-in synthetic swath (400 x 48)");
});

window.addEventListener("resize", () => { if (lastResult) render(lastResult); });

const syn = makeSynthetic();
setImage(syn.im, syn.ncols, syn.nrows, "built-in synthetic swath (400 x 48)");
