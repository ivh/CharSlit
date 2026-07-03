// Minimal FITS and .npy readers for the demo. Returns row-major Float64Array
// with row 0 first (FITS storage order; drawn bottom-up by the UI).

export const MAX_PIXELS = 2048 * 2048;

export function parseFits(buffer) {
  const td = new TextDecoder("ascii");
  let pos = 0;
  let first = true;

  const readHeader = () => {
    const cards = {};
    for (;;) {
      if (pos + 2880 > buffer.byteLength) throw new Error("truncated FITS header");
      const block = td.decode(new Uint8Array(buffer, pos, 2880));
      pos += 2880;
      for (let i = 0; i < 36; i++) {
        const card = block.slice(i * 80, (i + 1) * 80);
        const key = card.slice(0, 8).trim();
        if (key === "END") return cards;
        if (!key || card[8] !== "=") continue;
        cards[key] = card.slice(10).split("/")[0].trim();
      }
    }
  };
  const num = (h, k, d) => (h[k] !== undefined ? parseFloat(h[k]) : d);

  while (pos < buffer.byteLength) {
    const h = readHeader();
    const bitpix = num(h, "BITPIX", 0);
    const naxis = num(h, "NAXIS", 0);
    const dims = [];
    let ndata = naxis > 0 ? 1 : 0;
    for (let i = 1; i <= naxis; i++) {
      const n = num(h, "NAXIS" + i, 0);
      dims.push(n);
      ndata *= n;
    }
    const isImage = first || (h.XTENSION || "").includes("IMAGE");

    if (isImage && naxis >= 2 && dims[0] > 1 && dims[1] > 1 && ndata > 0) {
      const w = dims[0], ht = dims[1];   // first 2D plane of an N-D cube
      if (w * ht > MAX_PIXELS)
        throw new Error(`image too large (${w}x${ht}, cap ${(MAX_PIXELS / 1e6).toFixed(1)} Mpx)`);
      const bscale = num(h, "BSCALE", 1), bzero = num(h, "BZERO", 0);
      const bytes = Math.abs(bitpix) / 8;
      if (pos + w * ht * bytes > buffer.byteLength) throw new Error("truncated FITS data");
      const dv = new DataView(buffer, pos);
      const out = new Float64Array(w * ht);
      for (let i = 0; i < w * ht; i++) {
        let v;
        switch (bitpix) {   // FITS is big-endian, DataView's default
          case 8: v = dv.getUint8(i); break;
          case 16: v = dv.getInt16(i * 2); break;
          case 32: v = dv.getInt32(i * 4); break;
          case -32: v = dv.getFloat32(i * 4); break;
          case -64: v = dv.getFloat64(i * 8); break;
          default: throw new Error("unsupported BITPIX " + bitpix);
        }
        out[i] = v * bscale + bzero;
      }
      return { data: out, ncols: w, nrows: ht };
    }

    pos += Math.ceil(ndata * Math.abs(bitpix) / 8 / 2880) * 2880;
    first = false;
  }
  throw new Error("no 2D image HDU found");
}

export function parseNpy(buffer) {
  const dv = new DataView(buffer);
  const td = new TextDecoder("ascii");
  const magic = new Uint8Array(buffer, 0, 6);
  if (magic[0] !== 0x93 || td.decode(magic.subarray(1)) !== "NUMPY")
    throw new Error("not a .npy file");
  const major = dv.getUint8(6);
  const hlen = major >= 2 ? dv.getUint32(8, true) : dv.getUint16(8, true);
  const off0 = major >= 2 ? 12 : 10;
  const header = td.decode(new Uint8Array(buffer, off0, hlen));
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const fortran = /'fortran_order':\s*True/.test(header);
  const shape = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1]
    .split(",").map((s) => parseInt(s, 10)).filter((n) => !isNaN(n));
  if (!descr || !shape || shape.length !== 2) throw new Error("need a 2D .npy array");
  const [nr, nc] = shape;
  if (nr * nc > MAX_PIXELS)
    throw new Error(`array too large (cap ${(MAX_PIXELS / 1e6).toFixed(1)} Mpx)`);

  const le = descr[0] !== ">";
  const getters = {
    f8: (o) => dv.getFloat64(o, le), f4: (o) => dv.getFloat32(o, le),
    i8: (o) => Number(dv.getBigInt64(o, le)), i4: (o) => dv.getInt32(o, le),
    i2: (o) => dv.getInt16(o, le), u2: (o) => dv.getUint16(o, le),
    u1: (o) => dv.getUint8(o), i1: (o) => dv.getInt8(o),
  };
  const t = descr.slice(1);
  if (!(t in getters)) throw new Error("unsupported dtype " + descr);
  const size = parseInt(t.slice(1), 10);
  const base = off0 + hlen;
  const get = getters[t];
  const out = new Float64Array(nr * nc);
  for (let i = 0; i < nr * nc; i++) {
    // fortran order stores column-major: element (r,c) sits at c*nr + r
    const j = fortran ? (i % nc) * nr + Math.floor(i / nc) : i;
    out[i] = get(base + j * size);
  }
  return { data: out, ncols: nc, nrows: nr };
}
