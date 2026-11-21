// SPDX-License-Identifier: LGPL-3.0-or-later
// XBeach WebGPU solver — derived from XBeach (c) Deltares (LGPL v2.1+).
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * I/O helpers for the headless (Deno) runner.
 * Output uses the same raw layout as XBeach outputformat=fortran: one
 * float64 record of (nx+1)*(ny+1) values per output time, column-major
 * (i fastest), file <mnem>.dat — so the python compare tooling reads both
 * models identically.
 */

export async function readDepFile(
  path: string,
  nx: number,
  ny: number,
): Promise<Float64Array> {
  const text = await Deno.readTextFile(path);
  const vals = text.split(/\s+/).filter(Boolean).map(Number);
  const n = (nx + 1) * (ny + 1);
  if (vals.length < n) {
    throw new Error(`${path}: expected ${n} values, got ${vals.length}`);
  }
  // file is row-per-j, i fastest -> matches our column-major layout directly
  return new Float64Array(vals.slice(0, n));
}

export class DatWriter {
  private files = new Map<string, Deno.FsFile>();
  constructor(public dir: string) {}

  async append(name: string, data: Float32Array | Float64Array) {
    let f = this.files.get(name);
    if (!f) {
      f = await Deno.open(`${this.dir}/${name}.dat`, {
        create: true,
        write: true,
        truncate: true,
      });
      this.files.set(name, f);
    }
    const f64 = data instanceof Float64Array ? data : Float64Array.from(data);
    const bytes = new Uint8Array(f64.buffer, f64.byteOffset, f64.byteLength);
    let written = 0;
    while (written < bytes.length) written += await f.write(bytes.subarray(written));
  }

  close() {
    for (const f of this.files.values()) f.close();
    this.files.clear();
  }
}
