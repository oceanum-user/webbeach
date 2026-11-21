// SPDX-License-Identifier: LGPL-3.0-or-later
// XBeach WebGPU solver — derived from XBeach (c) Deltares (LGPL v2.1+).
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * Thin WebGPU helpers: device acquisition, storage buffers, pipeline cache.
 * Pure WebGPU API — runs in Deno and in the browser.
 */

export class Gpu {
  device!: GPUDevice;
  adapterInfo = "";
  layout?: GPUPipelineLayout;
  bglLayout?: GPUBindGroupLayout;
  private pipelines = new Map<string, GPUComputePipeline>();

  /** Shared explicit layout: binding 0 uniform, 1-8 storage (1,3 read-only). */
  makeSharedLayout() {
    const entries: GPUBindGroupLayoutEntry[] = [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ];
    for (let b = 1; b <= 9; b++) {
      entries.push({
        binding: b,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: b === 1 || b === 3 || b === 9 ? "read-only-storage" : "storage" },
      });
    }
    this.bglLayout = this.device.createBindGroupLayout({ entries });
    this.layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.bglLayout] });
  }

  static async init(): Promise<Gpu> {
    const g = new Gpu();
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("WebGPU adapter not available");
    g.adapterInfo = `${adapter.info?.vendor ?? "?"} ${adapter.info?.description ?? ""}`.trim();
    g.device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBuffersPerShaderStage: Math.min(
          10,
          adapter.limits.maxStorageBuffersPerShaderStage,
        ),
        maxComputeInvocationsPerWorkgroup: 256,
        maxStorageBufferBindingSize: Math.min(
          512 * 1024 * 1024,
          adapter.limits.maxStorageBufferBindingSize,
        ),
        // the s2 field pack is the largest single buffer (~352 B/node); the
        // default 256 MiB maxBufferSize would cap runs at ~760k nodes even
        // though the binding size above allows twice that
        maxBufferSize: Math.min(
          512 * 1024 * 1024,
          adapter.limits.maxBufferSize,
        ),
      },
    });
    return g;
  }

  buffer(name: string, byteLength: number, usage?: GPUBufferUsageFlags): GPUBuffer {
    return this.device.createBuffer({
      label: name,
      size: Math.max(byteLength, 4),
      usage: usage ??
        (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST),
    });
  }

  upload(buf: GPUBuffer, data: Float32Array | Uint32Array | Int32Array, offset = 0) {
    this.device.queue.writeBuffer(buf, offset, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
  }

  uploadF64(buf: GPUBuffer, data: Float64Array) {
    this.upload(buf, new Float32Array(data));
  }

  /** Compile (with cache) a compute pipeline from WGSL source. */
  pipeline(key: string, code: string, entryPoint = "main"): GPUComputePipeline {
    let p = this.pipelines.get(key);
    if (!p) {
      const module = this.device.createShaderModule({ label: key, code });
      p = this.device.createComputePipeline({
        label: key,
        layout: this.layout ?? "auto",
        compute: { module, entryPoint },
      });
      this.pipelines.set(key, p);
    }
    return p;
  }

  bind(pipeline: GPUComputePipeline, buffers: GPUBuffer[], group = 0): GPUBindGroup {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(group),
      entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
  }

  async read(buf: GPUBuffer, byteLength = buf.size): Promise<Float32Array> {
    const staging = this.device.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, staging, 0, byteLength);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.destroy();
    return out;
  }

  async readU32(buf: GPUBuffer, byteLength = buf.size): Promise<Uint32Array> {
    const f = await this.read(buf, byteLength);
    return new Uint32Array(f.buffer);
  }
}

/** Dispatch geometry for n threads with the given workgroup size.
 * WebGPU caps each dispatch dimension at 65535 workgroups, so a large 1-D
 * grid is folded into 2-D: [gx, gy] with gx*gy*wg >= n. Every kernel entry
 * linearises gid back with num_workgroups (see shader preambles) — a run of
 * ~350k+ nodes used to die here with "dispatch group size must be <= 65535".
 */
export const MAX_GROUPS_PER_DIM = 65535;
export function groups(n: number, wg = 64): [number, number] {
  const g = Math.ceil(n / wg);
  const gy = Math.ceil(g / MAX_GROUPS_PER_DIM);
  return [Math.ceil(g / gy), gy];
}
