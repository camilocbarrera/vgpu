/**
 * Shared public types with no runtime weight and no dependency on the `Gpu` facade.
 *
 * They live outside `gpu.ts` so feature modules (compute, storage, ping-pong, uniforms, ...)
 * can be imported — and tree-shaken — without pulling the object that owns every factory.
 */
import type { ShaderSource } from "@vgpu/wgsl";
import type { VGPUError } from "./errors.ts";
import type { Target } from "./target.ts";

export interface ComputeOptions {
  readonly label?: string;
  readonly set?: Record<string, unknown>;
  /**
   * Initial values for instance-owned bindings, keyed by WGSL binding name. Declaring a binding here
   * pins it value-owned at construction: its storage is created here and only `.set()` writes it.
   */
  readonly values?: Record<string, unknown>;
  /**
   * Externally-owned resources, keyed by WGSL binding name. A binding declared here is external from
   * construction — `.set()` on it throws VGPU-R1-EXTERNAL-BINDING and `.bind()` swaps its identity.
   */
  readonly bindings?: Record<string, unknown>;
  /** Values for WGSL `override` constants, keyed by name (or by numeric id as a string when the override has @id). Immutable after construction. */
  readonly constants?: Readonly<Record<string, number | boolean>>;
  /** Compute entry point to use when the shader has several. Defaults to the first @compute entry point. */
  readonly entry?: string;
  /** Shader source, only used by the single-argument `compute(gpu, { shader, ... })` form; the two-argument form passes it separately. */
  readonly shader?: string | ShaderSource;
}
export interface DispatchOptions {
  /** GPU-driven dispatch: read the workgroup counts from a buffer instead of CPU-side counts. */
  readonly indirect: StorageBuffer | { readonly buffer: StorageBuffer; readonly offset?: number };
}
export interface Compute {
  /** Binding-scoped write: names a complete instance-owned binding and writes its bytes. */
  set(binding: string, value: unknown): this;
  // Overload order is public API surface: `Parameters<Compute["set"]>` resolves to the LAST overload,
  // so the legacy flat bag stays last and every type derived from it keeps its current shape.
  set(values: Record<string, unknown>): this;
  /** Identity swap of an externally-owned binding. Dedupes by identity; rebuilds exactly that group. */
  bind(binding: string, resource: unknown): this;
  dispatch(x: number, y?: number, z?: number): void;
  dispatch(opts: DispatchOptions): void;
  dispatchOnce(x: number, y?: number, z?: number): Promise<void>;
  dispatchOnce(opts: DispatchOptions): Promise<void>;
}
export type StorageAccess = "read" | "read-write";
export interface StorageOptions {
  /** Binding access for shader reflection. Defaults to "read-write". */
  readonly access?: StorageAccess;
  /** Adds the "indirect" buffer usage so the buffer can supply GPU-read draw/dispatch arguments. Defaults to false. */
  readonly indirect?: boolean;
}
export interface StorageBuffer {
  readonly size: number;
  readonly access: StorageAccess;
  /**
   * The underlying `GPUBuffer` — the single `.gpu` accessor spelling every vgpu object uses
   * (`gpu.gpu` is the `GPUDevice`, `Texture.gpu` a `GPUTexture`, `prepared.gpu` the pipeline). It is
   * what makes this buffer usable with `f.copyBuffer()` and inside an `f.raw()` block, and with raw
   * WebGPU calls vgpu does not wrap.
   *
   * It is a real handle, so `destroy()` is reachable through it: the same trade-off `Buffer.gpu` and
   * `Texture.gpu` already accept. Destroying it by hand leaves the owning `StorageBuffer` pointing at
   * a destroyed buffer — let `gpu.dispose()` own the lifetime instead.
   */
  readonly gpu: GPUBuffer;
  read(): Promise<ArrayBuffer>;
  write(data: BufferSource): void;
}
export interface PingPongTargets { readonly read: Target; readonly write: Target; swap(): void }
export interface PingPongStorage { readonly read: StorageBuffer; readonly write: StorageBuffer; swap(): void }
export interface SharedUniforms<T extends Record<string, unknown> = Record<string, unknown>> { set(values: Partial<T>): void }
export type GpuErrorListener = (error: VGPUError) => void;
