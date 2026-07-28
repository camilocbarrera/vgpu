/**
 * TEMPORARY compatibility bridge (T202-01 … T202-04).
 *
 * The core `Gpu` now lives in `kernel.ts`: device, `disposed`, `onError`, `settled`, `dispose`.
 * Everything below re-attaches the 0.1.x methods to that object so tests, examples and docs keep
 * compiling while the free-function families land (T202-02 render, T202-03 non-render) and the
 * first-party consumers migrate (T202-04).
 *
 * MUST BE DELETED IN T202-05, together with `LegacyGpuMethods` and this file: after the clean cut
 * the public surface is `createGpu()` from `kernel.ts` plus free functions. Nothing new should be
 * added here, and no code outside this file should rely on these methods.
 *
 * The bridge is already lazy: no cache, frame runner, query ring or surface exists until the
 * corresponding method is called, so the lifetime/ownership semantics are the final ones.
 */
import type { ShaderSource } from "@vgpu/wgsl";
import { bundle, type Bundle, type BundleOptions, type BundleRecorder } from "./bundle.ts";
import { draw, type Draw, type DrawOptions } from "./draw.ts";
import { frame, frameLoop, type Frame, type FrameLoopCallback, type FrameLoopHandle, type FrameLoopOptions } from "./frame.ts";
import { effect, type Effect, type EffectOptions } from "./effect.ts";
import { geometry as geometryFactory, Geometry, type GeometryOptions } from "./scene/geometry-descriptor.ts";
import type { SceneGeometry } from "./scene/geometry.ts";
import type { Target, TargetOptions, TargetTextureOptions } from "./target.ts";
import { target as createTarget } from "./target-offscreen.ts";
import { compute as computeFactory } from "./compute.ts";
import { storage as storageFactory } from "./storage.ts";
import { pingPong as pingPongFactory, pingPongStorage as pingPongStorageFactory } from "./ping-pong.ts";
import { toWgsl } from "./shader-source.ts";
import { uniforms as uniformsFactory } from "./uniforms.ts";
import { surface, type Surface, type SurfaceCanvas, type SurfaceOptions } from "./surface.ts";
import { sampler } from "./sampler.ts";
import { unsupportedError } from "./errors.ts";
import { timer as timerFactory, type Timer } from "./timer.ts";
import { visibility as visibilityFactory, type Visibility, type VisibilityOptions } from "./visibility.ts";
import type { ClearColor } from "./target-utils.ts";
import type { Compute, ComputeOptions, PingPongStorage, PingPongTargets, SharedUniforms, StorageAccess, StorageBuffer, StorageOptions } from "./api-types.ts";
import { frameState } from "./frame-state.ts";
import { renderService } from "./render-service.ts";
import { createCoreGpu, kernelOf, type AdapterFactory, type EntryKind, type Gpu as CoreGpu, type InitOptions, type Kernel } from "./kernel.ts";

export type { AdapterFactory, InitOptions } from "./kernel.ts";
export type { Compute, ComputeOptions, DispatchOptions, GpuErrorListener, PingPongStorage, PingPongTargets, SharedUniforms, StorageAccess, StorageBuffer, StorageOptions } from "./api-types.ts";

/**
 * @deprecated TEMPORARY — the callable shape of `gpu.frame`. Replaced by the free functions
 * `frame(gpu, cb)` and `frameLoop(gpu, cb, opts)`; deleted with the rest of the bridge in T202-05.
 */
export interface LegacyFrameRunner {
  (cb?: (frame: Frame) => void): Frame;
  frame(cb?: (frame: Frame) => void): Frame;
  loop(cb: FrameLoopCallback, opts?: FrameLoopOptions): FrameLoopHandle;
}

/**
 * @deprecated TEMPORARY — removed in T202-05. Each member becomes a free function that takes the
 * `Gpu` as first argument (`draw(gpu, opts)`, `frame(gpu, cb)`, `createSurface(gpu, canvas)`, ...).
 */
export interface LegacyGpuMethods {
  time: number;
  deltaTime: number;
  frameCount: number;
  clearColor: ClearColor;
  surface(canvas: SurfaceCanvas, opts?: SurfaceOptions): Surface;
  effect(source: string | ShaderSource, opts?: EffectOptions): Effect;
  draw(opts: DrawOptions): Draw;
  target(opts: TargetOptions): Target;
  readonly frame: LegacyFrameRunner;
  sampler(desc?: GPUSamplerDescriptor): GPUSampler;
  geometry(geometry: SceneGeometry): Geometry;
  geometry(options: GeometryOptions): Geometry;
  compute(source: string | ShaderSource, opts?: ComputeOptions): Compute;
  storage(bytes: number, access?: StorageAccess | StorageOptions): StorageBuffer;
  /** GPU pass timing. Needs the "timestamp-query" device feature — request it at init: init({ requiredFeatures: ["timestamp-query"] }). */
  timer(): Timer;
  /** Occlusion query results for visibility culling. Core WebGPU — no device feature required. Pass the instance as FramePassOptions.visibility and wrap proxy draws in pass.occlusion(handle, body). */
  visibility(options?: VisibilityOptions): Visibility;
  pingPong(width: number, height: number, opts?: TargetTextureOptions): PingPongTargets;
  pingPongStorage(bytes: number): PingPongStorage;
  uniforms<T extends Record<string, unknown>>(values: T): SharedUniforms<T>;
  bundle(opts: BundleOptions, cb: (recorder: BundleRecorder) => void): Bundle;
}

/** Ring-1 facade shared by browser, node, and mock entrypoints. Shrinks to {@link CoreGpu} in T202-05. */
export interface Gpu extends CoreGpu, LegacyGpuMethods {}

export async function createGpu(entry: EntryKind, opts: InitOptions = {}, adapterFactory?: AdapterFactory): Promise<Gpu> {
  return installLegacyBridge(await createCoreGpu(entry, opts, adapterFactory));
}

/** @deprecated TEMPORARY — see the file header. Deleted in T202-05. */
function installLegacyBridge(core: CoreGpu): Gpu {
  const kernel = kernelOf(core);
  let runner: LegacyGpuMethods["frame"] | undefined;

  const methods = {
    surface(canvas: SurfaceCanvas, opts: SurfaceOptions = {}): Surface { return surface(core, canvas, opts); },
    effect(source: string | ShaderSource, opts: EffectOptions = {}): Effect {
      // The free function reports itself as `effect`; keep the 0.1.x wording for the method.
      if (hasGeometry(opts)) throw unsupportedError("gpu.effect", "gpu.effect() never accepts vertex buffers; use gpu.draw({ shader, geometry: gpu.geometry(descriptor) }).");
      return effect(core, source, opts);
    },
    draw(opts: DrawOptions): Draw { return draw(core, opts); },
    target(opts: TargetOptions): Target { return createTarget(core, opts); },
    sampler(desc?: GPUSamplerDescriptor): GPUSampler { return sampler(core, desc); },
    geometry(input: SceneGeometry | GeometryOptions): Geometry { return geometryFactory(core, input); },
    compute(source: string | ShaderSource, opts: ComputeOptions = {}): Compute { return computeFactory(core, source, opts); },
    storage(bytes: number, access: StorageAccess | StorageOptions = "read-write"): StorageBuffer { return storageFactory(core, bytes, access); },
    timer(): Timer { return timerFactory(core); },
    visibility(options: VisibilityOptions = {}): Visibility { return visibilityFactory(core, options); },
    pingPong(width: number, height: number, opts: TargetTextureOptions = {}): PingPongTargets { return pingPongFactory(core, width, height, opts); },
    pingPongStorage(bytes: number): PingPongStorage { return pingPongStorageFactory(core, bytes); },
    uniforms<T extends Record<string, unknown>>(values: T): SharedUniforms<T> { return uniformsFactory(core, values); },
    bundle(opts: BundleOptions, cb: (recorder: BundleRecorder) => void): Bundle { return bundle(core, opts, cb); },
  };

  Object.defineProperties(core, {
    time: { get: () => frameState(kernel).time, set: (value: number) => { frameState(kernel).time = value; }, enumerable: true, configurable: true },
    deltaTime: { get: () => frameState(kernel).deltaTime, set: (value: number) => { frameState(kernel).deltaTime = value; }, enumerable: true, configurable: true },
    frameCount: { get: () => frameState(kernel).frameCount, set: (value: number) => { frameState(kernel).frameCount = value; }, enumerable: true, configurable: true },
    clearColor: { get: () => frameState(kernel).clearColor, set: (value: ClearColor) => { frameState(kernel).clearColor = value; }, enumerable: true, configurable: true },
    frame: { get: () => (runner ??= legacyFrameRunner(core)), enumerable: true, configurable: true },
  });
  return Object.assign(core, methods) as Gpu;
}

/**
 * @deprecated TEMPORARY — `gpu.frame(cb)` is `frame(gpu, cb)` and `gpu.frame.loop(cb)` is
 * `frameLoop(gpu, cb)`. Callable object over the free functions: the runner itself (clock,
 * reentrancy guard, loop registrations) lives in the kernel, so both spellings drive the same one.
 */
function legacyFrameRunner(core: CoreGpu): LegacyGpuMethods["frame"] {
  const callable = ((cb?: (f: Frame) => void) => frame(core, cb)) as LegacyGpuMethods["frame"];
  callable.frame = (cb?: (f: Frame) => void) => frame(core, cb);
  callable.loop = (cb: FrameLoopCallback, opts?: FrameLoopOptions) => frameLoop(core, cb, opts);
  return callable;
}

function hasGeometry(opts: EffectOptions): boolean {
  return "geometry" in (opts as Record<string, unknown>);
}
