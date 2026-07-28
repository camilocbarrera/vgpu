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
import { createBundle, type Bundle, type BundleOptions, type BundleRecorder } from "./bundle.ts";
import { InternalDraw, type Draw, type DrawOptions } from "./draw.ts";
import { Frame, FrameRunner } from "./frame.ts";
import { InternalEffect, type Effect, type EffectOptions } from "./effect.ts";
import { createGeometry } from "./scene/geometry-factory.ts";
import { Geometry, type GeometryOptions } from "./scene/geometry-descriptor.ts";
import type { SceneGeometry } from "./scene/geometry.ts";
import { OffscreenTarget, type Target, type TargetOptions, type TargetTextureOptions } from "./target.ts";
import { surfaceDuplicateError, unsupportedError } from "./errors.ts";
import { ComputePipeline } from "./compute.ts";
import { createStorageBuffer } from "./storage.ts";
import { createPingPongStorage, createPingPongTargets } from "./ping-pong.ts";
import { toWgsl } from "./shader-source.ts";
import { createSharedUniforms } from "./uniforms.ts";
import { CanvasSurface, type Surface, type SurfaceCanvas, type SurfaceOptions } from "./surface.ts";
import { createTimer, type Timer } from "./timer.ts";
import { createVisibility, type Visibility, type VisibilityOptions } from "./visibility.ts";
import type { ClearColor } from "./target-utils.ts";
import type { Compute, ComputeOptions, PingPongStorage, PingPongTargets, SharedUniforms, StorageAccess, StorageBuffer, StorageOptions } from "./api-types.ts";
import { frameState } from "./frame-state.ts";
import { renderService } from "./render-service.ts";
import { createCoreGpu, kernelOf, type AdapterFactory, type EntryKind, type Gpu as CoreGpu, type InitOptions, type Kernel } from "./kernel.ts";

export type { AdapterFactory, InitOptions } from "./kernel.ts";
export type { Compute, ComputeOptions, DispatchOptions, GpuErrorListener, PingPongStorage, PingPongTargets, SharedUniforms, StorageAccess, StorageBuffer, StorageOptions } from "./api-types.ts";

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
  readonly frame: FrameRunner & ((cb?: (frame: Frame) => void) => Frame);
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
  const device = kernel.device;
  let runner: LegacyGpuMethods["frame"] | undefined;
  let surfaces: Map<SurfaceCanvas, CanvasSurface> | undefined;
  const errorSink = (error: Parameters<Kernel["reportError"]>[0]) => kernel.reportError(error);
  const trackSettled = (promise: Promise<unknown>) => kernel.trackDelivery(promise);

  const methods = {
    surface(canvas: SurfaceCanvas, opts: SurfaceOptions = {}): Surface {
      surfaces ??= new Map<SurfaceCanvas, CanvasSurface>();
      const open = surfaces;
      const existing = open.get(canvas);
      if (existing && !existing.disposed) throw surfaceDuplicateError(existing.label);
      const surface = new CanvasSurface(device, canvas, opts, (s) => {
        if (open.get(s.canvas) === s) open.delete(s.canvas);
        releaseAutoResize();
        releaseOwnership();
      });
      // Surfaces resize themselves right after the frame clock advances, and go down with the gpu
      // before its caches and its device.
      const releaseAutoResize = frameState(kernel).onAdvance(() => surface.applyAutoResize());
      const releaseOwnership = kernel.own("resource", () => surface.dispose());
      open.set(canvas, surface);
      return surface;
    },
    effect(source: string | ShaderSource, opts: EffectOptions = {}): Effect {
      if (hasGeometry(opts)) throw unsupportedError("gpu.effect", "gpu.effect() never accepts vertex buffers; use gpu.draw({ shader, geometry: gpu.geometry(descriptor) }).");
      const render = renderService(kernel);
      return new InternalEffect(device, toWgsl(source), opts, render.binds, undefined, render.pipelines, render.shaderModules, render.pipelineLayouts, errorSink, trackSettled);
    },
    draw(opts: DrawOptions): Draw {
      const shader = toWgsl(opts.shader);
      const render = renderService(kernel);
      return new InternalDraw(device, shader, { ...opts, shader }, render.binds, undefined, render.pipelines, render.shaderModules, render.pipelineLayouts, errorSink, trackSettled);
    },
    target(opts: TargetOptions): Target { return new OffscreenTarget(device, opts); },
    sampler(desc?: GPUSamplerDescriptor): GPUSampler { return renderService(kernel).sampler(desc); },
    geometry(input: SceneGeometry | GeometryOptions): Geometry {
      return isGeometryOptions(input) ? new Geometry(device, input) : createGeometry(device, input);
    },
    compute(source: string | ShaderSource, opts: ComputeOptions = {}): Compute { return new ComputePipeline(device, toWgsl(source), opts, renderService(kernel).binds); },
    storage(bytes: number, access: StorageAccess | StorageOptions = "read-write"): StorageBuffer {
      const opts = typeof access === "string" ? { access } : access;
      return createStorageBuffer(device, bytes, opts.access ?? "read-write", undefined, opts.indirect ?? false);
    },
    timer(): Timer {
      let release = (): void => undefined;
      const timer: Timer = createTimer(device, {
        trackSettled,
        errorSink,
        onDispose: () => { release(); },
      });
      release = kernel.own("resource", () => timer.dispose());
      return timer;
    },
    visibility(options: VisibilityOptions = {}): Visibility {
      let release = (): void => undefined;
      const visibility: Visibility = createVisibility(device, options, () => frameState(kernel).frameCount, {
        trackSettled,
        errorSink,
        onDispose: () => { release(); },
      });
      release = kernel.own("resource", () => visibility.dispose());
      return visibility;
    },
    pingPong(width: number, height: number, opts: TargetTextureOptions = {}): PingPongTargets { return createPingPongTargets(device, width, height, opts); },
    pingPongStorage(bytes: number): PingPongStorage { return createPingPongStorage(device, bytes); },
    uniforms<T extends Record<string, unknown>>(values: T): SharedUniforms<T> { return createSharedUniforms(device, values); },
    bundle(opts: BundleOptions, cb: (recorder: BundleRecorder) => void): Bundle { return createBundle(device, opts, cb); },
  };

  Object.defineProperties(core, {
    time: { get: () => frameState(kernel).time, set: (value: number) => { frameState(kernel).time = value; }, enumerable: true, configurable: true },
    deltaTime: { get: () => frameState(kernel).deltaTime, set: (value: number) => { frameState(kernel).deltaTime = value; }, enumerable: true, configurable: true },
    frameCount: { get: () => frameState(kernel).frameCount, set: (value: number) => { frameState(kernel).frameCount = value; }, enumerable: true, configurable: true },
    clearColor: { get: () => frameState(kernel).clearColor, set: (value: ClearColor) => { frameState(kernel).clearColor = value; }, enumerable: true, configurable: true },
    frame: { get: () => (runner ??= createFrameRunner(kernel)), enumerable: true, configurable: true },
  });
  return Object.assign(core, methods) as Gpu;
}

/**
 * @deprecated TEMPORARY — moves to `frame.ts` as `frame(gpu, cb)` / `frameLoop(gpu, cb)` in T202-02.
 * Loops register in the kernel's `scheduler` phase so `dispose()` stops them before touching the device.
 */
function createFrameRunner(kernel: Kernel): LegacyGpuMethods["frame"] {
  const state = frameState(kernel);
  const runner = new FrameRunner(
    () => new Frame(kernel.device, undefined, (error) => kernel.reportError(error), (promise) => kernel.trackDelivery(promise), () => state.clearColor),
    () => state.advance(),
    (handle) => kernel.own("scheduler", () => handle.stop()),
  );
  const callable = ((cb?: (frame: Frame) => void) => runner.frame(cb)) as LegacyGpuMethods["frame"];
  Object.setPrototypeOf(callable, FrameRunner.prototype);
  Object.assign(callable, runner);
  callable.frame = runner.frame.bind(runner);
  callable.loop = runner.loop.bind(runner);
  return callable;
}

function hasGeometry(opts: EffectOptions): boolean {
  return "geometry" in (opts as Record<string, unknown>);
}

function isGeometryOptions(value: SceneGeometry | GeometryOptions): value is GeometryOptions {
  return typeof value === "object" && value !== null && "buffers" in value;
}
