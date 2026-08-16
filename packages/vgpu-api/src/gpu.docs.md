# Gpu

The main API (`vgpu`) context returned by `init()`. It owns device lifetime and the frame clock; every resource — canvas surfaces, offscreen targets, render, compute, storage, uniforms, samplers, and bundles — is created by a free function that takes the `Gpu` as its first argument.

## Import

```ts
import type { Gpu } from "vgpu";
import { init } from "vgpu/mock";
```

## Signature

```ts
import type { Bundle, BundleOptions, BundleRecorder, Clock, Compute, ComputeOptions, Draw, DrawOptions, Effect, EffectOptions, Frame, FrameLoopHandle, FrameLoopOptions, FrameOptions, Geometry, GeometryOptions, GeometryRecipe, GpuErrorListener, PingPongStorage, PingPongTargets, PrepareRequest, RenderDestination, RenderOncePass, SharedUniforms, StorageAccess, StorageBuffer, StorageOptions, Surface, SurfaceOptions, Target, TargetOptions, TargetTextureOptions, Timer, Visibility, VisibilityOptions } from "vgpu";
import type { Device } from "vgpu/core";
import type { ShaderSource } from "vgpu";

interface Gpu {
  readonly device: Device;
  readonly gpu: GPUDevice;
  /** True once `dispose()` ran. Reads stay legal; new work does not. */
  readonly disposed: boolean;
  /** Resolves on REAL device loss only — never after an intentional `dispose()`. Terminal signal. */
  readonly lost: Promise<GPUDeviceLostInfo | undefined>;
  dispose(): void;
  onError(cb: GpuErrorListener): () => void;
  settled(): Promise<void>;
}

// The creation API: named exports of `vgpu`, `vgpu/node` and `vgpu/mock`, all gpu-first.
// Every shader factory takes exactly ONE positional input after `gpu`.
declare function surface(gpu: Gpu, canvas: HTMLCanvasElement | OffscreenCanvas, opts?: SurfaceOptions): Surface;
declare function effect(gpu: Gpu, input: string | ShaderSource | EffectOptions): Effect;
declare function draw(gpu: Gpu, input: DrawOptions): Draw;
declare function target(gpu: Gpu, opts: TargetOptions): Target;
declare function frame(gpu: Gpu, cb?: (frame: Frame) => void, opts?: FrameOptions): Frame;
declare function frameLoop(gpu: Gpu, cb: (frame: Frame) => void, opts?: FrameLoopOptions): FrameLoopHandle;
declare function prepare(gpu: Gpu, requests: PrepareRequest | readonly PrepareRequest[]): Promise<unknown>;
declare function renderOnce(gpu: Gpu, target: RenderDestination, body: (pass: RenderOncePass) => void): Promise<void>;
declare function sampler(gpu: Gpu, desc?: GPUSamplerDescriptor): GPUSampler;
declare function geometry(gpu: Gpu, input: GeometryOptions | GeometryRecipe): Geometry;
declare function compute(gpu: Gpu, input: string | ShaderSource | ComputeOptions): Compute;
declare function storage(gpu: Gpu, bytes: number, access?: StorageAccess | StorageOptions): StorageBuffer;
declare function timer(gpu: Gpu): Timer;
declare function visibility(gpu: Gpu, options?: VisibilityOptions): Visibility;
declare function pingPong(gpu: Gpu, width: number, height: number, opts?: TargetTextureOptions): PingPongTargets;
declare function pingPongStorage(gpu: Gpu, bytes: number): PingPongStorage;
declare function uniform<T extends Record<string, unknown>>(gpu: Gpu, values: T): SharedUniforms<T>;
declare function bundle(gpu: Gpu, opts: BundleOptions, record: (recorder: BundleRecorder) => void): Bundle;
declare function clock(gpu: Gpu): Clock;
```

## Parameters

`Gpu` is an object, not a callable constructor: it carries no creation methods. Every factory below takes it as `gpu`, its first argument.

| Param | Type | Required | Default | Notes |
|---|---|---:|---|---|
| surface.canvas | `HTMLCanvasElement \| OffscreenCanvas` | ✔ | — | Canvas-like object with a `webgpu` context. A canvas may have one live `Surface`. |
| surface.opts | `SurfaceOptions` | ✖ | `{}` | Per-surface canvas format, size, DPR, and auto-resize behavior. |
| effect.input | `string \| ShaderSource \| EffectOptions` | ✔ | — | One positional input: a bare WGSL string / `ShaderSource` (shorthand for `{ shader }`), or the full `EffectOptions` object. Compose a module-exported options object with app resources by spread: `effect(gpu, { ...bloom, bindings: { globals } })`. |
| draw.opts | `DrawOptions` | ✔ | — | Includes required `shader`; see `DrawOptions`. |
| target.opts | `TargetOptions` | ✔ | — | Offscreen target options. `size` is required. |
| frame.cb | `(frame: Frame) => void` | ✖ | `undefined` | If provided, submits automatically in `finally`; if omitted, caller must call `frame.submit()`. |
| sampler.desc | `GPUSamplerDescriptor` | ✖ | `undefined` | Cached by descriptor. `sampler(gpu)` is the canonical default sampler. |
| geometry.input | `GeometryOptions \\| GeometryRecipe` | ✔ | — | A raw buffer descriptor, or a `vgpu/scene` recipe such as `box()` or `plane()`. |
| compute.input | `string \| ShaderSource \| ComputeOptions` | ✔ | — | One positional input: a bare WGSL string / `ShaderSource` (shorthand for `{ shader }`) or the full `ComputeOptions` object. Must contain a `@compute` entry point. `compute()` creates no pipeline at construction. |
| prepare.requests | `PrepareRequest \| readonly PrepareRequest[]` | ✔ | — | One combination per request: `{ draw, target }`, `{ compute }` or `{ bundle }`. The one spelling for readiness. See `prepare`. |
| renderOnce.target | `RenderDestination` | ✔ | — | Standalone render with its own encoder and exactly one submit; awaits async pipeline readiness. |
| storage.bytes | `number` | ✔ | — | Byte size for a main API (`vgpu`) storage buffer. |
| storage.access | `StorageAccess \| StorageOptions` | ✖ | `"read-write"` | Access string, or a `StorageOptions` bag `{ access?, indirect? }`. See `Compute` for storage buffer semantics, including `{ indirect: true }` for GPU-driven draw/dispatch arguments. |
| timer | — | — | — | No parameters. GPU pass timing; needs the `"timestamp-query"` device feature. See `Timer` for feature gating, spans, and result delivery. |
| visibility.options | `VisibilityOptions` | ✖ | `{}` | Occlusion queries for visibility culling — core WebGPU, no device feature required. See `Visibility` for capacity and handle semantics. |
| pingPong.width | `number` | ✔ | — | Floored and clamped to at least `1`. |
| pingPong.height | `number` | ✔ | — | Floored and clamped to at least `1`. |
| pingPong.opts | `TargetTextureOptions` | ✖ | `{}` | Texture/attachment options only; size comes from positional width/height. |
| pingPongStorage.bytes | `number` | ✔ | — | Creates two `"read-write"` storage buffers. |
| uniform.values | `Record<string, unknown>` | ✔ | — | One shared uniform resource, usable across pipelines. Storage is zero-initialized; `globals.set({ time })` is one write every bound pipeline observes. |
| bundle.opts | `BundleOptions` | ✔ | — | Requires a `target` or target signature. |
| bundle.cb | `(recorder: BundleRecorder) => void` | ✔ | — | Records bundle commands immediately. |
| onError.cb | `GpuErrorListener` | ✔ | — | Receives asynchronous vgpu errors; returns an unsubscribe function. |
| clock | — | — | — | No parameters. The frame clock of this gpu: `{ time, deltaTime, frameCount, advance(dtSeconds) }`, one instance per gpu. See `Clock`. |

**Returns:** each factory returns the resource named in its signature. `dispose()` and frame/pass callbacks return `void`.

**Throws:** `VGPU-GPU-DISPOSED` when any factory (or `clock(gpu)`) runs after `gpu.dispose()` — the device and everything it owned are gone, so the handle it would return could only fail later; create resources before disposing, or `init()` a new gpu; `VGPU-GPU-FOREIGN` when the first argument was not created by `init()` (a plain object, a `GPUDevice`, a gpu from another library): it carries no vgpu kernel, so pass the object returned by `init()` from `vgpu`, `vgpu/node` or `vgpu/mock`; `VGPU-LIMIT-STORAGE-VERTEX` / `VGPU-LIMIT-STORAGE-FRAGMENT` when a selected render entry exceeds its granted storage-buffer limit. The structured detail reports `stage`, `entryPoint`, `count`, `limit`, and each counted binding's `name`, `group`, and `binding`; request a supported limit or reduce/move the data; `VGPU-SHADER-SOURCE-INVALID` for malformed `ShaderSource`; `VGPU-SET-TEXTURE-FILTERABILITY` when a known facade texture format cannot satisfy an ordinarily sampled float binding (detail reports format, texture binding/name/label, and paired sampler identity); `VGPU-RING1-UNSUPPORTED` for unsupported effect/compute/target cases; `VGPU-PIPELINE-PENDING` when a synchronous encode meets a combination that was never prepared (the default `pendingPipelines: "throw"`) — `await prepare(gpu, [{ draw, target }])` first, or opt in to inline compilation with `pendingPipelines: "sync"`; `VGPU-PREPARE-FAILED` when `prepare()` rejects, enumerating every failed combination; `VGPU-DEVICE-LOST` for **any** operation on the object graph of a lost device (see [Device loss is terminal](#device-loss-is-terminal)); `VGPU-R1-EXTERNAL-BINDING` when `.set()` names an externally-bound binding (update the resource instead); `VGPU-SURFACE-NOT-BINDABLE` when a `Surface` is used as a texture binding; `VGPU-TARGET-REQUIRED` when one-shot drawing needs an explicit target; `VGPU-TARGET-SIZE-REQUIRED` for runtime JS calls to `target(gpu)` without `size`; `VGPU-SURFACE-*` errors from `surface()`, surface resize, surface readback, or using disposed surfaces; plus method-specific `VGPU-R1-*`, `VGPU-R3-*`, and `VGPU-R4-*` errors documented on `Effect`, `Draw`, `Compute`, `Frame`, `Bundle`, `Target`, and `SharedUniforms`.

## Examples

```ts
import { init, draw, frame, prepare, target } from "vgpu/mock";

const gpu = await init();
const colorTarget = target(gpu, { size: [128, 128], depth: true });
const drawable = draw(gpu, {
  shader: `
    @vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
      var p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
      return vec4f(p[vi], 0, 1);
    }
    @fragment fn fs_main() -> @location(0) vec4f { return vec4f(1, 0, 1, 1); }
  `,
});

// Readiness is a property of the (renderable, target signature) COMBINATION:
await prepare(gpu, [{ draw: drawable, target: colorTarget }]);

frame(gpu, (currentFrame) => {
  currentFrame.pass({ target: colorTarget, clear: [0, 0, 0, 1] }, (pass) => pass.draw(drawable));
});
```

```ts
import { init, effect, frameLoop, prepare, surface } from "vgpu";

declare const canvas: HTMLCanvasElement;

const gpu = await init();
const canvasSurface = surface(gpu, canvas, { dpr: [1, 2] });
const wave = effect(gpu, `@fragment fn fs_main() -> @location(0) vec4f { return vec4f(0.2, 0.4, 1.0, 1.0); }`);

await prepare(gpu, [{ draw: wave, target: canvasSurface }]);

frameLoop(gpu, (frame) => {
  frame.pass({ target: canvasSurface }, (pass) => pass.draw(wave));
});
```

## Error delivery

`gpu.onError(cb)` subscribes to asynchronous vgpu errors and returns an unsubscribe function. Listeners run in subscription order; removing one stops future deliveries; a throwing listener is reported to `console.error` without stopping the rest. If no listener is registered, vgpu reports the error to `console.error` by default.

`gpu.settled()` is a **snapshot**, and the only place GPU completion is observable. At the moment of
the call it snapshots everything vgpu has already started and resolves when all of it has finished:

1. `GPUQueue.onSubmittedWorkDone()` for every submission made **before** the call — unconditionally,
   not only when other bookkeeping happens to be in flight;
2. pipeline compilation already started (`prepare()`, `"skip"` background compiles);
3. readback mapping already started by vgpu (`target.read()`, timer/occlusion query rings);
4. pending `gpu.onError` deliveries, flushed before it resolves — so a caller may
   `await gpu.settled()` and then read the errors it collected.

Deliberately excluded: work submitted, compiled or requested **after** the call (the snapshot does
not extend), and any readback the application did not start. It **never rejects** — failures arrive
through `gpu.onError` — and calling it twice is meaningful: each call takes a new snapshot.

## Device loss is terminal

`gpu.lost` resolves **only on real device loss**, with the platform's `GPUDeviceLostInfo`, and it
resolves proactively — even if the application is idle. It is the terminal signal:

- Loss is terminal for the `Gpu` **and for every object created from it**. vgpu never restores,
  reattaches or re-points an existing `Surface`, `Target`, texture, buffer, uniform/storage
  resource, geometry, `Draw`/`Effect`/`Compute` instance, bind group, prepared pipeline or bundle
  onto a replacement device.
- On loss vgpu stops every frame loop it owns, marks the device-local object graph terminal, and
  from then on **every** operation on that graph throws `VGPU-DEVICE-LOST`: `fx.set()` / `fx.bind()`,
  `frame(gpu, cb)` / `frameLoop()`, `prepare()`, a bundle replay, `target.read()`.
- The application rebuilds: a new `Gpu`, new surfaces/targets, textures/buffers/uniforms, new
  instances and bind groups, and new prepared pipelines and bundles. What is reusable is
  **application data**, not vgpu objects — a GPU-independent options object can be re-instantiated
  as-is, while any options value holding a device-local object (`bindings: { input: oldTexture }`,
  a `values` entry carrying a `Target`) must be reconstructed. The canvas itself is reused: the new
  device reconfigures its context.
- An intentional `gpu.dispose()` is **not** a loss: `gpu.lost` never resolves after it, even though
  the native `GPUDevice.lost` settles with reason `"destroyed"`. That keeps the canonical recovery
  pattern from firing a rebuild during teardown. Intentional disposal is observed synchronously
  through `gpu.disposed`. There is deliberately no `onLost()` callback — one spelling per semantic.

```ts
import { init, effect, prepare, surface } from "vgpu";
import type { EffectOptions } from "vgpu";

declare const canvas: HTMLCanvasElement;
declare const bloom: EffectOptions & { readonly shader: string };

async function createGpuState(target: HTMLCanvasElement) {
  const gpu = await init();
  const screen = surface(gpu, target, { depth: true, sampleCount: 4 }); // same canvas, new device
  const fx = effect(gpu, bloom);                    // GPU-independent module options: reused as-is
  await prepare(gpu, [{ draw: fx, target: screen }]);
  return { gpu, screen, fx };
}

let state = await createGpuState(canvas);
await state.gpu.lost;                   // proactive: resolves even if the app is idle
state = await createGpuState(canvas);   // rebuild everything device-local
```

## Notes

- There is no implicit screen property and no implicit default target. Pass `target` explicitly to frame passes and one-shot draws.
- Canvas-specific `size`, `dpr`, and `autoResize` live on `surface(gpu, canvas, opts)`, not on `init()`; `depth` and `sampleCount` are surface options too.
- `gpu.gpu` is the `GPUDevice`, and `.gpu` is the one spelling for a raw handle everywhere: `Texture.gpu`, `Buffer.gpu`, `prepared.gpu` (the pipeline or bundle a `prepare()` handle carries). There is no second `.raw` spelling and no per-object `compile()` / `pipelineFor()`.
- Time is explicit JS state, and it lives on the clock, not on the context: read `clock(gpu).time` / `.deltaTime` / `.frameCount` and pass them through `.set()` or a shared `uniform(gpu, ...)` when shaders need them.
- Every factory rejects a disposed gpu with `VGPU-GPU-DISPOSED`, and an object vgpu did not create with `VGPU-GPU-FOREIGN`. Both are thrown synchronously, from the call that made the mistake.
- **See also:** `init`, `Clock`, `Surface`, `Effect`, `Draw`, `Compute`, `Frame`, `Target`, `Bundle`, `SharedUniforms`, `Timer`, `Visibility`.

## Sampled float texture layouts

vgpu infers sampled-texture layouts per selected WGSL entry point. A non-multisampled `texture_*<f32>` used by `textureSample*` or `textureGather*` with an ordinary sampler receives WebGPU `sampleType: "float"`; a texture used only by `textureLoad` remains `"unfilterable-float"`. Calls through helper functions are included.

The WGSL `f32` scalar type does not make every concrete texture format filterable. In particular, `r32float`, `rg32float`, and `rgba32float` require the device's `float32-filterable` feature for ordinary sampling. Use a filterable format, request that feature when supported, or use `textureLoad` without a sampler.
