# Effect

Fullscreen-fragment render unit created by `effect(gpu)`. Use it for post-processing, gradients, blurs, and screen/target copies; use `draw(gpu)` for meshes, vertex buffers, instancing, or explicit vertex counts.

## Import

```ts
import type { Effect, EffectOptions } from "vgpu";
```

## Signature

```ts
import type { CompileTarget, DrawCallOptions, Gpu, ShaderSource } from "vgpu";

type BlendPreset = "alpha" | "additive" | "premultiplied";
interface BlendComponentOptions { readonly src: GPUBlendFactor; readonly dst: GPUBlendFactor; readonly op?: GPUBlendOperation; }
interface BlendOptions { readonly color: BlendComponentOptions; readonly alpha?: BlendComponentOptions; }

interface EffectOptions {
  readonly shader?: string | ShaderSource;
  /** Instance-owned bindings: storage created here, written with `.set()`. */
  readonly values?: Record<string, unknown>;
  /** Externally-owned bindings: identity swapped with `.bind()`. */
  readonly bindings?: Record<string, unknown>;
  readonly label?: string;
  readonly blend?: BlendPreset | BlendOptions;
  readonly writeMask?: readonly ("r" | "g" | "b" | "a")[];
}

interface Effect {
  /** The `GPURenderPipeline` of the last prepared combination, or `undefined` while none is. */
  readonly gpu: GPURenderPipeline | undefined;
  /** Binding-scoped byte write on an instance-owned binding. Never rebuilds a bind group. */
  set(binding: string, value: unknown): this;
  /** Identity swap of an externally-owned binding. Dedupes by identity; rebuilds exactly that group. */
  bind(binding: string, resource: unknown): this;
}

// One positional input after `gpu`: a bare shader (shorthand for `{ shader }`) or the options object.
declare function effect(gpu: Gpu, input: string | ShaderSource | EffectOptions): Effect;
```

## Parameters

| Param | Type | Required | Default | Notes |
|---|---|---:|---|---|
| effect.input | `string \| ShaderSource \| EffectOptions` | ✔ | — | **One positional input.** A bare WGSL string or `ShaderSource` is shorthand for `{ shader }`. Compose a module-exported options object with app-side resources by plain object spread — `effect(gpu, { ...bloom, bindings: { globals } })` — never a second options argument. If no `@vertex` entry exists, vgpu injects a fullscreen triangle vertex stage and provides `@location(0) uv`. |
| input.shader | `string \| ShaderSource` | ✔ | — | WGSL string or loader-produced `ShaderSource`. The field is spelled `shader` in all three shader factories (`draw`, `effect`, `compute`); there is no `wgsl` / `code` / `source` spelling. |
| input.values | `Record<string, unknown>` | ✖ | `undefined` | Initial values of **instance-owned** bindings, keyed by WGSL binding name. Declaring a binding here pins it value-owned at construction and pins its type; storage is created here and zero-initialized. Only `.set()` writes it. |
| input.bindings | `Record<string, unknown>` | ✖ | `undefined` | **Externally-owned** resources (a shared `uniform()`, a `Target`, a `StorageBuffer`, a sampler), keyed by WGSL binding name. `.set()` on one of these throws `VGPU-R1-EXTERNAL-BINDING`; `.bind()` swaps its identity. |
| input.label | `string` | ✖ | `"effect"` | Used in shader reflection labels, GPU object labels, and `VGPU-*` error `where` fields. |
| input.blend | `"alpha" \| "additive" \| "premultiplied" \| BlendOptions` | ✖ | `undefined` | Constructor-only blend state passed through to the fullscreen draw. Presets and defaults match `DrawOptions.blend`; omitted explicit `alpha` copies `color`, and `op` defaults to `"add"`. |
| input.writeMask | `readonly ("r" \| "g" \| "b" \| "a")[]` | ✖ | all channels | Constructor-only color channel mask. Omit for RGBA; `[]` writes no channels; `["r","g","b"]` skips alpha. |
| effect.set.binding | `string` | ✔ | — | Names a **complete instance-owned WGSL binding**. Struct-typed bindings accept a partial (absent fields keep their last value, merged CPU-side into one struct rewrite); a non-struct binding (`mat4x4f`, `vec4f`, `array<f32,N>`) takes its complete value. |
| effect.bind.resource | `unknown` | ✔ | — | New identity for an externally-owned binding. Re-binding the same resource is free (deduped by identity); a `Surface` is rejected with `VGPU-SURFACE-NOT-BINDABLE`. |

The `uv` varying that `effect(gpu)` injects is top-origin: `(0, 0)` is the
top-left corner and `v` grows downward — the same convention as WebGPU texture
coordinates, `@builtin(position)`, and `target.read()`. Sampling any texture
with this `uv` needs no flip: a pass that samples `src` at `uv` reproduces the
image exactly. If you are porting a WebGL or Shadertoy shader that assumes
`v` grows upward, invert once at the boundary (`1.0 - uv.y`) and keep
everything else flip-free.

**Returns:** `effect(gpu, input)` returns `Effect`; `effect.set()` and `effect.bind()` return the same `Effect`.

**Throws:** `VGPU-PIPELINE-PENDING` when a synchronous encode meets a `(effect, target signature)` combination that was never prepared — the default `pendingPipelines: "throw"` never compiles implicitly, so `await prepare(gpu, [{ draw: fx, target }])` first (or opt in with `pendingPipelines: "sync"`); `VGPU-R1-EXTERNAL-BINDING` when `.set()` names a binding declared in `bindings` — update the resource itself (`globals.set({ time })`) or swap identity with `.bind()`; `VGPU-SURFACE-NOT-BINDABLE` when a `Surface` is passed as a binding (bind a `Target`, or read back with `surface.read()`); `VGPU-BLEND-INVALID` for an unknown blend preset or malformed blend object; `VGPU-WRITEMASK-INVALID` for a non-array or unknown write mask channel; `VGPU-RING1-UNSUPPORTED` when `effect(gpu)` receives mesh/vertex data; `VGPU-SHADER-SOURCE-INVALID` for malformed `ShaderSource`; `VGPU-R1-BINDING-NEVER-SET` when a reflected binding has no value at draw time; `VGPU-DEVICE-LOST` for every operation — including `.set()` / `.bind()` — after the device was lost; `VGPU-SET-TEXTURE-FILTERABILITY` when an ordinarily sampled facade texture is not filterable (structured detail names its format/binding and paired sampler; use a filterable format, request `float32-filterable`, or use `textureLoad` without a sampler). Asynchronous draw validation errors are delivered through `gpu.onError`; tests can `await gpu.settled()`.

## Examples

```ts
import { init, clock, effect, frame, prepare, target, uniform } from "vgpu/mock";

const gpu = await init();
const colorTarget = target(gpu, { size: [64, 64] });
const globals = uniform(gpu, { time: 0 });
const shader = effect(gpu, {
  label: "wave",
  shader: `
    struct Params { speed: f32 }
    struct Globals { time: f32 }
    @group(0) @binding(0) var<uniform> params: Params;
    @group(0) @binding(1) var<uniform> globals: Globals;

    @fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
      return vec4f(uv, sin(globals.time * params.speed) * 0.5 + 0.5, 1);
    }
  `,
  values: { params: { speed: 2 } },   // instance-owned → .set()
  bindings: { globals },              // external → update the resource
});

await prepare(gpu, [{ draw: shader, target: colorTarget }]);

shader.set("params", { speed: 3 });        // bytes only; bind groups untouched
globals.set({ time: clock(gpu).time });    // one write, every pipeline bound to it sees it
frame(gpu, (currentFrame) => currentFrame.pass(colorTarget, shader));
```

```ts
// A reusable module: options objects are plain typed data — no gpu, no wiring.
import type { EffectOptions } from "vgpu";

export const bloom = {
  shader: `@fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f { return vec4f(uv, 0, 1); }`,
  blend: "additive",
} satisfies EffectOptions;
```

```ts
import { init, effect, renderOnce, target, uniform } from "vgpu/mock";
import type { EffectOptions } from "vgpu";

declare const bloom: EffectOptions & { readonly shader: string };

const gpu = await init();
const screen = target(gpu, { size: [32, 32] });
const globals = uniform(gpu, { time: 0 });

const fx = effect(gpu, bloom);                                  // instantiate the module as-is
const hdr = effect(gpu, { ...bloom, bindings: { globals } });    // composition is object spread

// Standalone render: own encoder, exactly one submit, async pipeline readiness.
await renderOnce(gpu, screen, (p) => {
  p.draw(fx);
  p.draw(hdr);
});
```

## Readiness and ownership

**Readiness is a property of a combination, never of an object.** An `Effect` can be ready for the
screen, uncompiled for an HDR target and failed for a `depth24plus-stencil8` target at the same
time, so there is no `ready` / `pending` field, no per-target status map, and no per-object
`compile()` / `compileSync()` / `pipelineFor()`. `prepare()` is the query and the one spelling:

```ts
import { init, effect, prepare, target } from "vgpu/mock";

const gpu = await init();
const screen = target(gpu, { size: [64, 64] });
const fx = effect(gpu, `@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1); }`);

const prepared = await prepare(gpu, { draw: fx, target: screen });
prepared.draw;       // the renderable, echoed back — the handle identifies the combination
prepared.signature;  // resolved TargetSignature { colors, depth, sampleCount }
prepared.gpu;        // GPURenderPipeline — the only low-level pipeline escape hatch
```

**Ownership is fixed at construction**, and the two update paths never overlap:

| Update | Spelling | Cost | Effect on bundles |
|---|---|---|---|
| bytes of an instance-owned binding | `fx.set("params", { intensity })` | one buffer write | bundle stays `ready` |
| identity of an external binding | `fx.bind("input", nextTexture)` | rebuilds exactly that group | bundle goes `stale` |
| bytes of a shared resource | `globals.set({ time })` | one write, all pipelines | bundle stays `ready` |

A binding declared in the WGSL that appears in neither `values` nor `bindings` is **instance-owned
(value-owned) by default** — its storage is created at construction and zero-initialized — which is
why `fx.set("params", …)` is legal without declaring `values`. Ownership is never decided by call
order, so the old `VGPU-R1-OWNERSHIP-FLIP` class is unrepresentable in this model. Owned→external
transitions are not supported: recreate the instance (a plain options object makes that cheap).

## Notes

- A fragment-only effect is internally implemented as a `Draw` with an injected fullscreen triangle. Fragment-only resources receive fragment visibility only, so storage does not consume `maxStorageBuffersInVertexStage`.
- `blend` and `writeMask` are immutable pipeline state, fixed at `effect(gpu)` construction, and apply uniformly to every color target. Use them for overlays, glow, UI, and other loaded-pass compositing. For explicit blends, `op` defaults to `"add"` and omitted `alpha` copies `color`.
- Inside `frame(gpu)`, draw through `frame.pass()` — `p.draw()` accepts `Draw | Effect`, so there is no `p.effect()`. For a standalone render outside any frame use `renderOnce(gpu, target, cb)`: it owns its encoder, submits exactly once, and awaits async pipeline readiness instead of stalling.
- There is no implicit screen target. Browser code should create a `Surface` and pass it as the pass destination; a `Surface` is a presentation destination, not a bindable texture.
- Do not rely on implicit uniforms like time or resolution; pass `clock(gpu).time`, `target.size`, or `target.texelSize` explicitly through `.set()` or a shared `uniform(gpu, …)`.
- **See also:** `effect`, `Draw`, `prepare`, `FramePass.draw`, `Surface`, `Target`, `SharedUniforms`.
