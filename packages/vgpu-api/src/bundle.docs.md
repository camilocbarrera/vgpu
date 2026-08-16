# Bundle

Main API (`vgpu`) render bundle recorded by `bundle(gpu, { target }, cb)`. A bundle captures a **logical command list** plus the attachment formats, sample count and bind-group identities it was recorded against; the native `GPURenderBundle` is materialized by `prepare(gpu, [{ bundle }])`. Readiness is observable on `bundle.status`, and replaying a non-`ready` bundle follows the one `pendingPipelines` chain — never a silent hot-path compile.

## Import

```ts
import type { Bundle, BundleOptions, BundleRecorder, BundleStatus } from "vgpu";
```

## Signature

```ts
import type { CompileTarget, Draw, DrawCallOptions, Effect, VGPUError } from "vgpu";

interface BundleOptions {
  readonly target: CompileTarget;   // a Target/Surface, or a bare TargetSignature
  readonly label?: string;
}

interface BundleRecorder {
  draw(drawable: Draw | Effect, opts?: DrawCallOptions): void;
}

type BundleStatus = "pending-pipelines" | "ready" | "stale" | "failed" | "disposed";

interface Bundle {
  readonly id: string;
  /** The native bundle once it exists; `undefined` until the first successful `prepare()`. */
  readonly gpu: GPURenderBundle | undefined;
  readonly status: BundleStatus;
  /** The retained failure of the last `prepare()`; defined ONLY while status === "failed". */
  readonly error: VGPUError | undefined;
  /** Replaces the logical command list, synchronously. Never compiles; never throws for readiness. */
  rebuild(record: (recorder: BundleRecorder) => void): void;
  /** Terminal: releases the native bundle and the retained recording. A second call is a no-op. */
  dispose(): void;
}
```

## Parameters

| Param | Type | Required | Default | Notes |
|---|---|---:|---|---|
| bundle.opts | `BundleOptions` | ✔ | — | Recording options. |
| opts.target | `Target \| Surface \| TargetSignature` | ✔ | — | Formats, depth format, and sample count are frozen at construction. Signature form is `{ colors: [...], depth?, sampleCount? }`; `colors` is required. Recording needs only formats — never `getCurrentTexture()` — so recording against a `Surface` **outside** `frame()` is legal, exactly like `prepare()`. |
| opts.label | `string` | ✖ | `` `bundle${n}` `` | Bundle id and GPU label. Auto id increments from `bundle1`. |
| bundle.cb | `(recorder: BundleRecorder) => void` | ✔ | — | Called immediately to record the logical command list. Under the default policy it compiles nothing: the bundle is born `"pending-pipelines"`. |
| recorder.draw.drawable | `Draw \| Effect` | ✔ | — | Draw or fullscreen effect to record. |
| recorder.draw.opts | `DrawCallOptions` | ✖ | `{}` | Counts and offsets captured in the recorded commands. `indirect` records fine — render bundle encoders support `drawIndirect`/`drawIndexedIndirect` — and the GPU re-reads the argument buffer on every replay. |
| bundle.rebuild.record | `(recorder: BundleRecorder) => void` | ✔ | — | **Synchronous.** Replaces the logical command list — only for when the command list itself changes. It clears a retained `error`, never compiles, and leaves the bundle `stale` (or `pending-pipelines` if the new recording introduces an uncompiled combination). `rebuild()` is *not* how a `stale` bundle becomes `ready`. |
| framePass.bundles.bundles | `readonly Bundle[]` | ✔ | — | Replayed bundles; must be created by `bundle()`. |

**Returns:** `bundle(gpu, opts, cb)` returns a `Bundle` whose `status` starts at `"pending-pipelines"`; `BundleRecorder.draw()`, `rebuild()`, `dispose()` and `FramePass.bundles()` return `void`.

**Throws:** `VGPU-PIPELINE-PENDING` when a `pending-pipelines` bundle is replayed under the default `pendingPipelines: "throw"` — `await prepare(gpu, [{ bundle }])` first; `VGPU-R3-BUNDLE-STALE` when a `stale` bundle is replayed under `"throw"` (the message names `prepare(gpu, { bundle })`; **no `rebuild()` is required**), or when the replay target's formats/depth/sample count differ from the recorded signature; `VGPU-BUNDLE-DISPOSED` when a disposed bundle is replayed — under **every** policy; the retained `bundle.error` rethrown (with its `cause`) when a `failed` bundle is replayed; `VGPU-PREPARE-FAILED` when `prepare()` cannot compile a combination of the recording (the bundle moves to `failed` and retains the error); `VGPU-R3-BUNDLE-INVALID` when replay receives an object not created by `bundle()`; `VGPU-BUNDLE-BLEND-CONSTANT` when recording a draw with `blendConstant` (the blend constant is render-pass state that render bundle encoders cannot set; encode such draws in a frame pass instead); `VGPU-BUNDLE-STENCIL-REF` when recording a draw whose `stencil` has `ref` (the stencil reference is likewise render-pass state; stencil state without `ref` records fine); `VGPU-SURFACE-DISPOSED` when replaying against a disposed surface; `VGPU-DEVICE-LOST` for any replay after device loss; draw binding errors such as `VGPU-R1-BINDING-NEVER-SET` can throw during recording. Signature mismatch messages print both recorded and actual signature keys.

## Examples

```ts
import { init, bundle, draw, frame, prepare, target } from "vgpu/mock";

const gpu = await init();
const colorTarget = target(gpu, { size: [64, 64] });
const drawable = draw(gpu, { shader: `
  @vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    var p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
    return vec4f(p[vi], 0, 1);
  }
  @fragment fn fs_main() -> @location(0) vec4f { return vec4f(1, 1, 0, 1); }
` });

const statics = bundle(gpu, { target: colorTarget, label: "static" }, (recorded) => {
  recorded.draw(drawable);        // hundreds of draws, recorded once
});

statics.status;                   // "pending-pipelines" — construction never materializes the bundle
const prepared = await prepare(gpu, { bundle: statics });
statics.status;                   // "ready"
prepared.gpu;                     // GPURenderBundle

frame(gpu, (currentFrame) => {
  currentFrame.pass({ target: colorTarget }, (pass) => pass.bundles(statics));
});
```

Byte updates keep a bundle `ready`; identity updates make it `stale`, and `prepare()` re-encodes the
retained recording:

```ts
import { init, bundle, effect, frame, prepare, target } from "vgpu/mock";

const gpu = await init();
const screen = target(gpu, { size: [64, 64] });
const scene = target(gpu, { size: [64, 64], format: "rgba16float" });
const fx = effect(gpu, {
  shader: `
    struct Params { intensity: f32 }
    @group(0) @binding(0) var<uniform> params: Params;
    @group(0) @binding(1) var src: texture_2d<f32>;
    @fragment fn fs_main() -> @location(0) vec4f {
      return vec4f(textureLoad(src, vec2u(0, 0), 0).rgb * params.intensity, 1);
    }
  `,
  values: { params: { intensity: 1 } },
  bindings: { src: scene },       // bind the TARGET, not scene.color
});

const post = bundle(gpu, { target: screen }, (b) => b.draw(fx));
await prepare(gpu, [{ bundle: post }]);

fx.set("params", { intensity: 0.8 });   // byte update: bundle stays "ready"
frame(gpu, (f) => f.pass(screen, (p) => p.bundles(post)));

scene.resize([128, 128]);                // identity event: the binding auto-heals…
post.status;                             // …but the native bundle is "stale"
await prepare(gpu, [{ bundle: post }]);  // re-encodes the retained recording; no rebuild() needed
```

```ts
import { init, bundle, clock, effect, frame, pingPong, prepare } from "vgpu/mock";

const gpu = await init();
const ping = pingPong(gpu, 32, 32);
const shader = effect(gpu, `@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1); }`);
const even = bundle(gpu, { target: ping.write }, (b) => b.draw(shader));
ping.swap();
const odd = bundle(gpu, { target: ping.write }, (b) => b.draw(shader));
ping.swap();

await prepare(gpu, [{ bundle: even }, { bundle: odd }]);

frame(gpu, (currentFrame) => {
  currentFrame.pass({ target: ping.write }, (p) => p.bundles(clock(gpu).frameCount % 2 ? odd : even));
});
```

## Bundle state machine

`BundleStatus` is a frozen union. `pending-pipelines` means pipelines are still to compile (and the
native bundle still to encode); `stale` means the pipelines are compiled but the native bundle must
be re-encoded **from the retained logical recording** — and that re-encode is
`prepare(gpu, [{ bundle }])`, never `rebuild()`.

| From | Event | To |
|---|---|---|
| — | `bundle(gpu, { target }, rec)` | `pending-pipelines` (always: the native bundle is not materialized at construction) |
| `pending-pipelines` \| `stale` \| `failed` | `prepare(gpu, [{ bundle }])` succeeds | `ready` (compiles missing pipelines, then encodes the native bundle) |
| any non-`disposed` | `prepare()` fails | `failed`, retaining the error in `bundle.error` |
| `ready` | `.set()` byte update on a captured resource | `ready` (no transition) |
| `ready` | `.bind()` identity update, or a captured `Target` recreating its texture (resize) | `stale` |
| `pending-pipelines` \| `stale` \| `failed` | identity update | unchanged |
| any non-`disposed` | `rebuild(cb)` (synchronous) | clears `bundle.error`, then `stale` — or `pending-pipelines` when the new recording introduces an uncompiled combination. It never compiles and never throws for readiness reasons |
| any | `dispose()` | `disposed` (terminal; releases the native bundle and the retained recording; a second `dispose()` is a no-op) |

`prepare()` on a `ready` bundle is a no-op, and it **always retries** a `failed` one — there is no
"retryable failure" classification. During that retry `bundle.error` stays available and the status
stays `failed`; success clears the error and moves to `ready`. `rebuild()` clears the retained error
immediately, because it replaces the recording that failure belonged to. `bundle.error` is defined
only while `status === "failed"`.

**Replay of a non-`ready` bundle** follows the one `pendingPipelines` chain (call site → frame → gpu):

| Policy | `pending-pipelines` | `stale` | `failed` | `disposed` |
|---|---|---|---|---|
| `"throw"` *(default)* | `VGPU-PIPELINE-PENDING` | `VGPU-R3-BUNDLE-STALE` | rethrows the retained error (`cause` preserved) | `VGPU-BUNDLE-DISPOSED` |
| `"skip"` | skipped; async compilation starts/continues | skipped — never silently re-encoded | reported **once** via `gpu.onError`, then kept skipped | `VGPU-BUNDLE-DISPOSED` |
| `"sync"` | compiles + encodes inline (the stall this feature exists to avoid) | re-encodes from the retained recording (bounded: no compilation — the opt-in auto-heal) | does **not** retry; the retained error is thrown | `VGPU-BUNDLE-DISPOSED` |

Auto-healing a `stale` bundle on replay is deliberately **not** the default: it would hide a
re-encode in the hot path, against the purpose of bundles ("record once, replay cheap"). The opt-in
already exists — `"sync"` means "do the pending work inline".

## Signature-arm recording

`bundle(gpu, { target: { colors: ["bgra8unorm"], depth: "depth24plus", sampleCount: 4 } }, cb)`
records before a target exists. This relaxes only the replay target: any resources sampled by draws
still need to be bound before recording. It compiles nothing either — materialize with
`await prepare(gpu, [{ bundle }])` before the first replay.

For future canvas surfaces, use `navigator.gpu.getPreferredCanvasFormat()` when building the
signature. A bundle recorded for `bgra8unorm` will not replay on an `rgba8unorm` surface, and the
stale error prints both keys.

## Notes

- **`prepare()` is the one spelling for bundle readiness.** There is no per-bundle readiness promise and no `await bundle.ready`: `prepare(gpu, [{ bundle }])` compiles what is missing, encodes the native bundle, and is idempotent on a `ready` one. A `{ bundle }` request carries no `target` — a bundle froze its target signature at construction.
- Bundles match replay targets by render **signature**, not size. They survive resizing the target they draw onto: a resize that preserves color format, depth format and sample count leaves a bundle `ready`.
- A bundle that *samples* a resized `Target` moves to `stale` through the identity-change event, and its logical recording stays valid — `prepare(gpu, [{ bundle }])` re-encodes it with the new texture. Bindings auto-heal; native bundles require explicit preparation. Bind the `Target`, not `target.color`: a `Texture` snapshot has no recreation hook and goes silently stale.
- Bundles freeze bind group identities, not buffer contents. `instance.set(binding, value)` (bytes) is safe and keeps the bundle `ready`; `instance.bind(binding, resource)` swaps identity and stales it.
- `rebuild()` is reserved for replacing the command list itself (the set of draws changed). It is synchronous because it is called at the edge of a synchronous frame callback.
- Draws with `blendConstant` cannot be recorded: render bundle encoders have no way to set the pass blend constant. Recording throws `VGPU-BUNDLE-BLEND-CONSTANT`; use `FramePass.draw` for those draws.
- Draws whose `stencil` has `ref` cannot be recorded either: render bundle encoders have no way to set the pass stencil reference. Recording throws `VGPU-BUNDLE-STENCIL-REF`; stencil pipeline state without `ref` records fine.
- **See also:** `prepare`, `FramePass.bundles`, `Draw`, `Effect`, `Surface`, `Target`, `createRenderBundle`.
