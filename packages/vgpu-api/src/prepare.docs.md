# prepare

`prepare(gpu, requests)` is the **one** spelling for pipeline readiness. It compiles the pipelines a set of **combinations** needs — `{ draw, target }`, `{ compute }`, `{ bundle }` — through `createRenderPipelineAsync()` / `createComputePipelineAsync()`, and resolves with one handle per request. Encoding is synchronous, so compilation has to happen somewhere else: `prepare()` is that place.

## Import

```ts
import { prepare } from "vgpu";
import type { PendingPipelines, PrepareRequest, PreparedBundle, PreparedCompute, PreparedDraw, PreparedFor } from "vgpu";
```

## Signature

```ts
import type { Bundle, Compute, CompileTarget, Draw, Effect, Gpu, TargetSignature } from "vgpu";

type PrepareRequest =
  | { draw: Draw | Effect; target: CompileTarget }
  | { compute: Compute }
  | { bundle: Bundle };

type PreparedDraw    = { readonly draw: Draw | Effect; readonly signature: TargetSignature; readonly gpu: GPURenderPipeline };
type PreparedCompute = { readonly compute: Compute;                                         readonly gpu: GPUComputePipeline };
type PreparedBundle  = { readonly bundle: Bundle;      readonly signature: TargetSignature; readonly gpu: GPURenderBundle };

type PreparedFor<R> =
  R extends { compute: Compute } ? PreparedCompute :
  R extends { bundle: Bundle }   ? PreparedBundle  : PreparedDraw;

declare function prepare<const R extends PrepareRequest>(gpu: Gpu, request: R): Promise<PreparedFor<R>>;
declare function prepare<const R extends readonly PrepareRequest[]>(gpu: Gpu, requests: R): Promise<{ [K in keyof R]: PreparedFor<R[K]> }>;

/** The policy that decides what a SYNCHRONOUS encode does with an unprepared combination. */
type PendingPipelines = "throw" | "skip" | "sync";
```

## Parameters

| Param | Type | Required | Default | Notes |
|---|---|---:|---|---|
| gpu | `Gpu` | ✔ | — | The context that owns every object in the requests. |
| requests | `PrepareRequest \| readonly PrepareRequest[]` | ✔ | — | One entry per combination to warm. The array form preserves request order and tuple inference; the single-request form returns one handle. |
| request.draw | `Draw \| Effect` | ✔ in the `draw` arm | — | The renderable — the same `Draw \| Effect` union `p.draw()` takes. There is no `{ effect }` branch and no `render:` spelling: every request key is the name of the encode call it warms. |
| request.target | `Target \| Surface \| TargetSignature` | ✔ in the `draw` arm | — | The target signature side of the combination. A `Surface` is legal **outside** `frame()`: its signature comes from its configuration, not from `getCurrentTexture()`. A bare `{ colors, depth?, sampleCount? }` works too. |
| request.compute | `Compute` | ✔ in the `compute` arm | — | A compute request has no target signature — a compute pipeline needs none. |
| request.bundle | `Bundle` | ✔ in the `bundle` arm | — | Compiles the recording's missing pipelines and then **encodes the native `GPURenderBundle`**, moving the bundle to `ready`. A `{ bundle }` request carries no `target`: a bundle froze its signature at construction. |

**Returns:** a `Promise` of one handle per request — `PreparedDraw`, `PreparedCompute` or `PreparedBundle`, resolved per request through `PreparedFor<R>`. The array form resolves to a tuple in request order; the single-request form to one handle. Ignoring the result is valid: the happy path only needs the `await`.

**Throws:** the promise **rejects** with `VGPU-PREPARE-FAILED` when any requested combination fails to compile, enumerating **every** failure (renderable label + resolved signature + `cause`). Combinations that did compile stay cached, so re-preparing the succeeding subset performs no new `createRenderPipeline` call. Also `VGPU-DEVICE-LOST` when the device was lost (loss is terminal), `VGPU-SURFACE-DISPOSED` for a disposed surface (a disposed surface has no derivable signature), and `VGPU-GPU-DISPOSED` / `VGPU-GPU-FOREIGN` from the usual entry guards.

## Examples

The canonical happy path — one async line before the loop:

```ts
import { init, clock, draw, frameLoop, geometry, prepare, surface } from "vgpu";
import { box } from "vgpu/scene";

declare const canvas: HTMLCanvasElement;
declare const litWgsl: string;
declare function mvpFor(time: number): number[];

const gpu    = await init();
const screen = surface(gpu, canvas, { depth: true, sampleCount: 4 });
const mesh   = geometry(gpu, box());
const cube   = draw(gpu, { shader: litWgsl, geometry: mesh });

await prepare(gpu, [{ draw: cube, target: screen }]);   // the one async line the strict default asks for

frameLoop(gpu, (f) => {
  cube.set("camera", { viewProjection: mvpFor(clock(gpu).time) });  // binding-scoped byte write
  f.pass(screen, (p) => p.draw(cube));                  // prepared → draws; unprepared → VGPU-PIPELINE-PENDING
});
```

One handle per requested combination:

```ts
import { init, bundle, compute, draw, prepare, target } from "vgpu/mock";

const gpu = await init();
const screen = target(gpu, { size: [64, 64] });
const cubeDraw = draw(gpu, { shader: `@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1); }` });
const simulation = compute(gpu, `@compute @workgroup_size(1) fn cs_main() {}`);

const [cube, sim] = await prepare(gpu, [
  { draw: cubeDraw, target: screen },
  { compute: simulation },
]);

cube.draw;       // the renderable, echoed back — the handle identifies the combination
cube.signature;  // resolved TargetSignature { colors, depth, sampleCount }
cube.gpu;        // GPURenderPipeline — the same-agent low-level escape hatch
sim.gpu;         // GPUComputePipeline (a compute request has no target signature)

const forest = bundle(gpu, { target: screen }, (p) => p.draw(cubeDraw));
const prepared = await prepare(gpu, { bundle: forest });   // single request → single handle
prepared.gpu;    // GPURenderBundle
```

Combination-scoped readiness, and a failure that names every combination:

```ts
import { init, draw, prepare, target } from "vgpu/mock";

const gpu = await init();
const ldr = target(gpu, { size: [32, 32] });
const hdr = target(gpu, { size: [32, 32], format: "rgba16float" });
const mesh = draw(gpu, { shader: `@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1); }` });

await prepare(gpu, [{ draw: mesh, target: ldr }]);   // ready for `ldr`…
// …and still uncompiled for `hdr`: preparing one combination never prepares another.

try {
  await prepare(gpu, [{ draw: mesh, target: hdr }]);
} catch (error) {
  // VGPU-PREPARE-FAILED lists every failed combination; the ones that compiled stay cached.
  console.error(error);
}
```

Policy as an exception, not a habit:

```ts
import { init, draw, frame, prepare, target } from "vgpu/mock";

const gpu = await init();
const screen = target(gpu, { size: [64, 64] });
const world = draw(gpu, { shader: `@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1); }` });
const streamedMesh = draw(gpu, { shader: `@fragment fn fs_main() -> @location(0) vec4f { return vec4f(0.5); }` });

await prepare(gpu, [{ draw: world, target: screen }]);

frame(gpu, (f) => {
  f.pass(screen, (p) => {
    p.draw(world);                                           // prepared → draws
    p.draw(streamedMesh, { pendingPipelines: "skip" });       // appears when its pipeline is ready
  });
});
```

## Compilation paths

| Entry path | What happens when the required pipeline is not ready | Encoding / submission |
|---|---|---|
| `await prepare(gpu, requests)` | Uses `createRenderPipelineAsync()` / `createComputePipelineAsync()` and waits for readiness. | Prepares only; it does not submit a frame command buffer. |
| `await renderOnce(...)` / `await compute.dispatchOnce(...)` | Uses the corresponding async pipeline creation path, then continues once ready. | Own encoder and exactly one submit; resolves after **submit**, not after GPU completion. |
| Synchronous encode with `"throw"` *(default)* | Does **not** start compilation; throws `VGPU-PIPELINE-PENDING` immediately. | The affected command is not encoded. |
| Synchronous encode with `"skip"` | Starts or continues async compilation in the background and returns immediately. | The affected draw, dispatch or bundle replay is omitted from this frame. |
| Synchronous encode with `"sync"` | Uses immediate synchronous pipeline creation inline; a stale bundle is re-encoded inline. | The command is encoded in the current frame, but the call may stall. |

An already-prepared combination always reuses its cached pipeline; none of these paths creates another
one. The policy is resolved **call site → frame → gpu**:

- **`"throw"` (the default)** — uniform across browser, mock and node, so there is no dev/prod
  divergence. It does not start compilation and throws with an actionable message (*"await
  prepare(gpu, [{draw, target}]) before drawing, or opt in to inline compilation with
  pendingPipelines: 'sync'"*). Synchronous encode contexts **never** compile pipelines implicitly: a
  compilation stall is a choice, never an accident. Tests simply call `prepare()`, which is instant
  on the mock device.
- **`"skip"`** — the object is skipped this frame while async compilation starts or continues. A
  compilation failure moves it to `failed`, is reported **once** through `gpu.onError` with a stable
  code, and the object keeps being skipped; it never throws per frame. A skipped compute producer
  does **not** cascade: consumers read the resource's previous contents (or its zero-initialized
  state). vgpu is not a render graph and does not pretend to have the dependency graph to do
  otherwise.
- **`"sync"`** — do the pending work inline: immediate WebGPU pipeline creation, and for a `stale`
  bundle a bounded re-encode. May stall. It is the prototyping/porting escape hatch, and the stall is
  visible at the policy call site.

## Notes

- **Readiness is a property of a combination, never of an object.** A `Draw` can be ready for the screen, uncompiled for an HDR target and failed for a `depth24plus-stencil8` target at the same time. No `Draw`/`Effect`/`Compute` carries `ready`/`pending`/`failed` fields, and no per-target status map is exposed: `prepare()` **is** the query — idempotent, O(1) on an already-prepared combination, and it rejects with the failure.
- **Handles are data, not state.** There is no `prepared.status`: either `prepare()` rejects on failure (it does), which would make the field a dead constant, or the happy path would walk past a failure that reappears frames later in the encode path — exactly what the strict default exists to prevent.
- `prepared.gpu` is the **only** low-level pipeline escape hatch — no per-object `compile()` / `compileSync()` / `pipelineFor()`, no `DrawOptions.targets` precompile list, and no `compute.pipeline` field. It is the same object the encode path uses, and it is same-agent only, like every device-local object.
- **Request keys mirror the encode call they warm:** `p.draw(Draw | Effect)` → `{ draw, target }`; `f.compute(c)` → `{ compute }`; `p.bundles(b)` → `{ bundle }`.
- Every other pipeline-affecting input — geometry vertex layout, topology/strip format, cull/front face, depth/stencil/multisample/blend state, pipeline constants, entry points — is fixed at construction of the renderable, so `(renderable, signature)` is a complete key. That is why `prepare()` needs no `geometry:` or state axis. The internal pipeline cache key is not public API.
- After a completed `prepare()` on a combination, encoding it in a `frame()` creates **no** pipelines: no `createRenderPipeline` / `createComputePipeline` is issued during encode.
- A resize that preserves color format, depth format and sample count invalidates no prepared combination; changing any of the three does.
- Device loss is terminal: `prepare()` on a lost device throws `VGPU-DEVICE-LOST`, and prepared pipelines are never re-pointed at a replacement device. Rebuild the graph on a new `Gpu` and prepare again.
- **See also:** `renderOnce`, `Draw`, `Effect`, `Compute`, `Bundle`, `Frame`, `Surface`, `Target`, `init`.
