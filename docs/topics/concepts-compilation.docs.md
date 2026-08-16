---
title: Compilation
summary: Pipelines never compile behind your back — prepare() warms them on an async path, so no frame ever hitches.
relatedSymbols:
  - prepare
  - Draw
  - Effect
  - CompileTarget
prevNext:
  prev:
    title: Draws
    href: /concepts/draws
  next:
    title: Effects
    href: /concepts/effects
order: 30
---

# Compilation

WebGPU keys pipelines by shader **and** render signature — the tuple of color formats, depth format and sample count — so the same WGSL rendering into a canvas and into an MSAA target means two pipelines. Creating one is expensive, and encoding a frame is synchronous, so vgpu never does it implicitly: **the default policy `pendingPipelines: "throw"` fails fast instead of compiling inside your frame.** Compilation happens on an async path you chose — `prepare()`, `renderOnce()` / `dispatchOnce()`, or a background `"skip"` — and readiness belongs to the *combination* `(renderable, target)`, never to an object.

## Pre-warming with a target

Most of the time you already have the target in hand. `await prepare(gpu, [...])` warms exactly those
combinations and resolves with one handle per request:

```ts
import { init, draw, effect, frame, prepare, surface } from "vgpu";

const gpu = await init();
const canvas = document.querySelector("canvas")!;
const canvasSurface = surface(gpu, canvas);

// ---cut---
const ocean = effect(gpu, `
  @fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    return vec4f(uv, 0.8, 1.0);
  }
`);
const tri = draw(gpu, {
  shader: `
    struct Out { @builtin(position) position: vec4f }
    @vertex fn vs_main(@builtin(vertex_index) vi: u32) -> Out {
      var pts = array<vec2f, 3>(vec2f(-0.5, -0.5), vec2f(0.5, -0.5), vec2f(0.0, 0.5));
      var out: Out;
      out.position = vec4f(pts[vi], 0.0, 1.0);
      return out;
    }
    @fragment fn fs_main() -> @location(0) vec4f { return vec4f(1.0, 0.4, 0.2, 1.0); }
  `,
});

await prepare(gpu, [
  { draw: tri, target: canvasSurface },
  { draw: ocean, target: canvasSurface },   // `draw:` takes Draw | Effect, like p.draw()
]);

frame(gpu, (f) => {
  f.pass(canvasSurface, (p) => {
    p.draw(tri);      // prepared → encodes; unprepared → VGPU-PIPELINE-PENDING
    p.draw(ocean);
  });
});
```

Preparing against a `Surface` **outside** `frame()` is legal and is exactly this path: a surface's
signature comes from its configuration, not from `getCurrentTexture()`. Pipelines are cached per
combination at the device level, so every frame after this one just encodes work, and calling
`prepare()` again on a warm combination is free.

Readiness is per combination, so one prepared target says nothing about another:

```ts
import { init, draw, prepare, target } from "vgpu";

const gpu = await init();
const mesh = draw(gpu, { shader: `@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1); }` });
const ldr = target(gpu, { size: [64, 64] });
const hdr = target(gpu, { size: [64, 64], format: "rgba16float" });

await prepare(gpu, [{ draw: mesh, target: ldr }]);   // ready for ldr, still uncompiled for hdr
await prepare(gpu, [{ draw: mesh, target: hdr }]);   // a second signature is a second pipeline
```

## Compiling without a target

Sometimes the target doesn't exist yet. Pass a signature object instead: `colors` is required, `depth`
and `sampleCount` are optional.

```ts
import { init, draw, geometry, prepare } from "vgpu";
import { box } from "vgpu/scene";

const gpu = await init();
const sceneShader = `/* vertex + fragment WGSL */`;
const msaaScene = draw(gpu, { shader: sceneShader, geometry: geometry(gpu, box({ size: 1 })) });

await prepare(gpu, [{
  draw: msaaScene,
  target: { colors: ['bgra8unorm'], depth: 'depth24plus', sampleCount: 4 },
}]);
```

> Good to know: surface formats are platform-dependent — `bgra8unorm` on most browsers, `rgba8unorm`
> on others. Preparing the wrong signature is not an error, it is just a warm-up you didn't need; but
> it does **not** cover the real target either, and under the default policy the real draw will throw
> instead of compiling silently. When in doubt, prepare against the actual target.

## Choosing a policy

`pendingPipelines` is resolved **call site → frame → gpu**, so you normally set it once:

```ts
import { init, draw, frame, prepare, target } from "vgpu";

const gpu = await init();                       // default: pendingPipelines: "throw"
const screen = target(gpu, { size: [64, 64] });
const world = draw(gpu, { shader: `@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1); }` });
const streamed = draw(gpu, { shader: `@fragment fn fs_main() -> @location(0) vec4f { return vec4f(0.5); }` });

await prepare(gpu, [{ draw: world, target: screen }]);

frame(gpu, (f) => {
  f.pass(screen, (p) => {
    p.draw(world);
    p.draw(streamed, { pendingPipelines: "skip" }); // exception: appears when its pipeline is ready
  });
});
```

| Policy | An uncompiled combination during a synchronous encode |
|---|---|
| `"throw"` *(default)* | Does not start compilation; throws `VGPU-PIPELINE-PENDING` and encodes nothing for that command. |
| `"skip"` | Starts/continues async compilation in the background and omits the command this frame. Never throws per frame; a failure is reported once through `gpu.onError`. |
| `"sync"` | Immediate inline pipeline creation — the porting/prototyping escape hatch, and the only value that can stall. |

`init({ pendingPipelines: "sync" })` reproduces the eager behavior of older versions app-wide, with
the stall visible at the policy call site. The default is the same in the browser, in Node and on the
mock device: no dev/prod divergence, and tests get readiness from `prepare()`, which is instant on
the mock adapter.

One-shot helpers do not need any of this: `await renderOnce(gpu, target, cb)` and
`await kernel.dispatchOnce(n)` always take the async readiness path, own their encoder and submit once.

## Errors

A failed `prepare()` **rejects**, and the error belongs to the call site — catch it where you scheduled
the warm-up. `VGPU-PREPARE-FAILED` enumerates every failed combination (renderable label, resolved
signature, `cause`), while the combinations that did compile stay cached, so re-preparing the good
subset is free:

```ts
import { init, draw, prepare } from "vgpu";

const gpu = await init();
const tri = draw(gpu, { shader: `@vertex fn vs_main() -> @builtin(position) vec4f { return vec4f(0); }` });

try {
  await prepare(gpu, [{ draw: tri, target: { colors: ['bgra8unorm'] } }]);
} catch (error) {
  console.error('Pipeline failed to compile', error);
}
```

There is no `prepared.status` to inspect: handles are data, and failure is the rejection. The
`"skip"` path is the one place a compilation failure is not thrown — it is reported **once** through
[`gpu.onError`](/reference/vgpu/gpu#onerror) with a stable code, the object keeps being skipped, and
`gpu.settled()` lets tests wait for those deliveries.

## Render bundles

A [bundle](/concepts/render-bundles) records a logical command list at construction and compiles
nothing: it is born `"pending-pipelines"`, and `await prepare(gpu, [{ bundle }])` compiles what is
missing and encodes the native bundle. See
[compilation at record time](/concepts/render-bundles#compilation-at-record-time) for that flow.
