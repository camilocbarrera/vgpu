---
title: Render bundles
summary: bundle(gpu, opts, record) records draws once; replaying them each frame skips re-encoding.
relatedSymbols:
  - Bundle
  - BundleOptions
  - BundleRecorder
prevNext:
  prev:
    title: Frames
    href: /concepts/frames
order: 70
---

# Render bundles

A render loop re-encodes every pipeline, bind group, and draw on every tick — even when nothing changed. A bundle records that command list once; replaying it each frame costs almost nothing.

Recording and materializing are two steps. `bundle(gpu, { target }, rec)` captures a **logical** command list and compiles nothing, so the bundle is born `"pending-pipelines"`; `await prepare(gpu, [{ bundle }])` compiles what is missing and encodes the native `GPURenderBundle`, moving it to `"ready"`. Readiness is observable on `bundle.status`.

## Record once, replay every frame

[`bundle(gpu)`](/reference/vgpu/bundle#bundle) records draws against a target and returns a [`Bundle`](/reference/vgpu/bundle#bundle). Replay it inside a pass with `pass.bundles()`:

```ts
import { init, bundle, clock, effect, frameLoop, prepare, surface } from "vgpu";

const gpu = await init();
const canvas = document.querySelector("canvas")!;
const canvasTarget = surface(gpu, canvas);
const ocean = effect(gpu, {
  shader: `
    struct Params { time: f32 }
    @group(0) @binding(0) var<uniform> params: Params;

    @fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
      return vec4f(0.1, 0.3, sin(params.time + uv.y) * 0.2 + 0.6, 1.0);
    }
  `,
  values: { params: { time: 0 } },
});
const boat = effect(gpu, `
  @fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    return vec4f(0.6, 0.4, 0.2, step(distance(uv, vec2f(0.5, 0.6)), 0.1));
  }
`);

// ---cut---
const scene = bundle(gpu, { target: canvasTarget }, (b) => {
  b.draw(ocean);
  b.draw(boat);
});                                    // recorded once, right here — and compiled: nothing yet
scene.status;                          // "pending-pipelines"

await prepare(gpu, [{ bundle: scene }]);  // compiles + encodes the native bundle
scene.status;                             // "ready"

const time = clock(gpu);
frameLoop(gpu, (frame) => {
  ocean.set("params", { time: time.time }); // byte update: the bundle stays "ready"
  frame.pass(canvasTarget, (pass) => pass.bundles(scene)); // replay — no re-encoding
});
```

Record what doesn't change, `.set()` what does: a bundle captures resource **identities**, not buffer
contents, so byte writes flow through on every replay and never invalidate it. Swapping an identity
(`ocean.bind("src", other)`) does invalidate it — see [resizes and sampled
targets](#resizes-and-sampled-targets).

> Good to know: draws inside a bundle can use different shaders and pipelines. What a bundle freezes is the target's render signature — color formats, depth format, sample count — plus bind groups, not a material or a target size.

## Compilation at record time

`bundle(gpu)` records the logical command list right when you call it, and **compiles nothing**:
construction never throws for pending pipelines. The bundle starts at `"pending-pipelines"`, and
`await prepare(gpu, [{ bundle }])` is what compiles the missing pipelines and encodes the native
bundle. A failure there rejects with `VGPU-PREPARE-FAILED`, moves the bundle to `"failed"` and retains
the error in `bundle.error`; `prepare()` always retries a failed bundle.

Recording needs only formats — never `getCurrentTexture()` — so recording against a `Surface` outside
`frame()` is legal, and the `target` option also takes a plain signature, so you can record during
load before the real target exists:

```ts
import { init, bundle, clock, effect, frameLoop, prepare, surface } from "vgpu";

const gpu = await init();
const canvas = document.querySelector("canvas")!;
const canvasTarget = surface(gpu, canvas);
const ocean = effect(gpu, `
  @fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    return vec4f(0.1, 0.3, 0.6, 1.0);
  }
`);
const boat = effect(gpu, `
  @fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    return vec4f(0.6, 0.4, 0.2, 1.0);
  }
`);

// ---cut---
const scene = bundle(gpu, { target: { colors: ['bgra8unorm'] } }, (b) => {
  b.draw(ocean);
  b.draw(boat);
});

// One request warms every pipeline of the recording AND encodes the native bundle:
await prepare(gpu, [{ bundle: scene }]);

frameLoop(gpu, (frame) => {
  frame.pass(canvasTarget, (pass) => pass.bundles(scene));
});
```

Three caveats. A `{ bundle }` request carries no `target` — the bundle froze its signature at
construction. Bindings must be declared before recording — the signature relaxes the target
requirement, not the resources. And replay targets must match the recorded signature exactly: when
they don't, the error prints both keys, which is how you catch a platform surface-format surprise
(`bgra8unorm` recorded, `rgba8unorm` actual).

Replaying a bundle that is not `"ready"` follows the one `pendingPipelines` chain, never a silent
hot-path compile:

| Policy | `"pending-pipelines"` | `"stale"` | `"failed"` |
|---|---|---|---|
| `"throw"` *(default)* | `VGPU-PIPELINE-PENDING` | `VGPU-R3-BUNDLE-STALE` | rethrows the retained error |
| `"skip"` | skipped; compilation continues in the background | skipped — never silently re-encoded | reported once via `gpu.onError`, then skipped |
| `"sync"` | compiles + encodes inline (the stall bundles exist to avoid) | re-encodes from the retained recording | does not retry |

A `disposed` bundle always throws `VGPU-BUNDLE-DISPOSED`, under every policy.

## Mix recorded and dynamic draws

A pass can replay bundles and encode fresh draws side by side:

```ts
import { init, bundle, clock, effect, frameLoop, prepare, surface } from "vgpu";

const gpu = await init();
const canvas = document.querySelector("canvas")!;
const canvasTarget = surface(gpu, canvas);
const ocean = effect(gpu, `
  @fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    return vec4f(0.1, 0.3, 0.6, 1.0);
  }
`);
const cursor = effect(gpu, {
  shader: `
    struct Params { pos: vec2f }
    @group(0) @binding(0) var<uniform> params: Params;

    @fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
      return vec4f(1.0, 1.0, 1.0, step(distance(uv, params.pos), 0.02));
    }
  `,
  values: { params: { pos: [0.5, 0.5] } },
});
const scene = bundle(gpu, { target: canvasTarget }, (b) => b.draw(ocean));

// ---cut---
// The bundle AND the dynamic draw each need their combination prepared:
await prepare(gpu, [{ bundle: scene }, { draw: cursor, target: canvasTarget }]);

frameLoop(gpu, (frame) => {
  frame.pass(canvasTarget, (pass) => {
    pass.bundles(scene); // the static part, replayed
    pass.draw(cursor); // the dynamic part, encoded fresh on top
  });
});
```

Some draws must stay on the dynamic side. Draws that set a `blendConstant` or a `stencil` `ref` cannot be recorded — bundle encoders have no way to set those pass-level values — so encode them with `pass.draw()`. A bundle also cannot replay inside a `depthReadOnly` pass, because bundles always record with writable depth. Indirect draws record fine: the GPU re-reads the argument buffer on every replay.

## Resizes and sampled targets

A bundle matches replay targets by render **signature**, not size: a resize that preserves color
format, depth format and sample count leaves it `"ready"` and invalidates no prepared pipeline.

What does invalidate the native bundle is an **identity** change among the resources it captured — a
`.bind()` swap, or a captured `Target` recreating its texture on resize. The bundle moves to
`"stale"`, and because its logical recording is still valid, `prepare(gpu, [{ bundle }])` re-encodes
it with the current resources. `rebuild()` is *not* needed for that: it exists only for when the
command list itself changes (a different set of draws), and it is synchronous, clears any retained
error, and leaves the bundle `"stale"`:

```ts
import { init, bundle, clock, effect, frameLoop, prepare, surface } from "vgpu";

const gpu = await init();
const canvas = document.querySelector("canvas")!;
const canvasTarget = surface(gpu, canvas);
const ocean = effect(gpu, `
  @fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    return vec4f(0.1, 0.3, 0.6, 1.0);
  }
`);

// ---cut---
const scene = bundle(gpu, { target: canvasTarget }, (b) => b.draw(ocean));
await prepare(gpu, [{ bundle: scene }]);

// A bundle that SAMPLES a resized target goes "stale" through the identity event.
// The binding auto-heals; the native bundle needs one explicit re-encode:
canvasTarget.onResize(async () => {
  if (scene.status === "stale") await prepare(gpu, [{ bundle: scene }]);
});

// Only when the command LIST changes (different draws) do you replace the recording:
scene.rebuild((b) => b.draw(ocean));   // synchronous; leaves the bundle "stale"
await prepare(gpu, [{ bundle: scene }]);

frameLoop(gpu, (frame) => {
  frame.pass(canvasTarget, (pass) => pass.bundles(scene));
});
```

Bindings auto-heal, native bundles require explicit preparation. That asymmetry is deliberate:
auto-healing a stale bundle on replay would hide a re-encode in the hot path, against the whole point
of "record once, replay cheap". The opt-in exists where the cost is visible —
`pendingPipelines: "sync"` means "do the pending work inline".

## When not to bother

Recording is not free, and a couple of draws per frame cost almost nothing to encode. Bundles pay off with many draws in a hot loop. The full ladder: `renderOnce(gpu, target, cb)` for a standalone one-shot render, `frame(gpu, cb)` to encode every pass, dispatch and copy into one submit, `bundle(gpu)` + `prepare()` to skip re-encoding what never changes.

See it live: the [batch rendering example](/examples/batch-rendering) packs four primitive types into one buffer and replays them from a single bundle.
