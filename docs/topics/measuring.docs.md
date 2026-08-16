# Measuring

Measure the thing you intend to optimize: CPU encoding, pipeline warm-up, bind-group churn, target memory, or shader cost. The public API makes those boundaries visible.

## CPU encoding vs replay

If the CPU is busy rebuilding the same render pass, compare a direct loop with a bundle:

```ts
import { bundle, frameLoop, init, prepare, target } from "vgpu/mock";
import type { Draw } from "vgpu";

const gpu = await init();
const screen = target(gpu, { size: [64, 64] });
declare const staticDraws: readonly Draw[];

// ---cut---
const staticScene = bundle(gpu, { target: screen }, (b) => {
  for (const drawable of staticDraws) b.draw(drawable);
});
await prepare(gpu, [{ bundle: staticScene }]);   // compiles + encodes the native bundle

frameLoop(gpu, (f) => f.pass(screen, (p) => p.bundles(staticScene)));
```

Replay is only cheap once the bundle is `ready`: recording compiles nothing, and replaying a
`pending-pipelines` or `stale` bundle throws under the default policy instead of quietly re-encoding
in the hot path.

## First-frame hitches

If the first visible frame stutters, warm the combinations it will encode — every `(renderable, target
signature)` pair, on an async path, before the loop starts:

```ts
import { draw, init, prepare, target } from "vgpu/mock";

const gpu = await init();
declare const WGSL: string;

// ---cut---
const hdr = target(gpu, { size: [256, 256], format: "rgba16float", depth: true, sampleCount: 4 });
const mesh = draw(gpu, { shader: WGSL });

await prepare(gpu, [{ draw: mesh, target: hdr }]);
```

Under the default `pendingPipelines: "throw"` a missed combination cannot hitch — it throws
`VGPU-PIPELINE-PENDING` instead, which turns a rare frame-time spike into a deterministic error at the
call site. If some content legitimately arrives late, name `pendingPipelines: "skip"` on that draw so
it is omitted until its pipeline is ready.

## Binding churn

If allocations or bind-group count grows every frame, look at which of the two update paths you are
using. `instance.set(binding, value)` writes bytes and **never** rebuilds a bind group;
`instance.bind(binding, resource)` swaps identity and rebuilds exactly the affected group (and stales
bundles that captured it). Move shared state to one `uniform(gpu, { … })` bound into every shader that
needs it — one write instead of N — and keep identities stable across frames.

## GPU pass cost

If a pass looks expensive, confirm it on the GPU before optimizing it. CPU time around `frame.pass(...)` measures encoding only — encoders record commands, they do not run them. Mark the pass with a `timer(gpu)` span instead:

```ts
import { createMockAdapter, frameLoop, init, timer } from "vgpu/mock";
import type { Draw, Target } from "vgpu";

declare const shadowMap: Target;
declare const scene: Target;
declare const casters: Draw;
declare const world: Draw;

// ---cut---
const gpu = await init({
  adapter: createMockAdapter({ features: ["timestamp-query"] }),
  requiredFeatures: ["timestamp-query"],
});
const spans = timer(gpu);
spans.onResults((results) => console.log(`shadows ${results.shadows}ms, main ${results.main}ms`));

frameLoop(gpu, (f) => {
  f.pass({ target: shadowMap, timer: spans.span("shadows") }, (p) => p.draw(casters));
  f.pass({ target: scene, timer: spans.span("main") }, (p) => p.draw(world));
});
```

Durations arrive through `onResults` in milliseconds, one or two frames after submit; readback never blocks a frame. A span times the whole pass, not individual draws — move suspect work into its own pass to isolate it. GPU completion itself is observable in exactly one place: `await gpu.settled()`, which snapshots the submissions, compilations and readbacks already started and never rejects.

## Correctness before speed

When measuring visual output, render one deterministic `frame(gpu, ...)` into an explicit target and read it back. Do not measure while also resizing, preparing pipelines, or creating temporary targets inside the loop — `prepare()` belongs before the measurement, not inside it.
