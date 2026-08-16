---
title: "Optimize a pass"
description: "Optimize one pass by first deciding what changes every frame."
---

## 0. Measure first

Attach a `timer(gpu)` span (requires a device with `"timestamp-query"`) and judge every change by the reported GPU milliseconds:

```ts
import { createMockAdapter, frameLoop, init, prepare, timer } from "vgpu/mock";
import type { Effect, Target } from "vgpu";

declare const screen: Target;
declare const fx: Effect;

// ---cut---
const gpu = await init({
  adapter: createMockAdapter({ features: ["timestamp-query"] }),
  requiredFeatures: ["timestamp-query"],
});
const spans = timer(gpu);
spans.onResults((results) => console.log(`pass ${results.pass}ms`));

await prepare(gpu, [{ draw: fx, target: screen }]);
frameLoop(gpu, (f) => f.pass({ target: screen, timer: spans.span("pass") }, (p) => p.draw(fx)));
```

## 1. Static commands

If the draw list is static, record it once and materialize it with `prepare()`:

```ts
import { bundle, frameLoop, init, prepare, target } from "vgpu/mock";
import type { Effect } from "vgpu";

const gpu = await init();
const screen = target(gpu, { size: [64, 64] });
declare const background: Effect;
declare const grid: Effect;

// ---cut---
const effectBundle = bundle(gpu, { target: screen }, (b) => {
  b.draw(background);
  b.draw(grid);
});
await prepare(gpu, [{ bundle: effectBundle }]);   // "pending-pipelines" → "ready"

frameLoop(gpu, (f) => f.pass(screen, (p) => p.bundles(effectBundle)));
```

## 2. Animated scalar/vector values

Keep the instance and write the bytes of the binding that changes — binding-scoped `.set()` never rebuilds a bind group, so a bundle that captured this effect stays `ready`:

```ts
import { clock, effect, frameLoop, init, prepare, target } from "vgpu/mock";

const gpu = await init();
const screen = target(gpu, { size: [64, 64] });
const WGSL = `
  struct Params { time: f32, exposure: f32 }
  @group(0) @binding(0) var<uniform> params: Params;
  @fragment fn fs_main() -> @location(0) vec4f { return vec4f(params.time * params.exposure); }
`;

// ---cut---
const fx = effect(gpu, { shader: WGSL, values: { params: { time: 0, exposure: 1 } } });
const time = clock(gpu);

await prepare(gpu, [{ draw: fx, target: screen }]);
frameLoop(gpu, (f) => {
  fx.set("params", { time: time.time });   // partial: exposure keeps its value
  f.pass(screen, fx);
});
```

## 3. Resources that swap

Use ping-pong rather than allocating a new target or storage buffer, and swap **identity** with `.bind()` — never `.set()`:

```ts
import { effect, frameLoop, init, pingPong, prepare } from "vgpu/mock";

const gpu = await init();
const WGSL = `
  @group(0) @binding(0) var src: texture_2d<f32>;
  @fragment fn fs_main() -> @location(0) vec4f { return textureLoad(src, vec2u(0, 0), 0); }
`;

// ---cut---
const state = pingPong(gpu, 512, 512, { format: "rgba16float" });
const step = effect(gpu, { shader: WGSL, bindings: { src: state.read } });

await prepare(gpu, [{ draw: step, target: state.write }]);
frameLoop(gpu, (f) => {
  step.bind("src", state.read);   // identity swap: rebuilds exactly that group
  f.pass(state.write, step);
  state.swap();
});
```

Both halves share one render signature, so one prepared combination covers the swap. Bind the
`Target`, not `target.color`: a target re-binds its new texture identity when it is recreated, a
texture snapshot goes silently stale.

## 4. Many objects

Use instancing for many copies of the same draw, or `draw.group()` with a manually claimed bind group plus dynamic `offsets` when every object needs a different uniform block.
