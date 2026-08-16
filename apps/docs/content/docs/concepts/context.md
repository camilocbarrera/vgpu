---
title: "Context"
description: "init() creates the Gpu context; every surface, target, effect, and frame is created from it."
---

Everything in vgpu starts from one call. `init()` requests the WebGPU adapter and device and returns a [`Gpu`](/reference/vgpu/gpu#gpu) context. Every other object — surfaces, targets, effects, draws, frames — is created from that context, so all of them share one device.

```ts
import { init, effect, prepare, renderOnce, surface, target } from "vgpu";

const gpu = await init();

const canvas = document.querySelector("canvas")!;
const canvasSurface = surface(gpu, canvas); // the canvas you present to
const colorTarget = target(gpu, { size: [256, 256] }); // an offscreen, bindable texture
const gradient = effect(gpu, `
  @fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    return vec4f(uv, 0.4, 1.0);
  }
`);

// Warm the (renderable, target) combination — the one async step of a vgpu app
await prepare(gpu, [{ draw: gradient, target: canvasSurface }]);

// …or render once standalone, which awaits pipeline readiness by itself
await renderOnce(gpu, canvasSurface, (p) => p.draw(gradient));
```

Both halves of the render destination come from the context: a `Surface` presents to a canvas, a
`Target` is an offscreen texture you can also **bind** into a later pass. A surface is never a
binding — that is the one asymmetry between them.

## Create resources once, draw every frame

Create the context, surface, effects and prepared pipelines once, up front. Encoding a frame should
encode work, not rebuild long-lived resources and never compile a pipeline: under the default
`pendingPipelines: "throw"` policy, an unprepared combination fails fast instead of stalling inside
your frame.

The expensive objects — context, surface, effect, prepared pipeline — live outside your render code
and are reused by every frame. What changes per frame is data: `instance.set(binding, value)` writes
bytes, and a shared `uniform(gpu, …)` updates every pipeline bound to it with one write.

The context is also where the lifecycle ends: `gpu.dispose()` tears everything down (observable
synchronously through `gpu.disposed`), while a real device loss resolves `gpu.lost` and is
**terminal** — every object created from that context throws `VGPU-DEVICE-LOST` afterwards, and the
application rebuilds the whole graph from a new `init()`.
