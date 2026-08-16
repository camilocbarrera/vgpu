---
title: "Performance model"
description: "vgpu's public API is organized around stable identities and explicit costs."
---

## Binding ownership

Ownership is fixed **at construction**, not by call order. A binding listed in `values` (or declared in the WGSL and listed in neither bag) is **instance-owned**: `instance.set(binding, value)` writes its bytes. A binding listed in `bindings` is **external**: its identity is swapped with `instance.bind(binding, resource)`, and a shared resource is updated on the resource itself (`globals.set({ time })`). `.set()` on an external binding fails with `VGPU-R1-EXTERNAL-BINDING`, naming the resource to update instead — the order-dependent ownership latch of 0.3 (`VGPU-R1-OWNERSHIP-FLIP`) is unrepresentable in this model. Owned→external transitions are not supported: recreate the instance, which is cheap because its input is a plain options object.

## Identity cache

Bind groups are cached by resource identity. `.set()` writes bytes and **never** rebuilds a bind group; `.bind()` changes identity and rebuilds exactly the affected group (it dedupes, so re-binding the same resource is free). A struct binding takes a partial and vgpu merges it CPU-side into a **single** buffer write, so N member updates cost one upload.

## Bundle staleness

Bundles freeze a logical command list plus bind-group identities. Buffer contents may change freely, and replay targets may resize as long as their render signature (color formats, depth format, sample count) matches. An identity change — a `.bind()` swap or a captured `Target` recreating its texture — moves the bundle to `stale`; the recording stays valid, so `prepare(gpu, [{ bundle }])` re-encodes it. `rebuild()` is only for when the command list itself changes. Replaying a non-`ready` bundle follows the `pendingPipelines` chain: by default `VGPU-PIPELINE-PENDING` / `VGPU-R3-BUNDLE-STALE` rather than a hidden hot-path re-encode.

## Claimed groups

`draw.group(group, bindGroup)` claims an entire reflected group. vgpu validates the group layout and forbids `set()` into that group. Dynamic offsets are passed at encode time, and the raw `offsets` knob is exclusively for claimed groups — vgpu knows neither their layout nor their sizes:

```ts
import { draw, frame, init, prepare, target } from "vgpu/mock";

const gpu = await init();
const screen = target(gpu, { size: [64, 64] });
const drawable = draw(gpu, { shader: `@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1); }` });
const offset = 256;

await prepare(gpu, [{ draw: drawable, target: screen }]);

// ---cut---
frame(gpu, (f) => {
  f.pass(screen, (p) => p.draw(drawable, { offsets: { 1: [offset] } }));
});
```

## Cost model defaults

- Warm every combination you will encode with `await prepare(gpu, [{ draw, target }, { compute }, { bundle }])`. Under the default `pendingPipelines: "throw"` a synchronous encode never compiles implicitly, so a stall is always a choice (`"sync"`) and never an accident.
- Share globals with one `uniform(gpu, { … })`: one write updates every pipeline bound to it.
- Prefer bytes over identity: `.set()` in the hot loop, `.bind()` only when the resource really changes.
- Use `instances` for repeated geometry.
- Use ping-pong for iterative read/write resources, and `.bind()` the swapped halves.
- Put depth, `sampleCount` and formats on targets (and surfaces), not in global state — they are the pipeline signature, so changing one invalidates the prepared combinations that used it.
- Record static draw lists into a bundle, `prepare()` it once, and replay it.
- Keep GPU completion out of the frame path: `await gpu.settled()` is a snapshot for tests, readbacks and teardown, not a per-frame call.
