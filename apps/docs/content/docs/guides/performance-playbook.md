---
title: "Performance playbook: write fast vgpu by default"
description: "This guide is for LLMs and humans writing shaders. Treat these as default shapes, not late-stage optimizations: each **After** snippet is the pattern to copy when the situation matches."
---

## 1. Bundles / replay (`bundle()` + `p.bundles`)

Use when static draws repeat every frame. Bundles freeze commands, bind groups, target formats, sample count, and attachment identity; they do **not** freeze buffer contents.

Before:
```text
frameLoop(gpu, (f) => f.pass({ target: scene }, (p) => {
  p.draw(floor);
  p.draw(walls);
  p.draw(player);
}));
```
After:
```text
const staticScene = bundle(gpu, { target: scene }, (b) => {
  b.draw(floor);
  b.draw(walls);
});
await prepare(gpu, [{ bundle: staticScene }, { draw: player, target: scene }]);
frameLoop(gpu, (f) => f.pass({ target: scene }, (p) => {
  p.bundles(staticScene);
  p.draw(player);
}));
```
Default: bundle static work once, `prepare()` it, and replay with `p.bundles(...)`. Recording compiles nothing (the bundle is born `"pending-pipelines"`); `prepare(gpu, [{ bundle }])` compiles the missing pipelines and encodes the native bundle. Byte updates keep it `"ready"`; an identity swap moves it to `"stale"`, which `prepare()` re-encodes from the retained recording.

## 2. Pipeline pre-warm (`prepare`)

Use before the first visible frame or route transition — and in fact always: under the default `pendingPipelines: "throw"` a synchronous encode never compiles implicitly, so an unprepared combination throws `VGPU-PIPELINE-PENDING` instead of hitching. Readiness belongs to the **combination** `(renderable, target signature)`, so one prepared target says nothing about another.

Before:
```text
const cube = draw(gpu, { shader: LIT_WGSL, geometry: geometry(gpu, box()) });
frame(gpu, (f) => f.pass({ target: scene }, (p) => p.draw(cube))); // VGPU-PIPELINE-PENDING
```
After:
```text
const scene = target(gpu, { size: [256, 256], format: "rgba16float", depth: true, sampleCount: 4 });
const cube = draw(gpu, { shader: LIT_WGSL, geometry: geometry(gpu, box()) });
await prepare(gpu, [{ draw: cube, target: scene }]);
frame(gpu, (f) => f.pass({ target: scene }, (p) => p.draw(cube)));
```
Default: `await prepare(gpu, [{ draw, target }])` for every combination a frame will encode — plus `{ compute }` for kernels and `{ bundle }` for bundles. It is idempotent, free on an already-warm combination, and it rejects with `VGPU-PREPARE-FAILED` listing every failure. There is no per-object `compile()` / `compileSync()` / `pipelineFor()` and no `targets: [...]` precompile list; inline compilation is reachable only as the explicit `pendingPipelines: "sync"` opt-in, where the stall is visible at the call site.

## 3. Manual group claim + dynamic offsets (`draw.group`)

Use for hundreds or thousands of objects that share one shader and one bind-group layout. `draw.group()` claims a reflected group; offsets travel per draw call.

Before:
```text
for (const obj of objects) {
  cube.set("model", { model: obj.model }); // one buffer write per object, every frame
  p.draw(cube);
}
```
After:
```text
import { UniformPool, type UniformLayout } from "vgpu/core";

type ObjectUniforms = { model: Float32Array };
const objectLayout: UniformLayout<ObjectUniforms> = {
  size: 64,
  bindGroupLayout: cube.layout(1, { dynamicOffsets: true }),
  encode(value, dst, byteOffset) {
    new Float32Array(dst, byteOffset, 16).set(value.model);
  },
};
const pool = new UniformPool(gpu.device, { capacityBytes: 1 << 20 });
const slot = pool.alloc(objectLayout);
cube.group(1, slot.bindGroup);

frameLoop(gpu, (f) => {
  pool.beginFrame(clock(gpu).frameCount);
  f.pass({ target: scene }, (p) => {
    for (const obj of objects) {
      const offset = slot.push({ model: obj.model });
      p.draw(cube, { offsets: { 1: [offset] } });
    }
  });
  pool.endFrame();
});
```
Default: for many per-object uniforms, allocate a `UniformPool` slot with an `encode(...)` function, call `pool.beginFrame(...)`, push values, draw with offsets, then `pool.endFrame()` before the frame submits.

## 4. `set()` in-place

Use for animated JS values. Ownership is fixed **at construction**: bindings in `values` (or declared in the WGSL and listed in neither bag) are instance-owned and written with `.set(binding, value)`; bindings in `bindings` are external, swapped with `.bind(binding, resource)` and updated on the resource itself. `.set()` writes bytes and never rebuilds a bind group, so bundles stay `"ready"`.

Before:
```text
const wave = effect(gpu, { shader: WAVE_WGSL, values: { params: { time: 0, speed: 2 } } });
frameLoop(gpu, (frame) => {
  wave.set("params", { time: clock(gpu).time, speed: 2 }); // re-writes `speed` every frame
  frame.pass(target, wave);
});
```
After:
```text
const wave = effect(gpu, { shader: WAVE_WGSL, values: { params: { time: 0, speed: 2 } } });
frameLoop(gpu, (frame) => {
  wave.set("params", { time: clock(gpu).time });  // partial: `speed` keeps its value
  frame.pass(target, wave);
});
```
Default: create once; update the binding that changed with `.set(binding, value)`. A struct takes a
partial and N fields collapse into **one** buffer write. `set()` performs no equality check — a value
written every frame is uploaded every frame, so hoist static and resize-class values out of the render
loop. Never `.set()` an external binding: that throws `VGPU-R1-EXTERNAL-BINDING` and names the
resource to update instead.

## 5. Bake static inputs once

Use when a heavy pass produces a texture that does not change every frame.

Before:
```text
frameLoop(gpu, (f) => {
  f.pass({ target: baked }, (p) => p.draw(heavyScene));  // re-rendered every frame
  post.set("params", { texel: baked.texelSize });
  f.pass({ target: screen }, (p) => p.draw(post));
});
```
After:
```text
const post = effect(gpu, {
  shader: POST_WGSL,
  values:   { params: { texel: baked.texelSize } },  // instance-owned → .set()
  bindings: { src: baked },                          // bind the TARGET, not baked.color
});
await prepare(gpu, [{ draw: heavyScene, target: baked }, { draw: post, target: screen }]);

frame(gpu, (f) => f.pass({ target: baked }, (p) => p.draw(heavyScene)));  // bake once
frameLoop(gpu, (f) => f.pass({ target: screen }, (p) => p.draw(post)));
```
Default: if an input is static, bake it outside the loop with one `frame(gpu, ...)`. Bind the `Target`
rather than `target.color`: a target re-binds its new texture identity when it is recreated on resize,
while a `Texture` snapshot goes **silently** stale.

## 6. Instancing (`instances`, `vertices`)

Use for N copies of the same geometry. `DrawOptions.instances/vertices/firstInstance` set defaults; `DrawCallOptions.instances/vertices/firstVertex/firstInstance` override per call. `instances: 0` is valid; indexed geometries ignore `vertices` and `firstVertex`.

Before:
```text
for (let i = 0; i < COUNT; i++) {
  particles.set("params", { particleIndex: i });
  p.draw(particles);
}
```
After:
```text
const particles = draw(gpu, {
  shader: PARTICLE_WGSL,
  instances: COUNT,
  vertices: 6,
  bindings: { particleBuffer },      // external storage buffer, identity fixed here
});
await prepare(gpu, [{ draw: particles, target: scene }]);
frameLoop(gpu, (f) => f.pass({ target: scene }, (p) => p.draw(particles)));
```
Default: one draw with `instances` beats N draw calls.

## 7. `uniform(gpu)` shared values

Use when many shaders consume the same time, camera, mouse, or exposure values.

Before:
```text
const time = clock(gpu);
wave.set("globals", { time: time.time, mouse });   // three instances, three writes
blur.set("globals", { time: time.time, mouse });
post.set("globals", { time: time.time, mouse });
```
After:
```text
const globals = uniform(gpu, { time: 0, mouse: [0, 0] });
const wave = effect(gpu, { shader: WAVE_WGSL, bindings: { globals } });
const blur = effect(gpu, { shader: BLUR_WGSL, bindings: { globals } });
await prepare(gpu, [{ draw: wave, target }, { draw: blur, target }]);
frameLoop(gpu, (frame) => {
  globals.set({ time: clock(gpu).time, mouse });   // ONE write, every pipeline sees it
  frame.pass(target, (pass) => {
    pass.draw(wave);
    pass.draw(blur);
  });
});
```
Default: shared values belong in one `uniform(gpu, …)` object, declared in `bindings` and updated on
the resource. Its storage is zero-initialized, and `uniform()` keeps a one-argument `.set()` because
the receiver *is* the binding — the plural `uniforms()` is a legacy alias of the same factory.

## 8. Ping-pong (`pingPong`) without churn + two bundles

Use for iterative effects. Ping-pong keeps two stable identities, so bind-group caches can reuse them.

Before:
```text
frameLoop(gpu, (f) => {
  const tmp = target(gpu, { size: [256, 256], format: "rgba16float" });  // new identity every frame
  sim.bind("src", previous);
  f.pass({ target: tmp }, (p) => p.draw(sim));
  previous = tmp;
});
```
After:
```text
const state = pingPong(gpu, 512, 512, { format: "rgba16float" });
const sim = effect(gpu, { shader: SIM_WGSL, bindings: { src: state.read } });
const even = bundle(gpu, { target: state.write }, (b) => b.draw(sim));
state.swap();
sim.bind("src", state.read);
const odd = bundle(gpu, { target: state.write }, (b) => b.draw(sim));
state.swap();
sim.bind("src", state.read);
await prepare(gpu, [{ bundle: even }, { bundle: odd }]);
let parity = 0;
frameLoop(gpu, (f) => {
  f.pass({ target: state.write }, (p) => p.bundles(parity === 0 ? even : odd));
  state.swap();
  parity ^= 1;
});
```
Default: create ping-pong resources once and `.bind()` the swapped halves — two stable identities, so
bind-group caches reuse them. If you bundle, record both parity cases, `prepare()` both, and replay the
matching one. Recording captures the identity current at record time, so each parity bundle keeps its
own; re-`prepare()` a bundle only when it reports `"stale"`.

## 9. MSAA/depth in the target

Use for 3D anti-aliasing and depth testing. Resolution, depth, color format, and sample count are target state — and the last three **are** the pipeline signature. `sampleCount` is the one spelling for multisampling, on both `target()` and `surface()`.

Before:
```text
const scene = target(gpu, { size: [256, 256], format: "rgba8unorm" });
const cube = draw(gpu, { shader: LIT_WGSL, geometry: geometry(gpu, box()) });
```
After:
```text
const scene = target(gpu, { size: [256, 256], format: "rgba16float", depth: true, sampleCount: 4 });
const cube = draw(gpu, { shader: LIT_WGSL, geometry: geometry(gpu, box()) });
await prepare(gpu, [{ draw: cube, target: scene }]);
frameLoop(gpu, (f) => f.pass({ target: scene, clear: [0, 0, 0, 1] }, (p) => p.draw(cube)));
```
Default: put depth and `sampleCount` on the target (or the surface — `surface(gpu, canvas, { depth: true, sampleCount: 4 })` owns its own attachments); do not invent global render settings. A resize that preserves color format, depth format and sample count invalidates no prepared pipeline and leaves bundles `"ready"`; changing any of the three does.

## 10. Back-face culling (`cull: "back"`)

Use for closed geometries. With the default `cull: "none"`, triangles facing away from the camera still rasterize; culling them drops roughly half of a closed geometry's fragment work.

Before:
```text
const cube = draw(gpu, { shader: LIT_WGSL, geometry: geometry(gpu, box()) });
```
After:
```text
const cube = draw(gpu, { shader: LIT_WGSL, geometry: geometry(gpu, box()), cull: "back" });
```
Default: `cull: "back"` for closed geometries. Keep `"none"` for planes, alpha-tested foliage, and anything seen from both sides.

## 11. Occlusion culling (`visibility()`)

Use for many-object scenes with large occluders. Query a cheap proxy — a bounding box — and skip the expensive draw when the GPU confirmed it was hidden.

Before:
```text
f.pass({ target: scene }, (p) => {
  p.draw(world);
  p.draw(statue); // full cost even when a wall hides it
});
```
After:
```text
const vis = visibility(gpu);
const qStatue = vis.query("statue");
await prepare(gpu, [
  { draw: world, target: scene },
  { draw: statueProxy, target: scene },
  { draw: statue, target: scene },
]);
frameLoop(gpu, (f) => {
  f.pass({ target: scene, visibility: vis }, (p) => {
    p.draw(world); // occluders first
    p.occlusion(qStatue, statueProxy); // cheap bounding-box proxy
    if (!qStatue.hidden) p.draw(statue);
  });
});
```
Default: draw occluders first, query proxies, condition real draws on `hidden`. Results lag one or two frames and `hidden` stays `false` until a query confirms zero passing samples, so the fallback is always to draw. The pass target needs `depth: true`.

## 12. Indirect draws and dispatches (`indirect`)

Use when the GPU decides the counts — compute-driven particles, culled instance lists. Reading counts back to the CPU stalls on a round-trip; `indirect` keeps them on the GPU.

Before:
```text
const data = await counts.read(); // GPU-to-CPU round-trip, a frame late
p.draw(particles, { instances: decodeCount(data) });
```
After:
```text
const args = storage(gpu, 16, { indirect: true });
await prepare(gpu, [{ compute: emit }, { draw: particles, target: scene }]);
frameLoop(gpu, (f) => {
  f.compute(emit, Math.ceil(COUNT / 64));  // compute writes the draw arguments into `args`…
  f.pass({ target: scene }, (p) => p.draw(particles, { indirect: args })); // …same encoder, same submit
});
```
Default: counts produced on the GPU stay on the GPU. Put the producing dispatch in the **same frame**
as the consuming draw — `f.compute()` shares the frame's encoder, so program order is execution order
and no second submit is needed. For standalone one-shot work outside a frame there is
`await emit.dispatchOnce(n)`, which owns its encoder and awaits pipeline readiness on its own. The same
option shape drives compute: `await sim.dispatchOnce({ indirect: args })`.

## 13. Time passes before optimizing (`timer()`)

Use before reaching for any pattern above. CPU timers see encoding only — encoders record commands, the GPU runs them later — so a "slow pass" verdict needs GPU timestamps.

Before:
```text
const t0 = performance.now();
frame(gpu, (f) => f.pass({ target: scene }, (p) => p.draw(world)));
const ms = performance.now() - t0; // encode + submit time, not GPU cost
```
After:
```text
const gpu = await init({ requiredFeatures: ["timestamp-query"] });
const spans = timer(gpu);
spans.onResults((results) => console.log(`main ${results.main}ms`));
await prepare(gpu, [{ draw: world, target: scene }]);
frameLoop(gpu, (f) => f.pass({ target: scene, timer: spans.span("main") }, (p) => p.draw(world)));
```
Default: attach `timer.span(name)` to each pass you plan to touch and optimize the worst milliseconds first. Open `measuring` for what else to measure.
