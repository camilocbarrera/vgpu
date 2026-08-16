---
title: "Authoring shaders for performance"
description: "Write WGSL so reflection can build stable layouts. Bindings should be explicit, structs should be host-shareable, and hot paths should avoid per-frame resource identity changes."
---

Layouts mirror what each entry point statically uses, the same way WebGPU's
`layout: 'auto'` behaves: bindings an entry never touches are omitted,
visibility covers only the stages that actually read a binding, and sampled
`f32` textures are declared filterable exactly when the shader samples them
through a filtering sampler (`textureLoad`-only access stays
unfilterable, keeping `rgba32float` readbacks valid). Mismatches fail eagerly
with structured errors — `VGPU-LIMIT-STORAGE-VERTEX`/`-FRAGMENT` when a
storage binding would exceed a device's per-stage limits, and
`VGPU-SET-TEXTURE-FILTERABILITY` when a non-filterable format meets a
filtering sampler — instead of surfacing as native pipeline failures.

## WGSL defaults

```wgsl
struct Globals {
  time: f32,
  mouse: vec2f,
  enabled: u32,
}
@group(0) @binding(0) var<uniform> globals: Globals;
```

- Use `u32` instead of `bool` in host-written uniforms; encode false/true as `0`/`1`.
- Put target resolution in a uniform value sourced from `target.size` or `target.texelSize`.
- Keep imported WGSL modules binding-free. Modules may export structs/functions/constants; entry shaders own `@group/@binding` declarations.
- Prefer storage buffers plus `instances` for many similar particles or sprites.

## JavaScript defaults

```ts
import { clock, draw, frameLoop, init, prepare, target, uniform } from "vgpu/mock";

const gpu = await init();
const screen = target(gpu, { size: [64, 64] });
const mouse: [number, number] = [0, 0];
const WGSL = `
  struct Globals { time: f32, mouse: vec2f, enabled: u32 }
  @group(0) @binding(0) var<uniform> globals: Globals;
  @vertex fn vs_main() -> @builtin(position) vec4f { return vec4f(globals.time); }
  @fragment fn fs_main() -> @location(0) vec4f { return vec4f(1); }
`;

// ---cut---
const globals = uniform(gpu, { time: 0, mouse: [0, 0], enabled: 1 });
const mesh = draw(gpu, { shader: WGSL, bindings: { globals } });  // external → declared once

await prepare(gpu, [{ draw: mesh, target: screen }]);             // warm the combination

frameLoop(gpu, (f) => {
  globals.set({ time: clock(gpu).time, mouse });                  // one write, every pipeline
  f.pass(screen, (p) => p.draw(mesh));
});
```

Use the performance playbook before writing a new shader: bundles for static draws, `prepare()` for
pipeline readiness, `draw.group()` for many objects, one shared `uniform(gpu, …)` for shared state,
ping-pong for iterative effects, and target-owned depth/`sampleCount`. Note where each update lands:
an **external** binding (`bindings`) is updated on the resource, an **instance-owned** one (`values`,
or simply undeclared) with `instance.set(binding, value)`.
