# SharedUniforms

One shared uniform resource created by `uniform(gpu, values)`, bindable into many pipelines at once. It adopts the binary WGSL layout lazily from the first compatible shader binding, reuses one stable buffer across shaders, and updates every pipeline bound to it with a **single** write.

## Import

```ts
import { uniform } from "vgpu";
import type { SharedUniforms } from "vgpu";
```

## Signature

```ts
import type { Gpu } from "vgpu";

interface SharedUniforms<T extends Record<string, unknown> = Record<string, unknown>> {
  set(values: Partial<T>): void;
}

// The receiver IS the binding, so `.set()` takes one argument.
declare function uniform<T extends Record<string, unknown>>(gpu: Gpu, values: T): SharedUniforms<T>;
```

## Parameters

| Param | Type | Required | Default | Notes |
|---|---|---:|---|---|
| uniform.values | `T extends Record<string, unknown>` | ✔ | — | Initial values are cloned. Storage is **zero-initialized** (the WebGPU guarantee), so no initial value is mandatory — pass `{}`-shaped defaults only where they are meaningful. Layout and buffer are not created until the object is first bound to a reflected uniform/storage buffer. |
| shared.set.values | `Partial<T>` | ✔ | — | Deep-merges plain objects and clones arrays/typed arrays before writing current values to the adopted layout. One call, one write, visible to **every** pipeline bound to this resource. |

**Returns:** `uniform(gpu, values)` returns `SharedUniforms<T>`; `shared.set()` returns `void`.

**Throws:** `VGPU-R1-SHARED-UNIFORMS-LAYOUT-MISMATCH` when a later shader declares a structurally different layout for the same shared object; `VGPU-RING1-UNSUPPORTED` when address spaces differ, the binding is not a buffer, the binding has no host-shareable layout, or the layout is runtime-sized; `VGPU-DEVICE-LOST` after a real device loss (loss is terminal — recreate the resource on the new `Gpu`); packing may throw core validation errors for values that do not match the adopted WGSL layout.

## Examples

```ts
import { init, clock, effect, draw, frame, prepare, target, uniform } from "vgpu/mock";

const gpu = await init();
const colorTarget = target(gpu, { size: [64, 64] });
const globals = uniform(gpu, { time: 0, mouse: [0, 0] });

const wave = effect(gpu, {
  shader: `
    struct Globals { time: f32, mouse: vec2f }
    @group(0) @binding(0) var<uniform> globals: Globals;
    @fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
      return vec4f(uv, sin(globals.time) * 0.5 + 0.5, 1);
    }
  `,
  bindings: { globals },      // external resource → declared in `bindings`
});
const overlay = draw(gpu, {
  shader: `
    struct Globals { time: f32, mouse: vec2f }
    @group(0) @binding(0) var<uniform> globals: Globals;
    @vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
      var p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
      return vec4f(p[vi] * globals.time, 0, 1);
    }
    @fragment fn fs_main() -> @location(0) vec4f { return vec4f(1); }
  `,
  bindings: { globals },      // the SAME resource in a second pipeline
});

await prepare(gpu, [
  { draw: wave, target: colorTarget },
  { draw: overlay, target: colorTarget },
]);

globals.set({ time: clock(gpu).time });   // ONE write — both pipelines see the update
frame(gpu, (currentFrame) => currentFrame.pass({ target: colorTarget }, (pass) => {
  pass.draw(wave);
  pass.draw(overlay);
}));
```

```ts
import { init, uniform } from "vgpu/mock";

const gpu = await init();
const globals = uniform(gpu, { exposure: 1, tint: [1, 1, 1] });
globals.set({ exposure: 1.25 });   // update the RESOURCE, never the instance
```

## Shared resource vs instance-owned values

Ownership is fixed at construction, and each side has exactly one update spelling:

| Where the value lives | Declared as | Update | Who sees it |
|---|---|---|---|
| shared across pipelines | `bindings: { globals }` | `globals.set({ time })` | every instance bound to it |
| owned by one instance | `values: { params: … }` (or simply not declared) | `instance.set("params", { … })` | that instance only |

`uniform()` keeps its **one-argument** `.set()` because the receiver *is* the binding — that is not a
second spelling of the binding-scoped `instance.set(binding, value)` form. Calling
`instance.set("globals", …)` on an externally-bound name is an error (`VGPU-R1-EXTERNAL-BINDING`) whose
message names the resource to update instead; swapping the resource *identity* is
`instance.bind("globals", otherUniform)`.

## Notes

- **`uniform()` (singular) is the spelling for a shared uniform resource.** The plural `uniforms()` of 0.3 is a legacy alias of the same factory kept only for the transition; new code uses `uniform(gpu, values)`.
- Storage is zero-initialized, so a shader reads zeros until the first `.set()` — no mandatory initial values, and no "never set" error class for a shared resource.
- The first shader to bind the object chooses the WGSL layout. Keep struct member names/types/order aligned for every later shader that reuses it.
- Use a shared `uniform()` for values like time, mouse, camera, exposure, and viewport data consumed by many passes: one write instead of N.
- If one shader needs a different layout, create a second `uniform(gpu, …)` object rather than mutating the first layout.
- Multiple writes before a submit do **not** snapshot: passes sharing the resource within one frame all see the last value written, because the writes land in the same buffer and all of them land before the single `queue.submit()`.
- A shared resource is device-local: after a device loss it is gone like everything else, and the replacement `Gpu` needs a new `uniform()` — reuse the CPU-side values, not the object.
- **See also:** `uniform`, `Effect.set`, `Draw.set`, `prepare`, `Uniform`, `StructuredUniform`.
