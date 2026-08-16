import { expect, test } from "vitest";
import { init, compute, draw, effect } from "../src/mock.ts";
import { drawReflection } from "../src/draw.ts";
import { effectDraw } from "../src/effect.ts";

const FRAGMENT = `
struct Params { value: f32 }
@group(0) @binding(0) var<uniform> params: Params;
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
  return vec4f(uv, params.value, 1.0);
}
`;

const DRAW = `
struct Params { value: f32 }
@group(0) @binding(0) var<uniform> params: Params;
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(pos[vi], 0.0, 1.0);
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(params.value); }
`;

const COMPUTE = `
struct Params { value: f32 }
@group(0) @binding(0) var<uniform> params: Params;
@compute @workgroup_size(1) fn main() { _ = params.value; }
`;

test("effect(gpu, ...) accepts string and ShaderSource with identical reflection", async () => {
  const gpu = await init();
  const fromString = effect(gpu, { shader: FRAGMENT, label: "shader" });
  const fromArtifact = effect(gpu, { shader: { version: 1, wgsl: FRAGMENT }, label: "shader" });

  expect(drawReflection(effectDraw(fromArtifact)).bindings.map(({ name, mangledName, group, binding, kind }) => ({ name, mangledName, group, binding, kind })))
    .toEqual(drawReflection(effectDraw(fromString)).bindings.map(({ name, mangledName, group, binding, kind }) => ({ name, mangledName, group, binding, kind })));
  gpu.dispose();
});

test("draw(gpu, ...) accepts ShaderSource and keeps Draw internals string-only", async () => {
  const gpu = await init();
  const drawable = draw(gpu, { shader: { version: 1, wgsl: DRAW }, label: "artifact-draw" });

  expect(drawReflection(drawable).bindings[0]).toMatchObject({ name: "params", group: 0, binding: 0 });
  gpu.dispose();
});

test("compute(gpu, ...) accepts ShaderSource", async () => {
  const gpu = await init();
  const job = compute(gpu, { shader: { version: 1, wgsl: COMPUTE }, label: "artifact-compute" });

  job.set({ params: { value: 1 } });
  gpu.dispose();
});

// T04-01 (unified-signature): a bare object without "version" is now dispatched as the additive
// single-argument options form (`effect(gpu, { shader, ... })`), not as a malformed ShaderSource —
// so a shape like `{ wgsl }` (no `version`, no `shader`) surfaces the actionable "requires
// options.shader" error instead of VGPU-SHADER-SOURCE-INVALID. An object WITH "version" still goes
// through the ShaderSource path and can still raise VGPU-SHADER-SOURCE-INVALID (see the "unsupported
// ShaderSource version" case below), so that error code is still reachable, just not from this shape.
test("a bare object without version or shader throws the options-form actionable error, not VGPU-SHADER-SOURCE-INVALID", async () => {
  const gpu = await init();

  expect(() => effect(gpu, { wgsl: FRAGMENT } as never)).toThrowError(
    "effect(gpu, options) requires options.shader; use effect(gpu, source, opts) for the two-argument form, or effect(gpu, { shader, ... }).",
  );
  gpu.dispose();
});

test("unsupported ShaderSource version throws VGPU-SHADER-SOURCE-INVALID", async () => {
  const gpu = await init();

  expect(() => effect(gpu, { version: 2, wgsl: FRAGMENT } as never)).toThrowError(
    "VGPU-SHADER-SOURCE-INVALID: unsupported ShaderSource v2; expected v1. Fix: update vgpu or regenerate it.",
  );
  gpu.dispose();
});

// T04-01 (unified-signature): the generic malformed-shader-source branch (toWgsl on a non-string,
// non-{version,wgsl} input) is still fully reachable — restoring coverage that the "bare object"
// test above no longer exercises now that a bare `{ wgsl }` object is classified as the
// options-form-missing-shader case instead.
test("VGPU-SHADER-SOURCE-INVALID is still reachable: draw(gpu, { shader }) with a malformed shader field", async () => {
  const gpu = await init();

  expect(() => draw(gpu, { shader: { wgsl: DRAW } as never })).toThrowError(
    /VGPU-SHADER-SOURCE-INVALID: expected WGSL or \{ version, wgsl \}, got .* Fix: configure @vgpu\/wgsl loader-vite or loader-webpack\./,
  );
  gpu.dispose();
});

test("VGPU-SHADER-SOURCE-INVALID is still reachable: effect(gpu, null | number) — not a string, not object-shaped", async () => {
  const gpu = await init();

  expect(() => effect(gpu, null as never)).toThrowError(/VGPU-SHADER-SOURCE-INVALID/);
  expect(() => effect(gpu, 7 as never)).toThrowError(/VGPU-SHADER-SOURCE-INVALID/);
  gpu.dispose();
});

test("VGPU-SHADER-SOURCE-INVALID is still reachable: a real ShaderSource shape with a malformed wgsl field", async () => {
  const gpu = await init();

  expect(() => effect(gpu, { version: 1, wgsl: 123 } as never)).toThrowError(
    /VGPU-SHADER-SOURCE-INVALID: expected WGSL or \{ version, wgsl \}, got .* Fix: configure @vgpu\/wgsl loader-vite or loader-webpack\./,
  );
  gpu.dispose();
});
