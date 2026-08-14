import { expect, test } from "vitest";
import { getMockGPUDeviceInstrumentation } from "@vgpu/core";
import { compute, draw, effect, init, storage, target } from "../src/mock.ts";
import { drawReflection } from "../src/draw.ts";
import { effectDraw } from "../src/effect.ts";

const EFFECT_SHADER = `
@fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f { return vec4f(uv, 0.0, 1.0); }
`;

const COMPUTE_SHADER = `
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3u) { data[id.x] = 1.0; }
`;

test("effect(gpu, { shader, ... }) shares the exact pipeline of effect(gpu, source, opts)", async () => {
  const gpu = await init();
  const colorTarget = target(gpu, { size: [2, 2] });

  const fromTwoArgs = effect(gpu, EFFECT_SHADER, { blend: "additive" });
  fromTwoArgs.draw(colorTarget);
  const afterFirst = getMockGPUDeviceInstrumentation(gpu.device.gpu).createRenderPipelineDescriptors.length;

  const fromOptionsObject = effect(gpu, { shader: EFFECT_SHADER, blend: "additive" });
  fromOptionsObject.draw(colorTarget);
  const instrumentation = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  // Same shader + same blend => same pipeline key => no second createRenderPipeline call.
  expect(instrumentation.createRenderPipelineDescriptors.length).toBe(afterFirst);
  expect(instrumentation.createRenderPipelineDescriptors.at(-1)?.fragment?.targets?.[0]).toMatchObject({
    blend: { color: { srcFactor: "one", dstFactor: "one", operation: "add" }, alpha: { srcFactor: "one", dstFactor: "one", operation: "add" } },
  });
  gpu.dispose();
});

test("compute(gpu, { shader, entry }) produces a Compute equivalent to compute(gpu, source, { entry })", async () => {
  const gpu = await init();
  const instrumentation = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  // compute() no longer compiles a pipeline at construction (contract #4, next/0.4 compute-async);
  // dispatch() compiles it lazily-sync the first time it runs. Bind data and dispatch before
  // inspecting the descriptor.
  const fromTwoArgs = compute(gpu, COMPUTE_SHADER, { entry: "main", label: "job" });
  fromTwoArgs.set({ data: storage(gpu, 16) });
  expect(() => fromTwoArgs.dispatch(1)).not.toThrow();
  const twoArgsDesc = instrumentation.createComputePipelineDescriptors.at(-1);
  expect(twoArgsDesc?.label).toBe("job.pipeline");
  expect(twoArgsDesc?.compute.entryPoint).toBe("main");

  const fromOptionsObject = compute(gpu, { shader: COMPUTE_SHADER, entry: "main", label: "job" });
  fromOptionsObject.set({ data: storage(gpu, 16) });
  expect(() => fromOptionsObject.dispatch(1)).not.toThrow();
  // Assert directly on the descriptor the options-object form itself produced, not just on parity
  // with the two-argument form's descriptor (kills a mutant that drops opts on the object-form path).
  const objectFormDesc = instrumentation.createComputePipelineDescriptors.at(-1);
  expect(objectFormDesc?.label).toBe("job.pipeline");
  expect(objectFormDesc?.compute.entryPoint).toBe("main");
  gpu.dispose();
});

test("effect(gpu, { shader, version, wgsl, blend }) picks shader over a spurious version/wgsl and honors sibling options", async () => {
  const gpu = await init();
  const colorTarget = target(gpu, { size: [2, 2] });
  const shaderWithBinding = `
struct Params { value: f32 }
@group(0) @binding(0) var<uniform> params: Params;
@fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f { return vec4f(uv, params.value, 1.0); }
`;
  const decoyWgsl = EFFECT_SHADER;

  // A real ShaderSource can never carry `shader`, so a spread like `{ ...artifact, shader, blend }`
  // must still be recognized as the options-object form: `shader` wins over the spurious `version`/`wgsl`.
  const fx = effect(gpu, { shader: shaderWithBinding, version: 1, wgsl: decoyWgsl, blend: "additive" } as never);

  // Only `shaderWithBinding` declares the `params` uniform; if the decoy `wgsl` had won, this binding
  // would not exist.
  expect(drawReflection(effectDraw(fx)).bindings.map((binding) => binding.name)).toContain("params");

  fx.set({ params: { value: 1 } });
  fx.draw(colorTarget);
  const desc = getMockGPUDeviceInstrumentation(gpu.device.gpu).createRenderPipelineDescriptors.at(-1);
  expect(desc?.fragment?.targets?.[0]).toMatchObject({
    blend: { color: { srcFactor: "one", dstFactor: "one", operation: "add" }, alpha: { srcFactor: "one", dstFactor: "one", operation: "add" } },
  });
  gpu.dispose();
});

test("compute(gpu, { shader, version, wgsl, entry, label }) picks shader over a spurious version/wgsl and honors entry/label", async () => {
  const gpu = await init();
  // The decoy has no `main` entry point: if `version`/`wgsl` won over `shader`, requesting `entry:
  // "main"` against the decoy would fail to resolve an entry point at construction time.
  const decoyWithoutMainEntry = `
@compute @workgroup_size(1) fn other() {}
`;

  const job = compute(gpu, { shader: COMPUTE_SHADER, version: 1, wgsl: decoyWithoutMainEntry, entry: "main", label: "amb" } as never);
  const instrumentation = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  // compute() no longer compiles a pipeline at construction (contract #4, next/0.4 compute-async);
  // dispatch() compiles it lazily-sync the first time it runs.
  job.set({ data: storage(gpu, 16) });
  expect(() => job.dispatch(1)).not.toThrow();

  const desc = instrumentation.createComputePipelineDescriptors.at(-1);
  expect(desc?.label).toBe("amb.pipeline");
  expect(desc?.compute.entryPoint).toBe("main");
  gpu.dispose();
});

test("effect(gpu, { blend }) without options.shader throws an actionable error, not malformedShaderSourceError", async () => {
  const gpu = await init();

  expect(() => effect(gpu, { blend: "additive" } as never)).toThrowError(
    /effect\(gpu, options\) requires options\.shader; use effect\(gpu, source, opts\) for the two-argument form, or effect\(gpu, \{ shader, \.\.\. \}\)\./,
  );
  gpu.dispose();
});

test("compute(gpu, { entry }) without options.shader throws an actionable error, not malformedShaderSourceError", async () => {
  const gpu = await init();

  expect(() => compute(gpu, { entry: "main" } as never)).toThrowError(
    /compute\(gpu, options\) requires options\.shader; use compute\(gpu, source, opts\) for the two-argument form, or compute\(gpu, \{ shader, \.\.\. \}\)\./,
  );
  gpu.dispose();
});

test("effect(gpu, shaderSourceArtifact) still works as a shorthand, not confused with the options-object form", async () => {
  const gpu = await init();
  const colorTarget = target(gpu, { size: [2, 2] });

  const shorthand = effect(gpu, { version: 1, wgsl: EFFECT_SHADER });
  expect(() => shorthand.draw(colorTarget)).not.toThrow();
  gpu.dispose();
});
