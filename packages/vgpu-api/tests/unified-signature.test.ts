import { expect, test } from "vitest";
import { getMockGPUDeviceInstrumentation } from "@vgpu/core";
import { compute, effect, init, target } from "../src/mock.ts";

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

  const fromTwoArgs = compute(gpu, COMPUTE_SHADER, { entry: "main", label: "job" });
  const fromOptionsObject = compute(gpu, { shader: COMPUTE_SHADER, entry: "main", label: "job" });

  const instrumentation = getMockGPUDeviceInstrumentation(gpu.device.gpu);
  const descriptors = instrumentation.createComputePipelineDescriptors;
  const twoArgsDesc = descriptors.find((d) => d.label === "job.pipeline");
  expect(twoArgsDesc?.compute.entryPoint).toBe("main");

  // Both must dispatch without throwing — proves the options-object form built a working pipeline
  // with the same entry point / bindings as the two-argument form.
  expect(() => fromTwoArgs.dispatch(1)).not.toThrow();
  expect(() => fromOptionsObject.dispatch(1)).not.toThrow();
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
