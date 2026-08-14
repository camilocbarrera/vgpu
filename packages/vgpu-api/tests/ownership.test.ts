import { expect, test, vi } from "vitest";
import { getMockGPUDeviceInstrumentation } from "@vgpu/core";
import { compute } from "../src/compute.ts";
import { draw } from "../src/draw.ts";
import { effect } from "../src/effect.ts";
import { frame } from "../src/frame.ts";
import { init } from "../src/mock.ts";
import { storage } from "../src/storage.ts";
import { target } from "../src/target-offscreen.ts";
import { uniform, uniforms } from "../src/uniforms.ts";

/**
 * Ownership contracts (#9, #10, #11 of issue #320 rev6 §6): `{ values, bindings }` fixes ownership at
 * construction, `.set(binding, value)` writes bytes, `.bind(binding, resource)` swaps identity.
 *
 * The assertions below are semantic (which group was rebuilt, which buffer received the bytes, which
 * error code fired) rather than raw call tallies wherever a tally would not pin the behavior down.
 */

const OWNERSHIP_SHADER = `
struct Params { intensity: f32, tint: vec3f }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var src: texture_2d<f32>;
struct Globals { time: f32 }
@group(1) @binding(0) var<uniform> globals: Globals;
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
  return textureLoad(src, vec2u(0, 0), 0) * params.intensity * globals.time + vec4f(params.tint, 1.0) + vec4f(uv, 0.0, 0.0);
}
`;

const MAT_SHADER = `
@group(0) @binding(0) var<uniform> mvp: mat4x4f;
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f { return mvp * vec4f(uv, 0.0, 1.0); }
`;

const CAMERA_SHADER = `
struct Camera { viewProjection: mat4x4f, exposure: f32 }
@group(0) @binding(0) var<uniform> camera: Camera;
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
  return camera.viewProjection * vec4f(uv, 0.0, camera.exposure);
}
`;

const AMBIGUOUS_SHADER = `
struct A { time: f32 }
struct B { time: f32 }
@group(0) @binding(0) var<uniform> a: A;
@group(0) @binding(1) var<uniform> b: B;
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f { return vec4f(a.time, b.time, uv); }
`;

const SIM_SHADER = `
struct Sim { dt: f32 }
@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read_write> particles: array<f32>;
@compute @workgroup_size(1) fn main() { particles[0] = sim.dt; }
`;

const GLOBALS_ONLY_SHADER = `
struct Globals { time: f32 }
@group(0) @binding(0) var<uniform> globals: Globals;
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f { return vec4f(globals.time, uv, 1.0); }
`;

function codeOf(fn: () => unknown): string {
  try { fn(); } catch (error) { return (error as { code?: string }).code ?? String(error); }
  throw new Error("expected a VGPUError, none was thrown");
}

function messageOf(fn: () => unknown): string {
  try { fn(); } catch (error) { return `${(error as Error).message} ${(error as { fix?: string }).fix ?? ""}`; }
  throw new Error("expected a VGPUError, none was thrown");
}

function groupLabels(mock: ReturnType<typeof getMockGPUDeviceInstrumentation>): string[] {
  return mock.createBindGroupDescriptors.map((descriptor) => descriptor.label ?? "");
}

// ---------------------------------------------------------------------------
// Contract #9 — .set() is bytes, .bind() is identity
// ---------------------------------------------------------------------------

test("contract #9: .set() on a value-owned binding never rebuilds a bind group", async () => {
  const gpu = await init();
  const source = target(gpu, { size: [4, 4] });
  const globals = uniform(gpu, { time: 0 });
  const screen = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, {
    shader: OWNERSHIP_SHADER,
    label: "fx",
    values: { params: { intensity: 1 } },
    bindings: { src: source, globals },
  });
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  frame(gpu, (f) => f.pass({ target: screen }, (p) => p.draw(fx)));
  const afterFirstFrame = mock.calls.createBindGroup;
  expect(afterFirstFrame).toBe(2);

  fx.set("params", { intensity: 0.5 });
  fx.set("params", { intensity: 0.25 });
  frame(gpu, (f) => f.pass({ target: screen }, (p) => p.draw(fx)));

  expect(mock.calls.createBindGroup).toBe(afterFirstFrame);
  gpu.dispose();
});

test("contract #9: .bind() rebuilds exactly the affected group", async () => {
  const gpu = await init();
  const source = target(gpu, { size: [4, 4] });
  const next = target(gpu, { size: [4, 4] });
  const globals = uniform(gpu, { time: 0 });
  const screen = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, { shader: OWNERSHIP_SHADER, label: "fx", values: { params: { intensity: 1 } }, bindings: { src: source, globals } });
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  frame(gpu, (f) => f.pass({ target: screen }, (p) => p.draw(fx)));
  expect(groupLabels(mock)).toEqual(["fx.group0", "fx.group1"]);

  fx.bind("src", next);
  frame(gpu, (f) => f.pass({ target: screen }, (p) => p.draw(fx)));

  // Exactly the group holding `src` is rebuilt; group1 keeps its cached bind group.
  expect(groupLabels(mock)).toEqual(["fx.group0", "fx.group1", "fx.group0"]);
  gpu.dispose();
});

test("contract #9: .bind() dedupes by identity", async () => {
  const gpu = await init();
  const source = target(gpu, { size: [4, 4] });
  const globals = uniform(gpu, { time: 0 });
  const screen = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, { shader: OWNERSHIP_SHADER, label: "fx", values: { params: { intensity: 1 } }, bindings: { src: source, globals } });
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  frame(gpu, (f) => f.pass({ target: screen }, (p) => p.draw(fx)));
  const baseline = mock.calls.createBindGroup;

  for (let i = 0; i < 5; i++) fx.bind("src", source);
  frame(gpu, (f) => f.pass({ target: screen }, (p) => p.draw(fx)));

  expect(mock.calls.createBindGroup).toBe(baseline);
  gpu.dispose();
});

test("contract #9: .set() of an externally-bound name fails with VGPU-R1-EXTERNAL-BINDING", async () => {
  const gpu = await init();
  const source = target(gpu, { size: [4, 4] });
  const other = target(gpu, { size: [4, 4] });
  const globals = uniform(gpu, { time: 0 });
  const fx = effect(gpu, { shader: OWNERSHIP_SHADER, label: "fx", bindings: { src: source, globals } });

  expect(codeOf(() => fx.set("src", other))).toBe("VGPU-R1-EXTERNAL-BINDING");
  expect(codeOf(() => fx.set("globals", { time: 1 }))).toBe("VGPU-R1-EXTERNAL-BINDING");
  // The message must name the binding and point at the resource to update instead.
  expect(messageOf(() => fx.set("globals", { time: 1 }))).toMatch(/globals/);
  expect(messageOf(() => fx.set("globals", { time: 1 }))).toMatch(/bind\(|\.set\(/);
  gpu.dispose();
});

test("a binding promoted by .bind() also rejects .set() with VGPU-R1-EXTERNAL-BINDING", async () => {
  const gpu = await init();
  const source = target(gpu, { size: [4, 4] });
  const globals = uniform(gpu, { time: 0 });
  const fx = effect(gpu, { shader: OWNERSHIP_SHADER, label: "fx", values: { params: { intensity: 1 } }, bindings: { globals } });

  fx.bind("src", source);
  expect(codeOf(() => fx.set("src", source))).toBe("VGPU-R1-EXTERNAL-BINDING");
  gpu.dispose();
});

test("the flat bag also refuses an externally declared binding", async () => {
  const gpu = await init();
  const source = target(gpu, { size: [4, 4] });
  const other = target(gpu, { size: [4, 4] });
  const globals = uniform(gpu, { time: 0 });
  const fx = effect(gpu, { shader: OWNERSHIP_SHADER, label: "fx", bindings: { src: source, globals } });

  expect(codeOf(() => fx.set({ src: other }))).toBe("VGPU-R1-EXTERNAL-BINDING");
  gpu.dispose();
});

test("owned -> external is not supported: .bind() on a value-owned binding throws", async () => {
  const gpu = await init();
  const globals = uniform(gpu, { time: 0 });
  const fx = effect(gpu, { shader: GLOBALS_ONLY_SHADER, label: "fx" });

  fx.set("globals", { time: 0.5 });
  expect(codeOf(() => fx.bind("globals", globals))).toBe("VGPU-R1-OWNERSHIP-FLIP");
  gpu.dispose();
});

test("a binding declared in values is value-owned from construction and refuses .bind()", async () => {
  const gpu = await init();
  const globals = uniform(gpu, { time: 0 });
  const fx = effect(gpu, { shader: GLOBALS_ONLY_SHADER, label: "fx", values: { globals: { time: 0 } } });

  expect(codeOf(() => fx.bind("globals", globals))).toBe("VGPU-R1-OWNERSHIP-FLIP");
  gpu.dispose();
});

test("declaring the same binding in values and bindings fails at construction", async () => {
  const gpu = await init();
  const globals = uniform(gpu, { time: 0 });

  expect(messageOf(() => effect(gpu, { shader: GLOBALS_ONLY_SHADER, label: "fx", values: { globals: { time: 0 } }, bindings: { globals } })))
    .toMatch(/globals/);
  gpu.dispose();
});

test("a binding declared in bindings never latches ownership: VGPU-R1-OWNERSHIP-FLIP is unreachable for it", async () => {
  const gpu = await init();
  const globals = uniform(gpu, { time: 0 });
  const fx = effect(gpu, { shader: GLOBALS_ONLY_SHADER, label: "fx", bindings: { globals } });

  // The legacy latch would report OWNERSHIP-FLIP here; construction-time ownership reports
  // EXTERNAL-BINDING instead, and it does so from the very first call.
  expect(codeOf(() => fx.set({ globals: { time: 1 } }))).toBe("VGPU-R1-EXTERNAL-BINDING");
  // ...and it stays EXTERNAL-BINDING on the second call too: there is no latch to flip.
  expect(codeOf(() => fx.set({ globals: { time: 2 } }))).toBe("VGPU-R1-EXTERNAL-BINDING");
  gpu.dispose();
});

// ---------------------------------------------------------------------------
// Contract #10 — binding-scoped .set()
// ---------------------------------------------------------------------------

test("contract #10: a non-struct binding accepts its complete value", async () => {
  const gpu = await init();
  const screen = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, { shader: MAT_SHADER, label: "mat" });
  const writeBuffer = vi.spyOn(gpu.device.gpu.queue, "writeBuffer");

  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  fx.set("mvp", identity);

  expect(writeBuffer).toHaveBeenCalledTimes(1);
  const [, offset, data] = writeBuffer.mock.calls[0]!;
  expect(offset).toBe(0);
  expect(new Float32Array(data as ArrayBuffer)).toEqual(new Float32Array(identity));
  frame(gpu, (f) => f.pass({ target: screen }, (p) => p.draw(fx)));
  writeBuffer.mockRestore();
  gpu.dispose();
});

test("contract #10: one call carrying N fields of a struct produces exactly one buffer write", async () => {
  const gpu = await init();
  const fx = effect(gpu, { shader: CAMERA_SHADER, label: "cam" });
  const writeBuffer = vi.spyOn(gpu.device.gpu.queue, "writeBuffer");

  fx.set("camera", { viewProjection: new Array(16).fill(0), exposure: 2 });

  expect(writeBuffer).toHaveBeenCalledTimes(1);
  writeBuffer.mockRestore();
  gpu.dispose();
});

test("contract #10: the binding-scoped form has no member-name shortcut", async () => {
  const gpu = await init();
  const fx = effect(gpu, { shader: CAMERA_SHADER, label: "cam" });

  // `exposure` is a member of `camera`; the scoped form only accepts binding names and says so.
  expect(messageOf(() => fx.set("exposure", 2))).toMatch(/camera/);
  gpu.dispose();
});

test("contract #10: a struct binding merges partials on the CPU and rewrites the complete struct", async () => {
  const gpu = await init();
  const fx = effect(gpu, { shader: CAMERA_SHADER, label: "cam" });

  fx.set("camera", { viewProjection: new Array(16).fill(0), exposure: 2 });
  const writeBuffer = vi.spyOn(gpu.device.gpu.queue, "writeBuffer");
  fx.set("camera", { exposure: 3 });

  expect(writeBuffer).toHaveBeenCalledTimes(1);
  const [, , data] = writeBuffer.mock.calls[0]!;
  // 16 floats of viewProjection then exposure: the absent field kept its previous value.
  expect(new Float32Array(data as ArrayBuffer)[16]).toBe(3);
  writeBuffer.mockRestore();
  gpu.dispose();
});

test("a binding absent from bindings is value-owned by default and takes .set() with no values declared", async () => {
  const gpu = await init();
  const screen = target(gpu, { size: [4, 4] });
  const cube = draw(gpu, { shader: CAMERA_SHADER, label: "cube", vertices: 3 });
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  cube.set("camera", { viewProjection: new Array(16).fill(1), exposure: 1 });
  frame(gpu, (f) => f.pass({ target: screen }, (p) => p.draw(cube)));
  const baseline = mock.calls.createBindGroup;
  cube.set("camera", { exposure: 2 });
  frame(gpu, (f) => f.pass({ target: screen }, (p) => p.draw(cube)));

  expect(baseline).toBe(1);
  expect(mock.calls.createBindGroup).toBe(baseline);
  gpu.dispose();
});

test("values declared at construction are written once, before any draw", async () => {
  const gpu = await init();
  const writeBuffer = vi.spyOn(gpu.device.gpu.queue, "writeBuffer");
  const fx = effect(gpu, { shader: CAMERA_SHADER, label: "cam", values: { camera: { exposure: 4 } } });

  expect(fx).toBeDefined();
  expect(writeBuffer).toHaveBeenCalledTimes(1);
  const [, , data] = writeBuffer.mock.calls[0]!;
  expect(new Float32Array(data as ArrayBuffer)[16]).toBe(4);
  writeBuffer.mockRestore();
  gpu.dispose();
});

// ---------------------------------------------------------------------------
// Contract #11 — uniform() shared across pipelines
// ---------------------------------------------------------------------------

test("contract #11: uniform() shared across two pipelines updates both with one write", async () => {
  const gpu = await init();
  const screen = target(gpu, { size: [4, 4] });
  const globals = uniform(gpu, { time: 0 });
  const a = effect(gpu, { shader: GLOBALS_ONLY_SHADER, label: "a", bindings: { globals } });
  const b = effect(gpu, { shader: GLOBALS_ONLY_SHADER, label: "b", bindings: { globals } });
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  frame(gpu, (f) => f.pass({ target: screen }, (p) => { p.draw(a); p.draw(b); }));
  const writeBuffer = vi.spyOn(gpu.device.gpu.queue, "writeBuffer");
  globals.set({ time: 1 });

  expect(writeBuffer).toHaveBeenCalledTimes(1);
  // One shared buffer behind both pipelines, and no bind group churn from the write.
  const bindGroups = mock.createBindGroupDescriptors.filter((descriptor) => descriptor.label === "a.group0" || descriptor.label === "b.group0");
  expect(bindGroups).toHaveLength(2);
  const buffers = bindGroups.map((descriptor) => ([...descriptor.entries][0]!.resource as GPUBufferBinding).buffer);
  expect(buffers[0]).toBe(buffers[1]);
  writeBuffer.mockRestore();
  gpu.dispose();
});

test("contract #11: uniform() storage is zero-initialized", async () => {
  const gpu = await init();
  const screen = target(gpu, { size: [4, 4] });
  const globals = uniform(gpu, {});
  const fx = effect(gpu, { shader: GLOBALS_ONLY_SHADER, label: "fx", bindings: { globals } });
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  // No .set() anywhere: the shared storage must already read as zeros.
  frame(gpu, (f) => f.pass({ target: screen }, (p) => p.draw(fx)));
  const descriptor = mock.createBindGroupDescriptors.find((entry) => entry.label === "fx.group0");
  const buffer = ([...descriptor!.entries][0]!.resource as GPUBufferBinding).buffer as unknown as { __vgpuMockBytes: Uint8Array };
  expect(buffer.__vgpuMockBytes.length).toBeGreaterThan(0);
  expect([...buffer.__vgpuMockBytes].every((byte) => byte === 0)).toBe(true);
  gpu.dispose();
});

test("uniform() and uniforms() are the same mechanism under two spellings", async () => {
  const gpu = await init();
  const singular = uniform(gpu, { time: 1 });
  const plural = uniforms(gpu, { time: 1 });

  expect(Object.getPrototypeOf(singular)).toBe(Object.getPrototypeOf(plural));
  gpu.dispose();
});

// ---------------------------------------------------------------------------
// Legacy regressions — the flat bag survives untouched
// ---------------------------------------------------------------------------

test("regression: the flat bag still sets several members at once", async () => {
  const gpu = await init();
  const screen = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, CAMERA_SHADER, { label: "cam" });
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  fx.set({ viewProjection: new Array(16).fill(0), exposure: 1 });
  frame(gpu, (f) => f.pass({ target: screen }, (p) => p.draw(fx)));
  fx.set({ exposure: 2 });
  frame(gpu, (f) => f.pass({ target: screen }, (p) => p.draw(fx)));

  expect(mock.calls.createBuffer).toBe(1);
  expect(mock.calls.createBindGroup).toBe(1);
  gpu.dispose();
});

test("regression: the flat bag still reports an ambiguous member name", async () => {
  const gpu = await init();
  const fx = effect(gpu, AMBIGUOUS_SHADER, { label: "amb" });

  expect(messageOf(() => fx.set({ time: 1 }))).toMatch(/ambiguous/);
  gpu.dispose();
});

test("regression: the legacy latch still fires for the flat bag", async () => {
  const gpu = await init();
  const globals = uniform(gpu, { time: 0 });
  const fx = effect(gpu, GLOBALS_ONLY_SHADER, { label: "fx" });

  fx.set({ globals: { time: 1 } });
  expect(codeOf(() => fx.set({ globals }))).toBe("VGPU-R1-OWNERSHIP-FLIP");
  gpu.dispose();
});

// ---------------------------------------------------------------------------
// compute() carries the same ownership surface
// ---------------------------------------------------------------------------

test("compute accepts values/bindings and the binding-scoped set()", async () => {
  const gpu = await init();
  const particles = storage(gpu, 64);
  const sim = compute(gpu, { shader: SIM_SHADER, label: "sim", values: { sim: { dt: 0.016 } }, bindings: { particles } });
  const writeBuffer = vi.spyOn(gpu.device.gpu.queue, "writeBuffer");

  sim.set("sim", { dt: 0.032 });

  expect(writeBuffer).toHaveBeenCalledTimes(1);
  expect(codeOf(() => sim.set("particles", particles))).toBe("VGPU-R1-EXTERNAL-BINDING");
  writeBuffer.mockRestore();
  sim.dispatch(1);
  gpu.dispose();
});

test("compute .bind() swaps a storage buffer identity", async () => {
  const gpu = await init();
  const first = storage(gpu, 64);
  const second = storage(gpu, 64);
  const sim = compute(gpu, { shader: SIM_SHADER, label: "sim", values: { sim: { dt: 0.016 } }, bindings: { particles: first } });
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  sim.dispatch(1);
  const baseline = mock.calls.createBindGroup;
  sim.bind("particles", first);
  sim.dispatch(1);
  expect(mock.calls.createBindGroup).toBe(baseline);

  sim.bind("particles", second);
  sim.dispatch(1);
  expect(mock.calls.createBindGroup).toBe(baseline + 1);
  gpu.dispose();
});

// ---------------------------------------------------------------------------
// Unknown names stay actionable
// ---------------------------------------------------------------------------

test(".set() and .bind() reject a name that is not a binding", async () => {
  const gpu = await init();
  const globals = uniform(gpu, { time: 0 });
  const fx = effect(gpu, { shader: GLOBALS_ONLY_SHADER, label: "fx" });

  expect(messageOf(() => fx.set("nope", 1))).toMatch(/'nope'/);
  expect(messageOf(() => fx.bind("nope", globals))).toMatch(/'nope'/);
  gpu.dispose();
});
