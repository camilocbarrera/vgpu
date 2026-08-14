import { afterEach, expect, test, vi } from "vitest";
import { init, target, draw, compute } from "../src/mock.ts";
import { kernelOf } from "../src/kernel.ts";

const SIMPLE_SHADER = `
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
  return vec4f(uv, 0.0, 1.0);
}
`;

const RAW_GROUP_SHADER = `
struct Globals { tint: f32 }
struct Obj { value: f32 }
@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var<uniform> obj: Obj;
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
  return vec4f(obj.value * globals.tint, uv, 1.0);
}
`;

const EMPTY_COMPUTE_SHADER = `
@compute @workgroup_size(1) fn main() {}
`;

afterEach(() => vi.restoreAllMocks());

test("gpu.settled awaits queue.onSubmittedWorkDone even with no validation, compilation or readback pending", async () => {
  const gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  // A standalone draw with no claimed raw groups: today it submits synchronously and never
  // registers anything with the kernel's pendingDeliveries/settledSources bookkeeping.
  const drawable = draw(gpu, { shader: SIMPLE_SHADER, label: "noPendingBookkeeping" });
  let resolveSubmitted!: () => void;
  let submitted = false;
  vi.spyOn(gpu.device.gpu.queue, "onSubmittedWorkDone").mockImplementation(() => new Promise<void>((resolve) => {
    resolveSubmitted = () => { submitted = true; resolve(); };
  }));

  drawable.draw(colorTarget);

  let settledDone = false;
  const settled = gpu.settled().then(() => { settledDone = true; });

  await Promise.resolve();
  await Promise.resolve();
  expect(settledDone).toBe(false);
  expect(submitted).toBe(false);

  resolveSubmitted();
  await settled;
  expect(settledDone).toBe(true);
  expect(submitted).toBe(true);
  gpu.dispose();
});

test("a submission made during/after a gpu.settled() call does not extend that call's wait (snapshot semantics)", async () => {
  const gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: SIMPLE_SHADER, label: "snapshotSemantics" });
  const resolvers: (() => void)[] = [];
  vi.spyOn(gpu.device.gpu.queue, "onSubmittedWorkDone").mockImplementation(() => new Promise<void>((resolve) => {
    resolvers.push(resolve);
  }));

  drawable.draw(colorTarget);

  const firstSettled = gpu.settled();
  expect(resolvers).toHaveLength(1);

  // A submission that happens after the settled() snapshot was taken must not be waited on by it.
  drawable.draw(colorTarget);

  let firstSettledDone = false;
  void firstSettled.then(() => { firstSettledDone = true; });
  await Promise.resolve();
  await Promise.resolve();
  expect(firstSettledDone).toBe(false);
  // Still exactly one onSubmittedWorkDone call belongs to the first settled() snapshot.
  expect(resolvers).toHaveLength(1);

  resolvers[0]!();
  await firstSettled;
  expect(firstSettledDone).toBe(true);
  gpu.dispose();
});

test("gpu.settled flushes pending onError deliveries before resolving", async () => {
  const gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: SIMPLE_SHADER, label: "settledFlush" });
  const nativeError = new Error("native createRenderPipeline failed");
  const errors: unknown[] = [];
  gpu.onError((error) => errors.push(error));
  vi.spyOn(gpu.device.gpu, "createRenderPipeline").mockImplementation(() => { throw nativeError; });

  expect(() => drawable.draw(colorTarget)).not.toThrow();
  await gpu.settled();

  expect(errors).toHaveLength(1);
  expect(errors[0]).toMatchObject({ code: "VGPU-COMPILE-FAILED", where: "settledFlush.pipelineFor", cause: nativeError });

  // Pin the actual mechanism, not just its outcome: the compile-error delivery above resolves on
  // its own microtask regardless of whether settled() waits for it, so a mutant that drops
  // `#pendingDeliveries` from settled()'s snapshot would not be caught by the assertions above.
  // A delivery that stays pending until we resolve it must hold settled() open.
  const kernel = kernelOf(gpu);
  let resolveDelivery!: () => void;
  const controlledDelivery = new Promise<void>((resolve) => { resolveDelivery = resolve; });
  let deliveryRan = false;
  kernel.trackDelivery(controlledDelivery.then(() => { deliveryRan = true; }));

  let secondSettledDone = false;
  const secondSettled = gpu.settled().then(() => { secondSettledDone = true; });
  await Promise.resolve();
  await Promise.resolve();
  expect(secondSettledDone).toBe(false);
  expect(deliveryRan).toBe(false);

  resolveDelivery();
  await secondSettled;
  expect(secondSettledDone).toBe(true);
  expect(deliveryRan).toBe(true);

  gpu.dispose();
});

test("gpu.settled never rejects even when queue completion itself reports a failed submission", async () => {
  const gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: SIMPLE_SHADER, label: "neverRejects" });
  vi.spyOn(gpu.device.gpu.queue, "onSubmittedWorkDone").mockImplementation(() => Promise.reject(new Error("submission failed")));

  drawable.draw(colorTarget);

  await expect(gpu.settled()).resolves.toBeUndefined();
  gpu.dispose();
});

test("regression: settled() still waits transitively on an in-flight claimed-bind-group validation", async () => {
  const gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  const { draw: drawable, popResolvers } = rawClaimedDrawWithDeferredScopes(gpu, "regressionClaim");
  const errors: unknown[] = [];
  gpu.onError((error) => errors.push(error));

  drawable.draw({ target: colorTarget, offsets: { 1: [0] } });

  let settledDone = false;
  const settled = gpu.settled().then(() => { settledDone = true; });
  await Promise.resolve();
  await Promise.resolve();
  expect(settledDone).toBe(false);

  expect(popResolvers.length).toBeGreaterThan(1);
  popResolvers[0]!(null);
  popResolvers[1]!({ message: "regression claim mismatch" } as GPUError);
  for (const resolve of popResolvers.slice(2)) resolve(null);

  await settled;
  expect(settledDone).toBe(true);
  expect(errors).toEqual([
    expect.objectContaining({ code: "VGPU-R4-GROUP-VALIDATION", where: "regressionClaim.draw", detail: { drawLabel: "regressionClaim", group: 1 } }),
  ]);
  gpu.dispose();
});

test("regression: compute's synchronous dispatch never awaits queue.onSubmittedWorkDone on its own resolution path", async () => {
  const gpu = await init();
  const sim = compute(gpu, EMPTY_COMPUTE_SHADER, { label: "settledQueueRegressionCompute" });
  const onSubmittedWorkDone = vi.spyOn(gpu.device.gpu.queue, "onSubmittedWorkDone");

  const result = sim.dispatch(1);

  expect(result).toBeUndefined();
  expect(onSubmittedWorkDone).not.toHaveBeenCalled();
  gpu.dispose();
});

test("QA-A5: device without onSubmittedWorkDone (old adapter/mock) does not crash settled()", async () => {
  const gpu = await init();
  const queue = gpu.device.gpu.queue as unknown as Record<string, unknown>;
  delete queue.onSubmittedWorkDone;
  expect((gpu.device.gpu.queue as unknown as Record<string, unknown>).onSubmittedWorkDone).toBeUndefined();
  await expect(gpu.settled()).resolves.toBeUndefined();
  gpu.dispose();
});

function rawClaimedDrawWithDeferredScopes(gpu: Awaited<ReturnType<typeof init>>, label: string) {
  const popResolvers: ((error: GPUError | null) => void)[] = [];
  const gpuDevice = gpu.device.gpu as GPUDevice & {
    pushErrorScope(filter: GPUErrorFilter): void;
    popErrorScope(): Promise<GPUError | null>;
  };
  gpuDevice.pushErrorScope = vi.fn();
  gpuDevice.popErrorScope = vi.fn(() => new Promise<GPUError | null>((resolve) => popResolvers.push(resolve)));

  const drawable = draw(gpu, { shader: `${RAW_GROUP_SHADER}
// ${label}`, label, set: { globals: { tint: 1 } } });
  const rawBuffer = gpu.device.gpu.createBuffer({ size: 4, usage: 64 });
  const rawLayout = gpu.device.gpu.createBindGroupLayout({
    label: `${label}.raw-static-layout`,
    entries: [{ binding: 0, visibility: 2, buffer: { type: "uniform", hasDynamicOffset: false, minBindingSize: 4 } }],
  });
  const rawBindGroup = gpu.device.gpu.createBindGroup({
    label: `${label}.raw-static-bind-group`,
    layout: rawLayout,
    entries: [{ binding: 0, resource: { buffer: rawBuffer, offset: 0, size: 4 } }],
  });
  drawable.layout(1, { dynamicOffsets: true });
  drawable.group(1, rawBindGroup);
  return { draw: drawable, popResolvers };
}
