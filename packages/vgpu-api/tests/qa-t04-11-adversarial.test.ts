/**
 * QA T04-11 — adversarial suite for contract #19 / design §9 "Device loss is terminal".
 *
 * Every test here exists because a MUTANT survived the branch's own suite (see
 * /tmp/debug/scripts/vgpu-t04-11-device-loss-qa/README.md), or because the instance-method matrix
 * needed pinning. Each test names the mutant it kills.
 */
import { afterEach, expect, test, vi } from "vitest";
import { createMockGPUDevice, Device, loseMockGPUDevice } from "@vgpu/core";
import { createMockAdapter, init, bundle, effect, frameLoop, geometry, prepare, storage, surface, target, timer, visibility } from "../src/mock.ts";
import { uniforms } from "../src/uniforms.ts";
import { compute } from "../src/compute.ts";
import { kernelOf, type Gpu } from "../src/kernel.ts";
import { FRAME_BUNDLE } from "../src/frame-protocols.ts";

const SOLID = `@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f { return vec4f(uv, 0.0, 1.0); }`;
const U_COMPUTE = `struct U { scale: f32 }; @group(0) @binding(0) var<uniform> u: U; @compute @workgroup_size(1) fn main() { let x = u.scale; }`;

type RafCallback = (timestamp: number) => void;
const originalRequestAnimationFrame = globalThis.requestAnimationFrame;
const originalCancelAnimationFrame = globalThis.cancelAnimationFrame;
afterEach(() => {
  globalThis.requestAnimationFrame = originalRequestAnimationFrame;
  globalThis.cancelAnimationFrame = originalCancelAnimationFrame;
});

async function lose(gpu: Gpu): Promise<void> {
  void loseMockGPUDevice(gpu.gpu, { reason: "unknown", message: "simulated loss" });
  await gpu.lost;
}

// --- The mock is the instrument: if it lies, every test above it is decoration ------------------

test("[kills M07] the mock's destroy() really resolves the native lost promise with reason \"destroyed\"", async () => {
  const native = createMockGPUDevice();
  const nativeLost = native.lost;

  native.destroy();

  // Without this, "device.lost stays pending after destroy()" passes for the wrong reason: there
  // would be no resolution to suppress in the first place.
  await expect(nativeLost).resolves.toMatchObject({ reason: "destroyed" });

  const device = new Device(createMockGPUDevice());
  device.destroy();
  await Promise.resolve();
  expect(await settledWithin(device.lost, 25)).toBe("pending");
});

test("[kills M06] loseMockGPUDevice hands back the NATIVE loss promise: a device is lost once, and the second call cannot re-describe it", async () => {
  const native = createMockGPUDevice();

  const first = await loseMockGPUDevice(native, { reason: "unknown", message: "the real one" });
  const second = await loseMockGPUDevice(native, { reason: "destroyed", message: "too late" });

  // Same object, not just same shape: the returned promise is `device.lost`, so a test that awaits
  // it has genuinely waited for the native resolution instead of one microtask of its own.
  expect(second).toBe(first);
  expect(first.message).toBe("the real one");
});

// --- Kernel phase discipline: loss stops schedulers, and only once ------------------------------

test("[kills M04] a real loss leaves the resource and service phases untouched — the app can still read its resources to report the failure", async () => {
  const gpu = await init();
  const kernel = kernelOf(gpu);
  let scheduler = 0, resource = 0, service = 0;
  kernel.own("scheduler", () => { scheduler += 1; });
  kernel.own("resource", () => { resource += 1; });
  kernel.own("service", () => { service += 1; });
  const scene = target(gpu, { size: [4, 4] });

  await lose(gpu);

  expect(scheduler).toBe(1);
  expect(resource).toBe(0);
  expect(service).toBe(0);
  expect(gpu.disposed).toBe(false);
  // Nothing was released: the textures are still there to be read by the app's error path, they
  // simply refuse new device work.
  expect(scene.size).toEqual([4, 4]);
  expect(() => scene.resize([8, 8])).toThrow(expect.objectContaining({ code: "VGPU-DEVICE-LOST" }));
});

test("[kills M21] the scheduler phase runs exactly once: a dispose() after a real loss does not stop the same loops twice", async () => {
  const gpu = await init();
  const kernel = kernelOf(gpu);
  let stops = 0;
  kernel.own("scheduler", () => { stops += 1; });

  await lose(gpu);
  expect(stops).toBe(1);

  gpu.dispose();
  // A disposer that already ran is gone from the set; running it again would double-stop loops and
  // double-free whatever a real disposer owns.
  expect(stops).toBe(1);
  expect(gpu.disposed).toBe(true);
});

test("loss-wins race: a dispose() landing after a resolved loss is idempotent and cannot un-resolve gpu.lost", async () => {
  const gpu = await init();
  const callbacks = mockAnimationFrames();
  let calls = 0;
  frameLoop(gpu, () => { calls += 1; });
  fire(callbacks, 1, 0);

  await lose(gpu);
  const info = await gpu.lost;

  expect(() => gpu.dispose()).not.toThrow();
  expect(() => gpu.dispose()).not.toThrow();
  expect(await gpu.lost).toBe(info);
  expect(gpu.disposed).toBe(true);
  expect(calls).toBe(1);
  // Spelling matrix: after a loss THEN a dispose, core's destroy() moves the device to "disposed",
  // so the door reports the disposal — the more recent, more local fact.
  expect(() => effect(gpu, SOLID)).toThrow(expect.objectContaining({ code: "VGPU-GPU-DISPOSED" }));
});

// --- Surface ------------------------------------------------------------------------------------

test("[kills M17] applyAutoResize refuses a lost device, naming surface.applyAutoResize", async () => {
  const gpu = await init();
  const canvas = canvasLike(8, 4);
  const view = surface(gpu, canvas);
  expect(view.autoResize).toBe(true);

  await lose(gpu);

  expect(() => (view as unknown as { applyAutoResize(): void }).applyAutoResize())
    .toThrow(expect.objectContaining({ code: "VGPU-DEVICE-LOST", where: "surface.applyAutoResize" }));
});

test("[kills M16] disposal is the more local fact: a disposed surface on a lost device keeps its own error", async () => {
  const gpu = await init();
  const view = surface(gpu, canvasLike(8, 4));
  view.dispose();

  await lose(gpu);

  expect(() => view.resize([16, 8])).toThrow(expect.objectContaining({ code: "VGPU-SURFACE-DISPOSED" }));
  expect(() => view.color).toThrow(expect.objectContaining({ code: "VGPU-SURFACE-DISPOSED" }));
  await expect(view.read()).rejects.toMatchObject({ code: "VGPU-SURFACE-DISPOSED" });
});

// --- Bundle -------------------------------------------------------------------------------------

test("[kills M09] the bundle's own prepare guard is load-bearing: the frame protocol path rejects too, not only prepare()'s door", async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, SOLID);
  const recorded = bundle(gpu, { target: scene }, (r) => r.draw(fx));
  await prepare(gpu, { bundle: recorded });
  expect(recorded.status).toBe("ready");

  await lose(gpu);

  // Reached without liveKernel: this is the guard inside RecordedBundle, and it must beat the
  // "already ready, resolve immediately" fast path.
  const protocol = (recorded as unknown as { [FRAME_BUNDLE]: { prepareCombination(): Promise<unknown> } })[FRAME_BUNDLE];
  await expect(protocol.prepareCombination()).rejects.toMatchObject({ code: "VGPU-DEVICE-LOST" });
});

// --- Instance matrix: what contract #19 really covers -------------------------------------------

test("instance writes are terminal after a real loss, transitively through the buffer guard", async () => {
  const gpu = await init();
  const warm = uniforms(gpu, { scale: 1 });
  compute(gpu, U_COMPUTE).set({ u: warm });        // adopts the layout, creates the backing buffer
  const buf = storage(gpu, 256, "read-write");
  const geom = geometry(gpu, { buffers: [{ data: new Float32Array(9), stride: 12, attributes: { position: "float32x3" as GPUVertexFormat } }] });

  warm.set({ scale: 2 });
  buf.write(new Float32Array([1, 2, 3, 4]));

  await lose(gpu);

  // These objects have no guard of their own — they inherit it from core `Buffer.write`/`Buffer.read`,
  // which is why the `where` names the buffer and not the caller's method. Documented, not silent.
  expect(() => warm.set({ scale: 3 })).toThrow(expect.objectContaining({ code: "VGPU-DEVICE-LOST" }));
  expect(() => buf.write(new Float32Array([5]))).toThrow(expect.objectContaining({ code: "VGPU-DEVICE-LOST" }));
  await expect(buf.read()).rejects.toMatchObject({ code: "VGPU-DEVICE-LOST" });
  expect(() => (geom as unknown as { write(d: BufferSource): void }).write(new Float32Array(9)))
    .toThrow(expect.objectContaining({ code: "VGPU-DEVICE-LOST" }));
});

test("every factory fails at the door after a real loss (the liveKernel decision)", async () => {
  const gpu = await init({ adapter: createMockAdapter({ features: ["timestamp-query"] }), requiredFeatures: ["timestamp-query"] });
  await lose(gpu);

  for (const [name, factory] of [
    ["uniforms", () => uniforms(gpu, { scale: 1 })],
    ["storage", () => storage(gpu, 64)],
    ["timer", () => timer(gpu)],
    ["visibility", () => visibility(gpu)],
    ["geometry", () => geometry(gpu, { buffers: [{ data: new Float32Array(3), stride: 12, attributes: { position: "float32x3" as GPUVertexFormat } }] })],
    ["target", () => target(gpu, { size: [2, 2] })],
    ["effect", () => effect(gpu, SOLID)],
  ] as const) {
    expect(factory, `${name}() must fail at the door`).toThrow(expect.objectContaining({ code: "VGPU-DEVICE-LOST", where: name }));
  }
});

test("KNOWN GAP (T04-11 scope decision): timer and visibility hand out handles after a real loss instead of throwing", async () => {
  const gpu = await init({ adapter: createMockAdapter({ features: ["timestamp-query"] }), requiredFeatures: ["timestamp-query"] });
  const t = timer(gpu);
  const vis = visibility(gpu);

  await lose(gpu);

  // Their private #assertUsable checks the object's OWN disposed flag, not the device state, so
  // these are pure bookkeeping calls that cannot fail. No device work is dropped (the query sets
  // are only touched from inside a frame, and frames are terminal), but by the letter of contract
  // #19 these should throw. Tripwire: if a later ticket closes the gap, this test must be updated.
  expect(() => t.span("pass")).not.toThrow();
  expect(() => t.onResults(() => undefined)).not.toThrow();
  expect(() => vis.query("thing")).not.toThrow();
  expect(() => vis.reset()).not.toThrow();
});

// --- Helpers -------------------------------------------------------------------------------------

async function settledWithin(promise: Promise<unknown>, ms: number): Promise<"settled" | "pending"> {
  return Promise.race([
    promise.then(() => "settled" as const, () => "settled" as const),
    new Promise<"pending">((resolve) => setTimeout(() => resolve("pending"), ms)),
  ]);
}

function mockAnimationFrames(): Map<number, RafCallback> {
  const callbacks = new Map<number, RafCallback>();
  let nextId = 1;
  globalThis.requestAnimationFrame = ((cb: FrameRequestCallback) => { const id = nextId++; callbacks.set(id, cb); return id; }) as typeof requestAnimationFrame;
  globalThis.cancelAnimationFrame = ((id: number) => { callbacks.delete(id); }) as typeof cancelAnimationFrame;
  return callbacks;
}

function fire(callbacks: Map<number, RafCallback>, id: number, timestamp: number): void {
  const cb = callbacks.get(id);
  callbacks.delete(id);
  cb?.(timestamp);
}

function canvasLike(width = 10, height = 5): HTMLCanvasElement {
  const context = { configure: vi.fn(), unconfigure: vi.fn(), getCurrentTexture: () => ({ createView: () => ({}) }) };
  const canvas: Record<string, unknown> = {
    width: 0, height: 0, clientWidth: width, clientHeight: height,
    getContext(kind: string) { if (kind !== "webgpu") return null; return { ...context, canvas }; },
  };
  return canvas as unknown as HTMLCanvasElement;
}
