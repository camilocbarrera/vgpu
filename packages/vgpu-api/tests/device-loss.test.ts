/**
 * Contract #19 + design rule §9 "Device loss is terminal".
 *
 * A REAL device loss (the native `GPUDevice.lost` promise resolving on its own) is graph-wide and
 * terminal: `gpu.lost` resolves with no intermediate operation, every frame loop vgpu owns stops by
 * itself, and every subsequent operation on that graph — surface, offscreen target, bundle, draw,
 * compute, frame — throws `VGPU-DEVICE-LOST`, naming the method the caller wrote. Nothing is ever
 * re-pointed at a replacement device and there is no API to ask for it.
 *
 * A deliberate `gpu.dispose()` is the OTHER semantic: `gpu.disposed` flips, `gpu.lost` stays pending
 * forever (WebGPU resolves the native promise with reason `"destroyed"`, and vgpu suppresses it),
 * and operations throw `VGPU-DEVICE-DISPOSED`. One spelling per semantic — there is no `onLost()`.
 */
import { afterEach, expect, test, vi } from "vitest";
import { loseMockGPUDevice } from "@vgpu/core";
import { init, bundle, effect, frame, frameLoop, prepare, surface, target } from "../src/mock.ts";
import type { Gpu } from "../src/kernel.ts";
import { FRAME_BUNDLE } from "../src/frame-protocols.ts";
import { createBundle, type Bundle } from "../src/bundle.ts";
import type { Target } from "../src/target.ts";

const SOLID = `
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f { return vec4f(uv, 0.0, 1.0); }
`;

type RafCallback = (timestamp: number) => void;
const originalRequestAnimationFrame = globalThis.requestAnimationFrame;
const originalCancelAnimationFrame = globalThis.cancelAnimationFrame;

afterEach(() => {
  globalThis.requestAnimationFrame = originalRequestAnimationFrame;
  globalThis.cancelAnimationFrame = originalCancelAnimationFrame;
});

// --- gpu.lost: resolves only on a real loss ------------------------------------------------------

test("gpu.lost resolves with the loss info after a real device loss, with no intermediate operation", async () => {
  const gpu = await init();

  void loseMockGPUDevice(gpu.gpu, { reason: "unknown", message: "simulated loss" });
  const info = await gpu.lost;

  expect(info?.reason).toBe("unknown");
  expect(info?.message).toBe("simulated loss");
  // Loss is not disposal: nothing was released, the graph is simply terminal.
  expect(gpu.disposed).toBe(false);
});

test("gpu.dispose() never resolves gpu.lost; it only flips gpu.disposed", async () => {
  const gpu = await init();

  gpu.dispose();

  expect(gpu.disposed).toBe(true);
  expect(await settledWithin(gpu.lost, 25)).toBe("pending");
  expect(() => effect(gpu, SOLID)).toThrow(expect.objectContaining({ code: "VGPU-GPU-DISPOSED" }));
});

test("dispose() after a real loss stays idempotent, and gpu.disposed only tracks dispose()", async () => {
  const gpu = await init();
  await lose(gpu);

  expect(gpu.disposed).toBe(false);
  expect(() => gpu.dispose()).not.toThrow();
  expect(() => gpu.dispose()).not.toThrow();
  expect(gpu.disposed).toBe(true);
  // The loss already resolved: dispose() cannot take that back.
  await expect(gpu.lost).resolves.toMatchObject({ reason: "unknown" });
});

// --- Frame loops stop by themselves --------------------------------------------------------------

test("a real device loss stops every frame loop the gpu owns, without disposing it", async () => {
  const callbacks = mockAnimationFrames();
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, SOLID);

  let calls = 0;
  frameLoop(gpu, (currentFrame) => { calls += 1; currentFrame.pass(scene, fx); });
  fire(callbacks, 1, 0);
  expect(calls).toBe(1);

  const queuedTick = callbacks.get(2);
  expect(queuedTick).toBeDefined();
  await lose(gpu);

  // The loop was cancelled by the loss itself, and the tick already queued when it landed returns
  // without re-entering the user callback (which would throw on every frame, forever).
  expect(callbacks.size).toBe(0);
  expect(() => queuedTick?.(16)).not.toThrow();
  expect(calls).toBe(1);
  expect(gpu.disposed).toBe(false);
});

test("the loops are stopped before anyone awaiting gpu.lost observes the loss", async () => {
  const callbacks = mockAnimationFrames();
  const gpu = await init();
  frameLoop(gpu, () => undefined);
  fire(callbacks, 1, 0);
  expect(callbacks.size).toBe(1);

  void loseMockGPUDevice(gpu.gpu, { reason: "unknown" });
  await gpu.lost;

  expect(callbacks.size).toBe(0);
});

// --- Every object of the graph is terminal -------------------------------------------------------

test("surface operations throw VGPU-DEVICE-LOST naming the surface method, not an internal device call", async () => {
  const gpu = await init();
  const canvas = canvasLike(8, 4);
  const view = surface(gpu, canvas, { depth: true });
  await lose(gpu);

  expect(() => view.resize([16, 8])).toThrow(expect.objectContaining({ code: "VGPU-DEVICE-LOST", where: "surface.resize" }));
  expect(() => view.color).toThrow(expect.objectContaining({ code: "VGPU-DEVICE-LOST", where: "surface.color" }));
  expect(() => view.depth).toThrow(expect.objectContaining({ code: "VGPU-DEVICE-LOST", where: "surface.depth" }));
  expect(() => view.renderPassDescriptor()).toThrow(expect.objectContaining({ code: "VGPU-DEVICE-LOST", where: "surface.renderPassDescriptor" }));
  await expect(view.read()).rejects.toMatchObject({ code: "VGPU-DEVICE-LOST", where: "surface.read" });
  await expect(view.readFloats()).rejects.toMatchObject({ code: "VGPU-DEVICE-LOST", where: "surface.readFloats" });
});

test("an offscreen target is terminal too, naming the target method", async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4] });
  await lose(gpu);

  expect(() => scene.resize([8, 8])).toThrow(expect.objectContaining({ code: "VGPU-DEVICE-LOST", where: "target.resize" }));
  expect(() => scene.renderPassDescriptor()).toThrow(expect.objectContaining({ code: "VGPU-DEVICE-LOST", where: "target.renderPassDescriptor" }));
  await expect(scene.read()).rejects.toMatchObject({ code: "VGPU-DEVICE-LOST", where: "target.read" });
  await expect(scene.readFloats()).rejects.toMatchObject({ code: "VGPU-DEVICE-LOST", where: "target.readFloats" });
});

test("a recorded bundle refuses to prepare, replay or rebuild after a real loss", async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, SOLID);
  const recorded = bundle(gpu, { target: scene }, (r) => r.draw(fx));
  await prepare(gpu, { bundle: recorded });
  expect(recorded.status).toBe("ready");

  await lose(gpu);

  await expect(prepare(gpu, { bundle: recorded })).rejects.toMatchObject({ code: "VGPU-DEVICE-LOST", where: "prepare" });
  expect(() => recorded.rebuild((r) => r.draw(fx))).toThrow(expect.objectContaining({ code: "VGPU-DEVICE-LOST" }));
  // Replay goes through the frame protocol: a lost device beats the "this bundle is ready" fast path.
  expect(() => replay(recorded, scene)).toThrow(expect.objectContaining({ code: "VGPU-DEVICE-LOST" }));
});

test("draw, compute-style set/bind, frame() and prepare() stay terminal after a loss (regression of the 34 existing guards)", async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, SOLID);
  await lose(gpu);

  expect(() => frame(gpu, () => undefined)).toThrow(expect.objectContaining({ code: "VGPU-DEVICE-LOST" }));
  expect(() => frameLoop(gpu, () => undefined)).toThrow(expect.objectContaining({ code: "VGPU-DEVICE-LOST" }));
  expect(() => fx.set({})).toThrow(expect.objectContaining({ code: "VGPU-DEVICE-LOST" }));
  expect(() => effect(gpu, SOLID)).toThrow(expect.objectContaining({ code: "VGPU-DEVICE-LOST" }));
  expect(() => target(gpu, { size: [2, 2] })).toThrow(expect.objectContaining({ code: "VGPU-DEVICE-LOST" }));
  await expect(prepare(gpu, { draw: fx, target: scene })).rejects.toMatchObject({ code: "VGPU-DEVICE-LOST" });
});

test("nothing is ever re-pointed: objects of a lost gpu stay terminal next to a brand new gpu", async () => {
  const lost = await init();
  const oldScene = target(lost, { size: [4, 4] });
  const oldEffect = effect(lost, SOLID);
  await lose(lost);

  const fresh = await init();
  // No restore(), no reattach(), no generation proxy: the old objects belong to the dead device forever.
  expect(Object.keys(lost)).not.toContain("restore");
  expect(() => frame(fresh, (f) => f.pass(oldScene, oldEffect))).toThrow(expect.objectContaining({ code: "VGPU-DEVICE-LOST" }));
  expect(() => oldScene.resize([8, 8])).toThrow(expect.objectContaining({ code: "VGPU-DEVICE-LOST" }));
  fresh.dispose();
});

// --- Races: dispose vs a loss landing at the same time -------------------------------------------

test("dispose() winning the race suppresses the loss the destroy itself causes", async () => {
  const callbacks = mockAnimationFrames();
  const gpu = await init();
  let calls = 0;
  frameLoop(gpu, () => { calls += 1; });

  gpu.dispose();
  // The native promise resolves right after destroy(); a late resolution must not turn a disposed
  // gpu into a "lost" one, nor resolve gpu.lost.
  void loseMockGPUDevice(gpu.gpu, { reason: "destroyed" });
  await tick();

  expect(await settledWithin(gpu.lost, 25)).toBe("pending");
  expect(gpu.disposed).toBe(true);
  expect(callbacks.size).toBe(0);
  expect(calls).toBe(0);
  expect(() => effect(gpu, SOLID)).toThrow(expect.objectContaining({ code: "VGPU-GPU-DISPOSED" }));
});

test("a prepare() in flight when the loss lands rejects with VGPU-DEVICE-LOST instead of encoding", async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, SOLID);
  const gate = deferred();
  const native = gpu.gpu as unknown as { createRenderPipelineAsync(desc: GPURenderPipelineDescriptor): Promise<GPURenderPipeline> };
  const original = native.createRenderPipelineAsync.bind(native);
  native.createRenderPipelineAsync = async (desc: GPURenderPipelineDescriptor) => { await gate.promise; return original(desc); };

  const recorded = bundle(gpu, { target: scene }, (r) => r.draw(fx));
  const preparing = prepare(gpu, { bundle: recorded });
  await lose(gpu);
  gate.resolve();

  // prepare() batches: the combination failed with the device error, reported as its cause.
  const failure = await preparing.then(() => undefined, (error: unknown) => error as { code: string; cause?: { code?: string } });
  expect(failure?.code).toBe("VGPU-PREPARE-FAILED");
  expect(failure?.cause?.code).toBe("VGPU-DEVICE-LOST");
  expect(recorded.gpu).toBeUndefined();
  expect(recorded.status).not.toBe("ready");
});

test("the bundle host guard is a capability, not a requirement: a host without it still records and replays", async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, SOLID);
  // The reduced host shape (`render-service.ts` builds one) defines no assertDeviceUsable: calling it
  // must stay optional, never a crash on an undefined member.
  const bare = createBundle({ device: { gpu: gpu.gpu } }, { target: scene }, (r) => r.draw(fx));

  expect(bare.status).toBe("pending-pipelines");
  expect(() => replay(bare, scene)).not.toThrow();
  expect(bare.status).toBe("ready");

  await lose(gpu);
  // Still no crash: a host that was given no guard simply cannot report the loss itself.
  expect(() => replay(bare, scene)).not.toThrow();
});

// --- Helpers -------------------------------------------------------------------------------------

/** Simulates a real (non-`destroy()`) device loss and waits until the whole graph observed it. */
async function lose(gpu: Gpu): Promise<void> {
  void loseMockGPUDevice(gpu.gpu, { reason: "unknown", message: "simulated loss" });
  await gpu.lost;
}

function replay(recorded: Bundle, on: Target): GPURenderBundle | undefined {
  return (recorded as unknown as { [FRAME_BUNDLE]: { resolveForReplay(target: Target): GPURenderBundle | undefined } })[FRAME_BUNDLE].resolveForReplay(on);
}

function deferred(): { promise: Promise<void>; resolve: () => void } {
  let resolve!: () => void;
  const promise = new Promise<void>((r) => { resolve = r; });
  return { promise, resolve };
}

async function tick(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
}

async function settledWithin(promise: Promise<unknown>, ms: number): Promise<"settled" | "pending"> {
  return Promise.race([
    promise.then(() => "settled" as const, () => "settled" as const),
    new Promise<"pending">((resolve) => setTimeout(() => resolve("pending"), ms)),
  ]);
}

function mockAnimationFrames(): Map<number, RafCallback> {
  const callbacks = new Map<number, RafCallback>();
  let nextId = 1;
  globalThis.requestAnimationFrame = ((cb: FrameRequestCallback) => {
    const id = nextId++;
    callbacks.set(id, cb);
    return id;
  }) as typeof requestAnimationFrame;
  globalThis.cancelAnimationFrame = ((id: number) => { callbacks.delete(id); }) as typeof cancelAnimationFrame;
  return callbacks;
}

function fire(callbacks: Map<number, RafCallback>, id: number, timestamp: number): void {
  const cb = callbacks.get(id);
  callbacks.delete(id);
  cb?.(timestamp);
}

function canvasLike(width = 10, height = 5): HTMLCanvasElement {
  const context = {
    configure: vi.fn(),
    unconfigure: vi.fn(),
    getCurrentTexture: () => ({ createView: () => ({}) }),
  };
  const canvas: Record<string, unknown> = {
    width: 0,
    height: 0,
    clientWidth: width,
    clientHeight: height,
    getContext(kind: string) {
      if (kind !== "webgpu") return null;
      return { ...context, canvas };
    },
  };
  return canvas as unknown as HTMLCanvasElement;
}
