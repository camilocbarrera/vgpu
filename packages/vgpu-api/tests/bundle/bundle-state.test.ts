/**
 * Contract #15 — the bundle state machine (`BundleStatus`, `bundle.error`, `rebuild()`,
 * `dispose()`) plus the `pendingPipelines` chain for replaying a non-`ready` bundle.
 *
 * One test per row of the transition table in the frozen design (issue #320, contract #15):
 * construction is always `pending-pipelines` (nothing is compiled and no native bundle is
 * encoded), `prepare()` is the one spelling that materializes it, an identity update only stales a
 * `ready` bundle, `rebuild()` replaces the logical recording without ever compiling, and
 * `dispose()` is terminal under every policy.
 */
import { expect, test, vi } from "vitest";
import { getMockGPUDeviceInstrumentation } from "@vgpu/core";
import { init, bundle, effect, frame, prepare, target } from "../../src/mock.ts";
import type { VGPUError } from "../../src/errors.ts";

const SOLID = `
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
  return vec4f(uv, 0.0, 1.0);
}
`;

const OTHER_SOLID = `
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
  return vec4f(1.0 - uv.x, uv.y, 0.5, 1.0);
}
`;

const TEXTURED = `
@group(0) @binding(0) var src: texture_2d<f32>;
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
  return textureLoad(src, vec2u(0, 0), 0) + vec4f(uv, 0.0, 0.0);
}
`;

const TEXTURED_FOG = `
struct Fog { fogDensity: f32 }
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<uniform> fog: Fog;
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
  return textureLoad(src, vec2u(0, 0), 0) * fog.fogDensity + vec4f(uv, 0.0, 0.0);
}
`;

function caught(run: () => unknown): VGPUError | undefined {
  try { run(); return undefined; }
  catch (error) { return error as VGPUError; }
}

async function rejection(run: () => Promise<unknown>): Promise<VGPUError> {
  try { await run(); throw new Error("expected a rejection"); }
  catch (error) { return error as VGPUError; }
}

// ---------------------------------------------------------------------------
// Row 1: construction → pending-pipelines, always.
// ---------------------------------------------------------------------------

test("bundle() records the logical command list without compiling or encoding anything", async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, SOLID, { label: "constructionFx" });
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  const recorded = bundle(gpu, { target: scene, label: "construction" }, (b) => b.draw(fx));

  expect(recorded.status).toBe("pending-pipelines");
  expect(recorded.gpu).toBeUndefined();
  expect(recorded.error).toBeUndefined();
  expect(mock.calls.createRenderPipeline).toBe(0);
  expect(mock.calls.createRenderPipelineAsync).toBe(0);
  expect(mock.calls.createRenderBundleEncoder).toBe(0);
  gpu.dispose();
});

test("a bundle whose draws already have their pipeline is still born pending-pipelines", async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, SOLID, { label: "warmFx" });
  await prepare(gpu, { draw: fx, target: scene });

  // "always: the native bundle is not materialized at construction" — readiness of the pipelines is
  // not the same fact as readiness of the native bundle.
  const recorded = bundle(gpu, { target: scene, label: "warmConstruction" }, (b) => b.draw(fx));

  expect(recorded.status).toBe("pending-pipelines");
  expect(recorded.gpu).toBeUndefined();
  gpu.dispose();
});

// ---------------------------------------------------------------------------
// Row 2: prepare() succeeds → ready. Plus contract #7 (bundle half): re-preparing is free.
// ---------------------------------------------------------------------------

test("prepare({ bundle }) compiles the recorded draws asynchronously and then encodes the native bundle", async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, SOLID, { label: "prepareFx" });
  const recorded = bundle(gpu, { target: scene, label: "preparedBundle" }, (b) => b.draw(fx));
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  const prepared = await prepare(gpu, { bundle: recorded });

  expect(mock.calls.createRenderPipelineAsync).toBe(1);
  expect(mock.calls.createRenderPipeline).toBe(0);
  expect(mock.calls.createRenderBundleEncoder).toBe(1);
  expect(recorded.status).toBe("ready");
  expect(recorded.gpu).toBeDefined();
  expect(prepared.gpu).toBe(recorded.gpu);
  expect(prepared.bundle).toBe(recorded);

  const again = await prepare(gpu, { bundle: recorded });

  // prepare() on a ready bundle is a no-op: no new pipeline, no re-encode.
  expect(mock.calls.createRenderPipelineAsync).toBe(1);
  expect(mock.calls.createRenderPipeline).toBe(0);
  expect(mock.calls.createRenderBundleEncoder).toBe(1);
  expect(again.gpu).toBe(recorded.gpu);
  expect(recorded.status).toBe("ready");
  gpu.dispose();
});

test("prepare() of the same bundle twice in one batch does not duplicate the work", async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, SOLID, { label: "batchFx" });
  const recorded = bundle(gpu, { target: scene, label: "batchBundle" }, (b) => b.draw(fx));
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  const [first, second] = await prepare(gpu, [{ bundle: recorded }, { bundle: recorded }]);

  expect(mock.calls.createRenderBundleEncoder).toBe(1);
  expect(mock.calls.createRenderPipelineAsync).toBe(1);
  expect(first.gpu).toBe(second.gpu);
  expect(recorded.status).toBe("ready");
  gpu.dispose();
});

// ---------------------------------------------------------------------------
// Row 3: prepare() fails → failed, retaining the error. And prepare() always retries.
// ---------------------------------------------------------------------------

test("a failed prepare({ bundle }) leaves the bundle failed with the error retained, and a later prepare retries it", async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, SOLID, { label: "failingFx" });
  const recorded = bundle(gpu, { target: scene, label: "failingBundle" }, (b) => b.draw(fx));
  const nativeError = new Error("async pipeline creation failed");
  const original = gpu.device.gpu.createRenderPipelineAsync.bind(gpu.device.gpu);
  const spy = vi.spyOn(gpu.device.gpu, "createRenderPipelineAsync").mockRejectedValue(nativeError);

  const failure = await rejection(() => prepare(gpu, { bundle: recorded }));

  expect(failure.code).toBe("VGPU-PREPARE-FAILED");
  expect(failure.message).toContain("failingBundle");
  expect(recorded.status).toBe("failed");
  expect(recorded.error?.code).toBe("VGPU-COMPILE-FAILED");
  expect(recorded.error?.where).toBe("bundle 'failingBundle' prepare");
  expect(recorded.gpu).toBeUndefined();

  spy.mockImplementation((desc: GPURenderPipelineDescriptor) => original(desc));
  const prepared = await prepare(gpu, { bundle: recorded });

  expect(recorded.status).toBe("ready");
  expect(recorded.error).toBeUndefined();
  expect(prepared.gpu).toBe(recorded.gpu);
  gpu.dispose();
});

// ---------------------------------------------------------------------------
// Rows 4-6: .set() does not transition, .bind() stales a ready bundle, an identity update on a
// non-ready bundle changes nothing.
// ---------------------------------------------------------------------------

test(".set() byte updates keep a ready bundle ready, .bind() identity updates make it stale", async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4], format: "rgba8unorm" });
  const first = target(gpu, { size: [4, 4] });
  const second = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, { shader: TEXTURED_FOG, label: "identityFx", bindings: { src: first }, set: { fogDensity: 0.1 } });
  const recorded = bundle(gpu, { target: scene, label: "identityBundle" }, (b) => b.draw(fx));
  await prepare(gpu, { bundle: recorded });
  const native = recorded.gpu;

  // Row 4, the real thing: .set() writes bytes into the buffer the bundle already captured, so every
  // recorded command and bind group stays valid — the bundle must NOT move, and must keep the very
  // same native bundle (that is the whole point of R3: value updates are free for bundles).
  fx.set({ fogDensity: 0.9 });
  expect(recorded.status).toBe("ready");
  expect(recorded.gpu).toBe(native);

  // Re-binding the SAME identity is not an identity change either.
  fx.bind("src", first);
  expect(recorded.status).toBe("ready");

  fx.bind("src", second);
  expect(recorded.status).toBe("stale");
  // The last valid native bundle is retained until something re-encodes it.
  expect(recorded.gpu).toBe(native);
  gpu.dispose();
});

test("an identity update on a bundle that is not ready leaves its status unchanged", async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4], format: "rgba8unorm" });
  const first = target(gpu, { size: [4, 4] });
  const second = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, { shader: TEXTURED, label: "pendingIdentityFx", bindings: { src: first } });
  const recorded = bundle(gpu, { target: scene, label: "pendingIdentityBundle" }, (b) => b.draw(fx));

  fx.bind("src", second);

  expect(recorded.status).toBe("pending-pipelines");
  gpu.dispose();
});

// ---------------------------------------------------------------------------
// Row 7: rebuild() replaces the logical recording, synchronously, without compiling.
// ---------------------------------------------------------------------------

test("rebuild() replaces the recording without compiling or encoding and lands on stale when every draw is ready", async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, SOLID, { label: "rebuildFx" });
  const recorded = bundle(gpu, { target: scene, label: "rebuildBundle" }, (b) => b.draw(fx));
  await prepare(gpu, { bundle: recorded });
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
  const baseline = { ...mock.calls };
  const native = recorded.gpu;

  recorded.rebuild((b) => b.draw(fx));

  expect(recorded.status).toBe("stale");
  expect(mock.calls.createRenderPipeline).toBe(baseline.createRenderPipeline);
  expect(mock.calls.createRenderPipelineAsync).toBe(baseline.createRenderPipelineAsync);
  expect(mock.calls.createRenderBundleEncoder).toBe(baseline.createRenderBundleEncoder);
  // The old native bundle stays the last valid one until a prepare()/"sync" replay re-encodes.
  expect(recorded.gpu).toBe(native);

  await prepare(gpu, { bundle: recorded });

  expect(recorded.status).toBe("ready");
  expect(mock.calls.createRenderBundleEncoder).toBe(baseline.createRenderBundleEncoder + 1);
  expect(recorded.gpu).not.toBe(native);
  gpu.dispose();
});

test("rebuild() goes to pending-pipelines when the new recording introduces an uncompiled combination", async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4] });
  const known = effect(gpu, SOLID, { label: "knownFx" });
  // A distinct shader source: two effects over the SAME source share a cached module and therefore
  // the same pipeline key, which would make the new draw ready for free.
  const introduced = effect(gpu, OTHER_SOLID, { label: "introducedFx" });
  const recorded = bundle(gpu, { target: scene, label: "growingBundle" }, (b) => b.draw(known));
  await prepare(gpu, { bundle: recorded });
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
  const baseline = { ...mock.calls };

  recorded.rebuild((b) => { b.draw(known); b.draw(introduced); });

  expect(recorded.status).toBe("pending-pipelines");
  expect(mock.calls.createRenderPipeline).toBe(baseline.createRenderPipeline);
  expect(mock.calls.createRenderPipelineAsync).toBe(baseline.createRenderPipelineAsync);
  expect(mock.calls.createRenderBundleEncoder).toBe(baseline.createRenderBundleEncoder);
  gpu.dispose();
});

test("rebuild() clears the retained error immediately because it replaces the recording that failed", async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, SOLID, { label: "clearedFx" });
  const recorded = bundle(gpu, { target: scene, label: "clearedBundle" }, (b) => b.draw(fx));
  const original = gpu.device.gpu.createRenderPipelineAsync.bind(gpu.device.gpu);
  const spy = vi.spyOn(gpu.device.gpu, "createRenderPipelineAsync").mockRejectedValue(new Error("nope"));
  await rejection(() => prepare(gpu, { bundle: recorded }));
  expect(recorded.error).toBeDefined();

  spy.mockImplementation((desc: GPURenderPipelineDescriptor) => original(desc));
  recorded.rebuild((b) => b.draw(fx));

  expect(recorded.error).toBeUndefined();
  expect(recorded.status).toBe("pending-pipelines");
  gpu.dispose();
});

// ---------------------------------------------------------------------------
// Replay of a non-ready bundle: the one pendingPipelines chain (frame → gpu for bundles).
// ---------------------------------------------------------------------------

test('replaying a pending-pipelines bundle under "throw" throws VGPU-PIPELINE-PENDING and compiles nothing', async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, SOLID, { label: "throwPendingFx" });
  const recorded = bundle(gpu, { target: scene, label: "throwPendingBundle" }, (b) => b.draw(fx));
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  const error = caught(() => frame(gpu, (f) => f.pass(scene, (p) => p.bundles(recorded)), { pendingPipelines: "throw" }));

  expect(error?.code).toBe("VGPU-PIPELINE-PENDING");
  expect(mock.calls.createRenderPipeline).toBe(0);
  expect(mock.calls.createRenderBundleEncoder).toBe(0);
  expect(recorded.status).toBe("pending-pipelines");
  gpu.dispose();
});

test('replaying a stale bundle under "throw" throws VGPU-R3-BUNDLE-STALE naming prepare()', async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4], format: "rgba8unorm" });
  const first = target(gpu, { size: [4, 4] });
  const second = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, { shader: TEXTURED, label: "throwStaleFx", bindings: { src: first } });
  const recorded = bundle(gpu, { target: scene, label: "throwStaleBundle" }, (b) => b.draw(fx));
  await prepare(gpu, { bundle: recorded });
  fx.bind("src", second);
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
  const encoders = mock.calls.createRenderBundleEncoder;

  const error = caught(() => frame(gpu, (f) => f.pass(scene, (p) => p.bundles(recorded)), { pendingPipelines: "throw" }));

  expect(error?.code).toBe("VGPU-R3-BUNDLE-STALE");
  expect(error?.message).toContain("prepare(gpu, { bundle: throwStaleBundle })");
  expect(mock.calls.createRenderBundleEncoder).toBe(encoders);
  expect(recorded.status).toBe("stale");
  gpu.dispose();
});

test('replaying a failed bundle under "throw" rethrows the retained error with its cause', async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, SOLID, { label: "throwFailedFx" });
  const recorded = bundle(gpu, { target: scene, label: "throwFailedBundle" }, (b) => b.draw(fx));
  vi.spyOn(gpu.device.gpu, "createRenderPipelineAsync").mockRejectedValue(new Error("compile said no"));
  await rejection(() => prepare(gpu, { bundle: recorded }));

  const error = caught(() => frame(gpu, (f) => f.pass(scene, (p) => p.bundles(recorded)), { pendingPipelines: "throw" }));

  expect(error).toBe(recorded.error);
  expect((error?.cause as VGPUError)?.code).toBe("VGPU-COMPILE-FAILED");
  gpu.dispose();
});

test('a pending-pipelines bundle under "skip" is omitted and its compilation continues in the background', async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, SOLID, { label: "skipPendingFx" });
  const recorded = bundle(gpu, { target: scene, label: "skipPendingBundle" }, (b) => b.draw(fx));
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  expect(() => frame(gpu, (f) => f.pass(scene, (p) => p.bundles(recorded)), { pendingPipelines: "skip" })).not.toThrow();

  // Skipped this frame: nothing was compiled synchronously and nothing was encoded inline.
  expect(mock.calls.createRenderPipeline).toBe(0);
  expect(mock.calls.createRenderBundleEncoder).toBe(0);
  expect(mock.calls.createRenderPipelineAsync).toBe(1);

  await gpu.settled();

  expect(recorded.status).toBe("ready");
  expect(mock.calls.createRenderBundleEncoder).toBe(1);
  gpu.dispose();
});

test('a stale bundle under "skip" is skipped and never silently re-encoded', async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4], format: "rgba8unorm" });
  const first = target(gpu, { size: [4, 4] });
  const second = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, { shader: TEXTURED, label: "skipStaleFx", bindings: { src: first } });
  const recorded = bundle(gpu, { target: scene, label: "skipStaleBundle" }, (b) => b.draw(fx));
  await prepare(gpu, { bundle: recorded });
  fx.bind("src", second);
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
  const encoders = mock.calls.createRenderBundleEncoder;

  expect(() => frame(gpu, (f) => f.pass(scene, (p) => p.bundles(recorded)), { pendingPipelines: "skip" })).not.toThrow();
  await gpu.settled();

  expect(recorded.status).toBe("stale");
  expect(mock.calls.createRenderBundleEncoder).toBe(encoders);
  gpu.dispose();
});

test('a failed bundle under "skip" is reported once through gpu.onError and keeps being skipped', async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, SOLID, { label: "skipFailedFx" });
  const recorded = bundle(gpu, { target: scene, label: "skipFailedBundle" }, (b) => b.draw(fx));
  vi.spyOn(gpu.device.gpu, "createRenderPipelineAsync").mockRejectedValue(new Error("still no"));
  await rejection(() => prepare(gpu, { bundle: recorded }));
  const errors: VGPUError[] = [];
  gpu.onError((error) => errors.push(error as VGPUError));

  expect(() => frame(gpu, (f) => f.pass(scene, (p) => p.bundles(recorded)), { pendingPipelines: "skip" })).not.toThrow();
  await gpu.settled();
  expect(() => frame(gpu, (f) => f.pass(scene, (p) => p.bundles(recorded)), { pendingPipelines: "skip" })).not.toThrow();
  await gpu.settled();

  expect(errors).toHaveLength(1);
  expect(errors[0]).toBe(recorded.error);
  expect(recorded.status).toBe("failed");
  gpu.dispose();
});

test('a pending-pipelines bundle under "sync" is compiled and encoded inline, which is the legacy eager behavior', async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, SOLID, { label: "syncPendingFx" });
  const recorded = bundle(gpu, { target: scene, label: "syncPendingBundle" }, (b) => b.draw(fx));
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  // No prepare() anywhere: the default policy of the train is "sync", so this is what every
  // existing program does today.
  expect(() => frame(gpu, (f) => f.pass(scene, (p) => p.bundles(recorded)))).not.toThrow();

  expect(recorded.status).toBe("ready");
  expect(recorded.gpu).toBeDefined();
  expect(mock.calls.createRenderPipeline).toBe(1);
  expect(mock.calls.createRenderBundleEncoder).toBe(1);

  // A second replay of a ready bundle re-encodes nothing.
  expect(() => frame(gpu, (f) => f.pass(scene, (p) => p.bundles(recorded)))).not.toThrow();
  expect(mock.calls.createRenderBundleEncoder).toBe(1);
  gpu.dispose();
});

test('a stale bundle under "sync" is re-encoded inline without compiling', async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4], format: "rgba8unorm" });
  const first = target(gpu, { size: [4, 4] });
  const second = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, { shader: TEXTURED, label: "syncStaleFx", bindings: { src: first } });
  const recorded = bundle(gpu, { target: scene, label: "syncStaleBundle" }, (b) => b.draw(fx));
  await prepare(gpu, { bundle: recorded });
  const native = recorded.gpu;
  fx.bind("src", second);
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
  const encoders = mock.calls.createRenderBundleEncoder;

  expect(() => frame(gpu, (f) => f.pass(scene, (p) => p.bundles(recorded)), { pendingPipelines: "sync" })).not.toThrow();

  expect(recorded.status).toBe("ready");
  expect(recorded.gpu).not.toBe(native);
  expect(mock.calls.createRenderBundleEncoder).toBe(encoders + 1);
  expect(mock.calls.createRenderPipeline).toBe(0);
  gpu.dispose();
});

test('a failed bundle under "sync" rethrows the retained error and does not retry', async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, SOLID, { label: "syncFailedFx" });
  const recorded = bundle(gpu, { target: scene, label: "syncFailedBundle" }, (b) => b.draw(fx));
  vi.spyOn(gpu.device.gpu, "createRenderPipelineAsync").mockRejectedValue(new Error("no pipeline"));
  await rejection(() => prepare(gpu, { bundle: recorded }));
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
  const asyncCalls = mock.calls.createRenderPipelineAsync;

  const error = caught(() => frame(gpu, (f) => f.pass(scene, (p) => p.bundles(recorded)), { pendingPipelines: "sync" }));

  expect(error).toBe(recorded.error);
  expect(mock.calls.createRenderPipelineAsync).toBe(asyncCalls);
  expect(mock.calls.createRenderPipeline).toBe(0);
  gpu.dispose();
});

test("a skipped bundle does not stop the rest of the pass from replaying", async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4] });
  const readyFx = effect(gpu, SOLID, { label: "mixedReadyFx" });
  const pendingFx = effect(gpu, SOLID, { label: "mixedPendingFx" });
  const ready = bundle(gpu, { target: scene, label: "mixedReady" }, (b) => b.draw(readyFx));
  await prepare(gpu, { bundle: ready });
  const pending = bundle(gpu, { target: scene, label: "mixedPending" }, (b) => b.draw(pendingFx));

  const executed: unknown[][] = [];
  const passes: GPURenderPassEncoder[] = [];
  const originalPass = gpu.device.gpu.createCommandEncoder.bind(gpu.device.gpu);
  vi.spyOn(gpu.device.gpu, "createCommandEncoder").mockImplementation((desc?: GPUCommandEncoderDescriptor) => {
    const encoder = originalPass(desc);
    const beginRenderPass = encoder.beginRenderPass.bind(encoder);
    encoder.beginRenderPass = (descriptor: GPURenderPassDescriptor) => {
      const pass = beginRenderPass(descriptor);
      passes.push(pass);
      const executeBundles = pass.executeBundles.bind(pass);
      pass.executeBundles = (list: Iterable<GPURenderBundle>) => { executed.push([...list]); return executeBundles(list); };
      return pass;
    };
    return encoder;
  });

  expect(() => frame(gpu, (f) => f.pass(scene, (p) => p.bundles(ready, pending)), { pendingPipelines: "skip" })).not.toThrow();

  expect(ready.status).toBe("ready");
  // The skipped entry must be FILTERED OUT, not passed to WebGPU as a hole: executeBundles(undefined)
  // is a native validation error waiting to happen, and the mock's no-op stub would never notice.
  expect(executed.length).toBe(1);
  expect(executed[0]).toEqual([ready.gpu]);
  expect(executed[0]?.every((entry) => entry !== undefined)).toBe(true);

  // ...and when EVERY bundle of the call is skipped there must be no executeBundles call at all:
  // executeBundles([]) is a native call for nothing, repeated every frame of a loop that is waiting
  // for its pipelines. The early return is the thing being asserted here.
  executed.length = 0;
  expect(() => frame(gpu, (f) => f.pass(scene, (p) => p.bundles(pending)), { pendingPipelines: "skip" })).not.toThrow();
  expect(executed).toEqual([]);
  vi.restoreAllMocks();
  gpu.dispose();
});

test('a "sync" replay hands the policy down to the recorded draws, not just to the bundle', async () => {
  // The gpu default is "throw" here, so a draw that is asked to encode without an explicit policy
  // refuses to compile. The frame says "sync", and that has to reach the recorded draws through the
  // encode — otherwise the bundle materializes its native encoder and the draw inside it throws.
  const gpu = await init({ pendingPipelines: "throw" });
  const scene = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, SOLID, { label: "policyDownFx" });
  const recorded = bundle(gpu, { target: scene, label: "policyDownBundle" }, (b) => b.draw(fx));
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
  const syncCompiles = mock.calls.createRenderPipeline;

  expect(() => frame(gpu, (f) => f.pass(scene, (p) => p.bundles(recorded)), { pendingPipelines: "sync" })).not.toThrow();

  expect(recorded.status).toBe("ready");
  expect(mock.calls.createRenderPipeline).toBe(syncCompiles + 1);
  gpu.dispose();
});

// ---------------------------------------------------------------------------
// Row 8: dispose() is terminal under every policy.
// ---------------------------------------------------------------------------

test("dispose() is terminal and idempotent, and every replay policy throws VGPU-BUNDLE-DISPOSED", async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, SOLID, { label: "disposedFx" });
  const recorded = bundle(gpu, { target: scene, label: "disposedBundle" }, (b) => b.draw(fx));
  await prepare(gpu, { bundle: recorded });

  recorded.dispose();
  expect(recorded.status).toBe("disposed");
  expect(recorded.gpu).toBeUndefined();
  expect(() => recorded.dispose()).not.toThrow();
  expect(recorded.status).toBe("disposed");

  for (const pendingPipelines of ["throw", "skip", "sync"] as const) {
    const error = caught(() => frame(gpu, (f) => f.pass(scene, (p) => p.bundles(recorded)), { pendingPipelines }));
    expect(error?.code).toBe("VGPU-BUNDLE-DISPOSED");
  }

  // prepare() on a disposed bundle fails loudly too — never a silent no-op.
  const failure = await rejection(() => prepare(gpu, { bundle: recorded }));
  expect(failure.code).toBe("VGPU-PREPARE-FAILED");
  expect((failure.cause as VGPUError).code).toBe("VGPU-BUNDLE-DISPOSED");

  expect(caught(() => recorded.rebuild((b) => b.draw(fx)))?.code).toBe("VGPU-BUNDLE-DISPOSED");
  gpu.dispose();
});

// ---------------------------------------------------------------------------
// Adversarial: events that arrive while a prepare() is in flight.
// ---------------------------------------------------------------------------

test("an identity update during an in-flight prepare() is not lost: the encode that follows it captures the new resource", async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4], format: "rgba8unorm" });
  const first = target(gpu, { size: [4, 4] });
  const second = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, { shader: TEXTURED, label: "raceFx", bindings: { src: first } });
  const recorded = bundle(gpu, { target: scene, label: "raceBundle" }, (b) => b.draw(fx));
  const original = gpu.device.gpu.createRenderPipelineAsync.bind(gpu.device.gpu);
  let release!: () => void;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  vi.spyOn(gpu.device.gpu, "createRenderPipelineAsync").mockImplementation(async (desc: GPURenderPipelineDescriptor) => {
    await gate;
    return original(desc);
  });

  const preparing = prepare(gpu, { bundle: recorded });
  fx.bind("src", second);
  release();
  await preparing;

  // The recording is replayed AFTER the identity change, so the native bundle it produced is the
  // up-to-date one: the bundle is ready, not stale.
  expect(recorded.status).toBe("ready");
  expect(() => frame(gpu, (f) => f.pass(scene, (p) => p.bundles(recorded)), { pendingPipelines: "throw" })).not.toThrow();
  gpu.dispose();
});

test("rebuild() during an in-flight prepare() makes that prepare finish on the new recording", async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4] });
  const before = effect(gpu, SOLID, { label: "beforeRebuildFx" });
  // A DIFFERENT shader on purpose: the new recording needs a pipeline the in-flight prepare never
  // warmed, which is exactly what the #generation re-check has to notice.
  const after = effect(gpu, OTHER_SOLID, { label: "afterRebuildFx" });
  const recorded = bundle(gpu, { target: scene, label: "rebuildRaceBundle" }, (b) => b.draw(before));
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
  const syncCompiles = mock.calls.createRenderPipeline;
  const original = gpu.device.gpu.createRenderPipelineAsync.bind(gpu.device.gpu);
  let release!: () => void;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  vi.spyOn(gpu.device.gpu, "createRenderPipelineAsync").mockImplementation(async (desc: GPURenderPipelineDescriptor) => {
    await gate;
    return original(desc);
  });

  const preparing = prepare(gpu, { bundle: recorded });
  recorded.rebuild((b) => b.draw(after));
  release();
  await preparing;

  expect(recorded.status).toBe("ready");
  // The pipeline of the draw the NEW recording names is the one that had to be compiled.
  expect(after.gpu).toBeDefined();
  // Load-bearing assert for the re-check: without it, prepare() encodes the new command list with
  // only the old recording warmed, so `after`'s pipeline is created SYNCHRONOUSLY inside the encode —
  // the stall prepare() exists to avoid. Zero synchronous creations is the proof it re-warmed.
  expect(mock.calls.createRenderPipeline).toBe(syncCompiles);
  expect(() => frame(gpu, (f) => f.pass(scene, (p) => p.bundles(recorded)), { pendingPipelines: "throw" })).not.toThrow();
  gpu.dispose();
});

test("rebuild() unregisters the draws it dropped, so their later identity changes no longer stale the bundle", async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4], format: "rgba8unorm" });
  const first = target(gpu, { size: [4, 4] });
  const second = target(gpu, { size: [4, 4] });
  const dropped = effect(gpu, { shader: TEXTURED, label: "droppedFx", bindings: { src: first } });
  const kept = effect(gpu, { shader: TEXTURED, label: "keptFx", bindings: { src: first } });
  const recorded = bundle(gpu, { target: scene, label: "dropBundle" }, (b) => b.draw(dropped));
  await prepare(gpu, { bundle: recorded });

  recorded.rebuild((b) => b.draw(kept));
  await prepare(gpu, { bundle: recorded });
  expect(recorded.status).toBe("ready");

  // The dropped draw is not part of the recording any more, so its identity changes are none of this
  // bundle's business. Registration is one-way in draw.ts, so getting this wrong is not a cosmetic
  // detail: the bundle would be stale FOREVER from a draw it no longer encodes, and no prepare()
  // could fix it because the re-encode does not name that draw.
  dropped.bind("src", second);
  expect(recorded.status).toBe("ready");
  expect(() => frame(gpu, (f) => f.pass(scene, (p) => p.bundles(recorded)), { pendingPipelines: "throw" })).not.toThrow();

  // ...while the draw the new recording DOES name still stales it.
  kept.bind("src", second);
  expect(recorded.status).toBe("stale");
  gpu.dispose();
});

test("dispose() unregisters the bundle from its draws: a later identity change reaches nothing", async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4], format: "rgba8unorm" });
  const first = target(gpu, { size: [4, 4] });
  const second = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, { shader: TEXTURED, label: "disposedFx", bindings: { src: first } });
  const recorded = bundle(gpu, { target: scene, label: "disposedDropBundle" }, (b) => b.draw(fx));
  await prepare(gpu, { bundle: recorded });

  recorded.dispose();
  // A live draw must not keep a disposed bundle reachable (that is a leak, and markStale() traffic
  // for a bundle that can never replay again). The observable half: the update lands without
  // throwing and the bundle stays terminal.
  expect(() => fx.bind("src", second)).not.toThrow();
  expect(recorded.status).toBe("disposed");
  expect(recorded.gpu).toBeUndefined();
  gpu.dispose();
});

test("dispose() during an in-flight prepare() makes that prepare fail with VGPU-BUNDLE-DISPOSED", async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [4, 4] });
  const fx = effect(gpu, SOLID, { label: "disposeRaceFx" });
  const recorded = bundle(gpu, { target: scene, label: "disposeRaceBundle" }, (b) => b.draw(fx));
  const original = gpu.device.gpu.createRenderPipelineAsync.bind(gpu.device.gpu);
  let release!: () => void;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  vi.spyOn(gpu.device.gpu, "createRenderPipelineAsync").mockImplementation(async (desc: GPURenderPipelineDescriptor) => {
    await gate;
    return original(desc);
  });
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  const preparing = prepare(gpu, { bundle: recorded });
  recorded.dispose();
  release();
  const failure = await rejection(() => preparing);

  expect(failure.code).toBe("VGPU-PREPARE-FAILED");
  expect((failure.cause as VGPUError).code).toBe("VGPU-BUNDLE-DISPOSED");
  expect(recorded.status).toBe("disposed");
  expect(mock.calls.createRenderBundleEncoder).toBe(0);
  gpu.dispose();
});

// ---------------------------------------------------------------------------
// Contract #16 — deferred on purpose: `uniformArray`/`slots` do not exist on this branch.
// ---------------------------------------------------------------------------

test.skip("contract #16: a bundle recorded with slots: at(0) replays with slot 0's offset (T04-07)", () => {
  // WebGPU freezes the dynamic offsets of setBindGroup when the GPURenderBundleEncoder records it,
  // so this contract holds for free once `uniformArray`/`slots` land. There is no slot mechanism on
  // this branch to exercise, and building one here would duplicate T04-07: the explicit
  // verification lives with the feature that introduces it.
});
