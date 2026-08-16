// T04-21 (the `pendingPipelines` default is now "throw"): this suite encodes without `prepare()`
// on purpose -- its subject is the descriptor/encoder behavior asserted below, not readiness -- so
// it takes the permanent `"sync"` opt-in, which is exactly the eager compile-on-encode these
// assertions were written against. The default itself is covered by pending-pipelines.test.ts,
// prepare.test.ts and prepare-corpus-throw.test.ts, which run under it.
import { expect, test, vi } from "vitest";
import { getMockGPUDeviceInstrumentation } from "@vgpu/core";
import { init, bundle, effect, frame, prepare, surface, target } from "../src/mock.ts";
import type { Frame } from "../src/mock.ts";

const SOLID = `
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
  return vec4f(uv, 0.0, 1.0);
}
`;

const TEXTURE = `
@group(0) @binding(0) var src: texture_2d<f32>;
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
  return textureLoad(src, vec2u(0, 0), 0);
}
`;

test("bundle(gpu, ...) can record against a target signature and replay on a compatible target", async () => {
  const gpu = await init({ pendingPipelines: "sync" });
  const scene = target(gpu, { size: [4, 4], format: "rgba8unorm" });
  const shader = effect(gpu, { shader: SOLID, label: "signatureFx" });

  const recorded = bundle(gpu, { target: { colors: ["rgba8unorm"] }, label: "signatureBundle" }, (b) => b.draw(shader));

  expect(() => frame(gpu, (currentFrame) => currentFrame.pass({ target: scene }, (p) => p.bundles(recorded)))).not.toThrow();
  gpu.dispose();
});

test("bundle replay target signature mismatches throw R3 stale with recorded and actual keys", async () => {
  const gpu = await init({ pendingPipelines: "sync" });
  const scene = target(gpu, { size: [4, 4], format: "bgra8unorm" });
  const shader = effect(gpu, { shader: SOLID, label: "mismatchFx" });
  const recorded = bundle(gpu, { target: { colors: ["rgba8unorm"] }, label: "signatureMismatch" }, (b) => b.draw(shader));

  expect(() => frame(gpu, (currentFrame) => currentFrame.pass({ target: scene }, (p) => p.bundles(recorded)))).toThrowError(
    "bundle 'signatureMismatch' is stale: the replay target signature does not match the recorded signature. Bundles freeze format/depth/sampleCount and bind groups.\n" +
      "  Recorded signature: rgba8unorm:none:1\n" +
      "  Actual signature: bgra8unorm:none:1\n" +
      "  Fix: re-record the bundle for this target → signatureMismatch = bundle(gpu, { target: scene }, ...)\n" +
      "  (re-recording is always your responsibility; the library only detects this).",
  );
  gpu.dispose();
});

test("bundle(gpu, ...) validates malformed signatures at record time", async () => {
  const gpu = await init({ pendingPipelines: "sync" });
  const shader = effect(gpu, SOLID);

  expect(() => bundle(gpu, { target: { colors: [] }, label: "badSignature" }, (b) => b.draw(shader))).toThrowError(/VGPU-COMPILE-SIGNATURE-INVALID|colors/);
  gpu.dispose();
});

test("bundle replay survives resize of the replay target when the signature is unchanged", async () => {
  const gpu = await init({ pendingPipelines: "sync" });
  const scene = target(gpu, { size: [4, 4], format: "rgba8unorm" });
  const shader = effect(gpu, { shader: SOLID, label: "resizeFx" });
  const recorded = bundle(gpu, { target: scene, label: "resizeBundle" }, (b) => b.draw(shader));

  scene.resize([8, 8]);

  expect(() => frame(gpu, (currentFrame) => currentFrame.pass({ target: scene }, (p) => p.bundles(recorded)))).not.toThrow();
  gpu.dispose();
});

test("precompiled draws record into signature bundles without sync pipeline creation", async () => {
  const gpu = await init({ pendingPipelines: "sync" });
  const shader = effect(gpu, { shader: SOLID, label: "precompiledFx" });
  const signature = { colors: ["rgba8unorm"] as const };
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  await shader.compile(signature);
  expect(mock.calls.createRenderPipelineAsync).toBe(1);
  expect(mock.calls.createRenderPipeline).toBe(0);

  bundle(gpu, { target: signature, label: "precompiledBundle" }, (b) => b.draw(shader));

  expect(mock.calls.createRenderPipelineAsync).toBe(1);
  expect(mock.calls.createRenderPipeline).toBe(0);
  gpu.dispose();
});

test("a bundle recording still requires draw resources to be set, at the moment it is encoded", async () => {
  const gpu = await init({ pendingPipelines: "sync" });
  const scene = target(gpu, { size: [4, 4], format: "rgba8unorm" });
  const post = effect(gpu, { shader: TEXTURE, label: "post" });

  // Contract #15, row 1: construction records the logical command list and encodes nothing, so an
  // unset binding is no longer discovered by bundle() itself — the encode the first replay performs
  // (or prepare()) is what resolves the bind groups, and that is where it fails.
  const recorded = bundle(gpu, { target: { colors: ["rgba8unorm"] }, label: "unsetTextureBundle" }, (b) => b.draw(post));
  expect(recorded.status).toBe("pending-pipelines");

  expect(() => frame(gpu, (currentFrame) => currentFrame.pass({ target: scene }, (p) => p.bundles(recorded)))).toThrowError(/VGPU-R1-BINDING-NEVER-SET|Unset/);
  gpu.dispose();
});

test("the first sync replay of a cold bundle uses the sync pipeline path and reports failures through gpu.onError", async () => {
  const gpu = await init({ pendingPipelines: "sync" });
  const scene = target(gpu, { size: [4, 4], format: "rgba8unorm" });
  const shader = effect(gpu, { shader: SOLID, label: "coldFailure" });
  const nativeError = new Error("sync pipeline failed during bundle recording");
  const errors: unknown[] = [];
  gpu.onError((error) => errors.push(error));
  vi.spyOn(gpu.device.gpu, "createRenderPipeline").mockImplementation(() => { throw nativeError; });

  // Contract #15: "construction never throws for pending pipelines" — and it never compiles either,
  // so nothing can fail here. The inline compile moved to the first replay, which under the train's
  // "sync" default reproduces exactly the eager behavior (and its error report) of before.
  const recorded = bundle(gpu, { target: { colors: ["rgba8unorm"] }, label: "coldFailureBundle" }, (b) => b.draw(shader));
  await gpu.settled();
  expect(errors).toEqual([]);

  expect(() => frame(gpu, (currentFrame) => currentFrame.pass({ target: scene }, (p) => p.bundles(recorded)))).not.toThrow();
  await gpu.settled();

  expect(errors).toEqual([
    expect.objectContaining({
      code: "VGPU-COMPILE-FAILED",
      where: "coldFailure.pipelineFor",
      cause: nativeError,
      detail: { signature: "rgba8unorm:none:1" },
    }),
  ]);
  gpu.dispose();
});

// ---------------------------------------------------------------------------
// Contract #8 / §4 — recording a bundle needs only formats, never `getCurrentTexture()`, so it is
// legal outside `frame()`, exactly like `prepare()`.
// ---------------------------------------------------------------------------

function bundleTestCanvas(ledger?: { currentTextures: number }): HTMLCanvasElement {
  const canvas: Record<string, unknown> = { width: 4, height: 4 };
  canvas.getContext = (kind: string) => kind === "webgpu" ? {
    configure: () => undefined,
    unconfigure: () => undefined,
    getCurrentTexture: () => { if (ledger) ledger.currentTextures += 1; return { createView: () => ({}) }; },
  } : null;
  return canvas as HTMLCanvasElement;
}

test("bundle(gpu, { target: surface }, ...) outside frame() is legal, exactly like prepare()", async () => {
  const gpu = await init({ pendingPipelines: "sync" });
  const ledger = { currentTextures: 0 };
  const canvasSurface = surface(gpu, bundleTestCanvas(ledger));
  const shader = effect(gpu, { shader: SOLID, label: "outOfFrameFx" });

  let recorded: ReturnType<typeof bundle> | undefined;
  expect(() => { recorded = bundle(gpu, { target: canvasSurface, label: "outOfFrameBundle" }, (b) => b.draw(shader)); }).not.toThrow();
  expect(recorded!.status).toBe("pending-pipelines");

  // `prepare()` materializes it — still without ever opening a frame.
  await prepare(gpu, [{ bundle: recorded! }]);
  expect(recorded!.status).toBe("ready");
  // The clause is literal about the REASON this is legal — "needs only formats, never
  // `getCurrentTexture()`". Without this the whole path could be re-implemented on the presentation
  // texture and every assertion above would still pass (mutation-verified).
  expect(ledger.currentTextures).toBe(0);
  gpu.dispose();
});

test("Frame.pass() over a surface, called through an escaped Frame after its callback already submitted, still throws VGPU-SURFACE-NOT-IN-FRAME", async () => {
  const gpu = await init({ pendingPipelines: "sync" });
  const canvasSurface = surface(gpu, bundleTestCanvas());
  const shader = effect(gpu, { shader: SOLID, label: "escapedFramePassFx" });
  let escaped: Frame | undefined;

  frame(gpu, (f) => {
    escaped = f;
    f.pass(canvasSurface, (p) => p.draw(shader));
  });

  expect(() => escaped!.pass(canvasSurface, (p) => p.draw(shader))).toThrowError(
    expect.objectContaining({ code: "VGPU-SURFACE-NOT-IN-FRAME" }),
  );
  gpu.dispose();
});

test("bundle(gpu, { target: disposedSurface }, ...) still throws — VGPU-SURFACE-DISPOSED, not the removed frame guard", async () => {
  const gpu = await init({ pendingPipelines: "sync" });
  const canvasSurface = surface(gpu, bundleTestCanvas());
  const shader = effect(gpu, { shader: SOLID, label: "disposedSurfaceFx" });
  canvasSurface.dispose();

  // `normalizeSignature()` reads `surface.pipelineSignature`, whose getter asserts liveness before
  // returning anything — a disposed surface never reaches (or needs) the frame-activity guard.
  expect(() => bundle(gpu, { target: canvasSurface, label: "disposedSurfaceBundle" }, (b) => b.draw(shader))).toThrowError(
    expect.objectContaining({ code: "VGPU-SURFACE-DISPOSED" }),
  );
  gpu.dispose();
});
