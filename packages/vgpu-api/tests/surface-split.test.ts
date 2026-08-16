/**
 * Contract #23 (issue #320 rev6 §4c) — `Surface` is a presentation destination, `Target` is a bindable
 * resource.
 *
 * Two halves, both here:
 *  1. A `Surface` is NOT a texture binding. The rejection is **nominal and at runtime**: `bindings` is
 *     `Record<string, unknown>`, so structural typing is not the enforcement mechanism and the JS/`unknown`
 *     path must fail exactly like the typed one.
 *  2. Resize invalidation is by **signature**, not by identity: a resize that preserves colors, depth and
 *     sample count compiles no new pipeline and leaves prepared bundles `ready`; a destination whose
 *     signature differs is a different prepared combination.
 *
 * Plus the multisampling vocabulary unification: `target({ sampleCount })` is the WebGPU spelling of the
 * legacy `target({ msaa })`, and the two may not disagree.
 */
import { expect, test, vi } from "vitest";
import { getMockGPUDeviceInstrumentation } from "@vgpu/core";
import { bundle, draw, effect, frame, init, prepare, surface, target } from "../src/mock.ts";
import { normalizeSignature, signatureKeyOf } from "../src/pipeline-store.ts";

const TEXTURED = `
@group(0) @binding(0) var src: texture_2d<f32>;
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
  return textureLoad(src, vec2u(0, 0), 0) + vec4f(uv, 0.0, 0.0);
}
`;

const TEXTURED_DRAW = `
@group(0) @binding(0) var src: texture_2d<f32>;
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(pos[vi], 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4f { return textureLoad(src, vec2u(0, 0), 0); }
`;

const SOLID = `
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f { return vec4f(uv, 0.0, 1.0); }
`;

/** Minimal layout-backed canvas with a stable presentation texture, like the other surface suites use. */
function canvasLike(width = 8, height = 4): HTMLCanvasElement {
  const texture = { createView: () => ({}) };
  const canvas: Record<string, unknown> = {
    clientWidth: width,
    clientHeight: height,
    width: 0,
    height: 0,
    getContext(kind: string) {
      if (kind !== "webgpu") return null;
      return { canvas, configure: vi.fn(), unconfigure: vi.fn(), getCurrentTexture: () => texture };
    },
  };
  return canvas as unknown as HTMLCanvasElement;
}

function codeOf(fn: () => unknown): string {
  try { fn(); } catch (error) { return (error as { code?: string }).code ?? String(error); }
  return "NO-THROW";
}

function errorOf(fn: () => unknown): { code?: string; message?: string; fix?: string } | undefined {
  try { fn(); } catch (error) { return error as { code?: string; message?: string; fix?: string }; }
  return undefined;
}

function pipelineCalls(gpu: { device: { gpu: GPUDevice } }): { sync: number; async: number } {
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
  return { sync: mock.calls.createRenderPipeline, async: mock.calls.createRenderPipelineAsync };
}

// ---------------------------------------------------------------------------
// #23, half 1 — a Surface is not a texture binding.
// ---------------------------------------------------------------------------

test("a Surface passed as a texture binding is rejected with VGPU-SURFACE-NOT-BINDABLE, through bindings and through .bind()", async () => {
  const gpu = await init();
  const screen = surface(gpu, canvasLike());
  const scene = target(gpu, { size: [8, 4] });

  expect(codeOf(() => effect(gpu, { shader: TEXTURED, label: "bound-at-construction", bindings: { src: screen } }))).toBe("VGPU-SURFACE-NOT-BINDABLE");

  const fx = effect(gpu, { shader: TEXTURED, label: "reboundFx", bindings: { src: scene } });
  expect(codeOf(() => fx.bind("src", screen))).toBe("VGPU-SURFACE-NOT-BINDABLE");

  const cube = draw(gpu, { shader: TEXTURED_DRAW, label: "reboundDraw", bindings: { src: scene } });
  expect(codeOf(() => cube.bind("src", screen))).toBe("VGPU-SURFACE-NOT-BINDABLE");

  gpu.dispose();
});

test("the surface rejection is actionable: it names the presentation-only rule and the target(gpu, ...) fix", async () => {
  const gpu = await init();
  const screen = surface(gpu, canvasLike());

  const error = errorOf(() => effect(gpu, { shader: TEXTURED, label: "actionableFx", bindings: { src: screen } }));

  expect(error?.code).toBe("VGPU-SURFACE-NOT-BINDABLE");
  expect(`${error?.message} ${error?.fix ?? ""}`).toMatch(/presentation-only/i);
  expect(`${error?.message} ${error?.fix ?? ""}`).toMatch(/target\(gpu/);
  gpu.dispose();
});

test("structural typing is not the enforcement mechanism: the same rejection fires through the JS/unknown path", async () => {
  const gpu = await init();
  const screen = surface(gpu, canvasLike());
  const scene = target(gpu, { size: [8, 4] });
  // The seam a JavaScript caller (or any `unknown`-typed plumbing) goes through: no type ever names
  // `Surface` here, so only the runtime guard can reject it.
  const bindThroughUnknown = (value: unknown) => effect(gpu, { shader: TEXTURED, label: "unknownFx", bindings: { src: value } });
  const rebindThroughUnknown = (value: unknown) => effect(gpu, { shader: TEXTURED, label: "unknownRebindFx", bindings: { src: scene } }).bind("src", value);

  expect(codeOf(() => bindThroughUnknown(screen))).toBe("VGPU-SURFACE-NOT-BINDABLE");
  expect(codeOf(() => rebindThroughUnknown(screen))).toBe("VGPU-SURFACE-NOT-BINDABLE");
  // A composed options bag must not hide it either: the guard is on the value, not on the call shape.
  const composed = { ...{ shader: TEXTURED, label: "composedFx" }, bindings: { src: screen as unknown } };
  expect(codeOf(() => effect(gpu, composed))).toBe("VGPU-SURFACE-NOT-BINDABLE");

  gpu.dispose();
});

test("a Target is still bindable exactly as before, and so is the Texture a surface hands out", async () => {
  const gpu = await init();
  const screen = surface(gpu, canvasLike());
  const scene = target(gpu, { size: [8, 4] });
  const other = target(gpu, { size: [8, 4] });

  const fx = effect(gpu, { shader: TEXTURED, label: "targetBindingFx", bindings: { src: scene } });
  expect(codeOf(() => fx.bind("src", other))).toBe("NO-THROW");

  // Documented debt, not the contract: `surface.color` is a `Texture`, and a `Texture` stays bindable.
  // The split rejects the Surface, never the texture the app pulled out of it (adj-lifecycle V4 — the
  // identity of that texture changes on every access, which is why binding it is a footgun, not an error).
  expect(codeOf(() => fx.bind("src", screen.color))).toBe("NO-THROW");

  gpu.dispose();
});

test("a Surface is still a render destination: pass(), prepare({ target }) and bundle({ target }) all accept it", async () => {
  const gpu = await init();
  const screen = surface(gpu, canvasLike());
  const fx = effect(gpu, SOLID, { label: "destinationFx" });

  await prepare(gpu, [{ draw: fx, target: screen }]);
  // Recording a bundle over a surface stays a frame-scoped operation (VGPU-SURFACE-NOT-IN-FRAME outside
  // one) — unchanged by the split: what matters here is that the surface is still a legal bundle target.
  expect(codeOf(() => frame(gpu, () => { bundle(gpu, { target: screen, label: "surfaceBundle" }, (b) => b.draw(fx)); }))).toBe("NO-THROW");

  expect(codeOf(() => frame(gpu, (f) => f.pass(screen, (p) => p.draw(fx)), { pendingPipelines: "throw" }))).toBe("NO-THROW");
  expect(codeOf(() => frame(gpu, (f) => f.pass({ target: screen }, (p) => p.draw(fx)), { pendingPipelines: "throw" }))).toBe("NO-THROW");

  gpu.dispose();
});

// ---------------------------------------------------------------------------
// #23, half 2 — invalidation is by signature, not by identity.
// ---------------------------------------------------------------------------

test("a resize that preserves colors, depth and sampleCount compiles no pipeline and leaves the bundle ready", async () => {
  const gpu = await init();
  const scene = target(gpu, { size: [8, 4], depth: true, label: "resizedScene" });
  const fx = effect(gpu, SOLID, { label: "resizedFx" });
  const recorded = bundle(gpu, { target: scene, label: "resizedBundle" }, (b) => b.draw(fx));
  await prepare(gpu, { bundle: recorded });
  expect(recorded.status).toBe("ready");
  const before = pipelineCalls(gpu);
  const signatureBefore = signatureKeyOf(normalizeSignature(scene));

  scene.resize([16, 8]);

  expect(signatureKeyOf(normalizeSignature(scene))).toBe(signatureBefore);
  expect(pipelineCalls(gpu)).toEqual(before);
  expect(recorded.status).toBe("ready");
  // The strictest policy is the proof: replaying under "throw" would report VGPU-PIPELINE-PENDING if the
  // resize had invalidated the prepared combination.
  expect(codeOf(() => frame(gpu, (f) => f.pass(scene, (p) => p.bundles(recorded)), { pendingPipelines: "throw" }))).toBe("NO-THROW");
  expect(pipelineCalls(gpu)).toEqual(before);

  gpu.dispose();
});

test("changing the sample count is a different signature, so the prepared combination does not cover it", async () => {
  const gpu = await init();
  // A resize can never change the sample count (it is fixed by the options bag), so the signature change
  // this half of the contract talks about is observed across destinations: preparing one sample count
  // prepares that combination and no other.
  const single = target(gpu, { size: [8, 4], label: "singleSampled" });
  const multi = target(gpu, { size: [8, 4], sampleCount: 4, label: "multiSampled" });
  const fx = effect(gpu, SOLID, { label: "signatureFx" });
  await prepare(gpu, [{ draw: fx, target: single }]);

  expect(signatureKeyOf(normalizeSignature(single))).not.toBe(signatureKeyOf(normalizeSignature(multi)));
  expect(codeOf(() => frame(gpu, (f) => f.pass(single, (p) => p.draw(fx)), { pendingPipelines: "throw" }))).toBe("NO-THROW");
  expect(codeOf(() => frame(gpu, (f) => f.pass(multi, (p) => p.draw(fx)), { pendingPipelines: "throw" }))).toBe("VGPU-PIPELINE-PENDING");

  gpu.dispose();
});

// ---------------------------------------------------------------------------
// Multisampling vocabulary: `sampleCount` is the WebGPU spelling of the legacy `msaa`.
// ---------------------------------------------------------------------------

test("target({ sampleCount: 4 }) and the legacy target({ msaa: true }) are the same target", async () => {
  const gpu = await init();
  const legacy = target(gpu, { size: [8, 4], depth: true, msaa: true, label: "legacyMsaa" });
  const modern = target(gpu, { size: [8, 4], depth: true, sampleCount: 4, label: "modernSampleCount" });

  expect(modern.sampleCount).toBe(4);
  expect(legacy.sampleCount).toBe(4);
  expect(signatureKeyOf(normalizeSignature(modern))).toBe(signatureKeyOf(normalizeSignature(legacy)));
  // Same attachment shape: an MSAA color that resolves into the sampleable one.
  const modernColor = (modern.renderPassDescriptor().colorAttachments as GPURenderPassColorAttachment[])[0];
  const legacyColor = (legacy.renderPassDescriptor().colorAttachments as GPURenderPassColorAttachment[])[0];
  expect(!!modernColor?.resolveTarget).toBe(!!legacyColor?.resolveTarget);
  expect(modernColor?.storeOp).toBe(legacyColor?.storeOp);

  expect(target(gpu, { size: [8, 4], sampleCount: 1, label: "explicitSingle" }).sampleCount).toBe(1);
  expect(target(gpu, { size: [8, 4], label: "defaultSingle" }).sampleCount).toBe(1);
  gpu.dispose();
});

test("msaa and sampleCount may not disagree, and each keeps its own invalid-value error", async () => {
  const gpu = await init();

  expect(codeOf(() => target(gpu, { size: [8, 4], msaa: true, sampleCount: 1 }))).toBe("VGPU-TARGET-SAMPLE-COUNT-CONFLICT");
  expect(codeOf(() => target(gpu, { size: [8, 4], msaa: false, sampleCount: 4 }))).toBe("VGPU-TARGET-SAMPLE-COUNT-CONFLICT");
  // Agreeing spellings are accepted — they describe the same target.
  expect(target(gpu, { size: [8, 4], msaa: true, sampleCount: 4 }).sampleCount).toBe(4);
  expect(target(gpu, { size: [8, 4], msaa: false, sampleCount: 1 }).sampleCount).toBe(1);

  expect(codeOf(() => target(gpu, { size: [8, 4], sampleCount: 2 as unknown as 1 | 4 }))).toBe("VGPU-TARGET-SAMPLE-COUNT-INVALID");
  expect(codeOf(() => target(gpu, { size: [8, 4], msaa: 2 as unknown as 4 }))).toBe("VGPU-TARGET-MSAA-INVALID");
  gpu.dispose();
});
