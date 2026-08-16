// T04-21 (the `pendingPipelines` default is now "throw"): this suite encodes without `prepare()`
// on purpose -- its subject is the descriptor/encoder behavior asserted below, not readiness -- so
// it takes the permanent `"sync"` opt-in, which is exactly the eager compile-on-encode these
// assertions were written against. The default itself is covered by pending-pipelines.test.ts,
// prepare.test.ts and prepare-corpus-throw.test.ts, which run under it.
import { expect, test, vi } from "vitest";
import { getMockGPUDeviceInstrumentation } from "@vgpu/core";
import { init, draw, effect, frame, surface, target } from "../src/mock.ts";
import type { InternalDraw } from "../src/draw.ts";
import { normalizeSignature, signatureKeyOf } from "../src/pipeline-store.ts";
import { depthFormatFor } from "../src/target-utils.ts";

const WGSL = `
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(pos[vi], 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1.0); }
`;

const FRAGMENT_ONLY = `
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1.0); }
`;

interface CanvasProbe {
  readonly canvas: HTMLCanvasElement;
  /** Stable view identity of the canvas presentation texture. */
  readonly view: object;
  readonly configure: ReturnType<typeof vi.fn>;
  currentTextures: number;
  /** Resizes the canvas behind the surface's back, the way layout or app code does. */
  setSize(width: number, height: number): void;
}

interface TextureLedger {
  created: number;
  destroyed: number;
  readonly labels: string[];
}

/**
 * Counts the textures a surface OWNS. The presentation texture is wrapped directly (`new Texture(…)`,
 * "external"), never created through the device, so this ledger sees depth and MSAA attachments only.
 */
function textureLedger(gpu: { device: unknown }): TextureLedger {
  const ledger: TextureLedger = { created: 0, destroyed: 0, labels: [] };
  const device = gpu.device as { createTexture(opts: Record<string, unknown>): { destroy(): void } };
  const original = device.createTexture.bind(device);
  device.createTexture = (opts: Record<string, unknown>) => {
    const texture = original(opts);
    ledger.created += 1;
    ledger.labels.push(String(opts.label ?? ""));
    const destroy = texture.destroy.bind(texture);
    let counted = false;
    (texture as { destroy: () => void }).destroy = () => { if (!counted) { counted = true; ledger.destroyed += 1; } destroy(); };
    return texture;
  };
  return ledger;
}

/** Canvas mock with STABLE presentation texture/view identity, so resolveTarget can be compared by reference. */
function canvasProbe(width = 8, height = 4): CanvasProbe {
  const view = { __canvasView: true };
  const texture = { createView: () => view };
  const configure = vi.fn();
  const probe = { view, configure, currentTextures: 0 } as { view: object; configure: ReturnType<typeof vi.fn>; currentTextures: number; canvas: HTMLCanvasElement };
  const canvas: Record<string, unknown> = {
    width,
    height,
    getContext(kind: string) {
      if (kind !== "webgpu") return null;
      return {
        canvas,
        configure,
        unconfigure: () => undefined,
        getCurrentTexture: () => { probe.currentTextures += 1; return texture; },
      };
    },
  };
  probe.canvas = canvas as unknown as HTMLCanvasElement;
  (probe as unknown as { setSize(w: number, h: number): void }).setSize = (w: number, h: number) => {
    canvas.width = w;
    canvas.height = h;
  };
  return probe as CanvasProbe;
}

function expectNotInFrame(fn: () => unknown): void {
  try { fn(); }
  catch (error) {
    expect(error).toMatchObject({ code: "VGPU-SURFACE-NOT-IN-FRAME" });
    return;
  }
  throw new Error("Expected VGPU-SURFACE-NOT-IN-FRAME");
}

// Contract #8 (half "resolves"): compiling against a surface outside frame() is legal.
test("compile/compileSync/pipelineForAsync against a surface outside frame() never throw VGPU-SURFACE-NOT-IN-FRAME", async () => {
  const gpu = await init({ pendingPipelines: "sync" });
  try {
    const probe = canvasProbe();
    const canvasSurface = surface(gpu, probe.canvas);
    const drawable = draw(gpu, { shader: WGSL, label: "outside" });
    const shader = effect(gpu, { shader: FRAGMENT_ONLY, label: "outside-effect" });

    await expect(drawable.compile(canvasSurface)).resolves.toBe(drawable);
    expect(drawable.compileSync(canvasSurface)).toBe(drawable);
    await expect((drawable as unknown as InternalDraw).pipelineForAsync(canvasSurface)).resolves.toBeDefined();
    await expect(shader.compile(canvasSurface)).resolves.toBe(shader);
    expect(shader.compileSync(canvasSurface)).toBe(shader);

    // The whole point of the clause: no current texture is ever fetched while only compiling.
    expect(probe.currentTextures).toBe(0);
  } finally {
    gpu.dispose();
  }
});

// Contract #8 (half "signature equals"): the signature resolved outside a frame is the encoding one.
test("the surface signature resolved outside frame() equals the one used encoding inside frame()", async () => {
  const gpu = await init({ pendingPipelines: "sync" });
  try {
    const probe = canvasProbe();
    const canvasSurface = surface(gpu, probe.canvas);
    const drawable = draw(gpu, { shader: WGSL, label: "stable-signature" });
    const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

    const outside = normalizeSignature(canvasSurface);
    expect(outside).toEqual({ colors: [canvasSurface.format], depth: undefined, sampleCount: 1 });

    await drawable.compile(canvasSurface);
    expect(mock.calls.createRenderPipelineAsync).toBe(1);
    expect(mock.calls.createRenderPipeline).toBe(0);

    let inside: unknown;
    frame(gpu, (currentFrame) => {
      inside = normalizeSignature(canvasSurface);
      currentFrame.pass(canvasSurface, drawable);
    });

    expect(inside).toEqual(outside);
    expect(signatureKeyOf(inside as never)).toBe(signatureKeyOf(outside));
    // Same key ⇒ the pipeline compiled outside the frame is the one the encode path reused.
    expect(mock.calls.createRenderPipeline).toBe(0);
    expect(mock.calls.createRenderPipelineAsync).toBe(1);
  } finally {
    gpu.dispose();
  }
});

// Encode still needs the current texture: the standalone one-shot draw keeps the guard.
test("standalone draw()/effect().draw() against a surface outside frame() still throws VGPU-SURFACE-NOT-IN-FRAME", async () => {
  const gpu = await init({ pendingPipelines: "sync" });
  try {
    const probe = canvasProbe();
    const canvasSurface = surface(gpu, probe.canvas);
    const drawable = draw(gpu, { shader: WGSL, label: "encode-guard" });
    const shader = effect(gpu, { shader: FRAGMENT_ONLY, label: "encode-guard-effect" });

    expectNotInFrame(() => drawable.draw(canvasSurface));
    expectNotInFrame(() => shader.draw(canvasSurface));
    expect(() => frame(gpu, (currentFrame) => currentFrame.pass(canvasSurface, drawable))).not.toThrow();
  } finally {
    gpu.dispose();
  }
});

// Design rule: depth on surface() resolves through the shared depthFormatFor.
test("surface({ depth: true }) owns a depth attachment resolved through depthFormatFor, like target({ depth: true })", async () => {
  const gpu = await init({ pendingPipelines: "sync" });
  try {
    const probe = canvasProbe();
    const canvasSurface = surface(gpu, probe.canvas, { depth: true });
    const offscreen = target(gpu, { size: [8, 4], depth: true });

    expect(canvasSurface.depth?.format).toBe(depthFormatFor({ depth: true }));
    expect(canvasSurface.depth?.format).toBe(offscreen.depth?.format);
    expect(canvasSurface.depth?.size).toEqual([8, 4]);
    expect(normalizeSignature(canvasSurface).depth).toBe("depth24plus");

    const descriptor = canvasSurface.renderPassDescriptor();
    expect(descriptor.depthStencilAttachment).toBeDefined();
    expect(descriptor.depthStencilAttachment?.depthLoadOp).toBe("clear");
    expect(descriptor.depthStencilAttachment?.depthStoreOp).toBe("store");

    // Explicit depth formats pass through unchanged, stencil aspect included.
    const stencilSurface = surface(gpu, canvasProbe().canvas, { depth: "depth24plus-stencil8" });
    expect(stencilSurface.depth?.format).toBe("depth24plus-stencil8");
    expect(stencilSurface.renderPassDescriptor().depthStencilAttachment?.stencilLoadOp).toBe("clear");
  } finally {
    gpu.dispose();
  }
});

// Design rule: MSAA resolves into the presentation texture.
test("surface({ sampleCount: 4 }) renders into an internal multisample attachment and resolves into the canvas texture", async () => {
  const gpu = await init({ pendingPipelines: "sync" });
  try {
    const probe = canvasProbe();
    const canvasSurface = surface(gpu, probe.canvas, { sampleCount: 4, depth: true });

    expect(canvasSurface.sampleCount).toBe(4);
    expect(normalizeSignature(canvasSurface).sampleCount).toBe(4);
    expect(canvasSurface.depth?.sampleCount).toBe(4);

    const descriptor = canvasSurface.renderPassDescriptor();
    const color = descriptor.colorAttachments[0]!;
    expect(color.resolveTarget).toBe(probe.view);
    expect(color.view).not.toBe(probe.view);
    expect(color.storeOp).toBe("discard");
    expect(descriptor.depthStencilAttachment?.depthStoreOp).toBe("discard");

    // The pipeline compiled for this surface is multisampled — and it is derivable outside a frame.
    const drawable = draw(gpu, { shader: WGSL, label: "msaa-surface" });
    await drawable.compile(canvasSurface);
    const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
    expect(mock.createRenderPipelineAsyncDescriptors.at(-1)?.multisample?.count).toBe(4);
    expect(mock.createRenderPipelineAsyncDescriptors.at(-1)?.depthStencil?.format).toBe("depth24plus");
  } finally {
    gpu.dispose();
  }
});

// Contract #23 (partial): resize invalidates by signature, not identity.
test("a resize that preserves format, depth and sample count keeps the signature and creates no pipeline", async () => {
  const gpu = await init({ pendingPipelines: "sync" });
  try {
    const probe = canvasProbe();
    const canvasSurface = surface(gpu, probe.canvas, { depth: true, sampleCount: 4 });
    const drawable = draw(gpu, { shader: WGSL, label: "resize-stable" });
    const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

    const before = signatureKeyOf(normalizeSignature(canvasSurface));
    await drawable.compile(canvasSurface);
    const createdBefore = mock.calls.createRenderPipeline + mock.calls.createRenderPipelineAsync;

    canvasSurface.resize([16, 9]);

    expect(canvasSurface.size).toEqual([16, 9]);
    expect(signatureKeyOf(normalizeSignature(canvasSurface))).toBe(before);
    // Internal attachments follow the new size while the signature stays put.
    expect(canvasSurface.depth?.size).toEqual([16, 9]);

    frame(gpu, (currentFrame) => currentFrame.pass(canvasSurface, drawable));
    expect(mock.calls.createRenderPipeline + mock.calls.createRenderPipelineAsync).toBe(createdBefore);
  } finally {
    gpu.dispose();
  }
});

// Zero regression: a plain surface behaves exactly as in 0.3.0.
test("surface(gpu, canvas) without the new options stays color-only, sampleCount 1, and allocates no textures", async () => {
  const gpu = await init({ pendingPipelines: "sync" });
  try {
    const probe = canvasProbe();
    const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
    const texturesBefore = mock.calls.createBuffer; // buffers untouched; textures counted below
    const canvasSurface = surface(gpu, probe.canvas);

    expect(canvasSurface.depth).toBeUndefined();
    expect(canvasSurface.sampleCount).toBe(1);
    expect(canvasSurface.colors).toHaveLength(1);

    const descriptor = canvasSurface.renderPassDescriptor();
    expect(descriptor.depthStencilAttachment).toBeUndefined();
    expect(descriptor.colorAttachments).toHaveLength(1);
    const color = descriptor.colorAttachments[0]!;
    expect(color.view).toBe(probe.view);
    expect(color.resolveTarget).toBeUndefined();
    expect(color.loadOp).toBe("clear");
    expect(color.storeOp).toBe("store");
    expect(mock.calls.createBuffer).toBe(texturesBefore);
  } finally {
    gpu.dispose();
  }
});

// A frame-independent signature is not a lifetime-independent one: disposal still wins.
test("a disposed surface rejects every signature consumer, compile included", async () => {
  const gpu = await init({ pendingPipelines: "sync" });
  try {
    const probe = canvasProbe();
    const canvasSurface = surface(gpu, probe.canvas, { depth: true });
    const drawable = draw(gpu, { shader: WGSL, label: "disposed" });
    canvasSurface.dispose();

    const expectDisposed = (fn: () => unknown): void => expect(fn).toThrowError(expect.objectContaining({ code: "VGPU-SURFACE-DISPOSED" }));
    expectDisposed(() => normalizeSignature(canvasSurface));
    expectDisposed(() => drawable.compileSync(canvasSurface));
    expectDisposed(() => (drawable as unknown as InternalDraw).pipelineFor(canvasSurface));
    // compile() resolves its key before it awaits anything, so it reports this synchronously too.
    expectDisposed(() => drawable.compile(canvasSurface));
  } finally {
    gpu.dispose();
  }
});

// The attachments a consumer caches views over are recreated by BOTH resize paths, so both announce it.
test("onTexturesRecreated fires when the canvas is resized behind the surface's back, not only through resize()", async () => {
  const gpu = await init({ pendingPipelines: "sync" });
  try {
    const probe = canvasProbe();
    const canvasSurface = surface(gpu, probe.canvas, { depth: true });
    let recreated = 0;
    // Same shape set-resources.ts consumes (RecreatingTarget): the hook is on the surface, not on the Target interface.
    (canvasSurface as unknown as { onTexturesRecreated(cb: () => void): () => void }).onTexturesRecreated(() => { recreated += 1; });

    const first = canvasSurface.depth;
    // Canvas resized directly: never goes through resize(), so only the lazy sync notices it.
    (probe.canvas as unknown as { width: number }).width = 32;
    const afterDrift = canvasSurface.depth;
    expect(afterDrift).not.toBe(first);
    expect(afterDrift?.size).toEqual([32, 4]);
    expect(recreated).toBe(1);

    canvasSurface.resize([48, 24]);
    expect(recreated).toBe(2);
    expect(canvasSurface.depth?.size).toEqual([48, 24]);
  } finally {
    gpu.dispose();
  }
});

// Attachment sizes obey the same 1×1 floor every other size on this surface does.
test("a zero-sized canvas still allocates a valid 1x1 attachment and does not re-create it on every read", async () => {
  const gpu = await init({ pendingPipelines: "sync" });
  try {
    const probe = canvasProbe(0, 0);
    const canvasSurface = surface(gpu, probe.canvas, { depth: true });
    const first = canvasSurface.depth;
    expect(first?.size).toEqual([1, 1]);
    // Sanitized on both sides of the comparison: no thrash between the 0×0 canvas and the 1×1 attachment.
    expect(canvasSurface.depth).toBe(first);
  } finally {
    gpu.dispose();
  }
});

// frame.ts branches that a Surface could never reach before it owned depth/MSAA.
test("pass options that depend on depth and MSAA now apply to surfaces exactly as they do to targets", async () => {
  const gpu = await init({ pendingPipelines: "sync" });
  try {
    const msaaDepth = surface(gpu, canvasProbe().canvas, { depth: true, sampleCount: 4 });
    const plainDepth = surface(gpu, canvasProbe().canvas, { depth: true });
    const stencil = surface(gpu, canvasProbe().canvas, { depth: "depth24plus-stencil8" });
    const colorOnly = surface(gpu, canvasProbe().canvas);
    const drawable = draw(gpu, { shader: WGSL, label: "pass-matrix" });

    // Empty body on purpose: this pins frame.pass's own option validation, not the separate rule that
    // rejects a depth-writing pipeline inside a read-only pass.
    const run = (options: Record<string, unknown>): unknown => {
      try { frame(gpu, (currentFrame) => currentFrame.pass(options as never, () => undefined)); return "ok"; }
      catch (error) { return (error as { code?: string }).code; }
    };
    void drawable;

    // MSAA discards what it resolves, so neither preserving nor reading depth back is coherent.
    expect(run({ target: msaaDepth, clear: false })).toBe("VGPU-PASS-PRESERVE-MSAA");
    expect(run({ target: msaaDepth, depthReadOnly: true })).toBe("VGPU-PASS-DEPTH-READONLY-MSAA");
    // Depth-dependent options are live on a depth surface and dead options on a color-only one.
    expect(run({ target: plainDepth, clearDepth: 0.5 })).toBe("ok");
    expect(run({ target: colorOnly, clearDepth: 0.5 })).toBe("VGPU-PASS-CLEARDEPTH-INVALID");
    expect(run({ target: plainDepth, depthReadOnly: true })).toBe("ok");
    expect(run({ target: colorOnly, depthReadOnly: true })).toBe("VGPU-PASS-DEPTH-READONLY");
    // clearStencil follows the stencil aspect of the surface's own depth format.
    expect(run({ target: stencil, clearStencil: 1 })).toBe("ok");
    expect(run({ target: plainDepth, clearStencil: 1 })).toBe("VGPU-PASS-CLEARSTENCIL-INVALID");
  } finally {
    gpu.dispose();
  }
});

// Donated from the independent QA probe suite (M7): the DoD's dispose clause, previously uncaught by
// any test in this branch — mutating #destroyAttachments to a no-op survived the whole suite.
test("dispose destroys the attachments the surface owns exactly once, and stays idempotent", async () => {
  const gpu = await init({ pendingPipelines: "sync" });
  try {
    const ledger = textureLedger(gpu);
    const canvasSurface = surface(gpu, canvasProbe().canvas, { depth: true, sampleCount: 4 });
    expect(ledger.created).toBe(2);

    canvasSurface.dispose();
    canvasSurface.dispose();
    canvasSurface.dispose();

    expect(ledger.destroyed).toBe(2);
    expect(canvasSurface.disposed).toBe(true);
  } finally {
    gpu.dispose();
  }
});

// Donated from the independent QA probe suite (A6): every recreate must destroy the pair it replaced,
// through both resize paths, or a long-lived canvas leaks a texture per resize.
test("repeated resizes and canvas drift never leak an attachment", async () => {
  const gpu = await init({ pendingPipelines: "sync" });
  try {
    const ledger = textureLedger(gpu);
    const probe = canvasProbe();
    const canvasSurface = surface(gpu, probe.canvas, { depth: true, sampleCount: 4 });

    for (let i = 1; i <= 25; i += 1) {
      if (i % 2 === 0) canvasSurface.resize([8 + i, 4 + i]);
      else { probe.setSize(8 + i, 4 + i); void canvasSurface.depth; }
      // Exactly one live pair (depth + msaa) at every step, never a second one.
      expect(ledger.created - ledger.destroyed).toBe(2);
    }

    canvasSurface.dispose();
    expect(ledger.created).toBe(ledger.destroyed);
    expect(ledger.labels.filter((label) => label.includes("msaa")).length).toBe(ledger.created / 2);
  } finally {
    gpu.dispose();
  }
});

// Donated from the independent QA probe suite (M10): the duck-typed branch must hand out a fresh
// signature, never the surface's own state, and must still normalize a plain TargetSignature.
test("normalizeSignature returns a non-aliased signature a caller cannot use to poison the surface", async () => {
  const gpu = await init({ pendingPipelines: "sync" });
  try {
    const canvasSurface = surface(gpu, canvasProbe().canvas, { depth: true, sampleCount: 4 });
    const first = normalizeSignature(canvasSurface);
    const second = normalizeSignature(canvasSurface);
    expect(first).not.toBe(second);
    expect(first.colors).not.toBe(second.colors);

    (first.colors as GPUTextureFormat[])[0] = "r8unorm";
    expect(normalizeSignature(canvasSurface).colors[0]).toBe(canvasSurface.format);

    // A plain TargetSignature has no pipelineSignature to prefer, so it still takes the normal path.
    const plain = normalizeSignature({ colors: ["rgba8unorm"], depth: "depth24plus", sampleCount: 4 } as never);
    expect(plain).toEqual({ colors: ["rgba8unorm"], depth: "depth24plus", sampleCount: 4 });
  } finally {
    gpu.dispose();
  }
});
