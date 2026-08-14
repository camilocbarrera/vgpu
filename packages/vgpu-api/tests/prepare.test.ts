import { expect, test, vi } from "vitest";
import { getMockGPUDeviceInstrumentation } from "@vgpu/core";
import { bundle, compute, draw, effect, frame, init, prepare, storage, surface, target } from "../src/mock.ts";
import { normalizeSignature } from "../src/pipeline-store.ts";

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

const BOUND_WGSL = `
@group(0) @binding(0) var<storage, read> data: array<f32>;
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  return vec4f(data[vi], 0.0, 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1.0); }
`;

const COMPUTE_WGSL = `
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) { data[id.x] = 1.0; }
`;

function surfaceCanvas(): HTMLCanvasElement {
  const canvas: Record<string, unknown> = { width: 4, height: 4 };
  canvas.getContext = (kind: string) => kind === "webgpu" ? {
    configure: () => undefined,
    getCurrentTexture: () => ({ createView: () => ({}) }),
  } : null;
  return canvas as HTMLCanvasElement;
}

/**
 * Records every pipeline the encode path binds, so `prepared.gpu` can be asserted by IDENTITY
 * against the object a real `pass.draw()` uses — not against a call count.
 */
/** Returns the error a synchronous call threw, so assertions can pin `code` instead of message text. */
function caught(fn: () => unknown): { code?: string; message?: string } | undefined {
  try { fn(); }
  catch (error) { return error as { code?: string; message?: string }; }
  return undefined;
}

function recordEncodedPipelines(device: GPUDevice): GPURenderPipeline[] {
  const bound: GPURenderPipeline[] = [];
  const createCommandEncoder = device.createCommandEncoder.bind(device);
  vi.spyOn(device, "createCommandEncoder").mockImplementation((desc?: GPUCommandEncoderDescriptor) => {
    const encoder = createCommandEncoder(desc);
    const beginRenderPass = encoder.beginRenderPass.bind(encoder);
    encoder.beginRenderPass = (passDesc: GPURenderPassDescriptor) => {
      const pass = beginRenderPass(passDesc);
      const setPipeline = pass.setPipeline.bind(pass);
      pass.setPipeline = (pipeline: GPURenderPipeline) => { bound.push(pipeline); setPipeline(pipeline); };
      return pass;
    };
    return encoder;
  });
  return bound;
}

// Contract #2 — "After a completed prepare() on a {draw, target} combination, encoding that
// combination in a frame() creates no pipelines."
test("contract #2: a prepared {draw, target} encodes in frame() without creating any pipeline", async () => {
  const gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: WGSL, label: "prepared" });
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  await prepare(gpu, { draw: drawable, target: colorTarget });
  const asyncAfterPrepare = mock.calls.createRenderPipelineAsync;
  expect(asyncAfterPrepare).toBe(1);
  expect(mock.calls.createRenderPipeline).toBe(0);

  frame(gpu, (f) => f.pass(colorTarget, (p) => p.draw(drawable)));

  expect(mock.calls.createRenderPipeline).toBe(0);
  expect(mock.calls.createRenderPipelineAsync).toBe(asyncAfterPrepare);
  gpu.dispose();
});

test("contract #2: a prepared {effect, target} encodes in frame() without creating any pipeline", async () => {
  const gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  const shader = effect(gpu, FRAGMENT_ONLY, { label: "preparedFx" });
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  const prepared = await prepare(gpu, { draw: shader, target: colorTarget });
  expect(prepared.draw).toBe(shader);
  expect(mock.calls.createRenderPipelineAsync).toBe(1);

  frame(gpu, (f) => f.pass(colorTarget, (p) => p.draw(shader)));

  expect(mock.calls.createRenderPipeline).toBe(0);
  expect(mock.calls.createRenderPipelineAsync).toBe(1);
  gpu.dispose();
});

// Contract #6 — combination-scoped readiness.
test("contract #6: preparing {draw, a} does not prepare {draw, b}", async () => {
  const gpu = await init();
  const plain = target(gpu, { size: [4, 4] });
  const depthTarget = target(gpu, { size: [4, 4], depth: true });
  const drawable = draw(gpu, { shader: WGSL, label: "scoped" });
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  const preparedA = await prepare(gpu, { draw: drawable, target: plain });
  expect(mock.calls.createRenderPipelineAsync).toBe(1);

  // The other combination is NOT ready: encoding it with the default "sync" policy has to compile.
  frame(gpu, (f) => f.pass(depthTarget, (p) => p.draw(drawable)));
  expect(mock.calls.createRenderPipeline).toBe(1);

  const preparedB = await prepare(gpu, { draw: drawable, target: depthTarget });
  expect(preparedB.gpu).not.toBe(preparedA.gpu);
  expect(preparedA.signature.depth).toBeUndefined();
  expect(preparedB.signature.depth).toBe("depth24plus");
  gpu.dispose();
});

test("contract #6: a failure on {draw, b} neither throws for {draw, a} nor mutates state observable through the draw", async () => {
  const gpu = await init();
  const plain = target(gpu, { size: [4, 4] });
  const depthTarget = target(gpu, { size: [4, 4], depth: true });
  const drawable = draw(gpu, { shader: WGSL, label: "isolated" });
  const nativeError = new Error("depth pipeline rejected");
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
  const createAsync = gpu.device.gpu.createRenderPipelineAsync.bind(gpu.device.gpu);
  vi.spyOn(gpu.device.gpu, "createRenderPipelineAsync").mockImplementation((desc: GPURenderPipelineDescriptor) => {
    if (desc.depthStencil) return Promise.reject(nativeError);
    return createAsync(desc);
  });

  const preparedA = await prepare(gpu, { draw: drawable, target: plain });
  expect(preparedA.gpu).toBeDefined();
  const pipelineA = drawable.gpu;

  await expect(prepare(gpu, { draw: drawable, target: depthTarget })).rejects.toMatchObject({ code: "VGPU-PREPARE-FAILED" });

  // The good combination is untouched: same pipeline object, still encodes with zero new pipelines.
  expect(drawable.gpu).toBe(pipelineA);
  const before = { sync: mock.calls.createRenderPipeline, async: mock.calls.createRenderPipelineAsync };
  expect(() => frame(gpu, (f) => f.pass(plain, (p) => p.draw(drawable)))).not.toThrow();
  expect(mock.calls.createRenderPipeline).toBe(before.sync);
  expect(mock.calls.createRenderPipelineAsync).toBe(before.async);
  await expect(prepare(gpu, { draw: drawable, target: plain })).resolves.toMatchObject({ gpu: preparedA.gpu });
  gpu.dispose();
});

// Contract #7 — handles.
test("contract #7: the array form preserves order, and prepared.gpu is the object the encode path binds", async () => {
  const gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  const first = draw(gpu, { shader: WGSL, label: "first" });
  const second = effect(gpu, FRAGMENT_ONLY, { label: "second" });
  const kernel = compute(gpu, COMPUTE_WGSL, { label: "kernel" });
  kernel.set({ data: storage(gpu, 16) });
  const bound = recordEncodedPipelines(gpu.device.gpu);

  const [a, b, c] = await prepare(gpu, [
    { draw: first, target: colorTarget },
    { draw: second, target: colorTarget },
    { compute: kernel },
  ]);

  // Order preserved, and each handle echoes the renderable it was requested with.
  expect(a.draw).toBe(first);
  expect(b.draw).toBe(second);
  expect(c.compute).toBe(kernel);
  expect(a.gpu).not.toBe(b.gpu);

  frame(gpu, (f) => f.pass(colorTarget, (p) => { p.draw(first); p.draw(second); }));

  // Identity, not counts: the encode path bound exactly the handles prepare() returned.
  expect(bound).toEqual([a.gpu, b.gpu]);
  gpu.dispose();
});

test("contract #7: prepare() rejects with VGPU-PREPARE-FAILED listing every failed combination", async () => {
  const gpu = await init();
  const plain = target(gpu, { size: [4, 4] });
  const depthTarget = target(gpu, { size: [4, 4], depth: true });
  const ok = draw(gpu, { shader: WGSL, label: "ok" });
  const bad = draw(gpu, { shader: WGSL, label: "bad" });
  const nativeError = new Error("nope");
  const createAsync = gpu.device.gpu.createRenderPipelineAsync.bind(gpu.device.gpu);
  vi.spyOn(gpu.device.gpu, "createRenderPipelineAsync").mockImplementation((desc: GPURenderPipelineDescriptor) => {
    if (desc.depthStencil) return Promise.reject(nativeError);
    return createAsync(desc);
  });

  const rejection = await prepare(gpu, [
    { draw: ok, target: plain },
    { draw: bad, target: depthTarget },
    { draw: ok, target: depthTarget },
  ]).then(() => undefined, (error: unknown) => error) as { code: string; message: string; detail: { failures: { label: string; signature?: string; cause: { code?: string; cause?: unknown } }[] } };

  expect(rejection.code).toBe("VGPU-PREPARE-FAILED");
  // Every failed combination is enumerated — both of them, in request order, with the signature
  // each one resolved to. The succeeding one is NOT in the list.
  expect(rejection.detail.failures.map(({ label, signature }) => ({ label, signature }))).toEqual([
    { label: "bad", signature: "rgba8unorm:depth24plus:1" },
    { label: "ok", signature: "rgba8unorm:depth24plus:1" },
  ]);
  // Each failure keeps the whole diagnosis chain: the vgpu compile error, carrying the native cause.
  for (const failure of rejection.detail.failures) {
    expect(failure.cause).toMatchObject({ code: "VGPU-COMPILE-FAILED", cause: nativeError });
  }
  expect(rejection.message).toContain("bad");
  expect(rejection.message).toContain("rgba8unorm:depth24plus:1");

  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
  const asyncCalls = mock.calls.createRenderPipelineAsync;
  // The subset that did compile stays cached: re-preparing it performs no new pipeline creation.
  await expect(prepare(gpu, [{ draw: ok, target: plain }])).resolves.toHaveLength(1);
  expect(mock.calls.createRenderPipelineAsync).toBe(asyncCalls);
  expect(mock.calls.createRenderPipeline).toBe(0);
  gpu.dispose();
});

test("contract #7: a failed combination is retried by a later prepare(), never poisoned", async () => {
  const gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: WGSL, label: "retry" });
  const createAsync = gpu.device.gpu.createRenderPipelineAsync.bind(gpu.device.gpu);
  let fail = true;
  vi.spyOn(gpu.device.gpu, "createRenderPipelineAsync").mockImplementation((desc: GPURenderPipelineDescriptor) => {
    if (fail) return Promise.reject(new Error("transient"));
    return createAsync(desc);
  });

  await expect(prepare(gpu, { draw: drawable, target: colorTarget })).rejects.toMatchObject({ code: "VGPU-PREPARE-FAILED" });
  fail = false;
  const prepared = await prepare(gpu, { draw: drawable, target: colorTarget });
  expect(prepared.gpu).toBeDefined();
  gpu.dispose();
});

test("contract #7: two concurrent prepare() calls on the same combination share one native compile", async () => {
  const gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: WGSL, label: "concurrent" });
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  const [a, b] = await Promise.all([
    prepare(gpu, { draw: drawable, target: colorTarget }),
    prepare(gpu, { draw: drawable, target: colorTarget }),
  ]);

  expect(a.gpu).toBe(b.gpu);
  expect(mock.calls.createRenderPipelineAsync).toBe(1);
  expect(mock.calls.createRenderPipeline).toBe(0);
  gpu.dispose();
});

// Contract #8 — Surface outside frame().
test("contract #8: prepare() against a Surface outside frame() resolves with the in-frame signature", async () => {
  const gpu = await init();
  const canvasSurface = surface(gpu, surfaceCanvas());
  const drawable = draw(gpu, { shader: WGSL, label: "surfaced" });
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
  const bound = recordEncodedPipelines(gpu.device.gpu);

  const prepared = await prepare(gpu, { draw: drawable, target: canvasSurface });

  expect(prepared.signature).toEqual(normalizeSignature(canvasSurface));
  expect(mock.calls.createRenderPipelineAsync).toBe(1);

  // Same signature inside a frame => the prepared pipeline is reused verbatim.
  frame(gpu, (f) => f.pass(canvasSurface, (p) => p.draw(drawable)));
  expect(bound).toEqual([prepared.gpu]);
  expect(mock.calls.createRenderPipeline).toBe(0);
  gpu.dispose();
});

// Contract #4 (the half T04-04 left open): {compute} in prepare().
test("contract #4: prepare({ compute }) compiles with createComputePipelineAsync and dispatch reuses it", async () => {
  const gpu = await init();
  const kernel = compute(gpu, COMPUTE_WGSL, { label: "sim" });
  kernel.set({ data: storage(gpu, 16) });
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  const prepared = await prepare(gpu, { compute: kernel });

  expect(prepared.compute).toBe(kernel);
  expect(prepared.gpu).toBeDefined();
  expect(mock.calls.createComputePipelineAsync).toBe(1);
  expect(mock.calls.createComputePipeline).toBe(0);

  kernel.dispatch(1);
  expect(mock.calls.createComputePipeline).toBe(0);
  expect(mock.calls.createComputePipelineAsync).toBe(1);

  // Idempotent: re-preparing a ready compute creates nothing.
  await expect(prepare(gpu, { compute: kernel })).resolves.toMatchObject({ gpu: prepared.gpu });
  expect(mock.calls.createComputePipelineAsync).toBe(1);
  gpu.dispose();
});

test("contract #4: prepare({ compute }) does not recompile a kernel a previous dispatch already compiled", async () => {
  const gpu = await init();
  const kernel = compute(gpu, COMPUTE_WGSL, { label: "warmed" });
  kernel.set({ data: storage(gpu, 16) });
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  kernel.dispatch(1);
  expect(mock.calls.createComputePipeline).toBe(1);

  await prepare(gpu, { compute: kernel });
  expect(mock.calls.createComputePipelineAsync).toBe(0);
  expect(mock.calls.createComputePipeline).toBe(1);
  gpu.dispose();
});

test("prepare() reports a failed compute combination through VGPU-PREPARE-FAILED", async () => {
  const gpu = await init();
  const kernel = compute(gpu, COMPUTE_WGSL, { label: "brokenSim" });
  kernel.set({ data: storage(gpu, 16) });
  const nativeError = new Error("compute compile failed");
  vi.spyOn(gpu.device.gpu, "createComputePipelineAsync").mockRejectedValue(nativeError);

  await expect(prepare(gpu, [{ compute: kernel }])).rejects.toMatchObject({
    code: "VGPU-PREPARE-FAILED",
    detail: { failures: [{ label: "brokenSim", cause: nativeError }] },
  });
  gpu.dispose();
});

test("prepare() rejects a disposed gpu and an unknown request shape", async () => {
  const gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: WGSL, label: "guarded" });

  await expect(prepare(gpu, { draw: drawable } as never)).rejects.toMatchObject({ code: "VGPU-PREPARE-FAILED" });

  gpu.dispose();
  await expect(prepare(gpu, { draw: drawable, target: colorTarget })).rejects.toMatchObject({ code: "VGPU-GPU-DISPOSED" });
});

test("gpu.dispose() with a prepare() in flight rejects it instead of leaving it hanging", async () => {
  const gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: WGSL, label: "disposedMidFlight" });
  vi.spyOn(gpu.device.gpu, "createRenderPipelineAsync").mockImplementation(() => new Promise<GPURenderPipeline>(() => undefined));
  const unhandled: unknown[] = [];
  const onUnhandled = (reason: unknown) => { unhandled.push(reason); };
  process.on("unhandledRejection", onUnhandled);

  try {
    const pending = prepare(gpu, { draw: drawable, target: colorTarget });
    gpu.dispose();

    const rejection = await pending.then(() => undefined, (error: unknown) => error) as { code: string; detail: { failures: { cause: { code?: string } }[] } };
    expect(rejection.code).toBe("VGPU-PREPARE-FAILED");
    expect(rejection.detail.failures[0]?.cause).toMatchObject({ code: "VGPU-COMPILE-DISPOSED" });
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(unhandled).toEqual([]);
  } finally {
    process.off("unhandledRejection", onUnhandled);
  }
});

// The T04-02 interaction: a surface signature comes from its configuration, so a resize landing
// while prepare() is resolving cannot change the signature the combination was keyed on.
test("a Surface resize while prepare() is in flight does not change the resolved signature", async () => {
  const gpu = await init();
  const canvas = surfaceCanvas();
  const canvasSurface = surface(gpu, canvas);
  const drawable = draw(gpu, { shader: WGSL, label: "resizedMidFlight" });
  let resolveNative!: (pipeline: GPURenderPipeline) => void;
  const createAsync = gpu.device.gpu.createRenderPipelineAsync.bind(gpu.device.gpu);
  vi.spyOn(gpu.device.gpu, "createRenderPipelineAsync").mockImplementation(async (desc: GPURenderPipelineDescriptor) => {
    const real = await createAsync(desc);
    return new Promise<GPURenderPipeline>((resolve) => { resolveNative = () => resolve(real); });
  });
  const signatureBefore = normalizeSignature(canvasSurface);

  const pending = prepare(gpu, { draw: drawable, target: canvasSurface });
  await Promise.resolve();
  (canvas as { width: number }).width = 16;
  (canvas as { height: number }).height = 16;
  frame(gpu, () => undefined);
  resolveNative({} as GPURenderPipeline);

  const prepared = await pending;
  expect(prepared.signature).toEqual(signatureBefore);
  expect(prepared.signature).toEqual(normalizeSignature(canvasSurface));

  // And the resized surface still encodes with that very pipeline: no new one is created.
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
  const before = mock.calls.createRenderPipeline;
  frame(gpu, (f) => f.pass(canvasSurface, (p) => p.draw(drawable)));
  expect(mock.calls.createRenderPipeline).toBe(before);
  gpu.dispose();
});

// Trimmed bundle scope: prepare() re-encodes a stale bundle from its retained recording.
test("prepare({ bundle }) re-encodes a bundle made stale by a binding identity change", async () => {
  const gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: BOUND_WGSL, label: "bundled" });
  drawable.set({ data: storage(gpu, 16) });
  const recorded = bundle(gpu, { target: colorTarget, label: "forest" }, (r) => r.draw(drawable));
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
  const encodersAfterRecord = mock.calls.createRenderBundleEncoder;
  const nativeBefore = recorded.gpu;

  drawable.set({ data: storage(gpu, 16) });
  expect(caught(() => frame(gpu, (f) => f.pass(colorTarget, (p) => p.bundles(recorded))))?.code).toBe("VGPU-R3-BUNDLE-STALE");

  const prepared = await prepare(gpu, { bundle: recorded });

  expect(mock.calls.createRenderBundleEncoder).toBe(encodersAfterRecord + 1);
  expect(prepared.bundle).toBe(recorded);
  expect(prepared.gpu).toBe(recorded.gpu);
  expect(recorded.gpu).not.toBe(nativeBefore);
  expect(() => frame(gpu, (f) => f.pass(colorTarget, (p) => p.bundles(recorded)))).not.toThrow();
  gpu.dispose();
});

test("prepare({ bundle }) on a fresh bundle is a no-op that returns the recorded native bundle", async () => {
  const gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: WGSL, label: "fresh" });
  const recorded = bundle(gpu, { target: colorTarget, label: "freshBundle" }, (r) => r.draw(drawable));
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
  const encoders = mock.calls.createRenderBundleEncoder;

  const prepared = await prepare(gpu, { bundle: recorded });

  expect(prepared.gpu).toBe(recorded.gpu);
  expect(prepared.signature).toEqual(normalizeSignature(colorTarget));
  expect(mock.calls.createRenderBundleEncoder).toBe(encoders);
  gpu.dispose();
});

test("prepare({ bundle }) does not auto-heal a real signature mismatch", async () => {
  const gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  const otherTarget = target(gpu, { size: [4, 4], depth: true });
  const drawable = draw(gpu, { shader: WGSL, label: "mismatch" });
  const recorded = bundle(gpu, { target: colorTarget, label: "mismatchBundle" }, (r) => r.draw(drawable));

  await prepare(gpu, { bundle: recorded });

  expect(caught(() => frame(gpu, (f) => f.pass(otherTarget, (p) => p.bundles(recorded))))?.code).toBe("VGPU-R3-BUNDLE-STALE");
  gpu.dispose();
});
