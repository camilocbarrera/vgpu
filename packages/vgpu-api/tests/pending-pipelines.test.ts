import { expect, test, vi } from "vitest";
import { getMockGPUDeviceInstrumentation } from "@vgpu/core";
import { draw, effect, frame, init, prepare, target } from "../src/mock.ts";
import { DEFAULT_PENDING_PIPELINES } from "../src/pending-pipelines.ts";

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

/** The exact actionable text the design mandates for VGPU-PIPELINE-PENDING. */
const PENDING_FIX = "await prepare(gpu, [{draw, target}]) before drawing, or opt in to inline compilation with pendingPipelines: 'sync'";

function countPipelines(gpu: { device: { gpu: GPUDevice } }): { sync: number; async: number } {
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
  return { sync: mock.calls.createRenderPipeline, async: mock.calls.createRenderPipelineAsync };
}

/** Returns the error a synchronous call threw, so assertions can pin `code` instead of message text. */
function caught(fn: () => unknown): { code?: string; fix?: string; message?: string } | undefined {
  try { fn(); }
  catch (error) { return error as { code?: string; fix?: string; message?: string }; }
  return undefined;
}

function boundPipelines(device: GPUDevice): GPURenderPipeline[] {
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

// AC #3 of the frozen design, in its final form (T04-21 flipped the constant): with no policy named
// anywhere, an unprepared encode throws and compiles NOTHING. Pinned on the exported constant AND on
// behavior, because either one alone can go green for the wrong reason -- a constant nobody reads,
// or a throw that some other guard produced.
test("the default with no policy anywhere is \"throw\": an unprepared draw raises and compiles nothing", async () => {
  expect(DEFAULT_PENDING_PIPELINES).toBe("throw");
  const gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: WGSL, label: "defaultThrow" });
  const bound = boundPipelines(gpu.device.gpu);

  const error = caught(() => frame(gpu, (f) => f.pass(colorTarget, (p) => p.draw(drawable))));

  expect(error?.code).toBe("VGPU-PIPELINE-PENDING");
  expect(countPipelines(gpu)).toEqual({ sync: 0, async: 0 });
  expect(bound).toHaveLength(0);
  gpu.dispose();
});

// The other half of the flip, and the one that keeps "sync" covered forever: it stopped being the
// default, it did not stop being a value. This is the verbatim body of the pre-T04-21 default test,
// with the policy now named explicitly at init() -- so the eager compile-on-encode path every
// pre-flip program relied on still has a test that fails if it regresses.
test("\"sync\" is still a first-class opt-in: init({ pendingPipelines: \"sync\" }) compiles inline", async () => {
  const gpu = await init({ pendingPipelines: "sync" });
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: WGSL, label: "optInSync" });
  const bound = boundPipelines(gpu.device.gpu);

  expect(() => frame(gpu, (f) => f.pass(colorTarget, (p) => p.draw(drawable)))).not.toThrow();

  expect(countPipelines(gpu)).toEqual({ sync: 1, async: 0 });
  expect(bound).toHaveLength(1);
  gpu.dispose();
});

test("\"throw\" does not start compilation and reports the actionable VGPU-PIPELINE-PENDING message", async () => {
  const gpu = await init({ pendingPipelines: "throw" });
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: WGSL, label: "unprepared" });
  const bound = boundPipelines(gpu.device.gpu);

  let thrown: { code?: string; fix?: string; message?: string } | undefined;
  try { frame(gpu, (f) => f.pass(colorTarget, (p) => p.draw(drawable))); }
  catch (error) { thrown = error as { code?: string; fix?: string }; }

  expect(thrown?.code).toBe("VGPU-PIPELINE-PENDING");
  expect(thrown?.fix).toBe(PENDING_FIX);
  expect(thrown?.message).toContain("unprepared");
  expect(thrown?.message).toContain("rgba8unorm:none:1");
  // "does not start compilation": neither the sync nor the async native call happened.
  expect(countPipelines(gpu)).toEqual({ sync: 0, async: 0 });
  expect(bound).toEqual([]);
  gpu.dispose();
});

test("\"throw\" lets a prepared combination draw normally", async () => {
  const gpu = await init({ pendingPipelines: "throw" });
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: WGSL, label: "readyUnderThrow" });

  const prepared = await prepare(gpu, { draw: drawable, target: colorTarget });
  const bound = boundPipelines(gpu.device.gpu);

  expect(() => frame(gpu, (f) => f.pass(colorTarget, (p) => p.draw(drawable)))).not.toThrow();
  expect(bound).toEqual([prepared.gpu]);
  expect(countPipelines(gpu)).toEqual({ sync: 0, async: 1 });
  gpu.dispose();
});

test("\"skip\" omits the draw without throwing and starts background compilation", async () => {
  const gpu = await init({ pendingPipelines: "skip" });
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: WGSL, label: "streamed" });
  const bound = boundPipelines(gpu.device.gpu);

  expect(() => frame(gpu, (f) => f.pass(colorTarget, (p) => p.draw(drawable)))).not.toThrow();

  expect(bound).toEqual([]);
  expect(countPipelines(gpu)).toEqual({ sync: 0, async: 1 });

  // A second frame before the compile lands continues the same compilation, never a second one.
  expect(() => frame(gpu, (f) => f.pass(colorTarget, (p) => p.draw(drawable)))).not.toThrow();
  expect(countPipelines(gpu).async).toBe(1);

  await gpu.settled();

  // Once it landed, the same "skip" policy draws it — no inline compile ever happened.
  expect(() => frame(gpu, (f) => f.pass(colorTarget, (p) => p.draw(drawable)))).not.toThrow();
  expect(bound).toHaveLength(1);
  expect(countPipelines(gpu)).toEqual({ sync: 0, async: 1 });
  gpu.dispose();
});

test("\"skip\" reports a deterministic compile failure once through gpu.onError and keeps skipping", async () => {
  const gpu = await init({ pendingPipelines: "skip" });
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: WGSL, label: "brokenSkip" });
  const nativeError = new Error("shader is broken");
  const errors: { code: string }[] = [];
  gpu.onError((error) => errors.push(error as unknown as { code: string }));
  vi.spyOn(gpu.device.gpu, "createRenderPipelineAsync").mockRejectedValue(nativeError);

  for (let i = 0; i < 3; i++) {
    expect(() => frame(gpu, (f) => f.pass(colorTarget, (p) => p.draw(drawable)))).not.toThrow();
    await gpu.settled();
  }

  expect(errors).toHaveLength(1);
  expect(errors[0]).toMatchObject({ code: "VGPU-COMPILE-FAILED", cause: nativeError });
  gpu.dispose();
});

test("a \"skip\" failure never poisons prepare(): the retry is a fresh compile", async () => {
  const gpu = await init({ pendingPipelines: "skip" });
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: WGSL, label: "skipThenPrepare" });
  const createAsync = gpu.device.gpu.createRenderPipelineAsync.bind(gpu.device.gpu);
  let fail = true;
  gpu.onError(() => undefined);
  vi.spyOn(gpu.device.gpu, "createRenderPipelineAsync").mockImplementation((desc: GPURenderPipelineDescriptor) => {
    if (fail) return Promise.reject(new Error("transient"));
    return createAsync(desc);
  });

  frame(gpu, (f) => f.pass(colorTarget, (p) => p.draw(drawable)));
  await gpu.settled();
  fail = false;

  await expect(prepare(gpu, { draw: drawable, target: colorTarget })).resolves.toBeDefined();
  gpu.dispose();
});

test("the policy chain resolves call site over frame over gpu default", async () => {
  const gpu = await init({ pendingPipelines: "throw" });
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: WGSL, label: "chained" });

  // frame beats the gpu default: "sync" here compiles inline where the gpu default would throw.
  expect(() => frame(gpu, (f) => f.pass(colorTarget, (p) => p.draw(drawable)), { pendingPipelines: "sync" })).not.toThrow();
  expect(countPipelines(gpu)).toEqual({ sync: 1, async: 0 });

  // A different blend => a different pipeline key: two draws with identical state share ONE
  // combination, and a combination the previous frame already compiled is ready under every policy.
  const other = draw(gpu, { shader: WGSL, label: "chainedCallSite", blend: "additive" });
  // call site beats the frame: "throw" here throws inside a "sync" frame.
  let thrown: { code?: string } | undefined;
  try {
    frame(gpu, (f) => f.pass(colorTarget, (p) => p.draw(other, { pendingPipelines: "throw" })), { pendingPipelines: "sync" });
  } catch (error) { thrown = error as { code?: string }; }
  expect(thrown?.code).toBe("VGPU-PIPELINE-PENDING");
  expect(countPipelines(gpu)).toEqual({ sync: 1, async: 0 });

  // ...and the call site also beats a frame that would throw.
  expect(() => frame(gpu, (f) => f.pass(colorTarget, (p) => p.draw(other, { pendingPipelines: "sync" })), { pendingPipelines: "throw" })).not.toThrow();
  expect(countPipelines(gpu)).toEqual({ sync: 2, async: 0 });
  gpu.dispose();
});

test("the frame policy survives a manually driven frame (no callback)", async () => {
  const gpu = await init({ pendingPipelines: "throw" });
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: WGSL, label: "manualFrame" });

  const f = frame(gpu, undefined, { pendingPipelines: "sync" });
  expect(() => f.pass(colorTarget, (p) => p.draw(drawable))).not.toThrow();
  f.submit();

  expect(countPipelines(gpu)).toEqual({ sync: 1, async: 0 });
  gpu.dispose();
});

test("the call-site policy reaches the one-shot draw()/effect draw path too", async () => {
  const gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: WGSL, label: "oneShot" });
  const shader = effect(gpu, { shader: FRAGMENT_ONLY, label: "oneShotFx" });

  expect(caught(() => drawable.draw({ target: colorTarget, pendingPipelines: "throw" }))?.code).toBe("VGPU-PIPELINE-PENDING");
  expect(caught(() => shader.draw({ target: colorTarget, pendingPipelines: "throw" }))?.code).toBe("VGPU-PIPELINE-PENDING");
  expect(countPipelines(gpu)).toEqual({ sync: 0, async: 0 });

  expect(() => drawable.draw({ target: colorTarget, pendingPipelines: "skip" })).not.toThrow();
  expect(countPipelines(gpu)).toEqual({ sync: 0, async: 1 });
  gpu.dispose();
});

test("frameLoop accepts the same policy option as frame()", async () => {
  const gpu = await init({ pendingPipelines: "throw" });
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: WGSL, label: "loopPolicy" });
  const frames: number[] = [];
  const { frameLoop } = await import("../src/mock.ts");

  const handle = frameLoop(gpu, (f) => {
    frames.push(1);
    f.pass(colorTarget, (p) => p.draw(drawable));
    handle.stop();
  }, { pendingPipelines: "sync" });

  await new Promise((resolve) => setTimeout(resolve, 40));
  handle.stop();

  expect(frames.length).toBeGreaterThan(0);
  expect(countPipelines(gpu)).toEqual({ sync: 1, async: 0 });
  gpu.dispose();
});
