/**
 * `renderOnce()` — contract #20 and §7 of the frozen design.
 *
 * "`renderOnce()` / `dispatchOnce()` use async pipeline readiness, own their encoder and submit
 * exactly once; they resolve after readiness + submit and **never** await `onSubmittedWorkDone`."
 */
import { afterEach, expect, test, vi } from "vitest";
import { getMockGPUDeviceInstrumentation } from "@vgpu/core";
import { draw, effect, init, prepare, renderOnce, storage, surface, target } from "../src/mock.ts";

const WGSL = `
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(pos[vi], 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1.0); }
`;

const FRAGMENT_ONLY = `
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(0.5); }
`;

const BOUND_WGSL = `
@group(0) @binding(0) var<storage, read> data: array<f32>;
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  return vec4f(data[vi], 0.0, 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1.0); }
`;

afterEach(() => { vi.restoreAllMocks(); });

function canvasLike(width = 8, height = 4): HTMLCanvasElement {
  const context = {
    configure: vi.fn(),
    unconfigure: vi.fn(),
    getCurrentTexture: () => ({ createView: () => ({}) }),
  };
  const canvas: Record<string, unknown> = {
    width,
    height,
    getContext: (kind: string) => (kind === "webgpu" ? { ...context, canvas } : null),
  };
  return canvas as unknown as HTMLCanvasElement;
}

/** Pipelines bound by the encode path, in order — identity, not just call counts. */
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

function countSubmits(device: GPUDevice): { count: number } {
  const counter = { count: 0 };
  const submit = device.queue.submit.bind(device.queue);
  vi.spyOn(device.queue, "submit").mockImplementation((buffers: GPUCommandBuffer[]) => { counter.count += 1; submit(buffers); });
  return counter;
}

// --- Contract #20 — async readiness, own encoder, exactly one submit --------------------------

test("contract #20: renderOnce compiles through createRenderPipelineAsync and submits exactly once", async () => {
  const gpu = await init();
  try {
    const screen = target(gpu, { size: [4, 4] });
    const quad = draw(gpu, { shader: WGSL, label: "quad" });
    const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
    const submits = countSubmits(gpu.device.gpu);

    await renderOnce(gpu, screen, (p) => p.draw(quad));

    expect(mock.calls.createRenderPipelineAsync).toBe(1);
    expect(mock.calls.createRenderPipeline).toBe(0);
    expect(submits.count).toBe(1);
    expect(mock.calls.createCommandEncoder).toBe(1);
  } finally {
    gpu.dispose();
  }
});

test("contract #20: renderOnce resolves right after submit and never awaits onSubmittedWorkDone", async () => {
  const gpu = await init();
  try {
    const screen = target(gpu, { size: [4, 4] });
    const quad = draw(gpu, { shader: WGSL, label: "order" });
    const order: string[] = [];
    const onSubmittedWorkDone = vi.spyOn(gpu.device.gpu.queue, "onSubmittedWorkDone");
    const submit = gpu.device.gpu.queue.submit.bind(gpu.device.gpu.queue);
    vi.spyOn(gpu.device.gpu.queue, "submit").mockImplementation((buffers: GPUCommandBuffer[]) => { order.push("submit"); submit(buffers); });

    await renderOnce(gpu, screen, (p) => p.draw(quad));
    order.push("resolved");

    expect(order).toEqual(["submit", "resolved"]);
    expect(onSubmittedWorkDone).not.toHaveBeenCalled();
  } finally {
    gpu.dispose();
  }
});

test("renderOnce prepares every named draw in parallel and encodes them all in one pass", async () => {
  const gpu = await init();
  try {
    const screen = target(gpu, { size: [4, 4] });
    const a = draw(gpu, { shader: WGSL, label: "a" });
    const b = effect(gpu, FRAGMENT_ONLY, { label: "b" });
    const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
    const submits = countSubmits(gpu.device.gpu);
    const bound = recordEncodedPipelines(gpu.device.gpu);
    // Both compiles must be in flight together, not serialized: the second call starts before the
    // first resolves.
    const inFlight: number[] = [];
    let live = 0;
    const createAsync = gpu.device.gpu.createRenderPipelineAsync.bind(gpu.device.gpu);
    vi.spyOn(gpu.device.gpu, "createRenderPipelineAsync").mockImplementation(async (desc: GPURenderPipelineDescriptor) => {
      live += 1;
      inFlight.push(live);
      await Promise.resolve();
      const pipeline = await createAsync(desc);
      live -= 1;
      return pipeline;
    });

    await renderOnce(gpu, screen, (p) => { p.draw(a); p.draw(b); });

    expect(mock.calls.createRenderPipelineAsync).toBe(2);
    expect(mock.calls.createRenderPipeline).toBe(0);
    expect(Math.max(...inFlight)).toBe(2);
    expect(submits.count).toBe(1);
    expect(mock.calls.createCommandEncoder).toBe(1);
    expect(bound).toHaveLength(2);
    expect(new Set(bound).size).toBe(2);
  } finally {
    gpu.dispose();
  }
});

test("renderOnce reuses an already-prepared combination without compiling anything", async () => {
  const gpu = await init();
  try {
    const screen = target(gpu, { size: [4, 4] });
    const quad = draw(gpu, { shader: WGSL, label: "warm" });
    const prepared = await prepare(gpu, { draw: quad, target: screen });
    const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
    const asyncBefore = mock.calls.createRenderPipelineAsync;
    const bound = recordEncodedPipelines(gpu.device.gpu);
    const submits = countSubmits(gpu.device.gpu);

    await renderOnce(gpu, screen, (p) => p.draw(quad));

    expect(mock.calls.createRenderPipelineAsync).toBe(asyncBefore);
    expect(mock.calls.createRenderPipeline).toBe(0);
    expect(submits.count).toBe(1);
    expect(bound).toEqual([prepared.gpu]);
  } finally {
    gpu.dispose();
  }
});

test("renderOnce names the same draw twice: one compile, two encodes, one submit", async () => {
  const gpu = await init();
  try {
    const screen = target(gpu, { size: [4, 4] });
    const quad = draw(gpu, { shader: WGSL, label: "twice" });
    const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
    const bound = recordEncodedPipelines(gpu.device.gpu);
    const submits = countSubmits(gpu.device.gpu);

    await renderOnce(gpu, screen, (p) => { p.draw(quad); p.draw(quad); });

    expect(mock.calls.createRenderPipelineAsync).toBe(1);
    expect(mock.calls.createRenderPipeline).toBe(0);
    expect(bound).toHaveLength(2);
    expect(new Set(bound).size).toBe(1);
    expect(submits.count).toBe(1);
  } finally {
    gpu.dispose();
  }
});

// --- Failure paths: nothing is encoded, nothing is submitted -----------------------------------

test("renderOnce rejects with VGPU-PREPARE-FAILED and opens no encoder when a draw fails to compile", async () => {
  const gpu = await init();
  try {
    const screen = target(gpu, { size: [4, 4] });
    const good = draw(gpu, { shader: WGSL, label: "good" });
    const bad = effect(gpu, FRAGMENT_ONLY, { label: "bad" });
    const createAsync = gpu.device.gpu.createRenderPipelineAsync.bind(gpu.device.gpu);
    vi.spyOn(gpu.device.gpu, "createRenderPipelineAsync").mockImplementation(async (desc: GPURenderPipelineDescriptor) => {
      if (String(desc.label ?? "").includes("bad")) throw new Error("boom: fragment stage");
      return createAsync(desc);
    });
    const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
    const encodersBefore = mock.calls.createCommandEncoder;
    const submits = countSubmits(gpu.device.gpu);

    await expect(renderOnce(gpu, screen, (p) => { p.draw(good); p.draw(bad); }))
      .rejects.toMatchObject({ code: "VGPU-PREPARE-FAILED" });

    expect(mock.calls.createCommandEncoder).toBe(encodersBefore);
    expect(submits.count).toBe(0);
  } finally {
    gpu.dispose();
  }
});

test("renderOnce rejects and encodes nothing when the gpu is disposed while the compile is in flight", async () => {
  const gpu = await init();
  const screen = target(gpu, { size: [4, 4] });
  const quad = draw(gpu, { shader: WGSL, label: "lost" });
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
  const submits = countSubmits(gpu.device.gpu);
  const createAsync = gpu.device.gpu.createRenderPipelineAsync.bind(gpu.device.gpu);
  let disposeNow: (() => void) | undefined;
  vi.spyOn(gpu.device.gpu, "createRenderPipelineAsync").mockImplementation(async (desc: GPURenderPipelineDescriptor) => {
    const pipeline = await createAsync(desc);
    disposeNow?.();
    return pipeline;
  });
  disposeNow = () => gpu.dispose();

  const encodersBefore = mock.calls.createCommandEncoder;
  await expect(renderOnce(gpu, screen, (p) => p.draw(quad)))
    .rejects.toMatchObject({ code: expect.stringMatching(/VGPU-DEVICE-(DISPOSED|LOST)/) });
  expect(mock.calls.createCommandEncoder).toBe(encodersBefore);
  expect(submits.count).toBe(0);
});

test("renderOnce on a disposed gpu fails at the entry guard with VGPU-GPU-DISPOSED", async () => {
  const gpu = await init();
  const screen = target(gpu, { size: [4, 4] });
  const quad = draw(gpu, { shader: WGSL, label: "disposed" });
  gpu.dispose();
  await expect(renderOnce(gpu, screen, (p) => p.draw(quad))).rejects.toMatchObject({ code: "VGPU-GPU-DISPOSED" });
});

test("a body that throws rejects renderOnce without opening an encoder or submitting", async () => {
  const gpu = await init();
  try {
    const screen = target(gpu, { size: [4, 4] });
    const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
    const submits = countSubmits(gpu.device.gpu);
    const encodersBefore = mock.calls.createCommandEncoder;

    await expect(renderOnce(gpu, screen, () => { throw new Error("body failed"); })).rejects.toThrow(/body failed/);

    expect(mock.calls.createCommandEncoder).toBe(encodersBefore);
    expect(submits.count).toBe(0);
  } finally {
    gpu.dispose();
  }
});

test("a combination that fails synchronously is reported with its siblings and orphans none of them", async () => {
  const gpu = await init();
  try {
    const screen = target(gpu, { size: [4, 4] });
    const failing = draw(gpu, { shader: WGSL, label: "driver-fails" });
    // `alphaToCoverage` against a single-sample target makes `pipelineForAsync()` throw
    // SYNCHRONOUSLY out of its compile key, before any promise exists — and it is named AFTER a draw
    // whose compile is already in flight, which is what makes the sibling orphanable.
    const badCombo = draw(gpu, { shader: WGSL, label: "a2c", multisample: { alphaToCoverage: true } });
    const createAsync = gpu.device.gpu.createRenderPipelineAsync.bind(gpu.device.gpu);
    vi.spyOn(gpu.device.gpu, "createRenderPipelineAsync").mockImplementation(async (desc: GPURenderPipelineDescriptor) => {
      if (String(desc.label ?? "").includes("driver-fails")) { await new Promise((resolve) => setTimeout(resolve, 5)); throw new Error("driver said no"); }
      return createAsync(desc);
    });
    const unhandled: string[] = [];
    const onUnhandled = (reason: unknown) => { unhandled.push(String(reason)); };
    process.on("unhandledRejection", onUnhandled);

    try {
      // Both failures — the synchronous one and the driver's — land in one batched report...
      const error = await renderOnce(gpu, screen, (p) => { p.draw(failing); p.draw(badCombo); }).catch((e: unknown) => e);
      expect((error as { code?: string }).code).toBe("VGPU-PREPARE-FAILED");
      // ...and nothing escapes to the process: an orphaned rejection is a crash under Node's
      // defaults, and an order-dependent (so intermittent) one.
      await new Promise((resolve) => setTimeout(resolve, 60));
      expect(unhandled, `orphaned: ${unhandled.join(" | ")}`).toEqual([]);
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
  } finally {
    gpu.dispose();
  }
});

// --- Callback semantics: the body runs exactly once --------------------------------------------

test("the renderOnce body runs exactly once, so its side effects are not duplicated", async () => {
  const gpu = await init();
  try {
    const screen = target(gpu, { size: [4, 4] });
    const quad = draw(gpu, { shader: WGSL, label: "single-run" });
    let runs = 0;

    await renderOnce(gpu, screen, (p) => { runs += 1; p.draw(quad); });

    expect(runs).toBe(1);
  } finally {
    gpu.dispose();
  }
});

test("an identity update between the body and the submit is reflected by the encode", async () => {
  const gpu = await init();
  try {
    const screen = target(gpu, { size: [4, 4] });
    const first = storage(gpu, 16);
    const second = storage(gpu, 16);
    // `.bind()` is only legal on a binding declared external at construction (ownership is fixed
    // there, never by call order), which is exactly the identity-update shape this race is about.
    const bound = draw(gpu, { shader: BOUND_WGSL, label: "rebindable", bindings: { data: first } });
    const groups: GPUBindGroup[] = [];
    const createCommandEncoder = gpu.device.gpu.createCommandEncoder.bind(gpu.device.gpu);
    vi.spyOn(gpu.device.gpu, "createCommandEncoder").mockImplementation((desc?: GPUCommandEncoderDescriptor) => {
      const encoder = createCommandEncoder(desc);
      const beginRenderPass = encoder.beginRenderPass.bind(encoder);
      encoder.beginRenderPass = (passDesc: GPURenderPassDescriptor) => {
        const pass = beginRenderPass(passDesc);
        const setBindGroup = pass.setBindGroup.bind(pass);
        (pass as { setBindGroup: unknown }).setBindGroup = (group: number, bindGroup: GPUBindGroup, offsets?: number[]) => {
          groups.push(bindGroup);
          setBindGroup(group, bindGroup, offsets);
        };
        return pass;
      };
      return encoder;
    });

    await renderOnce(gpu, screen, (p) => p.draw(bound));

    const pending = renderOnce(gpu, screen, (p) => p.draw(bound));
    // The identity update lands after the body ran and before the real encode: the encode must read
    // the CURRENT state of the draw, exactly like a bundle materialization does.
    bound.bind("data", second);
    await pending;

    // A third, unchanged render proves the second encode picked up `second` (stable cache entry),
    // rather than merely churning bind groups.
    await renderOnce(gpu, screen, (p) => p.draw(bound));

    expect(groups).toHaveLength(3);
    expect(groups[1]).not.toBe(groups[0]);
    expect(groups[2]).toBe(groups[1]);
  } finally {
    gpu.dispose();
  }
});

test("a recorder retained past the body refuses to record more draws", async () => {
  const gpu = await init();
  try {
    const screen = target(gpu, { size: [4, 4] });
    const quad = draw(gpu, { shader: WGSL, label: "retained" });
    let escaped: { draw: (drawable: typeof quad) => void } | undefined;

    await renderOnce(gpu, screen, (p) => { escaped = p; p.draw(quad); });

    // A p.draw() from an async continuation would either be dropped silently (after the submit) or
    // silently join a render the caller already described (before it): it is refused instead.
    expect(() => escaped!.draw(quad)).toThrowError(/ran after the renderOnce\(\) body returned/);
  } finally {
    gpu.dispose();
  }
});

// --- §7: the design's own example targets a surface, outside any frame() -----------------------

test("renderOnce works against a surface outside frame()", async () => {
  const gpu = await init();
  try {
    const screen = surface(gpu, canvasLike(), { size: [8, 4] });
    const quad = effect(gpu, FRAGMENT_ONLY, { label: "surface-quad" });
    const submits = countSubmits(gpu.device.gpu);

    await renderOnce(gpu, screen, (p) => p.draw(quad));

    expect(submits.count).toBe(1);
  } finally {
    gpu.dispose();
  }
});
