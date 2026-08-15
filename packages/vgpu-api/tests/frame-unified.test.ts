/**
 * Contracts of the unified frame (issue #320 §2 / §2a): one encoder and exactly one
 * `queue.submit()` for `f.compute()` + `f.copyBuffer()` + `f.raw()` + `f.pass()`.
 *
 * - #1  — a compute dispatch and a render pass recorded in the same `frame()` produce exactly one
 *         `queue.submit()`, in program order.
 * - #17 — `f.compute()` validates integer workgroup counts >= 0 with the same stable code
 *         `dispatchOnce()` already uses.
 * - #18 — `f.copyBuffer()` defaults `size` to `source.size - sourceOffset`; a range/alignment
 *         violation or `source === destination` fails with a stable code and encodes nothing.
 * - #24 — a thenable return aborts the frame (`VGPU-ASYNC-FRAME-CALLBACK`, no submit at all) and an
 *         escaped `Frame` then throws `VGPU-FRAME-CLOSED`.
 * - #25 — `f.raw()` borrows the frame's open encoder; inside a managed pass or reentrantly it throws
 *         `VGPU-FRAME-ENCODER-LOCKED` *before* invoking the callback.
 */
import { afterEach, expect, test, vi } from "vitest";
import { getMockGPUDeviceInstrumentation } from "@vgpu/core";
import { compute, draw, frame, init, storage, target } from "../src/mock.ts";
import type { Frame } from "../src/frame.ts";

const RENDER_WGSL = `
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(pos[vi], 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1.0); }
`;

const COMPUTE_WGSL = `
@compute @workgroup_size(1) fn main() {}
`;

/** Same shape compute/aliasing.test.ts uses: one read binding plus one writable binding to trip on. */
const ALIASING_WGSL = `
@group(0) @binding(0) var<storage, read> src: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4f>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  dst[id.x] = src[id.x];
}
`;

let gpu: Awaited<ReturnType<typeof init>> | undefined;

afterEach(() => {
  vi.restoreAllMocks();
  gpu?.dispose();
  gpu = undefined;
});

type EncodeOp = readonly [name: string, ...args: unknown[]];

/**
 * Records, in encode order, every command the frame's own encoder receives plus every
 * `queue.submit()`. `vgpu.frame` is the only label `Frame` uses, so a `dispatch()` that opened its
 * own encoder would not be counted here — which is exactly the distinction contract #1 needs.
 */
function spyFrameEncoder(device: GPUDevice): { readonly ops: EncodeOp[]; readonly submits: { count: number }; readonly frameEncoders: GPUCommandEncoder[] } {
  const ops: EncodeOp[] = [];
  const submits = { count: 0 };
  const frameEncoders: GPUCommandEncoder[] = [];
  const createCommandEncoder = device.createCommandEncoder.bind(device);
  vi.spyOn(device, "createCommandEncoder").mockImplementation((desc?: GPUCommandEncoderDescriptor) => {
    const encoder = createCommandEncoder(desc);
    if (desc?.label !== "vgpu.frame") return encoder;
    frameEncoders.push(encoder);
    const beginRenderPass = encoder.beginRenderPass.bind(encoder);
    const beginComputePass = encoder.beginComputePass.bind(encoder);
    const copyBufferToBuffer = encoder.copyBufferToBuffer.bind(encoder);
    (encoder as { beginRenderPass: unknown }).beginRenderPass = (passDesc: GPURenderPassDescriptor) => {
      ops.push(["beginRenderPass"]);
      return beginRenderPass(passDesc);
    };
    (encoder as { beginComputePass: unknown }).beginComputePass = (passDesc?: GPUComputePassDescriptor) => {
      ops.push(["beginComputePass"]);
      return beginComputePass(passDesc);
    };
    (encoder as { copyBufferToBuffer: unknown }).copyBufferToBuffer = (source: GPUBuffer, sourceOffset: number, destination: GPUBuffer, destinationOffset: number, size?: number) => {
      ops.push(["copyBufferToBuffer", source, sourceOffset, destination, destinationOffset, size]);
      return copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size);
    };
    return encoder;
  });
  const submit = device.queue.submit.bind(device.queue);
  vi.spyOn(device.queue, "submit").mockImplementation((buffers: Iterable<GPUCommandBuffer>) => {
    submits.count += 1;
    submit(buffers);
  });
  return { ops, submits, frameEncoders };
}

/** Returns the error a synchronous call threw, so assertions pin `code` instead of message text. */
function caught(fn: () => unknown): { code?: string; message?: string } | undefined {
  try { fn(); }
  catch (error) { return error as { code?: string; message?: string }; }
  return undefined;
}

/** GPUBufferUsage is a WebGPU global the mock environment does not install; the flags are spec constants. */
const BUFFER_USAGE = { COPY_SRC: 0x0004, COPY_DST: 0x0008 } as const;

function names(ops: readonly EncodeOp[]): string[] {
  return ops.map((op) => op[0]);
}

// --- Contract #1 — one encoder, one submit, program order ---------------------------------------

test("a compute dispatch and a render pass in the same frame() produce exactly one submit, in program order (contract #1)", async () => {
  gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  const sim = compute(gpu, COMPUTE_WGSL, { label: "sim" });
  const drawable = draw(gpu, { shader: RENDER_WGSL, label: "dots" });
  const spy = spyFrameEncoder(gpu.device.gpu);

  frame(gpu, (f) => {
    f.compute(sim, 2);
    f.pass(colorTarget, (p) => p.draw(drawable));
  });

  expect(spy.submits.count).toBe(1);
  expect(names(spy.ops)).toEqual(["beginComputePass", "beginRenderPass"]);
});

test("program order is the encode order, render first (contract #1)", async () => {
  gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  const sim = compute(gpu, COMPUTE_WGSL, { label: "sim" });
  const drawable = draw(gpu, { shader: RENDER_WGSL, label: "dots" });
  const spy = spyFrameEncoder(gpu.device.gpu);

  frame(gpu, (f) => {
    f.pass(colorTarget, (p) => p.draw(drawable));
    f.compute(sim, 1, 2, 3);
  });

  expect(spy.submits.count).toBe(1);
  expect(names(spy.ops)).toEqual(["beginRenderPass", "beginComputePass"]);
});

test("f.compute() dispatches the resolved workgroup counts and never finishes the borrowed encoder", async () => {
  gpu = await init();
  const sim = compute(gpu, COMPUTE_WGSL, { label: "sim" });
  const dispatched: Array<readonly [number, number, number]> = [];
  const createCommandEncoder = gpu.device.gpu.createCommandEncoder.bind(gpu.device.gpu);
  let finishes = 0;
  vi.spyOn(gpu.device.gpu, "createCommandEncoder").mockImplementation((desc?: GPUCommandEncoderDescriptor) => {
    const encoder = createCommandEncoder(desc);
    if (desc?.label !== "vgpu.frame") return encoder;
    const beginComputePass = encoder.beginComputePass.bind(encoder);
    const finish = encoder.finish.bind(encoder);
    (encoder as { beginComputePass: unknown }).beginComputePass = (passDesc?: GPUComputePassDescriptor) => {
      const pass = beginComputePass(passDesc);
      const dispatchWorkgroups = pass.dispatchWorkgroups.bind(pass);
      (pass as { dispatchWorkgroups: unknown }).dispatchWorkgroups = (x: number, y?: number, z?: number) => {
        dispatched.push([x, y ?? 1, z ?? 1]);
        dispatchWorkgroups(x, y, z);
      };
      return pass;
    };
    (encoder as { finish: unknown }).finish = (finishDesc?: GPUCommandBufferDescriptor) => {
      finishes += 1;
      return finish(finishDesc);
    };
    return encoder;
  });

  frame(gpu, (f) => {
    f.compute(sim, 4);
    f.compute(sim, 4, 5);
    f.compute(sim, 4, 5, 6);
  });

  expect(dispatched).toEqual([[4, 1, 1], [4, 5, 1], [4, 5, 6]]);
  // Exactly one finish() — the frame's, at submit time. `encodeForFrame` never finishes the borrow.
  expect(finishes).toBe(1);
});

// --- Contract #17 — workgroup count validation is shared with dispatchOnce ----------------------

test("f.compute() rejects non-integer and negative workgroup counts with the same code as dispatchOnce (contract #17)", async () => {
  gpu = await init();
  const sim = compute(gpu, COMPUTE_WGSL, { label: "sim" });
  const spy = spyFrameEncoder(gpu.device.gpu);

  const legacy = await dispatchOnceError(sim);
  expect(legacy?.code).toBe("VGPU-R1-DISPATCH-COUNT");

  for (const bad of [-1, 1.5, Number.NaN] as const) {
    const error = caught(() => frame(gpu!, (f) => f.compute(sim, bad)));
    expect(error?.code).toBe(legacy?.code);
  }
  const yError = caught(() => frame(gpu!, (f) => f.compute(sim, 1, -2)));
  expect(yError?.code).toBe("VGPU-R1-DISPATCH-COUNT");
  const zError = caught(() => frame(gpu!, (f) => f.compute(sim, 1, 1, 0.5)));
  expect(zError?.code).toBe("VGPU-R1-DISPATCH-COUNT");

  // Nothing was encoded: the validation runs before the compute pass opens.
  expect(names(spy.ops)).toEqual([]);
});

async function dispatchOnceError(sim: ReturnType<typeof compute>): Promise<{ code?: string } | undefined> {
  try { await sim.dispatchOnce(-1); }
  catch (error) { return error as { code?: string }; }
  return undefined;
}

// --- Contract #18 — f.copyBuffer() -------------------------------------------------------------

test("f.copyBuffer() with size omitted copies source.size - sourceOffset bytes (contract #18)", async () => {
  gpu = await init();
  const source = storage(gpu, 64);
  const destination = storage(gpu, 64);
  const spy = spyFrameEncoder(gpu.device.gpu);

  frame(gpu, (f) => {
    f.copyBuffer({ source, destination });
    f.copyBuffer({ source, destination, sourceOffset: 16 });
    f.copyBuffer({ source, destination, sourceOffset: 16, destinationOffset: 8, size: 4 });
  });

  expect(spy.ops).toEqual([
    ["copyBufferToBuffer", source.gpu, 0, destination.gpu, 0, 64],
    ["copyBufferToBuffer", source.gpu, 16, destination.gpu, 0, 48],
    ["copyBufferToBuffer", source.gpu, 16, destination.gpu, 8, 4],
  ]);
  expect(spy.submits.count).toBe(1);
});

test("f.copyBuffer() rejects range, alignment and self-copy violations with a stable code and encodes nothing (contract #18)", async () => {
  gpu = await init();
  const source = storage(gpu, 64);
  const destination = storage(gpu, 32);
  const spy = spyFrameEncoder(gpu.device.gpu);

  const violations: Array<() => void> = [
    // sourceOffset is not a multiple of 4
    () => frame(gpu!, (f) => f.copyBuffer({ source, destination, sourceOffset: 2, size: 4 })),
    // destinationOffset is not a multiple of 4
    () => frame(gpu!, (f) => f.copyBuffer({ source, destination, destinationOffset: 3, size: 4 })),
    // size is not a multiple of 4
    () => frame(gpu!, (f) => f.copyBuffer({ source, destination, size: 6 })),
    // sourceOffset + size > source.size
    () => frame(gpu!, (f) => f.copyBuffer({ source, destination, sourceOffset: 60, size: 8 })),
    // destinationOffset + size > destination.size — the implicit source remainder is too big
    () => frame(gpu!, (f) => f.copyBuffer({ source, destination })),
    // destinationOffset + size > destination.size, explicit size
    () => frame(gpu!, (f) => f.copyBuffer({ source, destination, destinationOffset: 28, size: 8 })),
    // source === destination
    () => frame(gpu!, (f) => f.copyBuffer({ source, destination: source, size: 4 })),
    // negative offset
    () => frame(gpu!, (f) => f.copyBuffer({ source, destination, sourceOffset: -4, size: 4 })),
  ];

  for (const violation of violations) {
    expect(caught(violation)?.code).toBe("VGPU-COPY-BUFFER-INVALID");
  }
  expect(names(spy.ops)).toEqual([]);
});

test("f.copyBuffer() accepts any { gpu, size } pair, not only a StorageBuffer (contract #18)", async () => {
  gpu = await init();
  const source = storage(gpu, 16);
  const raw = gpu.gpu.createBuffer({ size: 16, usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.COPY_SRC });
  const spy = spyFrameEncoder(gpu.device.gpu);

  frame(gpu, (f) => { f.copyBuffer({ source, destination: { gpu: raw, size: 16 } }); });

  expect(spy.ops).toEqual([["copyBufferToBuffer", source.gpu, 0, raw, 0, 16]]);
});

// --- Contract #24 — synchronous callbacks ------------------------------------------------------

test("a thenable-returning frame callback throws VGPU-ASYNC-FRAME-CALLBACK and submits nothing (contract #24)", async () => {
  gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: RENDER_WGSL, label: "dots" });
  const sim = compute(gpu, COMPUTE_WGSL, { label: "sim" });
  const spy = spyFrameEncoder(gpu.device.gpu);
  let escaped: Frame | undefined;

  const error = caught(() => frame(gpu!, ((f: Frame) => {
    escaped = f;
    f.pass(colorTarget, (p) => p.draw(drawable));
    return Promise.resolve();
  }) as never));

  expect(error?.code).toBe("VGPU-ASYNC-FRAME-CALLBACK");
  // Not a partial submit: the recorded render pass is discarded with the encoder.
  expect(spy.submits.count).toBe(0);
  expect(names(spy.ops)).toEqual(["beginRenderPass"]);

  // Every entry point of the escaped frame is closed.
  expect(caught(() => escaped!.pass(colorTarget, () => undefined))?.code).toBe("VGPU-FRAME-CLOSED");
  expect(caught(() => escaped!.submit())?.code).toBe("VGPU-FRAME-CLOSED");
  expect(caught(() => escaped!.cancel())?.code).toBe("VGPU-FRAME-CLOSED");
  expect(caught(() => escaped!.raw(() => undefined))?.code).toBe("VGPU-FRAME-CLOSED");
  expect(caught(() => escaped!.copyBuffer({ source: storage(gpu!, 4), destination: storage(gpu!, 4) }))?.code).toBe("VGPU-FRAME-CLOSED");
  expect(caught(() => escaped!.compute(sim, 1))?.code).toBe("VGPU-FRAME-CLOSED");
  expect(spy.submits.count).toBe(0);
});

test("any thenable counts, not only a real promise (contract #24)", async () => {
  gpu = await init();
  const spy = spyFrameEncoder(gpu.device.gpu);

  const error = caught(() => frame(gpu!, (() => ({ then() { /* thenable */ } })) as never));

  expect(error?.code).toBe("VGPU-ASYNC-FRAME-CALLBACK");
  expect(spy.submits.count).toBe(0);
});

test("frameLoop() rejects a thenable callback the same way (contract #24)", async () => {
  gpu = await init();
  const spy = spyFrameEncoder(gpu.device.gpu);
  const requested: FrameRequestCallback[] = [];
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => { requested.push(cb); return requested.length; });
  vi.stubGlobal("cancelAnimationFrame", () => undefined);

  const { frameLoop } = await import("../src/mock.ts");
  const handle = frameLoop(gpu, (() => Promise.resolve()) as never);
  const error = caught(() => requested[0]?.(0));
  handle.stop();
  vi.unstubAllGlobals();

  expect(error?.code).toBe("VGPU-ASYNC-FRAME-CALLBACK");
  expect(spy.submits.count).toBe(0);
});

test("a callback that throws synchronously still submits the prefix it recorded (preexisting behavior)", async () => {
  gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: RENDER_WGSL, label: "dots" });
  const spy = spyFrameEncoder(gpu.device.gpu);

  const error = caught(() => frame(gpu!, (f) => {
    f.pass(colorTarget, (p) => p.draw(drawable));
    throw new Error("boom");
  }));

  expect(error?.message).toBe("boom");
  // Documented behavior of frame(): it submits on the way out of a throw. Only the thenable case discards.
  expect(spy.submits.count).toBe(1);
});

// --- Contract #25 — f.raw() --------------------------------------------------------------------

test("f.raw() lends the frame's own encoder between managed passes, keeping one submit (contract #25)", async () => {
  gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: RENDER_WGSL, label: "dots" });
  const sim = compute(gpu, COMPUTE_WGSL, { label: "sim" });
  const scratchSource = storage(gpu, 16);
  const scratchDestination = storage(gpu, 16);
  const spy = spyFrameEncoder(gpu.device.gpu);
  const borrowed: unknown[] = [];

  frame(gpu, (f) => {
    // The mock command encoder implements only the copy/pass/finish subset, so the raw block uses
    // copyBufferToBuffer: what matters here is the encoder identity and the program order, not which
    // command is recorded.
    f.raw((encoder) => {
      borrowed.push(encoder);
      encoder.copyBufferToBuffer(scratchSource.gpu, 0, scratchDestination.gpu, 0, 16);
    });
    f.compute(sim, 1);
    f.pass(colorTarget, (p) => p.draw(drawable));
  });

  // Identity, not a wrapper: the callback receives the very encoder the frame will finish and submit,
  // which is what makes program order and the single submit structural rather than best-effort.
  expect(borrowed).toHaveLength(1);
  expect(borrowed[0]).toBe(spy.frameEncoders[0]);
  expect(spy.submits.count).toBe(1);
  expect(names(spy.ops)).toEqual(["copyBufferToBuffer", "beginComputePass", "beginRenderPass"]);
});

test("f.raw() inside f.pass() throws VGPU-FRAME-ENCODER-LOCKED before invoking the callback (contract #25)", async () => {
  gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: RENDER_WGSL, label: "dots" });
  let ran = false;

  const error = caught(() => frame(gpu!, (f) => {
    f.pass(colorTarget, (p) => {
      p.draw(drawable);
      f.raw(() => { ran = true; });
    });
  }));

  expect(error?.code).toBe("VGPU-FRAME-ENCODER-LOCKED");
  expect(ran).toBe(false);
});

test("f.raw() reentrantly from another f.raw() callback throws VGPU-FRAME-ENCODER-LOCKED before invoking the callback (contract #25)", async () => {
  gpu = await init();
  let ran = false;

  const error = caught(() => frame(gpu!, (f) => {
    f.raw(() => { f.raw(() => { ran = true; }); });
  }));

  expect(error?.code).toBe("VGPU-FRAME-ENCODER-LOCKED");
  expect(ran).toBe(false);
});

test("f.compute() and f.copyBuffer() inside a managed pass are rejected the same way (contract #25 boundary)", async () => {
  gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  const sim = compute(gpu, COMPUTE_WGSL, { label: "sim" });
  const source = storage(gpu, 16);
  const destination = storage(gpu, 16);

  const computeError = caught(() => frame(gpu!, (f) => {
    f.pass(colorTarget, () => { f.compute(sim, 1); });
  }));
  const copyError = caught(() => frame(gpu!, (f) => {
    f.pass(colorTarget, () => { f.copyBuffer({ source, destination }); });
  }));

  expect(computeError?.code).toBe("VGPU-FRAME-ENCODER-LOCKED");
  expect(copyError?.code).toBe("VGPU-FRAME-ENCODER-LOCKED");
});

test("the borrow ends when the raw callback returns: a second f.raw() after it is legal (contract #25)", async () => {
  gpu = await init();
  const spy = spyFrameEncoder(gpu.device.gpu);
  const seen: string[] = [];

  frame(gpu, (f) => {
    f.raw(() => { seen.push("first"); });
    f.raw(() => { seen.push("second"); });
  });

  expect(seen).toEqual(["first", "second"]);
  expect(spy.submits.count).toBe(1);
});

test("a raw callback returning a thenable aborts the whole frame, exactly like the top-level case (contract #25 -> #24)", async () => {
  gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: RENDER_WGSL, label: "dots" });
  const spy = spyFrameEncoder(gpu.device.gpu);
  let escaped: Frame | undefined;

  const error = caught(() => frame(gpu!, (f) => {
    escaped = f;
    f.pass(colorTarget, (p) => p.draw(drawable));
    f.raw((() => Promise.resolve()) as never);
  }));

  expect(error?.code).toBe("VGPU-ASYNC-FRAME-CALLBACK");
  expect(spy.submits.count).toBe(0);
  expect(caught(() => escaped!.raw(() => undefined))?.code).toBe("VGPU-FRAME-CLOSED");
});

// --- Regression: the frame that uses none of the new methods behaves exactly as before ---------

test("a frame with only f.pass() still submits exactly once, unchanged", async () => {
  gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: RENDER_WGSL, label: "dots" });
  const spy = spyFrameEncoder(gpu.device.gpu);

  frame(gpu, (f) => f.pass(colorTarget, (p) => p.draw(drawable)));

  expect(spy.submits.count).toBe(1);
  expect(names(spy.ops)).toEqual(["beginRenderPass"]);
  expect(getMockGPUDeviceInstrumentation(gpu.device.gpu).calls.createComputePipeline).toBe(0);
});

// --- pendingPipelines policy for f.compute() (contract #3/#4 through the frame) -----------------

test("f.compute() compiles inline under the effective \"sync\" default (contract #3)", async () => {
  gpu = await init();
  const sim = compute(gpu, COMPUTE_WGSL, { label: "sim" });
  const spy = spyFrameEncoder(gpu.device.gpu);

  frame(gpu, (f) => f.compute(sim, 1));

  const calls = getMockGPUDeviceInstrumentation(gpu.device.gpu).calls;
  expect(calls.createComputePipeline).toBe(1);
  expect(calls.createComputePipelineAsync).toBe(0);
  expect(names(spy.ops)).toEqual(["beginComputePass"]);
});

test("f.compute() under \"throw\" reports VGPU-PIPELINE-PENDING and starts no compilation (contract #3)", async () => {
  gpu = await init();
  const sim = compute(gpu, COMPUTE_WGSL, { label: "sim" });
  const spy = spyFrameEncoder(gpu.device.gpu);

  const error = caught(() => frame(gpu!, (f) => f.compute(sim, 1), { pendingPipelines: "throw" }));

  expect(error?.code).toBe("VGPU-PIPELINE-PENDING");
  const calls = getMockGPUDeviceInstrumentation(gpu.device.gpu).calls;
  expect(calls.createComputePipeline).toBe(0);
  expect(calls.createComputePipelineAsync).toBe(0);
  expect(names(spy.ops)).toEqual([]);
});

test("f.compute() under \"skip\" starts async compilation, encodes nothing and never throws (contract #3)", async () => {
  gpu = await init();
  const sim = compute(gpu, COMPUTE_WGSL, { label: "sim" });
  const spy = spyFrameEncoder(gpu.device.gpu);

  expect(() => frame(gpu!, (f) => f.compute(sim, 1), { pendingPipelines: "skip" })).not.toThrow();

  const calls = getMockGPUDeviceInstrumentation(gpu.device.gpu).calls;
  expect(calls.createComputePipeline).toBe(0);
  expect(calls.createComputePipelineAsync).toBe(1);
  expect(names(spy.ops)).toEqual([]);
  expect(spy.submits.count).toBe(1);

  // Once the background compile lands, the next frame encodes the dispatch without compiling again.
  await sim.dispatchOnce(1);
  frame(gpu, (f) => f.compute(sim, 1), { pendingPipelines: "skip" });
  expect(getMockGPUDeviceInstrumentation(gpu.device.gpu).calls.createComputePipelineAsync).toBe(1);
  expect(names(spy.ops)).toEqual(["beginComputePass"]);
});

test("the call-site policy wins over the frame policy for f.compute() (contract #3 chain)", async () => {
  gpu = await init();
  const sim = compute(gpu, COMPUTE_WGSL, { label: "sim" });

  frame(gpu, (f) => f.compute(sim, 1, undefined, undefined, { pendingPipelines: "skip" }), { pendingPipelines: "throw" });

  const calls = getMockGPUDeviceInstrumentation(gpu.device.gpu).calls;
  expect(calls.createComputePipeline).toBe(0);
  expect(calls.createComputePipelineAsync).toBe(1);
});

test("f.compute() rejects a value that is not a Compute created by this library", async () => {
  gpu = await init();

  const error = caught(() => frame(gpu!, (f) => f.compute({ dispatch() {} } as never, 1)));

  expect(error).toBeInstanceOf(TypeError);
  expect(error?.message).toContain("f.compute()");
});

test("the gpu-wide default applies when neither the call site nor the frame names a policy (contract #3 chain)", async () => {
  // Last link of the chain: with both overrides absent, f.compute() must read init()'s default
  // rather than falling back to a hardcoded "sync".
  gpu = await init({ pendingPipelines: "throw" });
  const sim = compute(gpu, COMPUTE_WGSL, { label: "sim" });
  const spy = spyFrameEncoder(gpu.device.gpu);

  const error = caught(() => frame(gpu!, (f) => f.compute(sim, 1)));

  expect(error?.code).toBe("VGPU-PIPELINE-PENDING");
  expect(getMockGPUDeviceInstrumentation(gpu.device.gpu).calls.createComputePipeline).toBe(0);
  expect(names(spy.ops)).toEqual([]);
});

test("f.compute() runs the writable-storage aliasing guard and encodes nothing when it trips", async () => {
  // The guard is #preflightAliasing, reused verbatim from dispatch()/dispatchOnce(): the frame path
  // must not be a hole through which an aliased bind group reaches the queue.
  gpu = await init();
  const sim = compute(gpu, ALIASING_WGSL, { label: "sim" });
  const buffer = storage(gpu, 16);
  sim.set({ src: buffer, dst: buffer });
  const spy = spyFrameEncoder(gpu.device.gpu);

  const error = caught(() => frame(gpu!, (f) => f.compute(sim, 1)));

  expect(error?.code).toBe(caught(() => sim.dispatch(1))?.code);
  expect(error?.message).toContain("alias");
  expect(names(spy.ops)).toEqual([]);
});

test("an unexpected submit failure propagates instead of being swallowed", async () => {
  // frame()'s implicit submit swallows exactly two codes (device gone, frame closed). Anything else
  // must reach the caller: silently dropping a queue failure would hide a real bug.
  gpu = await init();
  const colorTarget = target(gpu, { size: [4, 4] });
  const drawable = draw(gpu, { shader: RENDER_WGSL, label: "dots" });
  vi.spyOn(gpu.device.gpu.queue, "submit").mockImplementation(() => {
    throw new Error("submit exploded");
  });

  expect(() => frame(gpu!, (f) => f.pass(colorTarget, (p) => p.draw(drawable)))).toThrowError("submit exploded");
});
