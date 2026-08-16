/**
 * Motivation #1 of the rev6.1 design, verified rather than asserted (T04-18's DoD).
 *
 * > "a dispatch issued inside a frame callback submits before the frame's own encoder... Program
 * > order and execution order diverge."
 *
 * `frame-unified.test.ts` already pins the `f.compute()` side of contract #1 (one encoder, one
 * submit). What it does not pin is the CONTRAST that makes T04-18's migration worth doing: what the
 * legacy `Compute.dispatch()` actually does to the observable command order when it is called from
 * inside a frame callback, and that `f.compute()` is what removes it. Both arms are recorded here
 * off the same mock device, on one timeline, so the claim is a diff between two recordings and not a
 * pair of independent "it compiles" checks.
 *
 * The two shapes are deliberately different failures:
 *
 *  - **The `fft-ocean-surface` shape** (compute passes first, then the render that consumes them).
 *    Program order survives here — WebGPU queue submits are ordered, and the dispatch's own submit
 *    happens first — so the cost of the legacy form is the EXTRA SUBMITS, one per dispatch, not a
 *    wrong result. Worth stating plainly: this is why the corpus renders correctly today.
 *  - **The inverted shape** (a pass first, then a dispatch). Here the legacy form really does
 *    reorder: the dispatch opens its own encoder and submits it *during* the callback, while the
 *    pass recorded before it is still sitting in the frame's encoder, unsubmitted. Execution order
 *    comes out as the REVERSE of program order. That is the divergence motivation #1 names, and
 *    `f.compute()` removes it.
 */
import { afterEach, expect, test, vi } from "vitest";
import { compute, draw, frame, init, target } from "../src/mock.ts";

const RENDER_WGSL = `
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(pos[vi], 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1.0); }
`;

const COMPUTE_WGSL = `@compute @workgroup_size(1) fn main() {}`;

let gpu: Awaited<ReturnType<typeof init>> | undefined;

afterEach(() => {
  vi.restoreAllMocks();
  gpu?.dispose();
  gpu = undefined;
});

/**
 * One ordered timeline of everything the device is told to do, across EVERY encoder.
 *
 * `spyFrameEncoder()` in `frame-unified.test.ts` deliberately watches only the encoder labelled
 * `vgpu.frame`, which is the right lens for "did this land in the frame's encoder". It is the wrong
 * lens here: the whole point is what happens on the OTHER encoders, so this one tags every encoder
 * by its label, follows each `finish()` to the command buffer it produces, and records submits with
 * the labels of the buffers they carry.
 */
function recordTimeline(device: GPUDevice): string[] {
  const timeline: string[] = [];
  const labelOfBuffer = new WeakMap<GPUCommandBuffer, string>();
  const createCommandEncoder = device.createCommandEncoder.bind(device);
  vi.spyOn(device, "createCommandEncoder").mockImplementation((desc?: GPUCommandEncoderDescriptor) => {
    const encoder = createCommandEncoder(desc);
    const label = desc?.label ?? "<unlabelled>";
    const beginRenderPass = encoder.beginRenderPass.bind(encoder);
    const beginComputePass = encoder.beginComputePass.bind(encoder);
    const finish = encoder.finish.bind(encoder);
    (encoder as { beginRenderPass: unknown }).beginRenderPass = (d: GPURenderPassDescriptor) => {
      timeline.push(`encode ${label}: renderPass`);
      return beginRenderPass(d);
    };
    (encoder as { beginComputePass: unknown }).beginComputePass = (d?: GPUComputePassDescriptor) => {
      timeline.push(`encode ${label}: computePass`);
      return beginComputePass(d);
    };
    (encoder as { finish: unknown }).finish = (d?: GPUCommandBufferDescriptor) => {
      const buffer = finish(d);
      labelOfBuffer.set(buffer, label);
      return buffer;
    };
    return encoder;
  });
  const submit = device.queue.submit.bind(device.queue);
  vi.spyOn(device.queue, "submit").mockImplementation((buffers: Iterable<GPUCommandBuffer>) => {
    const list = [...buffers];
    timeline.push(`submit  ${list.map((b) => labelOfBuffer.get(b) ?? "<unknown>").join(", ")}`);
    submit(list);
  });
  return timeline;
}

const submits = (timeline: readonly string[]) => timeline.filter((e) => e.startsWith("submit")).length;

/** The order the GPU actually runs the work in: passes, in the order their SUBMIT reached the queue. */
function executionOrder(timeline: readonly string[]): string[] {
  const pending = new Map<string, string[]>();
  const executed: string[] = [];
  for (const event of timeline) {
    const encoded = /^encode (?<label>.+): (?<kind>renderPass|computePass)$/u.exec(event);
    if (encoded) {
      const label = encoded.groups!["label"]!;
      pending.set(label, [...(pending.get(label) ?? []), encoded.groups!["kind"]!]);
      continue;
    }
    const submitted = /^submit {2}(?<labels>.+)$/u.exec(event);
    if (!submitted) continue;
    for (const label of submitted.groups!["labels"]!.split(", ")) {
      executed.push(...(pending.get(label) ?? []));
      pending.delete(label);
    }
  }
  return executed;
}

// --- the fft-ocean-surface shape: computes, then the render that consumes them -------------------

test("legacy dispatch() in a frame callback costs one extra submit per dispatch (the fft-ocean shape)", async () => {
  gpu = await init();
  const hdr = target(gpu, { size: [4, 4] });
  const rowPass = compute(gpu, { shader: COMPUTE_WGSL, label: "fft-row" });
  const colPass = compute(gpu, { shader: COMPUTE_WGSL, label: "fft-col" });
  const ocean = draw(gpu, { shader: RENDER_WGSL, label: "ocean" });
  const timeline = recordTimeline(gpu.device.gpu);

  frame(gpu, (f) => {
    rowPass.dispatch(1);
    colPass.dispatch(1);
    f.pass({ target: hdr }, (p) => p.draw(ocean));
  });

  expect(timeline).toEqual([
    "encode fft-row.encoder: computePass",
    "submit  fft-row.encoder",
    "encode fft-col.encoder: computePass",
    "submit  fft-col.encoder",
    "encode vgpu.frame: renderPass",
    "submit  vgpu.frame",
  ]);
  // Three submits for one frame, and the two computes are in flight before the frame's encoder is
  // even finished. Program order does survive here — which is exactly why the corpus renders
  // correctly today and why this migration is not a bug fix in this shape.
  expect(submits(timeline)).toBe(3);
  expect(executionOrder(timeline)).toEqual(["computePass", "computePass", "renderPass"]);
});

test("f.compute() collapses the same frame to one encoder and one submit, in program order (contract #1)", async () => {
  gpu = await init();
  const hdr = target(gpu, { size: [4, 4] });
  const rowPass = compute(gpu, { shader: COMPUTE_WGSL, label: "fft-row" });
  const colPass = compute(gpu, { shader: COMPUTE_WGSL, label: "fft-col" });
  const ocean = draw(gpu, { shader: RENDER_WGSL, label: "ocean" });
  const timeline = recordTimeline(gpu.device.gpu);

  frame(gpu, (f) => {
    f.compute(rowPass, 1);
    f.compute(colPass, 1);
    f.pass({ target: hdr }, (p) => p.draw(ocean));
  });

  expect(timeline).toEqual([
    "encode vgpu.frame: computePass",
    "encode vgpu.frame: computePass",
    "encode vgpu.frame: renderPass",
    "submit  vgpu.frame",
  ]);
  expect(submits(timeline)).toBe(1);
  expect(executionOrder(timeline)).toEqual(["computePass", "computePass", "renderPass"]);
});

// --- the inverted shape: this is where the legacy form is actually WRONG --------------------------

test("legacy dispatch() AFTER a pass inverts execution order against program order (motivation #1)", async () => {
  gpu = await init();
  const hdr = target(gpu, { size: [4, 4] });
  const sim = compute(gpu, { shader: COMPUTE_WGSL, label: "sim" });
  const ocean = draw(gpu, { shader: RENDER_WGSL, label: "ocean" });
  const timeline = recordTimeline(gpu.device.gpu);

  frame(gpu, (f) => {
    f.pass({ target: hdr }, (p) => p.draw(ocean));   // program order: render FIRST
    sim.dispatch(1);                                  // program order: compute SECOND
  });

  // The pass was recorded into the frame's encoder, which is not submitted until the callback
  // returns; the dispatch opened its own encoder and submitted it immediately, mid-callback.
  expect(timeline).toEqual([
    "encode vgpu.frame: renderPass",
    "encode sim.encoder: computePass",
    "submit  sim.encoder",
    "submit  vgpu.frame",
  ]);
  expect(submits(timeline)).toBe(2);
  // Program order says render-then-compute. Execution order says the opposite.
  expect(executionOrder(timeline)).toEqual(["computePass", "renderPass"]);
});

test("f.compute() in the same position keeps program order and execution order identical", async () => {
  gpu = await init();
  const hdr = target(gpu, { size: [4, 4] });
  const sim = compute(gpu, { shader: COMPUTE_WGSL, label: "sim" });
  const ocean = draw(gpu, { shader: RENDER_WGSL, label: "ocean" });
  const timeline = recordTimeline(gpu.device.gpu);

  frame(gpu, (f) => {
    f.pass({ target: hdr }, (p) => p.draw(ocean));
    f.compute(sim, 1);
  });

  expect(timeline).toEqual([
    "encode vgpu.frame: renderPass",
    "encode vgpu.frame: computePass",
    "submit  vgpu.frame",
  ]);
  expect(submits(timeline)).toBe(1);
  expect(executionOrder(timeline)).toEqual(["renderPass", "computePass"]);
});
