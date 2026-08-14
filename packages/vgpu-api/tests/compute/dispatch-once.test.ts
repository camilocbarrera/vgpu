import { afterEach, describe, expect, test, vi } from "vitest";
import { getMockGPUDeviceInstrumentation } from "@vgpu/core";
import { init } from "../../src/mock.ts";
import { compute } from "../../src/compute.ts";

const SHADER = `
@compute @workgroup_size(1) fn main() {}
`;

let gpu: Awaited<ReturnType<typeof init>> | undefined;

afterEach(() => {
  gpu?.dispose();
  gpu = undefined;
});

// --- Contract #4 — compute() performs no pipeline creation at construction ---------------------

describe("compute() construction does not compile a pipeline (contract #4)", () => {
  test("neither createComputePipeline nor createComputePipelineAsync run in the constructor", async () => {
    gpu = await init();
    const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
    compute(gpu, SHADER, { label: "ctor-only" });
    expect(mock.calls.createComputePipeline).toBe(0);
    expect(mock.calls.createComputePipelineAsync).toBe(0);
  });

  test("the concrete instance exposes no public `pipeline` field at runtime", async () => {
    gpu = await init();
    const sim = compute(gpu, SHADER, { label: "no-field" }) as unknown as Record<string, unknown>;
    expect(Object.prototype.hasOwnProperty.call(sim, "pipeline")).toBe(false);
    expect(sim.pipeline).toBeUndefined();
  });
});

// --- dispatch() stays lazy-sync: it compiles on first use, not construction, and memoizes -------

describe("dispatch() lazy-sync pipeline compilation", () => {
  test("dispatch() compiles exactly once on the first call and reuses it afterwards", async () => {
    gpu = await init();
    const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
    const sim = compute(gpu, SHADER, { label: "lazy" });
    expect(mock.calls.createComputePipeline).toBe(0);
    sim.dispatch(1);
    expect(mock.calls.createComputePipeline).toBe(1);
    sim.dispatch(1);
    sim.dispatch(2, 3, 4);
    expect(mock.calls.createComputePipeline).toBe(1);
    expect(mock.calls.createComputePipelineAsync).toBe(0);
  });
});

// --- dispatchOnce(): async readiness, own encoder, submit exactly once, never onSubmittedWorkDone -

describe("dispatchOnce() async pipeline readiness (contract #20)", () => {
  test("uses createComputePipelineAsync when the pipeline is not ready yet", async () => {
    gpu = await init();
    const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
    const sim = compute(gpu, SHADER, { label: "once" });
    await sim.dispatchOnce(1);
    expect(mock.calls.createComputePipelineAsync).toBe(1);
    expect(mock.calls.createComputePipeline).toBe(0);
  });

  test("resolves right after submit and never awaits queue.onSubmittedWorkDone", async () => {
    gpu = await init();
    const sim = compute(gpu, SHADER, { label: "readiness" });
    const order: string[] = [];
    const onSubmittedWorkDone = vi.spyOn(gpu.device.gpu.queue, "onSubmittedWorkDone");
    const originalSubmit = gpu.device.gpu.queue.submit.bind(gpu.device.gpu.queue);
    vi.spyOn(gpu.device.gpu.queue, "submit").mockImplementation((buffers: GPUCommandBuffer[]) => {
      order.push("submit");
      return originalSubmit(buffers);
    });
    await sim.dispatchOnce(8);
    order.push("resolved");
    expect(order).toEqual(["submit", "resolved"]);
    expect(onSubmittedWorkDone).not.toHaveBeenCalled();
    vi.restoreAllMocks();
  });

  test("dispatchOnce() and dispatch() interleaved on the same instance share one compiled pipeline", async () => {
    gpu = await init();
    const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
    const sim = compute(gpu, SHADER, { label: "shared-async-first" });
    await sim.dispatchOnce(1);
    expect(mock.calls.createComputePipelineAsync).toBe(1);
    sim.dispatch(1); // legacy sync dispatch must reuse the pipeline dispatchOnce already compiled
    expect(mock.calls.createComputePipeline).toBe(0);
    await sim.dispatchOnce(1); // a second dispatchOnce must not recompile either
    expect(mock.calls.createComputePipelineAsync).toBe(1);
  });

  test("dispatch() first, then dispatchOnce(): the async path reuses the sync-compiled pipeline", async () => {
    gpu = await init();
    const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
    const sim = compute(gpu, SHADER, { label: "shared-sync-first" });
    sim.dispatch(1);
    expect(mock.calls.createComputePipeline).toBe(1);
    await sim.dispatchOnce(1);
    expect(mock.calls.createComputePipeline).toBe(1);
    expect(mock.calls.createComputePipelineAsync).toBe(0);
  });

  test("dispatch() racing an un-awaited dispatchOnce() on the same instance settles on one memoized pipeline", async () => {
    // #ensurePipeline() (sync) does not know about an in-flight #ensurePipelineAsync() compile, so this
    // interleaving can still trigger both createComputePipeline and createComputePipelineAsync once each —
    // but whichever result lands first must win (##= in the async .then()) and stay memoized: no further
    // calls, sync or async, may compile again afterwards.
    gpu = await init();
    const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
    const sim = compute(gpu, SHADER, { label: "race-mixed" });
    const pending = sim.dispatchOnce(1); // kicks off createComputePipelineAsync, not yet resolved
    sim.dispatch(1); // synchronous dispatch races ahead and compiles via createComputePipeline
    await pending;
    const totalCompilesAfterRace = mock.calls.createComputePipeline + mock.calls.createComputePipelineAsync;
    expect(totalCompilesAfterRace).toBeGreaterThanOrEqual(1);
    sim.dispatch(1);
    await sim.dispatchOnce(1);
    expect(mock.calls.createComputePipeline + mock.calls.createComputePipelineAsync).toBe(totalCompilesAfterRace);
  });
});

// --- Contract #17 — dispatch()/dispatchOnce() validate integer workgroup counts >= 0 -------------

describe("dispatch()/dispatchOnce() reject invalid workgroup counts with a stable error code (contract #17)", () => {
  test("dispatch() throws VGPU-R1-DISPATCH-COUNT for negative, fractional, or NaN counts", async () => {
    gpu = await init();
    const sim = compute(gpu, SHADER, { label: "invalid-sync" });
    for (const bad of [-1, 1.5, Number.NaN]) {
      expect(() => sim.dispatch(bad)).toThrowError(expect.objectContaining({ code: "VGPU-R1-DISPATCH-COUNT" }));
    }
    expect(() => sim.dispatch(1, -2)).toThrowError(expect.objectContaining({ code: "VGPU-R1-DISPATCH-COUNT" }));
    expect(() => sim.dispatch(1, 2, 1.2)).toThrowError(expect.objectContaining({ code: "VGPU-R1-DISPATCH-COUNT" }));
  });

  test("dispatchOnce() rejects with VGPU-R1-DISPATCH-COUNT for the same invalid counts", async () => {
    gpu = await init();
    const sim = compute(gpu, SHADER, { label: "invalid-async" });
    await expect(sim.dispatchOnce(-1)).rejects.toMatchObject({ code: "VGPU-R1-DISPATCH-COUNT" });
    await expect(sim.dispatchOnce(1.5)).rejects.toMatchObject({ code: "VGPU-R1-DISPATCH-COUNT" });
    await expect(sim.dispatchOnce(1, 2, -3)).rejects.toMatchObject({ code: "VGPU-R1-DISPATCH-COUNT" });
  });

  test("valid counts (including 0) never throw", async () => {
    gpu = await init();
    const sim = compute(gpu, SHADER, { label: "valid" });
    expect(() => sim.dispatch(0)).not.toThrow();
    await expect(sim.dispatchOnce(0, 0, 0)).resolves.toBeUndefined();
  });
});

// --- Lifecycle: a disposed gpu must fail with VGPU-DEVICE-DISPOSED, not a raw WebGPU error ----------

describe("dispose() interacting with the lazy/async pipeline paths", () => {
  test("dispatchOnce() rejects with VGPU-DEVICE-DISPOSED if the gpu was disposed before the call", async () => {
    gpu = await init();
    const sim = compute(gpu, SHADER, { label: "disposed-before" });
    gpu.dispose();
    await expect(sim.dispatchOnce(1)).rejects.toMatchObject({ code: "VGPU-DEVICE-DISPOSED" });
    gpu = undefined;
  });

  test("dispatch() throws VGPU-DEVICE-DISPOSED if the gpu was disposed before the call", async () => {
    gpu = await init();
    const sim = compute(gpu, SHADER, { label: "disposed-before-sync" });
    gpu.dispose();
    expect(() => sim.dispatch(1)).toThrowError(expect.objectContaining({ code: "VGPU-DEVICE-DISPOSED" }));
    gpu = undefined;
  });

  test("dispatchOnce() rejects with VGPU-DEVICE-DISPOSED (not a raw WebGPU error) when disposed while the async compile is in flight", async () => {
    gpu = await init();
    const sim = compute(gpu, SHADER, { label: "disposed-mid-flight" });
    let resolvePipeline!: (pipeline: GPUComputePipeline) => void;
    const pending = new Promise<GPUComputePipeline>((resolve) => { resolvePipeline = resolve; });
    vi.spyOn(gpu.device.gpu, "createComputePipelineAsync").mockReturnValue(pending);
    const dispatched = sim.dispatchOnce(1);
    gpu.dispose();
    resolvePipeline({} as GPUComputePipeline);
    await expect(dispatched).rejects.toMatchObject({ code: "VGPU-DEVICE-DISPOSED" });
    vi.restoreAllMocks();
    gpu = undefined;
  });
});
