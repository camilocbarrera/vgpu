import { expect, test } from "vitest";
import { createMockGPUDevice, Device, loseMockGPUDevice } from "../src/index.ts";

/**
 * Design §9 "Device loss is terminal", core half: a REAL device loss (the native `GPUDevice.lost`
 * promise resolving on its own) is observable through `device.lost`, while a deliberate
 * `dispose()`/`destroy()` — which also resolves the native promise, with reason `"destroyed"` —
 * must NEVER resolve it. Loss and disposal are two spellings for two different semantics.
 */

test("device.lost resolves with the native GPUDeviceLostInfo on a real loss, with no intermediate call", async () => {
  const device = new Device(createMockGPUDevice());

  void loseMockGPUDevice(device.gpu, { reason: "unknown", message: "simulated loss" });
  const info = await device.lost;

  expect(info?.reason).toBe("unknown");
  expect(info?.message).toBe("simulated loss");
});

test("every operation on a lost device throws VGPU-DEVICE-LOST carrying the loss info as cause", async () => {
  const device = new Device(createMockGPUDevice());
  const info = await loseMockGPUDevice(device.gpu, { reason: "unknown", message: "simulated loss" });
  await device.lost;

  expect(() => device.createBuffer({ size: 16, usage: ["storage"] })).toThrow(expect.objectContaining({ code: "VGPU-DEVICE-LOST", where: "Device.createBuffer" }));
  expect(() => device.createTexture({ size: [2, 2], format: "rgba8unorm", usage: ["render_attachment"] })).toThrow(expect.objectContaining({ code: "VGPU-DEVICE-LOST" }));
  try { device.createShader("@fragment fn main() -> @location(0) vec4f { return vec4f(1); }"); }
  catch (error) { expect((error as { cause?: unknown }).cause).toBe(info); }
});

test("device.lost never resolves after a deliberate destroy(), even though the native promise resolves with \"destroyed\"", async () => {
  const device = new Device(createMockGPUDevice());

  device.destroy();
  // The mock resolves its native `lost` on destroy(), exactly like a real device does.
  await Promise.resolve();

  expect(await settledWithin(device.lost, 25)).toBe("pending");
  expect(() => device.createBuffer({ size: 16, usage: ["storage"] })).toThrow(expect.objectContaining({ code: "VGPU-DEVICE-DISPOSED" }));
});

test("destroy() after a real loss stays idempotent and does not re-destroy the native device", async () => {
  let destroyed = 0;
  const gpu = createMockGPUDevice();
  const nativeDestroy = gpu.destroy.bind(gpu);
  gpu.destroy = () => { destroyed += 1; nativeDestroy(); };
  const device = new Device(gpu);

  void loseMockGPUDevice(gpu, { reason: "unknown" });
  await device.lost;
  expect(() => device.destroy()).not.toThrow();
  expect(() => device.destroy()).not.toThrow();

  // The device is already gone: destroying it again is the one call vgpu must not make.
  expect(destroyed).toBe(0);
  expect(() => device.createBuffer({ size: 16, usage: ["storage"] })).toThrow(expect.objectContaining({ code: "VGPU-DEVICE-DISPOSED" }));
});

test("a device whose native handle has no `lost` promise simply never resolves device.lost", async () => {
  const gpu = createMockGPUDevice() as GPUDevice & { lost?: unknown };
  delete gpu.lost;
  const device = new Device(gpu);

  expect(await settledWithin(device.lost, 25)).toBe("pending");
});

async function settledWithin(promise: Promise<unknown>, ms: number): Promise<"settled" | "pending"> {
  return Promise.race([
    promise.then(() => "settled" as const, () => "settled" as const),
    new Promise<"pending">((resolve) => setTimeout(() => resolve("pending"), ms)),
  ]);
}
