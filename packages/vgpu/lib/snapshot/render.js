import { SNAPSHOT_SIZE, REPRESENTATIVE_GRADIENT_WGSL } from "./shaders.js";

export const DEFAULT_SNAPSHOT_TIME = Math.PI / 4;
export const DEFAULT_SNAPSHOT_SPEED = 2;

/** @param api The `vgpu/node` module namespace: `init` plus the free functions this render needs. */
export async function renderRepresentativeSnapshot(api) {
  // `renderOnce` rather than `frame` (T04-21): it is the one-shot path, and it awaits readiness for
  // every combination it collected before encoding, so this tool needs no `pendingPipelines` opt-out
  // -- it renders under the shipped `"throw"` default, the way the headless recipe in the docs does.
  const { init, effect, renderOnce, target } = api;
  const gpu = await init();
  try {
    const colorTarget = target(gpu, { size: SNAPSHOT_SIZE, format: "rgba8unorm", label: "vgpu-snapshot-gradient" });
    const gradient = effect(gpu, REPRESENTATIVE_GRADIENT_WGSL, {
      label: "vgpu-snapshot-gradient",
      set: { speed: DEFAULT_SNAPSHOT_SPEED },
    });
    gradient.set({ time: DEFAULT_SNAPSHOT_TIME });
    await renderOnce(gpu, colorTarget, (encoder) => encoder.draw(gradient));
    return { pixels: await colorTarget.read(), width: SNAPSHOT_SIZE[0], height: SNAPSHOT_SIZE[1] };
  } finally {
    gpu.dispose();
  }
}
