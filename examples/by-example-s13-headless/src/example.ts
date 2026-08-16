import { init, effect, prepare, target } from "vgpu/node";

export const GRADIENT = /* wgsl */ `
struct Params { time: f32, speed: f32 }
@group(0) @binding(0) var<uniform> params: Params;
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
  return vec4f(uv.x + params.time * 0.1, uv.y, params.speed, 1.0);
}
`;

export async function renderGradientHeadless() {
  const gpu = await init();
  const colorTarget = target(gpu, { size: [8, 8], format: "rgba8unorm" });
  const p = effect(gpu, { shader: GRADIENT, label: "gradient" });
  p.set("params", { time: 1.25, speed: 1 });
  // No `frame()` here — the one-shot `p.draw({ target })` IS the first encode, and it resolves the
  // policy exactly like a frame draw does (draw.ts `encode()` -> `#pipelineForEncode`). So the
  // setup boundary is the line before it, not before a loop that does not exist.
  await prepare(gpu, [{ draw: p, target: colorTarget }]);
  p.draw({ target: colorTarget });
  return { gpu, target: colorTarget };
}
