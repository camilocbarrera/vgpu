import { init, effect, frame, prepare, target } from "vgpu/node";

export const WAVE = /* wgsl */ `
struct Params { time: f32, speed: f32 }
@group(0) @binding(0) var<uniform> params: Params;
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
  return vec4f(uv, sin(params.time * params.speed) * .5 + .5, 1);
}
`;

export async function runFullscreenExample() {
  const gpu = await init();
  const colorTarget = target(gpu, { size: [8, 8], format: "rgba8unorm" });
  const wave = effect(gpu, { shader: WAVE, label: "wave", set: { speed: 2 } });
  wave.set("params", { time: Math.PI / 4 });
  // One combination, one await, at the setup boundary: everything is built, nothing is encoded yet.
  await prepare(gpu, [{ draw: wave, target: colorTarget }]);
  frame(gpu, (currentFrame) => currentFrame.pass({ target: colorTarget }, (p) => p.draw(wave)));
  return { gpu, target: colorTarget };
}
