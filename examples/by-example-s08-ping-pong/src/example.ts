import { init, effect, frame, pingPong, prepare } from "vgpu/node";

export const FILL = /* wgsl */ `
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f { return vec4f(uv, 0.5, 1.0); }
`;
export const COPY = /* wgsl */ `
struct Params { texel: vec2f }
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
  return textureLoad(src, vec2u(vec2f(uv) / params.texel), 0);
}
`;

export async function runPingPongExample() {
  const gpu = await init();
  const buf = pingPong(gpu, 8, 8, { format: "rgba8unorm" });
  const fill = effect(gpu, { shader: FILL, label: "fill" });
  const copy = effect(gpu, { shader: COPY, label: "copy" });
  // `copy` runs AFTER swap(), so the half it writes into is today's `read` half — naming the real
  // object keeps the intent readable. Both halves share one signature (same format, same size), so
  // the two requests resolve to the same pipeline key: preparing is idempotent, not double work.
  // Nothing is *bound* here — `prepare()` takes the target to derive a signature, never to pin a
  // resource, which is why warming a ping-pong half cannot freeze it the way a construction-time
  // binding would.
  await prepare(gpu, [
    { draw: fill, target: buf.write },
    { draw: copy, target: buf.read },
  ]);
  frame(gpu, (currentFrame) => currentFrame.pass({ target: buf.write }, (p) => p.draw(fill)));
  buf.swap();
  frame(gpu, (currentFrame) => currentFrame.pass({ target: buf.write }, (p) => { copy.set({ src: buf.read, texel: buf.read.texelSize }); p.draw(copy); }));
  return { gpu, target: buf.write };
}
