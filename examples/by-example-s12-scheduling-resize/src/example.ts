import { init, effect, frame, prepare, target } from "vgpu/node";

export const POST = /* wgsl */ `
struct Params { time: f32, texel: vec2f }
@group(0) @binding(0) var<uniform> params: Params;
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f { return vec4f(params.texel, params.time, 1.0); }
`;

export async function runSchedulingResizeExample() {
  const gpu = await init();
  const baked = target(gpu, { size: [4, 4], format: "rgba8unorm" });
  const post = effect(gpu, { shader: POST, label: "post" });
  // ONE prepare covers both frames even though `baked.resize([8, 8])` runs between them: a target
  // signature is { colors, depth, sampleCount } — size is not in it, so a resize that keeps the
  // format cannot invalidate the pipeline this warms.
  await prepare(gpu, [{ draw: post, target: baked }]);
  frame(gpu, (f) => f.pass({ target: baked }, (p) => { post.set("params", { time: 0.25, texel: baked.texelSize }); p.draw(post); }));
  baked.resize([8, 8]);
  frame(gpu, (f) => f.pass({ target: baked }, (p) => { post.set("params", { time: 0.5, texel: baked.texelSize }); p.draw(post); }));
  return { gpu, target: baked };
}
