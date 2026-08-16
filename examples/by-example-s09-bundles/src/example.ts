import { init, bundle, effect, frame, prepare, target } from "vgpu/node";

export const FLOOR = /* wgsl */ `
struct Params { fogDensity: f32 }
@group(0) @binding(0) var<uniform> params: Params;
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f { return vec4f(params.fogDensity, uv.x, uv.y, 1.0); }
`;

export async function runBundlesExample() {
  const gpu = await init();
  const scene = target(gpu, { size: [8, 8], format: "rgba8unorm" });
  const floor = effect(gpu, { shader: FLOOR, label: "floor", set: { fogDensity: 0.2 } });
  const staticScene = bundle(gpu, { target: scene, label: "staticScene" }, (b) => { b.draw(floor); });

  // A `{ bundle }` request carries NO `target`: a bundle froze its target signature at
  // construction, so the combination is already complete. Preparing the bundle also warms the
  // pipelines of every draw it recorded (`floor`), which is why `floor` needs no request of its
  // own. One prepare covers BOTH frames below: the `floor.set(...)` between them writes bytes on
  // an instance-owned value, and byte writes do not stale a recorded bundle (only an identity
  // swap does).
  await prepare(gpu, [{ bundle: staticScene }]);

  frame(gpu, (currentFrame) => currentFrame.pass({ target: scene, clear: [0, 0, 0, 1] }, (p) => p.bundles(staticScene)));
  const before = new Uint8Array(await scene.read());

  floor.set("params", { fogDensity: 0.7 });
  frame(gpu, (currentFrame) => currentFrame.pass({ target: scene, clear: [0, 0, 0, 1] }, (p) => p.bundles(staticScene)));
  const after = new Uint8Array(await scene.read());

  return { gpu, target: scene, before, after };
}
