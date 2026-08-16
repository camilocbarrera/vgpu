import { init, effect, prepare, target } from "vgpu/mock";

const NEEDS_SAMPLER = /* wgsl */ `
@group(0) @binding(0) var samp: sampler;
fn useSampler(value: sampler) {}
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f { useSampler(samp); return vec4f(uv, 0.0, 1.0); }
`;

const SPEED = /* wgsl */ `
struct Params { speed: f32 }
@group(0) @binding(0) var<uniform> params: Params;
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f { return vec4f(params.speed, uv, 1.0); }
`;

export async function collectFixitMessages() {
  const gpu = await init();
  try {
    const missing = effect(gpu, { shader: NEEDS_SAMPLER, label: "lighting" });
    const ownership = effect(gpu, { shader: SPEED, label: "wave", set: { speed: 2 } });
    const messages: string[] = [];
    const colorTarget = target(gpu, { size: [4, 4] });
    // This example was originally left out of T04-19 on the theory that preparing `missing` would
    // make prepare() itself reject and REPLACE the fix-it this file exists to demonstrate. That
    // theory was never executed, and it is wrong in both halves: prepare() succeeds here (an unset
    // binding is a bind-time error, not a pipeline-creation one), and WITHOUT it the throw default
    // raises VGPU-PIPELINE-PENDING first, shadowing the fix-it entirely. So the await is what keeps
    // the lesson intact under T04-21: it settles the readiness question so the encode can reach the
    // binding check that is the actual subject.
    await prepare(gpu, [{ draw: missing, target: colorTarget }]);
    try { missing.draw({ target: colorTarget }); } catch (error) { messages.push(String((error as Error).message)); }
    try { ownership.set({ speed: gpu.device.createBuffer({ size: 4, usage: ["uniform", "copy_dst"] }) }); } catch (error) { messages.push(String((error as Error).message)); }
    return messages;
  } finally {
    gpu.dispose();
  }
}
