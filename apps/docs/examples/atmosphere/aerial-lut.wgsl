import { AERIAL_KM_PER_SLICE, AERIAL_LUT_SIZE, Atmosphere, Camera, cameraRay, integrateScattering, meanTransmittance } from "./atmosphere-common.wgsl";

@group(0) @binding(0) var<uniform> atmosphere: Atmosphere;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var transmittanceLut: texture_2d<f32>;
@group(0) @binding(3) var multiScatterLut: texture_2d<f32>;
@group(0) @binding(4) var lutSampler: sampler;
@group(0) @binding(5) var aerialLut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var aerialMieLut: texture_storage_3d<rgba16float, write>;

/** Froxel volume: xy = screen, z = quadratic depth slices. rgb = in-scattered luminance, a = 1 - transmittance. */
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let p = atmosphere;
  let uv = (vec2f(id.xy) + 0.5) / AERIAL_LUT_SIZE;
  let dir = cameraRay(camera, vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0));
  var slice = (f32(id.z) + 0.5) / AERIAL_LUT_SIZE;
  slice = slice * slice * AERIAL_LUT_SIZE;
  let tMax = slice * AERIAL_KM_PER_SLICE;
  let sampleCount = max(1.0, f32(id.z + 1u) * 2.0);
  let result = integrateScattering(p, camera.position, dir, p.sunDirection, tMax, sampleCount, true, false, true, transmittanceLut, multiScatterLut, lutSampler);
  textureStore(aerialLut, id, vec4f(result.luminance, 1.0 - meanTransmittance(result.transmittance)));
  textureStore(aerialMieLut, id, vec4f(result.directMie, 1.0));
}
