import { Atmosphere } from "./atmosphere-common.wgsl";
import { TERRAIN_MAP_EXTENT, sampleTerrainHeight } from "./terrain.wgsl";
import { CLOUD_SHADOW_MAP_SIZE, Clouds, cloudDensity, cloudRange } from "./clouds-common.wgsl";

@group(0) @binding(0) var<uniform> atmosphere: Atmosphere;
@group(0) @binding(1) var<uniform> clouds: Clouds;
@group(0) @binding(2) var terrainMap: texture_2d<f32>;
@group(0) @binding(3) var shapeNoise: texture_3d<f32>;
@group(0) @binding(4) var detailNoise: texture_3d<f32>;
@group(0) @binding(5) var weatherMap: texture_2d<f32>;
@group(0) @binding(6) var curlNoise: texture_2d<f32>;
@group(0) @binding(7) var lutSampler: sampler;
@group(0) @binding(8) var noiseSampler: sampler;
@group(0) @binding(9) var cloudShadowMap: texture_storage_2d<rgba16float, write>;

/** Samples through the cloud layer along the sun; the map's 390 m texels are coarser than this anyway. */
const SAMPLES: i32 = 8;
/** Extinction per unit density, 1/km; the same value the cloud march uses. */
const EXTINCTION: f32 = 32.0;

/**
 * Cloud shadow as a map over the terrain: for each texel, the transmittance of the cloud layer along the sun from
 * the terrain surface below. The terrain shading reads it where the surface is, and the aerial perspective reads it
 * for the air under the layer, which puts the clouds' shadows into the haze as well. Rebuilt every frame: the wind
 * moves the clouds and the sun may move too, and eight cheap density samples per texel cost little.
 */
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let p = atmosphere;
  let uv = (vec2f(id.xy) + 0.5) / CLOUD_SHADOW_MAP_SIZE;
  let xz = (uv - 0.5) * TERRAIN_MAP_EXTENT;
  var transmittance = 1.0;
  if (p.sunDirection.y > 0.02 && clouds.coverage > 0.0) {
    let surfaceRadius = p.groundRadius + sampleTerrainHeight(terrainMap, lutSampler, xz);
    let origin = vec3f(xz.x, sqrt(max(surfaceRadius * surfaceRadius - dot(xz, xz), 0.0)), xz.y);
    let range = cloudRange(clouds, origin, p.sunDirection, surfaceRadius);
    if (range.valid && range.end > range.start) {
      let step = (range.end - range.start) / f32(SAMPLES);
      var opticalDepth = 0.0;
      for (var i = 0; i < SAMPLES; i += 1) {
        let position = origin + p.sunDirection * (range.start + (f32(i) + 0.5) * step);
        let altitude = length(position) - p.groundRadius;
        opticalDepth += cloudDensity(clouds, shapeNoise, detailNoise, weatherMap, curlNoise, noiseSampler, position, altitude, 1e9, true) * step;
      }
      transmittance = exp(-EXTINCTION * opticalDepth);
    }
  }
  textureStore(cloudShadowMap, id.xy, vec4f(transmittance, 0.0, 0.0, 1.0));
}
