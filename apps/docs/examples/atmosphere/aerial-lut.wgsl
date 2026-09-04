import { AERIAL_KM_PER_SLICE, AERIAL_LUT_SIZE, Atmosphere, Camera, PLANET_RADIUS_OFFSET, SunShadow, cameraRay, meanTransmittance, miePhase, rayleighPhase, raySphere, sampleMedium, sampleMultiScatter, sampleTransmittance, sunShadowSample } from "./atmosphere-common.wgsl";
import { TERRAIN_MAP_EXTENT } from "./terrain.wgsl";
import { Clouds, sampleCloudShadow } from "./clouds-common.wgsl";

@group(0) @binding(0) var<uniform> atmosphere: Atmosphere;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var transmittanceLut: texture_2d<f32>;
@group(0) @binding(3) var multiScatterLut: texture_2d<f32>;
@group(0) @binding(4) var lutSampler: sampler;
@group(0) @binding(5) var aerialLut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var sunShadowMap0: texture_depth_2d;
@group(0) @binding(7) var aerialLossLut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(8) var<uniform> clouds: Clouds;
@group(0) @binding(9) var cloudShadowMap: texture_2d<f32>;
@group(0) @binding(10) var shadowSampler: sampler_comparison;
@group(0) @binding(11) var<uniform> sunShadow: SunShadow;
@group(0) @binding(12) var sunShadowMap1: texture_depth_2d;
@group(0) @binding(13) var sunShadowMap2: texture_depth_2d;

struct AerialResult { luminance: vec3f, loss: vec3f, transmittance: vec3f };

/**
 * The integrateScattering loop of atmosphere-common.wgsl (Mie/Rayleigh phase, multiple scattering, no ground) with
 * one addition: every sample also asks the sun's shadow map whether it sees the sun (one depth comparison; the map
 * is last frame's, a frame late when the sun moves), and the air under the cloud layer asks the cloud shadow map how
 * much of it, so only lit samples add single scattering. The single scattering
 * removed this way is accumulated separately, so sky pixels, which read the terrain-agnostic sky-view LUT, can take
 * it out too. The multiple-scattering ambient stays unshadowed.
 */
fn integrateAerial(p: Atmosphere, origin: vec3f, dir: vec3f, tMaxMax: f32, sampleCount: f32) -> AerialResult {
  var result = AerialResult(vec3f(0.0), vec3f(0.0), vec3f(1.0));
  let tBottom = raySphere(origin, dir, p.groundRadius);
  let tTop = raySphere(origin, dir, p.atmosphereRadius);
  var tMax = 0.0;
  if (tBottom < 0.0) {
    if (tTop < 0.0) { return result; }
    tMax = tTop;
  } else {
    tMax = tBottom;
    if (tTop > 0.0) { tMax = min(tTop, tBottom); }
  }
  tMax = min(tMax, tMaxMax);

  let cosTheta = dot(dir, p.sunDirection);
  let phaseMie = miePhase(p.mieG, cosTheta);
  let phaseRayleigh = rayleighPhase(cosTheta);
  let dt = tMax / sampleCount;
  var throughput = vec3f(1.0);
  for (var i = 0.0; i < sampleCount; i += 1.0) {
    let t = (i + 0.3) * dt;
    let position = origin + t * dir;
    let medium = sampleMedium(p, position);
    let viewHeight = length(position);
    let up = position / viewHeight;
    let sunZenithCos = dot(p.sunDirection, up);
    let sunTransmittance = sampleTransmittance(p, transmittanceLut, lutSampler, viewHeight, sunZenithCos);
    let earthShadow = select(1.0, 0.0, raySphere(position + up * PLANET_RADIUS_OFFSET, p.sunDirection, p.groundRadius) >= 0.0);
    var lit = sunShadowSample(sunShadow, sunShadowMap0, sunShadowMap1, sunShadowMap2, shadowSampler, position - vec3f(0.0, p.groundRadius, 0.0), 1.0);
    // The cloud shadow map is taken from the terrain surface; the air below the layer sees nearly the same column.
    if (viewHeight - p.groundRadius < clouds.bottom) { lit *= sampleCloudShadow(clouds, cloudShadowMap, lutSampler, position.xz, TERRAIN_MAP_EXTENT); }
    let multiScatter = sampleMultiScatter(p, multiScatterLut, lutSampler, viewHeight, sunZenithCos);
    let direct = p.sunIlluminance * (earthShadow * sunTransmittance * (medium.mie * phaseMie + medium.rayleigh * phaseRayleigh));
    let ambient = p.sunIlluminance * (multiScatter * medium.scattering);
    let extinction = max(medium.extinction, vec3f(1e-7));
    let stepTransmittance = exp(-extinction * dt);
    let integral = throughput * (1.0 - stepTransmittance) / extinction;
    result.luminance += (direct * lit + ambient) * integral;
    result.loss += direct * (1.0 - lit) * integral;
    throughput *= stepTransmittance;
  }
  result.transmittance = throughput;
  return result;
}

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
  let result = integrateAerial(p, camera.position, dir, tMax, sampleCount);
  textureStore(aerialLut, id, vec4f(result.luminance, 1.0 - meanTransmittance(result.transmittance)));
  textureStore(aerialLossLut, id, vec4f(result.loss, 0.0));
}
