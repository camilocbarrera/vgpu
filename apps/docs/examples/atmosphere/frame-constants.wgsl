import { Atmosphere, Camera, FrameConstants, PI, TERRAIN_TRANSMITTANCE_ENTRIES, sampleTransmittance, skyViewUv } from "./atmosphere-common.wgsl";
import { TERRAIN_MAX_HEIGHT } from "./terrain.wgsl";

@group(0) @binding(0) var<uniform> atmosphere: Atmosphere;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var transmittanceLut: texture_2d<f32>;
@group(0) @binding(3) var skyViewLut: texture_2d<f32>;
@group(0) @binding(4) var lutSampler: sampler;
@group(0) @binding(5) var<storage, read_write> frameConstants: FrameConstants;

/**
 * One workgroup per frame. Every scalar expression here is copied verbatim from the pass that used to evaluate it
 * per pixel, so those values are bit-identical. Runs before the sky-view pass, so skyAmbient reads last frame's LUT.
 * Each thread also bakes one entry of the terrain sun-transmittance table.
 */
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local: vec3u) {
  let p = atmosphere;
  let entryHeight = f32(local.x) / f32(TERRAIN_TRANSMITTANCE_ENTRIES - 1u) * TERRAIN_MAX_HEIGHT;
  frameConstants.terrainSunTransmittance[local.x] = vec4f(sampleTransmittance(p, transmittanceLut, lutSampler, p.groundRadius + entryHeight, p.sunDirection.y), 0.0);
  if (local.x != 0u) { return; }
  let viewHeight = length(camera.position);
  let up = camera.position / viewHeight;
  let vHorizon = sqrt(max(0.0, viewHeight * viewHeight - p.groundRadius * p.groundRadius));
  let beta = acos(clamp(vHorizon / viewHeight, -1.0, 1.0));
  let sunHorizontal = p.sunDirection - up * dot(p.sunDirection, up);
  let radius = camera.sunAngularRadius;
  frameConstants.skyAmbient = textureSampleLevel(skyViewLut, lutSampler, skyViewUv(p, viewHeight, 0.5, 0.0, false), 0.0).rgb;
  frameConstants.sunCosRadius = cos(radius);
  frameConstants.groundBounce = 0.15 * p.sunIlluminance * sampleTransmittance(p, transmittanceLut, lutSampler, p.groundRadius, p.sunDirection.y) * max(p.sunDirection.y, 0.0) / PI;
  frameConstants.sunSinRadius = sin(radius);
  frameConstants.sunHorizontal = sunHorizontal;
  frameConstants.sunHorizontalLength = length(sunHorizontal);
  frameConstants.beta = beta;
  frameConstants.zenithHorizonAngle = PI - beta;
  frameConstants.sunSolidAngle = PI * sin(radius) * sin(radius);
  // Below ~3.4 degrees of elevation the planet can shadow cloud samples up to 70 km away (local horizon tilts < 0.7 deg).
  frameConstants.planetShadowNeeded = select(0.0, 1.0, p.sunDirection.y < 0.06);
}
