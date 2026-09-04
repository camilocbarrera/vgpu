import { Atmosphere, Camera, FrameConstants, PI, sampleTransmittance, skyViewUv } from "./atmosphere-common.wgsl";

@group(0) @binding(0) var<uniform> atmosphere: Atmosphere;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var transmittanceLut: texture_2d<f32>;
@group(0) @binding(3) var skyViewLut: texture_2d<f32>;
@group(0) @binding(4) var lutSampler: sampler;
@group(0) @binding(5) var<storage, read_write> frameConstants: FrameConstants;

/**
 * One thread per frame. Every expression here is copied verbatim from the pass that used to evaluate it per
 * pixel, so the baked value is bit-identical. Runs before the sky-view pass, so skyAmbient reads last frame's LUT.
 */
@compute @workgroup_size(1)
fn main() {
  let p = atmosphere;
  let viewHeight = length(camera.position);
  let up = camera.position / viewHeight;
  let vHorizon = sqrt(max(0.0, viewHeight * viewHeight - p.groundRadius * p.groundRadius));
  let beta = acos(clamp(vHorizon / viewHeight, -1.0, 1.0));
  let sunHorizontal = p.sunDirection - up * dot(p.sunDirection, up);
  let radius = camera.sunAngularRadius;
  var f: FrameConstants;
  f.skyAmbient = textureSampleLevel(skyViewLut, lutSampler, skyViewUv(p, viewHeight, 0.5, 0.0, false), 0.0).rgb;
  f.sunCosRadius = cos(radius);
  f.groundBounce = 0.15 * p.sunIlluminance * sampleTransmittance(p, transmittanceLut, lutSampler, p.groundRadius, p.sunDirection.y) * max(p.sunDirection.y, 0.0) / PI;
  f.sunSinRadius = sin(radius);
  f.sunHorizontal = sunHorizontal;
  f.sunHorizontalLength = length(sunHorizontal);
  f.beta = beta;
  f.zenithHorizonAngle = PI - beta;
  f.sunSolidAngle = PI * sin(radius) * sin(radius);
  // Below ~3.4 degrees of elevation the planet can shadow cloud samples up to 70 km away (local horizon tilts < 0.7 deg).
  f.planetShadowNeeded = select(0.0, 1.0, p.sunDirection.y < 0.06);
  frameConstants = f;
}
