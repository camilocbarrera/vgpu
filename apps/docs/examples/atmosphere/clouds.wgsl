import { AERIAL_KM_PER_SLICE, AERIAL_LUT_SIZE, Atmosphere, Camera, PI, PLANET_RADIUS_OFFSET, cameraRay, raySphere, sampleTransmittance, skyViewUv } from "./atmosphere-common.wgsl";
import { Clouds, cloudDensity, heightFraction } from "./clouds-common.wgsl";

@group(0) @binding(0) var<uniform> atmosphere: Atmosphere;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<uniform> clouds: Clouds;
@group(0) @binding(3) var transmittanceLut: texture_2d<f32>;
@group(0) @binding(4) var skyViewLut: texture_2d<f32>;
@group(0) @binding(5) var aerialLut: texture_3d<f32>;
@group(0) @binding(6) var shapeNoise: texture_3d<f32>;
@group(0) @binding(7) var detailNoise: texture_3d<f32>;
@group(0) @binding(8) var weatherMap: texture_2d<f32>;
@group(0) @binding(9) var sceneHdr: texture_2d<f32>;
@group(0) @binding(10) var lutSampler: sampler;
@group(0) @binding(11) var noiseSampler: sampler;
@group(0) @binding(12) var history: texture_2d<f32>;
@group(0) @binding(13) var<uniform> reprojection: Reprojection;
@group(0) @binding(14) var curlNoise: texture_2d<f32>;

/**
 * Previous frame's camera basis: one texel in sixteen is re-marched per frame, the rest reproject from `history`.
 * `blend` < 1 accumulates each re-marched texel into its reprojected history with a sub-texel `jitter`,
 * which supersamples the edges over time; stills use blend 1 and no jitter so they stay deterministic.
 */
struct Reprojection {
  forward: vec3f, frame: f32,
  right: vec3f, tanHalfFov: f32,
  up: vec3f, aspect: f32,
  valid: f32, blend: f32, jitter: vec2f,
};

const MARCH_STEPS: i32 = 160;
const MIN_MARCH_STEPS: f32 = 80.0;
/** Densities below this are treated as the cloud surface and marched with half steps. */
const EDGE_DENSITY: f32 = 0.12;
const LIGHT_STEPS: i32 = 6;
const MAX_MARCH_DISTANCE: f32 = 70.0;
/** Extinction per unit density, 1/km (cumulus, ~0.03/m). */
const EXTINCTION: f32 = 32.0;
const ALBEDO: f32 = 0.97;
const FORWARD_G: f32 = 0.75;
const BACK_G: f32 = -0.3;
/** Multiple-scattering octaves (Wrenninge): per-octave scattering, extinction and phase-eccentricity multipliers. */
const MS_OCTAVES: i32 = 6;
const MS_SCATTER: f32 = 0.8;
const MS_EXTINCTION: f32 = 0.5;
const MS_PHASE: f32 = 0.5;

/** Both intersections with a sphere at the origin: (near, far), or (-1, -1) when missed. */
fn raySphereBoth(origin: vec3f, dir: vec3f, radius: f32) -> vec2f {
  let b = 2.0 * dot(dir, origin);
  let c = dot(origin, origin) - radius * radius;
  let delta = b * b - 4.0 * c;
  if (delta < 0.0) { return vec2f(-1.0); }
  let root = sqrt(delta);
  return vec2f((-b - root) * 0.5, (-b + root) * 0.5);
}

fn henyeyGreenstein(cosTheta: f32, g: f32) -> f32 {
  let g2 = g * g;
  return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
}

fn dualLobePhase(cosTheta: f32, octave: f32) -> f32 {
  return mix(henyeyGreenstein(cosTheta, FORWARD_G * octave), henyeyGreenstein(cosTheta, BACK_G * octave), 0.3);
}

/** Integer hash per pixel (PCG) for the march-start jitter; unstructured, so aliasing becomes fine grain. */
fn pixelJitter(p: vec2f) -> f32 {
  var v = vec2u(p) * vec2u(1664525u, 1013904223u);
  v.x += v.y * 3u; v.y += v.x * 5u;
  v ^= v >> vec2u(16u);
  v.x += v.y * 7u;
  return f32((v.x >> 8u) & 0xffffffu) / 16777215.0;
}

fn density(position: vec3f, viewDistance: f32, cheap: bool) -> f32 {
  let altitude = length(position) - atmosphere.groundRadius;
  return cloudDensity(clouds, shapeNoise, detailNoise, weatherMap, curlNoise, noiseSampler, position, altitude, viewDistance, cheap);
}

/** Optical depth toward the sun with doubling steps (20 m to 640 m); cheap samples skip erosion, so they are scaled down. */
fn lightOpticalDepth(position: vec3f, sunDir: vec3f) -> f32 {
  var depth = 0.0;
  var t = 0.0;
  var step = 0.02;
  for (var i = 0; i < LIGHT_STEPS; i += 1) {
    t += step * 0.5;
    depth += density(position + sunDir * t, 0.0, true) * step;
    t += step * 0.5;
    step *= 2.0;
  }
  return depth * EXTINCTION * 0.75;
}

/** Sum of attenuated scattering octaves; higher octaves see less extinction and a flatter phase. */
fn multiScatter(opticalDepth: f32, cosTheta: f32) -> f32 {
  var sum = 0.0;
  var scatter = 1.0;
  var extinction = 1.0;
  var phaseScale = 1.0;
  for (var i = 0; i < MS_OCTAVES; i += 1) {
    sum += scatter * exp(-opticalDepth * extinction) * dualLobePhase(cosTheta, phaseScale);
    scatter *= MS_SCATTER;
    extinction *= MS_EXTINCTION;
    phaseScale *= MS_PHASE;
  }
  return sum;
}

fn sampleAerial(uv: vec2f, distance: f32) -> vec4f {
  var slice = distance / AERIAL_KM_PER_SLICE;
  var weight = 1.0;
  if (slice < 0.5) { weight = saturate(slice * 2.0); slice = 0.5; }
  let w = sqrt(slice / AERIAL_LUT_SIZE);
  return weight * textureSampleLevel(aerialLut, lutSampler, vec3f(uv, w), 0.0);
}

struct MarchRange { start: f32, end: f32, valid: bool };

/** Entry/exit of the cloud shell for a camera below, inside or above it. */
fn cloudRange(origin: vec3f, dir: vec3f, viewHeight: f32) -> MarchRange {
  let rBottom = atmosphere.groundRadius + clouds.bottom;
  let rTop = atmosphere.groundRadius + clouds.top;
  let bottom = raySphereBoth(origin, dir, rBottom);
  let top = raySphereBoth(origin, dir, rTop);
  if (viewHeight < rBottom) {
    // Below the layer: the far bottom intersection is the entry, the far top intersection is the exit.
    return MarchRange(max(bottom.y, 0.0), top.y, top.y > 0.0);
  }
  if (viewHeight > rTop) {
    // Above the layer: enter at the near top intersection, exit at the near bottom one (or the far top one).
    if (top.x < 0.0) { return MarchRange(0.0, 0.0, false); }
    let exit = select(top.y, bottom.x, bottom.x > 0.0);
    return MarchRange(top.x, exit, true);
  }
  // Inside the layer.
  let exit = select(top.y, bottom.x, bottom.x > 0.0);
  return MarchRange(0.0, exit, true);
}

/**
 * Which texel of each 4x4 block is re-marched this frame (Bayer order so the refresh is spread out).
 * 4x4 rather than 2x2: GPUs shade 2x2 quads together, so one live pixel per quad would save nothing.
 */
fn updatesThisFrame(texel: vec2i, frame: i32) -> bool {
  let phase = (texel.x & 3) | ((texel.y & 3) << 2);
  var order = array<i32, 16>(0, 10, 2, 8, 5, 15, 7, 13, 1, 11, 3, 9, 4, 14, 6, 12);
  return order[frame % 16] == phase;
}

/** Rotation-only reprojection: clouds are far enough that the camera's translation between frames is negligible. */
fn reprojectedUv(dir: vec3f) -> vec2f {
  let z = dot(dir, reprojection.forward);
  if (z <= 1e-3) { return vec2f(-1.0); }
  let ndc = vec2f(dot(dir, reprojection.right) / (z * reprojection.tanHalfFov * reprojection.aspect), dot(dir, reprojection.up) / (z * reprojection.tanHalfFov));
  return vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

@fragment fn fs_main(@builtin(position) fragCoord: vec4f, @location(0) uv: vec2f) -> @location(0) vec4f {
  let p = atmosphere;
  let unjitteredDir = cameraRay(camera, vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0));
  let previousUv = reprojectedUv(unjitteredDir);
  let historyValid = reprojection.valid > 0.5 && all(previousUv >= vec2f(0.0)) && all(previousUv <= vec2f(1.0));
  if (historyValid && !updatesThisFrame(vec2i(fragCoord.xy), i32(reprojection.frame))) {
    return textureSampleLevel(history, lutSampler, previousUv, 0.0);
  }
  // Sub-texel jitter only matters when the result is blended into the history.
  let jitteredUv = uv + reprojection.jitter / vec2f(textureDimensions(history));
  let dir = cameraRay(camera, vec2f(jitteredUv.x * 2.0 - 1.0, 1.0 - jitteredUv.y * 2.0));
  let fresh = marchClouds(p, dir, fragCoord.xy, uv);
  if (historyValid && reprojection.blend < 1.0) {
    return mix(textureSampleLevel(history, lutSampler, previousUv, 0.0), fresh, reprojection.blend);
  }
  return fresh;
}

fn marchClouds(p: Atmosphere, dir: vec3f, fragCoord: vec2f, uv: vec2f) -> vec4f {
  let origin = camera.position;
  let viewHeight = length(origin);
  var range = cloudRange(origin, dir, viewHeight);
  let sceneDistance = textureSampleLevel(sceneHdr, lutSampler, uv, 0.0).a;
  if (sceneDistance > 0.0) { range.end = min(range.end, sceneDistance); }
  range.end = min(range.end, range.start + MAX_MARCH_DISTANCE);
  if (!range.valid || range.end <= range.start || clouds.coverage <= 0.0) { return vec4f(0.0, 0.0, 0.0, 1.0); }

  let cosTheta = dot(dir, p.sunDirection);
  let skyAmbient = textureSampleLevel(skyViewLut, lutSampler, skyViewUv(p, viewHeight, 0.5, 0.0, false), 0.0).rgb;
  let groundSunCos = max(p.sunDirection.y, 0.0);
  let groundBounce = 0.15 * p.sunIlluminance * sampleTransmittance(p, transmittanceLut, lutSampler, p.groundRadius, p.sunDirection.y) * groundSunCos / PI;

  // Rays near the horizon cross far more cloud than rays near the zenith, so the step budget follows the elevation.
  let stepBudget = mix(f32(MARCH_STEPS), MIN_MARCH_STEPS, abs(dir.y));
  let fineStep = max(0.02, (range.end - range.start) / stepBudget);
  let coarseStep = fineStep * 2.0;
  var t = range.start + fineStep * pixelJitter(fragCoord);
  var transmittance = 1.0;
  var luminance = vec3f(0.0);
  var depthSum = 0.0;
  var emptySamples = 0;
  var coarse = false;
  for (var i = 0; i < MARCH_STEPS; i += 1) {
    if (t >= range.end || transmittance < 0.01 || f32(i) >= stepBudget) { break; }
    let position = origin + dir * t;
    let sampleDensity = density(position, t, false);
    if (sampleDensity <= 0.0) {
      emptySamples += 1;
      // After a few empty fine samples switch to coarse stepping; the pull-back below restores precision.
      coarse = emptySamples > 3;
      t += select(fineStep, coarseStep, coarse);
      continue;
    }
    if (coarse) {
      // Coarse step hit density: back up one coarse step and resample finely.
      coarse = false;
      emptySamples = 0;
      t -= coarseStep - fineStep;
      continue;
    }
    emptySamples = 0;
    // Thin samples are the cloud's visible surface: halve the step there so the eroded detail resolves.
    let step = select(fineStep, fineStep * 0.5, sampleDensity < EDGE_DENSITY);
    let altitude = length(position) - p.groundRadius;
    let hf = heightFraction(clouds, altitude);
    let up = position / length(position);
    let sunZenithCos = dot(up, p.sunDirection);
    // Planet shadow: after sunset only clouds whose own horizon still shows the sun stay lit.
    let earthShadow = select(1.0, 0.0, raySphere(position + up * PLANET_RADIUS_OFFSET, p.sunDirection, p.groundRadius) >= 0.0);
    let sunTransmittance = sampleTransmittance(p, transmittanceLut, lutSampler, p.groundRadius + altitude, sunZenithCos) * earthShadow;
    let opticalDepth = lightOpticalDepth(position, p.sunDirection);
    let sunScatter = multiScatter(opticalDepth, cosTheta);
    let ambient = skyAmbient * mix(0.18, 0.75, hf) + groundBounce * (1.0 - hf) * 0.5;
    let extinction = EXTINCTION * sampleDensity;
    let scattered = ALBEDO * extinction * (p.sunIlluminance * sunTransmittance * sunScatter + ambient);
    let stepTransmittance = exp(-extinction * step);
    let integrated = (scattered - scattered * stepTransmittance) / extinction;
    luminance += transmittance * integrated;
    depthSum += t * transmittance * (1.0 - stepTransmittance);
    transmittance *= stepTransmittance;
    t += step;
  }

  let opacity = 1.0 - transmittance;
  if (opacity <= 0.0) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  // Aerial perspective at the transmittance-weighted mean depth, applied to the cloud's own contribution.
  let meanDepth = depthSum / opacity;
  let aerial = sampleAerial(uv, min(meanDepth, AERIAL_KM_PER_SLICE * AERIAL_LUT_SIZE));
  let color = luminance * (1.0 - aerial.a) + aerial.rgb * opacity;
  return vec4f(color, transmittance);
}
