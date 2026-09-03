import { AERIAL_KM_PER_SLICE, AERIAL_LUT_SIZE, Atmosphere, Camera, PI, cameraRay, raySphere, sampleTransmittance, skyViewUv } from "./atmosphere-common.wgsl";
import { TERRAIN_MAX_DISTANCE, TERRAIN_MAX_HEIGHT, terrainAlbedo, terrainHeight, terrainNormal } from "./terrain.wgsl";
import { Clouds, cloudShadow } from "./clouds-common.wgsl";

@group(0) @binding(0) var<uniform> atmosphere: Atmosphere;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var transmittanceLut: texture_2d<f32>;
@group(0) @binding(3) var skyViewLut: texture_2d<f32>;
@group(0) @binding(4) var aerialLut: texture_3d<f32>;
@group(0) @binding(5) var lutSampler: sampler;
@group(0) @binding(6) var<uniform> clouds: Clouds;
@group(0) @binding(7) var weatherMap: texture_2d<f32>;
@group(0) @binding(8) var noiseSampler: sampler;

const TERRAIN_STEPS: i32 = 256;
const SHADOW_STEPS: i32 = 20;

struct TerrainHit { distance: f32, position: vec3f, height: f32 };

fn sampleSkyView(p: Atmosphere, dir: vec3f, viewHeight: f32, intersectGround: bool) -> vec4f {
  let up = camera.position / viewHeight;
  let viewZenithCos = dot(dir, up);
  let sunHorizontal = p.sunDirection - up * dot(p.sunDirection, up);
  let dirHorizontal = dir - up * viewZenithCos;
  let sunLength = length(sunHorizontal);
  let dirLength = length(dirHorizontal);
  var lightViewCos = 1.0;
  if (sunLength > 1e-5 && dirLength > 1e-5) { lightViewCos = dot(sunHorizontal / sunLength, dirHorizontal / dirLength); }
  return textureSampleLevel(skyViewLut, lutSampler, skyViewUv(p, viewHeight, viewZenithCos, lightViewCos, intersectGround), 0.0);
}

fn sampleAerial(uv: vec2f, distance: f32) -> vec4f {
  var slice = distance / AERIAL_KM_PER_SLICE;
  var weight = 1.0;
  if (slice < 0.5) { weight = saturate(slice * 2.0); slice = 0.5; }
  let w = sqrt(slice / AERIAL_LUT_SIZE);
  return weight * textureSampleLevel(aerialLut, lutSampler, vec3f(uv, w), 0.0);
}

/** Sun disc with wavelength-dependent limb darkening, softened over one pixel. */
fn sunDisc(p: Atmosphere, dir: vec3f) -> vec3f {
  let cosAngle = dot(dir, p.sunDirection);
  let radius = camera.sunAngularRadius;
  let cosRadius = cos(radius);
  let edge = sin(radius) * camera.pixelAngle;
  let disc = smoothstep(cosRadius - edge, cosRadius + edge, cosAngle);
  if (disc <= 0.0) { return vec3f(0.0); }
  let angle = acos(clamp(cosAngle, -1.0, 1.0));
  let mu = sqrt(saturate(1.0 - (angle * angle) / (radius * radius)));
  let limb = 1.0 - vec3f(0.397, 0.503, 0.652) * (1.0 - mu);
  let solidAngle = PI * sin(radius) * sin(radius);
  return p.sunIlluminance / solidAngle * limb * disc;
}

/**
 * Analytic glare around the sun: two gaussian lobes carrying a small fraction of the solar illuminance,
 * tinted by the view transmittance so the halo reddens with the disc at sunset.
 */
fn sunGlare(p: Atmosphere, dir: vec3f) -> vec3f {
  let angle = acos(clamp(dot(dir, p.sunDirection), -1.0, 1.0));
  let wide = 0.0436;
  let tight = 0.0105;
  let lobe = 2e-3 / (2.0 * PI * wide * wide) * exp(-0.5 * angle * angle / (wide * wide))
    + 5e-4 / (2.0 * PI * tight * tight) * exp(-0.5 * angle * angle / (tight * tight));
  return p.sunIlluminance * lobe;
}

/** Altitude of a planet-centric point above the sphere; xz doubles as the tangent-plane terrain coordinate. */
fn altitudeOf(p: Atmosphere, position: vec3f) -> f32 { return length(position) - p.groundRadius; }

/** Sphere-tracing style march with distance-proportional steps and a bisection refinement. */
fn marchTerrain(p: Atmosphere, origin: vec3f, dir: vec3f) -> TerrainHit {
  var hit = TerrainHit(-1.0, vec3f(0.0), 0.0);
  let startAltitude = altitudeOf(p, origin);
  if (startAltitude > TERRAIN_MAX_HEIGHT && dir.y >= 0.0) { return hit; }
  var t = 0.0;
  var previousT = 0.0;
  // Skip the empty air above the highest possible peak.
  if (startAltitude > TERRAIN_MAX_HEIGHT) { t = (startAltitude - TERRAIN_MAX_HEIGHT) / max(-dir.y, 1e-3); previousT = t; }
  for (var i = 0; i < TERRAIN_STEPS; i += 1) {
    if (t > TERRAIN_MAX_DISTANCE) { break; }
    let position = origin + dir * t;
    let altitude = altitudeOf(p, position);
    let height = terrainHeight(position.xz);
    let delta = altitude - height;
    if (delta < 0.0) {
      var lo = previousT;
      var hi = t;
      for (var k = 0; k < 8; k += 1) {
        let mid = 0.5 * (lo + hi);
        let midPosition = origin + dir * mid;
        if (altitudeOf(p, midPosition) - terrainHeight(midPosition.xz) < 0.0) { hi = mid; } else { lo = mid; }
      }
      let finalPosition = origin + dir * hi;
      return TerrainHit(hi, finalPosition, terrainHeight(finalPosition.xz));
    }
    previousT = t;
    // Distance-proportional steps reach TERRAIN_MAX_DISTANCE within the budget even for grazing rays;
    // the clearance term slows down near the surface and the bisection above recovers precision.
    let distanceStep = 0.01 + t * 0.025;
    t += max(0.5 * distanceStep, min(delta * 0.6, distanceStep));
  }
  return hit;
}

/** Cheap soft shadow toward the sun; step size grows with distance. */
fn terrainShadow(p: Atmosphere, position: vec3f, sunDir: vec3f) -> f32 {
  if (sunDir.y <= 0.0) { return 0.0; }
  var t = 0.02;
  var shadow = 1.0;
  for (var i = 0; i < SHADOW_STEPS; i += 1) {
    let sample = position + sunDir * t;
    let altitude = altitudeOf(p, sample);
    if (altitude > TERRAIN_MAX_HEIGHT) { break; }
    let delta = altitude - terrainHeight(sample.xz);
    shadow = min(shadow, saturate(8.0 * delta / t));
    if (shadow <= 0.0) { break; }
    t += 0.03 + t * 0.35;
  }
  return shadow;
}

@fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let p = atmosphere;
  let dir = cameraRay(camera, vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0));
  let origin = camera.position;
  let viewHeight = length(origin);
  let tSphere = raySphere(origin, dir, p.groundRadius);
  let terrain = marchTerrain(p, origin, dir);
  let hitsGround = tSphere >= 0.0 || terrain.distance >= 0.0;
  let sky = sampleSkyView(p, dir, viewHeight, hitsGround);
  let skyAmbient = textureSampleLevel(skyViewLut, lutSampler, skyViewUv(p, viewHeight, 0.5, 0.0, false), 0.0).rgb;
  var color = sky.rgb;
  // Alpha carries the geometry distance (km) so the cloud pass can stop at terrain; -1 means sky.
  var hitDistance = -1.0;

  if (terrain.distance >= 0.0) {
    hitDistance = terrain.distance;
    let normal = terrainNormal(terrain.position.xz, max(0.004, terrain.distance * 0.0015));
    let sunZenithCos = dot(normal, p.sunDirection);
    let surfaceAltitude = p.groundRadius + terrain.height;
    let sunTransmittance = sampleTransmittance(p, transmittanceLut, lutSampler, surfaceAltitude, p.sunDirection.y);
    let shadow = terrainShadow(p, terrain.position, p.sunDirection) * cloudShadow(weatherMap, noiseSampler, clouds, terrain.position, terrain.height, p.sunDirection);
    let albedo = terrainAlbedo(terrain.height, normal, terrain.position.xz);
    let ambientOcclusion = 0.6 + 0.4 * normal.y;
    let lit = albedo * (p.sunIlluminance * sunTransmittance * max(sunZenithCos, 0.0) * shadow / PI + skyAmbient * ambientOcclusion);
    let aerial = sampleAerial(uv, terrain.distance);
    color = lit * (1.0 - aerial.a) + aerial.rgb;
  } else if (tSphere >= 0.0) {
    hitDistance = tSphere;
    let position = origin + tSphere * dir;
    let normal = normalize(position);
    let sunZenithCos = dot(normal, p.sunDirection);
    let sunTransmittance = sampleTransmittance(p, transmittanceLut, lutSampler, p.groundRadius, sunZenithCos);
    let albedo = terrainAlbedo(0.0, vec3f(0.0, 1.0, 0.0), position.xz);
    let shadow = cloudShadow(weatherMap, noiseSampler, clouds, position, 0.0, p.sunDirection);
    let ground = albedo * (p.sunIlluminance * sunTransmittance * max(sunZenithCos, 0.0) * shadow / PI + skyAmbient);
    if (tSphere < AERIAL_KM_PER_SLICE * AERIAL_LUT_SIZE) {
      let aerial = sampleAerial(uv, tSphere);
      color = ground * (1.0 - aerial.a) + aerial.rgb;
    } else {
      color = ground * sky.a + sky.rgb;
    }
  } else {
    let viewTransmittance = sampleTransmittance(p, transmittanceLut, lutSampler, viewHeight, dot(dir, origin / viewHeight));
    color += (sunDisc(p, dir) + sunGlare(p, dir)) * viewTransmittance;
  }
  return vec4f(color, hitDistance);
}
