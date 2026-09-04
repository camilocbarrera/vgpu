import { AERIAL_KM_PER_SLICE, AERIAL_LUT_SIZE, AERIAL_MAX_DISTANCE, Atmosphere, Camera, FrameConstants, PI, TERRAIN_TRANSMITTANCE_ENTRIES, cameraRay, raySphere, sampleTransmittance, skyViewUvFast } from "./atmosphere-common.wgsl";
import { TERRAIN_MAX_HEIGHT, TERRAIN_NEAR, sampleTerrainAlbedoNoise, sampleTerrainHeight, sampleTerrainNormal, terrainAlbedo } from "./terrain.wgsl";
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
@group(0) @binding(9) var terrainMap: texture_2d<f32>;
@group(0) @binding(10) var<storage, read> frame: FrameConstants;
@group(0) @binding(11) var terrainAlbedoMap: texture_2d<f32>;
@group(0) @binding(12) var aerialLossLut: texture_3d<f32>;
@group(0) @binding(13) var terrainDepth: texture_depth_2d;

fn height(xz: vec2f) -> f32 { return sampleTerrainHeight(terrainMap, lutSampler, xz); }

/** Sun transmittance at a terrain height from the per-frame table (linear between entries). */
fn terrainSunTransmittance(surfaceHeight: f32) -> vec3f {
  let x = saturate(surfaceHeight / TERRAIN_MAX_HEIGHT) * f32(TERRAIN_TRANSMITTANCE_ENTRIES - 1u);
  let index = u32(floor(x));
  let next = min(index + 1u, TERRAIN_TRANSMITTANCE_ENTRIES - 1u);
  return mix(frame.terrainSunTransmittance[index].rgb, frame.terrainSunTransmittance[next].rgb, fract(x));
}

const SHADOW_STEPS: i32 = 12;

struct TerrainHit { distance: f32, position: vec3f, height: f32 };

fn sampleSkyView(dir: vec3f, viewHeight: f32, intersectGround: bool) -> vec4f {
  let up = camera.position / viewHeight;
  let viewZenithCos = dot(dir, up);
  let dirHorizontal = dir - up * viewZenithCos;
  let dirLength = length(dirHorizontal);
  var lightViewCos = 1.0;
  if (frame.sunHorizontalLength > 1e-5 && dirLength > 1e-5) { lightViewCos = dot(frame.sunHorizontal / frame.sunHorizontalLength, dirHorizontal / dirLength); }
  return textureSampleLevel(skyViewLut, lutSampler, skyViewUvFast(frame, viewZenithCos, lightViewCos, intersectGround), 0.0);
}

struct AerialCoordinate { w: f32, weight: f32 };

/** Depth coordinate of the froxel volume; the first half slice fades in so the volume starts at zero. */
fn aerialCoordinate(distance: f32) -> AerialCoordinate {
  var slice = distance / AERIAL_KM_PER_SLICE;
  var weight = 1.0;
  if (slice < 0.5) { weight = saturate(slice * 2.0); slice = 0.5; }
  return AerialCoordinate(sqrt(slice / AERIAL_LUT_SIZE), weight);
}

fn sampleAerial(uv: vec2f, distance: f32) -> vec4f {
  let c = aerialCoordinate(distance);
  return c.weight * textureSampleLevel(aerialLut, lutSampler, vec3f(uv, c.w), 0.0);
}

/** Single scattering the aerial pass left out because the air was in terrain shadow (aerial-lut.wgsl). */
fn sampleAerialLoss(uv: vec2f, distance: f32) -> vec3f {
  let c = aerialCoordinate(distance);
  return c.weight * textureSampleLevel(aerialLossLut, lutSampler, vec3f(uv, c.w), 0.0).rgb;
}

/** Sun disc with wavelength-dependent limb darkening, softened over one pixel. */
fn sunDisc(p: Atmosphere, dir: vec3f) -> vec3f {
  let cosAngle = dot(dir, p.sunDirection);
  let radius = camera.sunAngularRadius;
  let edge = frame.sunSinRadius * camera.pixelAngle;
  let disc = smoothstep(frame.sunCosRadius - edge, frame.sunCosRadius + edge, cosAngle);
  if (disc <= 0.0) { return vec3f(0.0); }
  let angle = acos(clamp(cosAngle, -1.0, 1.0));
  let mu = sqrt(saturate(1.0 - (angle * angle) / (radius * radius)));
  let limb = 1.0 - vec3f(0.397, 0.503, 0.652) * (1.0 - mu);
  return p.sunIlluminance / frame.sunSolidAngle * limb * disc;
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

/**
 * The terrain surface under a pixel, from the depth the ring grid left in the prepass (terrain-depth.wgsl):
 * reversed-Z with depth = TERRAIN_NEAR / view depth, so the point lies on the pixel's ray at that view depth.
 */
fn terrainHit(p: Atmosphere, origin: vec3f, dir: vec3f, pixel: vec2i) -> TerrainHit {
  let depth = textureLoad(terrainDepth, pixel, 0);
  if (depth <= 0.0) { return TerrainHit(-1.0, vec3f(0.0), 0.0); }
  let distance = TERRAIN_NEAR / (depth * dot(dir, camera.forward));
  let position = origin + dir * distance;
  return TerrainHit(distance, position, altitudeOf(p, position));
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
    let delta = altitude - height(sample.xz);
    shadow = min(shadow, saturate(8.0 * delta / t));
    if (shadow <= 0.0) { break; }
    t += 0.04 + t * 0.6;
  }
  return shadow;
}

@fragment fn fs_main(@builtin(position) fragCoord: vec4f, @location(0) uv: vec2f) -> @location(0) vec4f {
  let p = atmosphere;
  let dir = cameraRay(camera, vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0));
  let origin = camera.position;
  let viewHeight = length(origin);
  let tSphere = raySphere(origin, dir, p.groundRadius);
  let terrain = terrainHit(p, origin, dir, vec2i(fragCoord.xy));
  let hitsGround = tSphere >= 0.0 || terrain.distance >= 0.0;
  let sky = sampleSkyView(dir, viewHeight, hitsGround);
  // The sky-view LUT knows nothing about terrain: take out the single scattering the aerial pass found in terrain
  // shadow along this ray, so the haze in front of a ridge that hides the sun stops glowing on sky pixels too.
  let skyLuminance = max(sky.rgb - sampleAerialLoss(uv, AERIAL_MAX_DISTANCE), vec3f(0.0));
  let skyAmbient = frame.skyAmbient;
  var color = skyLuminance;
  // Alpha carries the geometry distance (km) so the cloud pass can stop at terrain; -1 means sky.
  var hitDistance = -1.0;

  if (terrain.distance >= 0.0) {
    hitDistance = terrain.distance;
    let normal = sampleTerrainNormal(terrainMap, lutSampler, terrain.position.xz);
    let sunZenithCos = dot(normal, p.sunDirection);
    let sunTransmittance = terrainSunTransmittance(terrain.height);
    let shadow = terrainShadow(p, terrain.position, p.sunDirection) * cloudShadow(weatherMap, noiseSampler, clouds, terrain.position, terrain.height, p.sunDirection);
    let albedo = terrainAlbedo(terrain.height, normal, sampleTerrainAlbedoNoise(terrainAlbedoMap, lutSampler, terrain.position.xz));
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
    let albedo = terrainAlbedo(0.0, vec3f(0.0, 1.0, 0.0), sampleTerrainAlbedoNoise(terrainAlbedoMap, lutSampler, position.xz));
    let shadow = cloudShadow(weatherMap, noiseSampler, clouds, position, 0.0, p.sunDirection);
    let ground = albedo * (p.sunIlluminance * sunTransmittance * max(sunZenithCos, 0.0) * shadow / PI + skyAmbient);
    if (tSphere < AERIAL_MAX_DISTANCE) {
      let aerial = sampleAerial(uv, tSphere);
      color = ground * (1.0 - aerial.a) + aerial.rgb;
    } else {
      color = ground * sky.a + skyLuminance;
    }
  } else {
    let viewTransmittance = sampleTransmittance(p, transmittanceLut, lutSampler, viewHeight, dot(dir, origin / viewHeight));
    // The disc and its analytic glare are seen from the camera, so they hide behind local terrain as one.
    color += (sunDisc(p, dir) + sunGlare(p, dir)) * viewTransmittance * frame.sunTerrainVisibility;
  }
  return vec4f(color, hitDistance);
}
