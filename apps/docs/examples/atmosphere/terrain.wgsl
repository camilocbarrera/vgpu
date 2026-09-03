// Procedural ridged heightfield in kilometres: a flat valley around the camera, mountains from ~6 km out.

export const TERRAIN_MAX_DISTANCE: f32 = 90.0;
export const TERRAIN_MAX_HEIGHT: f32 = 3.2;
const TERRAIN_SCALE: f32 = 1.0 / 16.0;
const OCTAVES: i32 = 6;
const ROTATE = mat2x2f(vec2f(0.8, 0.6), vec2f(-0.6, 0.8));

fn hash2(p: vec2f) -> f32 {
  let h = dot(p, vec2f(127.1, 311.7));
  return fract(sin(h) * 43758.5453123);
}

fn valueNoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = hash2(i);
  let b = hash2(i + vec2f(1.0, 0.0));
  let c = hash2(i + vec2f(0.0, 1.0));
  let d = hash2(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 2.0 - 1.0;
}

/** Height above sea level (km) at a tangent-plane position (km). */
export fn terrainHeight(xz: vec2f) -> f32 {
  var p = xz * TERRAIN_SCALE;
  var amplitude = 1.0;
  var height = 0.0;
  var weight = 1.0;
  for (var i = 0; i < OCTAVES; i += 1) {
    let n = 1.0 - abs(valueNoise(p));
    height += n * n * amplitude * weight;
    weight = clamp(n * 1.2, 0.0, 1.0);
    p = ROTATE * p * 2.05 + vec2f(1.7, 9.2);
    amplitude *= 0.5;
  }
  let distance = length(xz);
  // Flat ground under the camera, foothills from ~1 km, full mountains beyond ~6 km, fading out before the march limit.
  let valley = smoothstep(1.0, 6.0, distance);
  let horizonFade = 1.0 - smoothstep(TERRAIN_MAX_DISTANCE * 0.7, TERRAIN_MAX_DISTANCE, distance);
  let rolling = 0.02 * (valueNoise(xz * 0.9) + 0.5 * valueNoise(xz * 2.3 + 4.0)) * smoothstep(0.1, 1.0, distance);
  let mountains = max(0.0, height - 0.55) * 2.6;
  return max(0.0, (mountains * valley + rolling) * horizonFade);
}

export fn terrainNormal(xz: vec2f, epsilon: f32) -> vec3f {
  let dx = terrainHeight(xz + vec2f(epsilon, 0.0)) - terrainHeight(xz - vec2f(epsilon, 0.0));
  let dz = terrainHeight(xz + vec2f(0.0, epsilon)) - terrainHeight(xz - vec2f(0.0, epsilon));
  return normalize(vec3f(-dx, 2.0 * epsilon, -dz));
}

/** Albedo by altitude and slope: grass in the plains, rock on steep faces, snow near the peaks. */
export fn terrainAlbedo(height: f32, normal: vec3f, xz: vec2f) -> vec3f {
  let grass = vec3f(0.11, 0.13, 0.05);
  let dry = vec3f(0.22, 0.17, 0.10);
  let rock = vec3f(0.23, 0.21, 0.19);
  let snow = vec3f(0.78, 0.80, 0.84);
  let variation = 0.5 + 0.5 * valueNoise(xz * 0.35);
  var albedo = mix(grass, dry, variation);
  let slope = 1.0 - normal.y;
  albedo = mix(albedo, rock, smoothstep(0.08, 0.35, slope));
  let snowLine = 1.9 + 0.35 * valueNoise(xz * 0.2);
  let snowAmount = smoothstep(snowLine, snowLine + 0.5, height) * (1.0 - smoothstep(0.35, 0.6, slope));
  return mix(albedo, snow, snowAmount);
}
