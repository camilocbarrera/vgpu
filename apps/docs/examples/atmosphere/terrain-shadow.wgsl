import { Atmosphere } from "./atmosphere-common.wgsl";
import { TERRAIN_MAP_EXTENT, TERRAIN_MAX_HEIGHT, TERRAIN_SHADOW_MAP_SIZE, sampleTerrainHeight } from "./terrain.wgsl";

@group(0) @binding(0) var<uniform> atmosphere: Atmosphere;
@group(0) @binding(1) var terrainMap: texture_2d<f32>;
@group(0) @binding(2) var lutSampler: sampler;
@group(0) @binding(3) var terrainShadowMap: texture_storage_2d<rgba16float, write>;

/** Sun-ray sample spacing (km): three heightmap texels, so a ridge a few hundred metres wide still registers. */
const STEP: f32 = 0.3;
const MAX_STEPS: i32 = 512;

/**
 * Terrain shadow as a height field: for each map texel, the altitude below which a point above it cannot see the sun.
 * A sun ray from altitude a over the texel clears the terrain when a exceeds the largest (terrain height - ray
 * altitude) along the way, so that maximum is the shadow height; the air is lit above it and dark below. Only the
 * sun direction changes it (the heightmap is static), so the renderer rebuilds it when the sun moves.
 */
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let p = atmosphere;
  let uv = (vec2f(id.xy) + 0.5) / TERRAIN_SHADOW_MAP_SIZE;
  let xz = (uv - 0.5) * TERRAIN_MAP_EXTENT;
  // Start on the sphere under this texel; xz doubles as the tangent-plane terrain coordinate.
  let origin = vec3f(xz.x, sqrt(max(p.groundRadius * p.groundRadius - dot(xz, xz), 0.0)), xz.y);
  let dir = p.sunDirection;
  let halfExtent = 0.5 * TERRAIN_MAP_EXTENT;
  // The surface itself is the first occluder: the shadow height is never below the terrain.
  var shadowHeight = sampleTerrainHeight(terrainMap, lutSampler, xz);
  for (var i = 1; i <= MAX_STEPS; i += 1) {
    let position = origin + dir * (f32(i) * STEP);
    // Beyond the map the terrain has faded to sea level, and the planet shadow is tested separately.
    if (abs(position.x) > halfExtent || abs(position.z) > halfExtent) { break; }
    let rayAltitude = length(position) - p.groundRadius;
    // Altitude along a straight line is convex, so once it is above the highest peak it stays there.
    if (rayAltitude > TERRAIN_MAX_HEIGHT) { break; }
    shadowHeight = max(shadowHeight, sampleTerrainHeight(terrainMap, lutSampler, position.xz) - rayAltitude);
  }
  textureStore(terrainShadowMap, id.xy, vec4f(shadowHeight, 0.0, 0.0, 1.0));
}
