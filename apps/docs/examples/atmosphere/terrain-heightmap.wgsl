import { TERRAIN_MAP_EXTENT, TERRAIN_MAP_SIZE, terrainHeight, terrainNormal } from "./terrain.wgsl";

@group(0) @binding(0) var terrainMap: texture_storage_2d<rgba16float, write>;

/** Bakes the procedural heightfield once: R = height (km), GBA = normal. Sampled by the terrain march instead of re-evaluating the fbm. */
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let uv = (vec2f(id.xy) + 0.5) / TERRAIN_MAP_SIZE;
  let xz = (uv - 0.5) * TERRAIN_MAP_EXTENT;
  let texel = TERRAIN_MAP_EXTENT / TERRAIN_MAP_SIZE;
  let normal = terrainNormal(xz, texel * 0.5);
  textureStore(terrainMap, id.xy, vec4f(terrainHeight(xz), normal));
}
