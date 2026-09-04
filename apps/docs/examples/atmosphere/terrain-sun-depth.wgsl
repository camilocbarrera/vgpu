import { Atmosphere, SunShadow } from "./atmosphere-common.wgsl";
import { terrainMeshVertex } from "./terrain.wgsl";

@group(0) @binding(0) var<uniform> atmosphere: Atmosphere;
@group(0) @binding(1) var<uniform> mesh: TerrainMesh;
@group(0) @binding(2) var terrainMap: texture_2d<f32>;
@group(0) @binding(3) var lutSampler: sampler;
@group(0) @binding(4) var<uniform> sunShadow: SunShadow;

struct TerrainMesh { columnOffset: u32, columns: u32 };

/**
 * The sun's shadow map: the whole ring grid rasterized from the sun with an orthographic projection (sunShadow.toShadow,
 * built in renderer.ts), depth only, nearest occluder wins. Terrain pixels and air samples then ask it whether they
 * see the sun with one depth comparison each, in place of a heightmap march. Rebuilt when the sun moves.
 */
@vertex fn vs_main(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> @builtin(position) vec4f {
  let fromGround = terrainMeshVertex(vertexIndex, instanceIndex, mesh.columnOffset, atmosphere.groundRadius, terrainMap, lutSampler);
  let shadow = sunShadow.toShadow * vec4f(fromGround, 1.0);
  return vec4f(shadow.xy, shadow.z, 1.0);
}

@fragment fn fs_main() -> @location(0) vec4f { return vec4f(0.0); }
