import { Atmosphere, Camera } from "./atmosphere-common.wgsl";
import { TERRAIN_MESH_COLUMNS, TERRAIN_MESH_RINGS, TERRAIN_NEAR, sampleTerrainHeight } from "./terrain.wgsl";

@group(0) @binding(0) var<uniform> atmosphere: Atmosphere;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<uniform> mesh: TerrainMesh;
@group(0) @binding(3) var terrainMap: texture_2d<f32>;
@group(0) @binding(4) var lutSampler: sampler;

/** The columns of the ring grid this frame draws (terrainSector in camera.ts). */
struct TerrainMesh { columnOffset: u32, columns: u32 };

/**
 * Depth prepass of the terrain. The surface is a static grid of rings around the camera's ground point (the camera
 * never leaves the axis): one triangle strip per azimuth column, `TERRAIN_MESH_RINGS + 1` ring vertices on each side,
 * no vertex buffers. Heights come from the baked heightmap in the vertex shader, so this is the same surface the
 * raymarch used to find, and scene.wgsl shades it once per pixel from the depth it leaves behind.
 * Reversed-Z with an infinite far plane: depth = TERRAIN_NEAR / view depth keeps precision from 1 m out to 400 km.
 */
const TAU: f32 = 6.28318530717959;
/** Ring layout: fine geometric steps over the flat valley, 265 m steps across the mountains (heightmap texels are 98 m), coarse steps over the bare sphere. */
const NEAR_RINGS: u32 = 128u;
const MID_RINGS: u32 = 320u;
const NEAR_RADIUS: f32 = 0.005;
const MID_RADIUS: f32 = 5.0;
const FAR_RADIUS: f32 = 90.0;
const LAST_RADIUS: f32 = 400.0;

fn ringRadius(ring: u32) -> f32 {
  if (ring == 0u) { return 0.0; }
  if (ring <= NEAR_RINGS) { return NEAR_RADIUS * pow(MID_RADIUS / NEAR_RADIUS, f32(ring - 1u) / f32(NEAR_RINGS - 1u)); }
  if (ring <= NEAR_RINGS + MID_RINGS) { return MID_RADIUS + (FAR_RADIUS - MID_RADIUS) * f32(ring - NEAR_RINGS) / f32(MID_RINGS); }
  return FAR_RADIUS * pow(LAST_RADIUS / FAR_RADIUS, f32(ring - NEAR_RINGS - MID_RINGS) / f32(TERRAIN_MESH_RINGS - NEAR_RINGS - MID_RINGS));
}

@vertex fn vs_main(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> @builtin(position) vec4f {
  let p = atmosphere;
  let ring = vertexIndex / 2u;
  let column = (mesh.columnOffset + instanceIndex + (vertexIndex & 1u)) % TERRAIN_MESH_COLUMNS;
  // Azimuth 0 is +Z, like the camera yaw and the sun azimuth.
  let theta = f32(column) * (TAU / f32(TERRAIN_MESH_COLUMNS));
  let radius = ringRadius(ring);
  let xz = vec2f(sin(theta), cos(theta)) * radius;
  let height = sampleTerrainHeight(terrainMap, lutSampler, xz);
  // The surface sits at altitude `height` over the sphere: y = sqrt((R + h)^2 - r^2). Its offset from the camera
  // (on the axis at altitude a) is formed without subtracting two 6360 km numbers, which f32 could not afford.
  let groundRadius = p.groundRadius;
  let cameraAltitude = camera.position.y - groundRadius;
  let surfaceRadius = groundRadius + height;
  let rr = dot(xz, xz);
  let y = sqrt(max(surfaceRadius * surfaceRadius - rr, 0.0));
  let relativeY = ((height - cameraAltitude) * (2.0 * groundRadius + height + cameraAltitude) - rr) / (y + groundRadius + cameraAltitude);
  let relative = vec3f(xz.x, relativeY, xz.y);
  let view = vec3f(dot(relative, camera.right), dot(relative, camera.up), dot(relative, camera.forward));
  return vec4f(view.x / (camera.tanHalfFov * camera.aspect), view.y / camera.tanHalfFov, TERRAIN_NEAR, view.z);
}

/** Depth only: the draw masks every color channel, scene.wgsl does the shading. */
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(0.0); }
