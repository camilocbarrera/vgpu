/**
 * Recipe bridge: turns a pure `SceneGeometry` recipe (`box()`, `sphere()`, ...) into GPU buffers.
 *
 * This is the only module that reaches the 15 mesh primitives, and nothing in the low-level path
 * imports it: `geometry(gpu, descriptor)` lives in `geometry-descriptor.ts` and stays free of the
 * primitive graph (~11 KB). The split is by ESM symbol, not by a runtime registry — importing
 * `geometryFromRecipe` is what pays for the generators, and a program that never mentions a recipe
 * never links them.
 */
import type { Device } from "@vgpu/core";
import {
  box as renderBox,
  capsule as renderCapsule,
  cone as renderCone,
  cylinder as renderCylinder,
  disk as renderDisk,
  dodecahedron as renderDodecahedron,
  fullscreenQuad as renderFullscreenQuad,
  icosahedron as renderIcosahedron,
  icosphere as renderIcosphere,
  octahedron as renderOctahedron,
  plane as renderPlane,
  ring as renderRing,
  sphere as renderSphere,
  tetrahedron as renderTetrahedron,
  torus as renderTorus,
  type MeshPrimitive,
  type VertexAttributes,
} from "./geometry-src/index.ts";
import type { SceneGeometry } from "./geometry.ts";
import { geometry as geometryFromDescriptor, ownGeometry, Geometry, type GeometryOptions } from "./geometry-descriptor.ts";
import type { Gpu } from "../kernel.ts";
import { liveKernel } from "../live-kernel.ts";

type PrimitiveFactory = (device: Device, geometry: SceneGeometry) => MeshPrimitive;

const primitiveFactories: { readonly [K in SceneGeometry["kind"]]: PrimitiveFactory } = {
  box: (device, geometry) => renderBox({ device, ...geometry.props }),
  capsule: (device, geometry) => renderCapsule({ device, radius: 0.5, height: 1, ...geometry.props }),
  cone: (device, geometry) => renderCone({ device, radius: 0.5, height: 1, ...geometry.props }),
  cylinder: (device, geometry) => renderCylinder({ device, radius: 0.5, height: 1, ...geometry.props }),
  disk: (device, geometry) => renderDisk({ device, radius: 0.5, ...geometry.props }),
  dodecahedron: (device, geometry) => renderDodecahedron({ device, radius: 0.5, ...geometry.props }),
  fullscreenQuad: (device, geometry) => renderFullscreenQuad({ device, ...geometry.props }),
  icosahedron: (device, geometry) => renderIcosahedron({ device, radius: 0.5, ...geometry.props }),
  icosphere: (device, geometry) => renderIcosphere({ device, radius: 0.5, ...geometry.props }),
  octahedron: (device, geometry) => renderOctahedron({ device, radius: 0.5, ...geometry.props }),
  plane: (device, geometry) => renderPlane({ device, ...geometry.props }),
  ring: (device, geometry) => renderRing({ device, innerRadius: 0.25, outerRadius: 0.5, ...geometry.props }),
  sphere: (device, geometry) => renderSphere({ device, ...geometry.props }),
  tetrahedron: (device, geometry) => renderTetrahedron({ device, radius: 0.5, ...geometry.props }),
  torus: (device, geometry) => renderTorus({ device, radius: 0.5, tube: 0.2, ...geometry.props }),
};

/** @internal Converts a pure scene geometry recipe into the vertex/index buffer contract consumed by draw(). */
export function createGeometry(device: Device, geometry: SceneGeometry): Geometry {
  const primitive = primitiveFor(device, geometry);
  return new Geometry(device, primitiveGeometryOptions(primitive));
}

/**
 * Uploads a mesh recipe (`box()`, `icosphere()`, ...) and returns the geometry that owns its buffers.
 *
 * Also accepts a plain descriptor, so a caller that already imports this module can build both kinds
 * through one symbol; the descriptor case is forwarded to `geometry(gpu, descriptor)` unchanged. Use
 * `geometry()` directly when the program has no recipes: that path never links the primitives.
 *
 * The buffers are destroyed by `gpu.dispose()` (or earlier by `geometry.destroy()`); a scene view
 * that receives the recipe instead uploads and owns its own copy.
 */
export function geometryFromRecipe(gpu: Gpu, recipe: SceneGeometry): Geometry;
export function geometryFromRecipe(gpu: Gpu, descriptor: GeometryOptions): Geometry;
export function geometryFromRecipe(gpu: Gpu, input: SceneGeometry | GeometryOptions): Geometry {
  if (isGeometryOptions(input)) return geometryFromDescriptor(gpu, input);
  const kernel = liveKernel(gpu, "geometryFromRecipe");
  return ownGeometry(kernel, createGeometry(kernel.device, input));
}

/** Recipes are `{ kind, props }`; descriptors always declare their vertex `buffers`. */
function isGeometryOptions(value: SceneGeometry | GeometryOptions): value is GeometryOptions {
  return "buffers" in value;
}

function primitiveFor(device: Device, geometry: SceneGeometry): MeshPrimitive {
  return primitiveFactories[geometry.kind](device, geometry);
}

function primitiveGeometryOptions(primitive: MeshPrimitive): GeometryOptions {
  const attrs = primitive.attributes;
  const attributes: Record<string, GPUVertexFormat | { readonly format: GPUVertexFormat; readonly offset?: number; readonly location?: number }> = {
    position: { ...attrs.position, location: 0 },
  };
  if (attrs.normal) attributes.normal = { ...attrs.normal, location: 1 };
  if (attrs.uv) attributes.uv = { ...attrs.uv, location: 2 };
  return {
    buffers: [{
      buffer: primitive.gpu?.vertexBuffer ?? primitive.vertexBuffer.gpu,
      stride: attrs.stride,
      attributes,
    }],
    vertexCount: primitive.vertexCount,
    indexBuffer: primitive.gpu?.indexBuffer ?? primitive.indexBuffer?.gpu,
    indexFormat: primitive.indexFormat,
    indexCount: primitive.indexCount,
  };
}
