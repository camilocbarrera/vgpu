/**
 * Low-level triangle path. The descriptor is intentionally inline: this fixture must prove that
 * `geometry(gpu, descriptor)` does not retain the scene recipe factory or its primitive meshes.
 * It uses the canonical gpu-first names that T202-05 publishes.
 */
import { draw, effect, frame, geometry, init, surface } from "vgpu";

const TRIANGLE = `
@vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let points = array<vec2f, 3>(vec2f(0.0, 0.5), vec2f(-0.5, -0.5), vec2f(0.5, -0.5));
  return vec4f(points[index], 0.0, 1.0);
}
@fragment fn fragment() -> @location(0) vec4f { return vec4f(1.0); }`;

export async function renderTriangle(canvas: HTMLCanvasElement) {
  const gpu = await init();
  const target = surface(gpu, canvas);
  const mesh = geometry(gpu, { topology: "triangle-list", vertexCount: 3 });
  const triangle = draw(gpu, { geometry: mesh, effect: effect(gpu, TRIANGLE) });
  frame(gpu, (next) => next.pass({ target }, (pass) => pass.draw(triangle)));
  return gpu;
}
