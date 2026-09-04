import { Atmosphere, Camera, cameraRay } from "./atmosphere-common.wgsl";
import { Clouds, cloudRange } from "./clouds-common.wgsl";
import { CloudOutput, Reprojection, compactCoordinate, compactSize, isLiveTexel, texelToCompact } from "./clouds-temporal.wgsl";

@group(0) @binding(0) var<uniform> atmosphere: Atmosphere;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<uniform> clouds: Clouds;
@group(0) @binding(3) var marchColor: texture_2d<f32>;
@group(0) @binding(4) var marchDepth: texture_2d<f32>;
@group(0) @binding(5) var history: texture_2d<f32>;
@group(0) @binding(6) var historyDepth: texture_2d<f32>;
@group(0) @binding(7) var lutSampler: sampler;
@group(0) @binding(8) var<uniform> reprojection: Reprojection;

// Builds this frame's cloud history: live texels take this frame's march from the compact target (blended into their
// reprojected history while accumulating), the others reproject last frame's history through the depth it stored.

/** Where the point at `depth` along this frame's ray fell on the previous frame's screen. */
fn reprojectedUv(dir: vec3f, depth: f32) -> vec2f {
  let relative = camera.position + dir * depth - reprojection.position;
  let z = dot(relative, reprojection.forward);
  if (z <= 1e-3) { return vec2f(-1.0); }
  let ndc = vec2f(dot(relative, reprojection.right) / (z * reprojection.tanHalfFov * reprojection.aspect), dot(relative, reprojection.up) / (z * reprojection.tanHalfFov));
  return vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

fn inside(uv: vec2f) -> bool { return all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0)); }

/**
 * Reprojection with parallax. The camera only moves along its axis, but a change of altitude still slides clouds a
 * few kilometres away by pixels per frame, which a rotation-only reprojection smears into ghosts. First guess the
 * depth from the middle of the cloud layer along this ray, then refine with the mean depth the history stored there.
 */
fn historyUv(dir: vec3f, origin: vec3f, viewHeight: f32) -> vec2f {
  let range = cloudRange(clouds, origin, dir, viewHeight);
  var depth = 1e4;
  if (range.valid && range.end > range.start) { depth = 0.5 * (range.start + range.end); }
  var uv = reprojectedUv(dir, depth);
  if (inside(uv)) {
    let storedDepth = textureSampleLevel(historyDepth, lutSampler, uv, 0.0).r;
    if (storedDepth > 0.0) { uv = reprojectedUv(dir, storedDepth); }
  }
  return uv;
}

/** This frame's march interpolated from the nearest live texels, for texels that have neither a march nor a history. */
fn nearestMarch(texel: vec2f, frameIndex: i32, period: i32) -> CloudOutput {
  let compact = vec2f(compactSize(vec2i(reprojection.size), period));
  let coordinate = clamp(compactCoordinate(texel, frameIndex, period), vec2f(0.0), compact - 1.0);
  let uv = (coordinate + 0.5) / vec2f(textureDimensions(marchColor));
  return CloudOutput(textureSampleLevel(marchColor, lutSampler, uv, 0.0), textureSampleLevel(marchDepth, lutSampler, uv, 0.0));
}

@fragment fn fs_main(@builtin(position) fragCoord: vec4f, @location(0) uv: vec2f) -> CloudOutput {
  let texel = vec2i(fragCoord.xy);
  let period = i32(reprojection.refreshPeriod);
  let frameIndex = i32(reprojection.frame);
  let dir = cameraRay(camera, vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0));
  let origin = camera.position;
  let previousUv = historyUv(dir, origin, length(origin));
  let historyValid = reprojection.valid > 0.5 && inside(previousUv);
  let live = isLiveTexel(texel, frameIndex, period);
  if (!live) {
    if (historyValid) {
      return CloudOutput(textureSampleLevel(history, lutSampler, previousUv, 0.0), textureSampleLevel(historyDepth, lutSampler, previousUv, 0.0));
    }
    // Newly visible (first frame, or a screen edge that just turned into view): filled from this frame's live
    // neighbours until its own turn comes.
    return nearestMarch(vec2f(fragCoord.xy), frameIndex, period);
  }
  let compact = texelToCompact(texel, period);
  let fresh = textureLoad(marchColor, compact, 0);
  let freshDepth = textureLoad(marchDepth, compact, 0).r;
  if (historyValid && reprojection.blend < 1.0) {
    let historyDepthValue = textureSampleLevel(historyDepth, lutSampler, previousUv, 0.0).r;
    // The depth is not blended: a fresh sample without cloud would pull the stored depth toward zero.
    let depth = select(historyDepthValue, freshDepth, freshDepth > 0.0);
    return CloudOutput(mix(textureSampleLevel(history, lutSampler, previousUv, 0.0), fresh, reprojection.blend), vec4f(depth, 0.0, 0.0, 0.0));
  }
  return CloudOutput(fresh, vec4f(freshDepth, 0.0, 0.0, 0.0));
}
