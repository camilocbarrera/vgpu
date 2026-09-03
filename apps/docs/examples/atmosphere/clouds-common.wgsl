import { remap } from "./noise-common.wgsl";

/** Cloud layer description shared by the cloud raymarch and the terrain cloud shadow. Distances in km. */
export struct Clouds {
  bottom: f32, top: f32, coverage: f32, density: f32,
  shapeScale: f32, detailScale: f32, weatherScale: f32, wind: f32,
  detailStrength: f32, groundRadius: f32, pad0: f32, pad1: f32,
};

export fn heightFraction(c: Clouds, altitude: f32) -> f32 {
  return saturate((altitude - c.bottom) / (c.top - c.bottom));
}

export fn sampleWeather(weather: texture_2d<f32>, weatherSampler: sampler, c: Clouds, xz: vec2f) -> vec4f {
  return textureSampleLevel(weather, weatherSampler, xz / c.weatherScale + vec2f(0.31 + c.wind * 0.004, 0.62 + c.wind * 0.001), 0.0);
}

/** Cumulus grows tall, stratus stays flat; both vanish at the layer bounds. */
fn heightGradient(hf: f32, cloudType: f32) -> f32 {
  let stratus = smoothstep(0.0, 0.08, hf) * (1.0 - smoothstep(0.2, 0.4, hf));
  let cumulus = smoothstep(0.0, 0.1, hf) * (1.0 - smoothstep(0.6, 1.0, hf));
  return mix(stratus, cumulus, cloudType);
}

/**
 * Cloud density in [0, density]. `position` is planet-centric; xz doubles as the tangent-plane coordinate.
 * `cheap` skips the erosion detail (used by light marches).
 */
export fn cloudDensity(
  c: Clouds, shape: texture_3d<f32>, detail: texture_3d<f32>, weather: texture_2d<f32>, noiseSampler: sampler,
  position: vec3f, altitude: f32, cheap: bool,
) -> f32 {
  let rawHf = heightFraction(c, altitude);
  if (rawHf <= 0.0 || rawHf >= 1.0) { return 0.0; }
  let w = sampleWeather(weather, noiseSampler, c, position.xz);
  let coverage = saturate(remap(w.r, 0.3, 0.75, 0.0, 1.0) * c.coverage * 1.2);
  if (coverage <= 0.0) { return 0.0; }
  // Tops vary with the weather so the deck is not one flat slab; keep this gentle, the top gradient
  // amplifies any interpolation creases of the weather texture into vertical streaks.
  let hf = rawHf / mix(0.7, 1.0, w.b);
  if (hf >= 1.0) { return 0.0; }
  let gradient = heightGradient(hf, w.g);
  if (gradient <= 0.0) { return 0.0; }
  // The constant offset just picks a pleasant patch of the tiled noise around the origin.
  let unwarped = vec3f(position.x, altitude, position.z) + vec3f(53.0 + c.wind * 0.02, 0.0, 29.0);
  // Gentle low-frequency domain warp hides the lattice of the tiled noise; keep the amplitude well
  // below the warp texel size or the piecewise-linear filtering tears the shape into creases.
  let warp = textureSampleLevel(shape, noiseSampler, unwarped / (c.shapeScale * 2.7) + vec3f(0.5, 0.2, 0.8), 0.0).gba - 0.5;
  let p = unwarped + warp * c.shapeScale * 0.22;
  let s = textureSampleLevel(shape, noiseSampler, p / c.shapeScale, 0.0);
  let lowFbm = s.g * 0.625 + s.b * 0.25 + s.a * 0.125;
  var base = saturate(remap(s.r, lowFbm - 1.0, 1.0, 0.0, 1.0)) * gradient;
  base = saturate(remap(base, 1.0 - coverage, 1.0, 0.0, 1.0));
  // A small floor keeps the air between clouds clear instead of a faint fog.
  base = saturate((base - 0.06) / 0.94);
  if (base <= 0.0 || cheap) { return base * c.density; }
  let d = textureSampleLevel(detail, noiseSampler, p / c.detailScale, 0.0).rgb;
  let detailFbm = d.r * 0.625 + d.g * 0.25 + d.b * 0.125;
  let modifier = mix(detailFbm, 1.0 - detailFbm, saturate(hf * 8.0)) * 0.35 * c.detailStrength;
  return saturate(remap(base, modifier, 1.0, 0.0, 1.0)) * c.density;
}

/** 2D approximation of the cloud shadow on the ground: coverage where the sun ray crosses mid-layer. */
export fn cloudShadow(weather: texture_2d<f32>, weatherSampler: sampler, c: Clouds, position: vec3f, altitude: f32, sunDir: vec3f) -> f32 {
  if (sunDir.y <= 0.02) { return 1.0; }
  let mid = 0.5 * (c.bottom + c.top);
  let t = max(0.0, (mid - altitude) / sunDir.y);
  let w = sampleWeather(weather, weatherSampler, c, (position + sunDir * t).xz);
  let coverage = saturate(remap(w.r, 0.3, 0.75, 0.0, 1.0) * c.coverage * 1.2);
  return 1.0 - 0.85 * smoothstep(0.35, 0.75, coverage);
}
