struct Present { exposure: f32, tonemap: f32, dither: f32, pad: f32 };

@group(0) @binding(0) var<uniform> present: Present;
@group(0) @binding(1) var sceneHdr: texture_2d<f32>;
@group(0) @binding(2) var linearSampler: sampler;

// AgX (Wende's minimal fit, sRGB working space).
const AGX_INSET = mat3x3f(
  vec3f(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
  vec3f(0.0784335999999992, 0.878468636469772, 0.0784336),
  vec3f(0.0792237451477643, 0.0791661274605434, 0.879142973793104),
);
const AGX_OUTSET = mat3x3f(
  vec3f(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
  vec3f(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
  vec3f(-0.0990297440797205, -0.0989611768448433, 1.15107367264116),
);
const AGX_MIN_EV: f32 = -12.47393;
const AGX_MAX_EV: f32 = 4.026069;

fn agxContrast(x: vec3f) -> vec3f {
  let x2 = x * x;
  let x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

fn tonemapAgx(color: vec3f) -> vec3f {
  var v = AGX_INSET * max(color, vec3f(1e-10));
  v = clamp(log2(v), vec3f(AGX_MIN_EV), vec3f(AGX_MAX_EV));
  v = (v - AGX_MIN_EV) / (AGX_MAX_EV - AGX_MIN_EV);
  v = agxContrast(v);
  v = AGX_OUTSET * v;
  return pow(max(v, vec3f(0.0)), vec3f(2.2));
}

fn tonemapAces(color: vec3f) -> vec3f {
  let x = color;
  return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

// Khronos PBR Neutral: keeps hue until highlights, then desaturates toward white.
fn tonemapNeutral(color: vec3f) -> vec3f {
  let startCompression = 0.8 - 0.04;
  let desaturation = 0.15;
  var c = color;
  let x = min(c.r, min(c.g, c.b));
  let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
  c -= offset;
  let peak = max(c.r, max(c.g, c.b));
  if (peak < startCompression) { return c; }
  let d = 1.0 - startCompression;
  let newPeak = 1.0 - d * d / (peak + d - startCompression);
  c *= newPeak / peak;
  let g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
  return mix(c, vec3f(newPeak), g);
}

fn linearToSrgb(value: vec3f) -> vec3f {
  let lo = value * 12.92;
  let hi = 1.055 * pow(value, vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, value <= vec3f(0.0031308));
}

fn hash(position: vec2f) -> f32 {
  let h = dot(position, vec2f(127.1, 311.7));
  return fract(sin(h) * 43758.5453123);
}

@fragment fn fs_main(@builtin(position) fragCoord: vec4f, @location(0) uv: vec2f) -> @location(0) vec4f {
  let hdr = textureSample(sceneHdr, linearSampler, uv).rgb * present.exposure;
  var color = hdr;
  if (present.tonemap < 0.5) { color = tonemapAgx(hdr); }
  else if (present.tonemap < 1.5) { color = tonemapAces(hdr); }
  else if (present.tonemap < 2.5) { color = tonemapNeutral(hdr); }
  color = linearToSrgb(saturate(color));
  color += (hash(fragCoord.xy) - 0.5) / 255.0 * present.dither;
  return vec4f(saturate(color), 1.0);
}
