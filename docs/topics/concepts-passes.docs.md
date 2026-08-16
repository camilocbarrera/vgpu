---
title: Passes
summary: A pass composites any number of draws into one target; a single shader can draw directly.
relatedSymbols:
  - Frame
  - FramePass
  - Effect
  - Draw
prevNext:
  prev:
    title: Effects
    href: /concepts/effects
  next:
    title: Frames
    href: /concepts/frames
order: 50
---

# Passes

A pass is a render-pass section inside a frame. It has one target, one clear color, and any number of draw calls. Open a pass by hand when you want to composite multiple draws into the same render target — here, an ocean and a boat rendered straight to the canvas:

```ts
import { init, effect, frame, prepare, surface } from "vgpu";

const gpu = await init();
const canvas = document.querySelector("canvas")!;
const canvasSurface = surface(gpu, canvas);

const oceanSource = `
  @fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    let wave = sin(uv.x * 24.0) * 0.01;
    let depth = smoothstep(0.4 + wave, 1.0, uv.y);
    return vec4f(0.1, 0.3 + depth * 0.2, 0.55 + depth * 0.3, 1.0);
  }
`;

// Draws only the hull pixels; discards everything else.
const boatSource = `
  @fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    let inHull = abs(uv.x - 0.5) < 0.12 && abs(uv.y - 0.42) < 0.05;
    if (!inHull) { discard; }
    return vec4f(0.45, 0.26, 0.13, 1.0);
  }
`;

// ---cut---
const ocean = effect(gpu, oceanSource);
const boat = effect(gpu, boatSource);

await prepare(gpu, [
  { draw: ocean, target: canvasSurface },
  { draw: boat, target: canvasSurface },
]);

frame(gpu, (currentFrame) => {
  currentFrame.pass({ target: canvasSurface, clear: [0, 0, 0, 1] }, (pass) => {
    pass.draw(ocean); // fill the canvas with water
    pass.draw(boat); // paint the boat on top — same target
  });
});
```

Both draws share one render pass and one target. Order inside the pass is paint order: the ocean fills the canvas first, then the boat draws on top of it.

> Good to know: [`FramePass.draw()`](/reference/vgpu/frame#framepass) accepts a fullscreen [`Effect`](/reference/vgpu/effect#effect) or an explicit [`Draw`](/reference/vgpu/draw#draw). Use `draw(gpu)` when you need meshes, vertex counts, instancing, or raw bind groups.

## One shader? Draw it directly

Now add postprocessing. The pass is the same — the only change is its target: an offscreen [`Target`](/reference/vgpu/target#target) with the same size as the canvas. The postprocessing effect declares that target as an external binding (`bindings: { src: scene }`), and a second pass in the **same** frame composites it to the screen:

```ts
import { init, effect, frame, prepare, sampler, surface, target } from "vgpu";

const gpu = await init();
const canvas = document.querySelector("canvas")!;
const canvasSurface = surface(gpu, canvas);

const ocean = effect(gpu, `
  @fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    let wave = sin(uv.x * 24.0) * 0.01;
    let depth = smoothstep(0.4 + wave, 1.0, uv.y);
    return vec4f(0.1, 0.3 + depth * 0.2, 0.55 + depth * 0.3, 1.0);
  }
`);
const boat = effect(gpu, `
  @fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    let inHull = abs(uv.x - 0.5) < 0.12 && abs(uv.y - 0.42) < 0.05;
    if (!inHull) { discard; }
    return vec4f(0.45, 0.26, 0.13, 1.0);
  }
`);
const postSource = `
  @group(0) @binding(0) var src: texture_2d<f32>;
  @group(0) @binding(1) var samp: sampler;

  @fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    let base = textureSampleLevel(src, samp, uv, 0.0);
    let vignette = 1.0 - 0.4 * length(uv - vec2f(0.5));
    return vec4f(base.rgb * vignette, 1.0);
  }
`;

// ---cut---
const scene = target(gpu, { size: [canvasSurface.size[0], canvasSurface.size[1]] });
const postprocessing = effect(gpu, {
  shader: postSource,
  bindings: {
    src: scene,   // bind the Target: a resize re-binds the new texture identity for you
    samp: sampler(gpu, { minFilter: 'linear', magFilter: 'linear' }),
  },
});

await prepare(gpu, [
  { draw: ocean, target: scene },
  { draw: boat, target: scene },
  { draw: postprocessing, target: canvasSurface },
]);

frame(gpu, (currentFrame) => {
  currentFrame.pass({ target: scene, clear: [0, 0, 0, 1] }, (pass) => {
    pass.draw(ocean);
    pass.draw(boat);
  });
  currentFrame.pass(canvasSurface, (pass) => pass.draw(postprocessing));
});
```

Two passes, one command encoder, one submit: the offscreen pass is encoded before the presentation
pass, so the scene is already rendered when postprocessing samples it — program order **is** execution
order. If you need the same work standalone, outside any frame,
`await renderOnce(gpu, canvasSurface, (p) => p.draw(postprocessing))` owns its encoder and submits
exactly once.

> Good to know: `frame.pass()` always needs a target. Use a canvas-backed [`Surface`](/reference/vgpu/surface#surface) from `surface(gpu, canvas)` or an offscreen [`Target`](/reference/vgpu/target#target) from `target(gpu, { size })`.

> Good to know: a pass takes more than a target and a clear color. [`FramePassOptions`](/reference/vgpu/frame#framepassoptions) also sets `clearDepth` (`0` for reversed-Z), `clearStencil`, a `viewport` or `scissor` rectangle for split-screen and partial redraws, `depthReadOnly` to depth-test while sampling the depth texture, a `timer` span for GPU timing, and `visibility` for occlusion queries.
