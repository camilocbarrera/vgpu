---
title: "External ticker"
description: "Drive the vgpu clock from GSAP, Motion, an XR frame callback or a fixed timestep with clock(gpu).advance(dt)."
---

# Driving vgpu with an external ticker — GSAP/Motion/XR

**`frame(gpu, cb)` per tick is the primitive; `frameLoop` is convenience sugar** for self-contained demos. Most real applications do not own the loop: a GSAP or Motion ticker, an XR session's frame callback, a host framework's render loop or a fixed-timestep physics loop already owns the timeline, and vgpu renders one explicit frame per tick inside it.

By default vgpu owns the clock: every `frame(gpu)` moves `clock(gpu).time` forward by the wall-clock delta since the last frame, and `frameLoop(gpu, cb)` schedules those frames on `requestAnimationFrame`. That is the right default for a page whose only animation is the render.

It stops being the right default the moment something else already owns the timeline: a GSAP or Motion ticker, an XR session's frame callback, a physics loop with a fixed timestep, or a test that must produce the same pixels twice. Two clocks running side by side drift, and drift shows up as animation that stutters against everything else on the page.

> **Why the strict compilation default matters most here.** A hidden synchronous pipeline compile inside a third-party tick is a dropped animation frame in a loop your code does not control. Under the default `pendingPipelines: "throw"` that stall is impossible: an unprepared combination fails fast instead of compiling, so every pipeline the ticker needs is warmed with `await prepare(gpu, [...])` **before** the ticker starts. Frame callbacks are synchronous for the same reason — there is no `await` you can slip into a tick.

The fix is one call. `clock(gpu).advance(dtSeconds)` moves the vgpu clock forward *now*, and claims that frame's tick: the next `frame(gpu)` counts the frame and runs its passes, but does not advance the clock again. One tick per frame, with the manual one winning.

```ts
import { init, clock, effect, frame, prepare, surface } from "vgpu";

declare const canvas: HTMLCanvasElement;
declare const gsap: { ticker: { add(cb: (time: number, deltaMs: number) => void): void } };

const gpu = await init();
const canvasSurface = surface(gpu, canvas);
const wave = effect(gpu, {
  shader: `
    struct Params { time: f32 }
    @group(0) @binding(0) var<uniform> params: Params;
    @fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
      return vec4f(uv, sin(params.time) * 0.5 + 0.5, 1.0);
    }
  `,
  values: { params: { time: 0 } },
});

// ---cut---
const time = clock(gpu);

// Warm every pipeline the ticker will need, BEFORE handing the loop over:
await prepare(gpu, [{ draw: wave, target: canvasSurface }]);

// GSAP owns the rAF; vgpu renders inside its tick, on GSAP's delta.
gsap.ticker.add((_total, deltaMs) => {
  time.advance(deltaMs / 1000);          // the clock moves here...
  wave.set("params", { time: time.time });
  frame(gpu, (f) => f.pass(canvasSurface, wave)); // ...and not again here
});
```

Note what disappears: there is no `frameLoop(gpu, ...)`. The external ticker is the loop, and `frame(gpu, cb)` is the render — encode, submit, done. The tick body stays fully synchronous: the clock advance, the byte writes and the frame all happen without a single `await`, which is exactly what the ticker contract requires.

## Motion, XR, and anything else with a delta

Every ticker hands you the same thing under a different name, so the shape never changes:

```ts
import { init, clock, frame } from "vgpu/mock";

const gpu = await init();
const time = clock(gpu);
declare function render(): void;

// ---cut---
// Motion (frame + delta):
//   frame.update(({ delta }) => { time.advance(delta / 1000); render(); });

// WebXR (absolute timestamps, one session frame at a time):
declare const session: { requestAnimationFrame(cb: (timestampMs: number, xrFrame: unknown) => void): number };
let previousMs: number | undefined;
const onXRFrame = (timestampMs: number) => {
  time.advance(previousMs === undefined ? 0 : (timestampMs - previousMs) / 1000);
  previousMs = timestampMs;
  frame(gpu, () => render());
  session.requestAnimationFrame(onXRFrame);
};
session.requestAnimationFrame(onXRFrame);
```

The first XR frame advances by `0`: there is no previous timestamp to measure against, and a made-up first delta is the classic source of a one-frame jump when the headset starts.

## Timescale: slow motion is a multiplication

Because the delta is yours, scaling it is the whole feature — no separate "speed" uniform threaded through every shader, and no second clock:

```ts
import { init, clock, frameLoop } from "vgpu/mock";

const gpu = await init();

// ---cut---
const time = clock(gpu);
let timescale = 1;         // 0 pauses, 0.25 is slow motion, 2 is fast forward
let previousMs = performance.now();

frameLoop(gpu, () => {
  const nowMs = performance.now();
  time.advance(((nowMs - previousMs) / 1000) * timescale);
  previousMs = nowMs;
  // ... render with time.time
});
```

`frameLoop` still schedules the frames; it just no longer decides what a frame is worth. `advance(0)` is legal and is the honest way to pause: the clock stops, frames keep rendering, `frameCount` keeps counting.

## Fixed timestep and determinism

A simulation that must not depend on frame rate advances in fixed steps and renders whatever the accumulator leaves behind:

```ts
import { init, clock, frame, frameLoop } from "vgpu/mock";

const gpu = await init();
declare function step(dt: number): void;
declare function render(): void;

// ---cut---
const time = clock(gpu);
const STEP = 1 / 120;                    // simulate at 120 Hz, render at display rate
let accumulator = 0;
let previousMs = performance.now();

frameLoop(gpu, () => {
  const nowMs = performance.now();
  accumulator += Math.min(0.25, (nowMs - previousMs) / 1000); // clamp: a hidden tab must not spiral
  previousMs = nowMs;

  let advanced = 0;
  while (accumulator >= STEP) {
    step(STEP);
    accumulator -= STEP;
    advanced += STEP;
  }
  time.advance(advanced);                // one advance per frame, however many steps ran
  render();
});
```

The same technique makes headless renders reproducible: drop the wall clock entirely and advance by a constant.

```ts
import { init, clock, effect, frame, target } from "vgpu/mock";

const gpu = await init();
const scene = target(gpu, { size: [64, 64] });
const shader = effect(gpu, `@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1); }`);

// ---cut---
const time = clock(gpu);
for (let i = 0; i < 90; i++) {
  time.advance(1 / 60);                  // frame 90 always lands on t = 1.5s
  frame(gpu, (f) => f.pass(scene, shader));
}
const pixels = await scene.read();       // same bytes on every machine, every run
void pixels;
```

## Rules of the technique

- **One advance per frame.** `advance()` before `frame()` is the pattern. Calling `advance()` twice before a single frame accumulates both deltas into `time` and leaves `deltaTime` at the last one — usually a bug in the ticker wiring.
- **Mixing is fine.** Skip `advance()` for a frame and that frame falls back to the wall-clock delta, measured from the previous tick. There is no mode to switch.
- **`frameCount` counts frames, not advances.** It only moves inside `frame()` / `frameLoop()`, so it stays a reliable "how many times did we render".
- **`advance()` takes seconds.** Most tickers hand out milliseconds — divide by 1000. Negative or non-finite deltas throw `VGPU-CLOCK-DELTA-INVALID` instead of quietly running time backwards.
- **Read the clock, don't cache the numbers.** `clock(gpu)` returns the same live object every time; `const time = clock(gpu)` outside the loop and `time.time` inside it always reads the current value.
- **Prepare before you hand over the loop.** `await prepare(gpu, [...])` every `(renderable, target)` combination the ticker will encode, and every `{ compute }` it will dispatch. Inside the tick the default policy throws `VGPU-PIPELINE-PENDING` rather than stalling; if some content genuinely arrives late, name `pendingPipelines: "skip"` on that call site (or on the frame) so it is omitted until its pipeline is ready.
- **The tick body must be synchronous.** `frame(gpu, cb)` rejects an `async` callback at the type level, and through JavaScript a thenable return throws `VGPU-ASYNC-FRAME-CALLBACK` and submits nothing at all. Load assets and prepare pipelines outside the tick.
- **One frame per tick, one submit per frame.** Everything the tick needs — compute (`f.compute`), buffer copies (`f.copyBuffer`), raw encoder commands (`f.raw`) and passes (`f.pass`) — goes into the *same* frame, so the ticker's tick maps to exactly one `queue.submit()`.
- **Stop the ticker on device loss.** Loss is terminal: vgpu stops the loops it owns, but an external ticker is yours to stop. `gpu.lost` resolves proactively — remove the tick callback there, then rebuild the whole GPU graph from a new `init()`.

## See also

- [Frames](concepts-frames.docs.md) — what `frame(gpu, cb)` and `frameLoop(gpu, cb)` actually do.
- `clock` — the full API of the clock, including its error codes.
