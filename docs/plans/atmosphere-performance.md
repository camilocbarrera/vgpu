# Atmosphere example: performance and temporal stability

Two problems reported on 2026-09-04 on an Apple laptop: the example holds 120 fps but heats the machine to ~92 °C,
and the temporal cloud update ghosts badly while the camera moves (altitude changes, fast drags), although it is
fine for stills. The two are linked: a shorter ghost needs more cloud texels per frame, and that budget has to come
from somewhere.

## Method

`?bench` on the preview page (`/preview/atmosphere?bench`) runs `examples/atmosphere/bench.ts` before the live loop:
for each pass it submits 24 frames that encode only that pass, waits for the queue once, and reports wall clock per
frame, at the default preset (golden hour, camera at 80 m) and at two device pixel ratios. The rows add up to the full
frame within noise. GPU timestamps were tried and rejected: on Apple GPUs the passes of a frame overlap, so a pass
timestamp pair also spans its predecessors (the trivial present pass "took" as long as the whole frame).

## Baseline

Apple GPU (Metal 3), Chromium, 1040x1160 CSS pixels, best of 3 runs of 24 frames. ms per frame.

| Pass | 1560x1740 (dpr 1.5, 2.7 Mpx) | 1040x1160 (dpr 1, 1.2 Mpx) |
|------|-----:|-----:|
| full frame | 8.7 | 4.1 |
| clouds: 1/16 of the texels marched | 5.9 | 2.8 |
| scene: terrain march + sky | 2.5 | 1.1 |
| luts: aerial + frame constants + sky-view | 0.15 | 0.16 |
| present: tonemap + cloud upsample | 0.17 | 0.08 |
| clouds: coverage 0 (reprojection only) | 0.11 | 0.06 |
| terrain shadow map (only when the sun moves) | 2.5 | 2.5 |

Reading: the cloud march is ~67 % of the frame and the terrain march ~28 %; everything else is noise. Both scale
linearly with the pixel count (2.25x pixels, 2.1x time). The terrain shadow map costs 2.5 ms per frame only while
the sun slider is being dragged. Note: with the `timestamp-query` feature enabled at init every pass ran ~1.6x
slower on this machine (14.3 ms full frame), one more reason the bench does not use it.

## Plan

1. Clouds at half resolution (`CLOUD_TUNING.renderScale` 2). The depth-aware upsample in present.wgsl was written for
   it. Expected: cloud pass /4, frame -45 %.
2. Frame cap and DPR: `frameLoop(gpu, cb, { fps: 60 })` and `dpr: [1, 1]`. Policy choices, each roughly halves the
   heat on its own; kept as options for the user.
3. Terrain march: reuse last frame's distance as the march start, or a max-mip of the heightmap for long steps.
4. Ghosting: reproject the cloud history with translation (needs the per-texel mean depth stored), and raise the
   refresh fraction with full blend while the camera or the lighting changes, back to 1/16 at rest.

## Results

Filled in as each step lands.
