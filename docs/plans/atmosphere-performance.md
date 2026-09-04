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

The report is headed by the power state (Battery API): a laptop GPU may clock differently unplugged, so numbers only
compare within one state. Steps 0 to 3 were measured plugged in; the same build re-measured on battery right after
step 3 gave 2.54 / 1.68 ms for the full frame against 2.60 / 1.67 plugged in, so on this machine the difference is
within noise at this load. Later rows say which state they were taken in.

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
   it. Expected: cloud pass /4, frame -45 %. Done.
2. Frame cap and DPR: `frameLoop(gpu, cb, { fps: 60 })` and `dpr: 1`. Policy choices, each roughly halves the heat
   on its own. Done, with an fps readout in the panel to see the effect of a change.
3. Terrain as a rasterized mesh instead of a raymarch, so models can join the same depth buffer later. Done.
4. Ghosting: reproject the cloud history with translation (needs the per-texel mean depth stored), and raise the
   refresh fraction with full blend while the camera or the lighting changes, back to 1/16 at rest. Done.
5. Compact the cloud march: render the live texels of the frame into a small buffer and resolve into the history.
   Done; the pass turned out latency-bound, see the finding below the results.

## Results

Same machine and method as the baseline. ms per frame.

| Step | 2.7 Mpx full | clouds | scene | 1.2 Mpx full | clouds | scene | Notes |
|------|-----:|-----:|-----:|-----:|-----:|-----:|-------|
| 0 baseline | 8.7 | 5.9 | 2.5 | 4.1 | 2.8 | 1.1 | |
| 1 clouds at half resolution | 4.6 | 1.7 | 2.5 | 2.5 | 1.2 | 1.1 | cloud edges against terrain stay clean (the cloud pass now reads the same scene pixel present.wgsl compares against); silhouettes a touch softer at 3x zoom, not visible at 1x |
| 2 dpr 1, 60 fps cap | | | | | | | policy: the live surface renders at 1.2 Mpx instead of 2.7, half the frames. The fps cap needed a 1 ms slack in vgpu's frame loop: a strict 1000/fps threshold dropped ticks that landed 0.1 ms short (48 fps measured with a 60 cap). On a display whose rate is not a multiple of the cap the loop settles on the nearest divisor (48 on 144 Hz) |

| 3 terrain as a mesh, depth prepass + deferred shading | 2.6 | 1.8 | 0.40 | 1.67 | 1.2 | 0.23 | the raymarch is gone: a static ring grid around the camera axis (4096 columns x 512 rings, generated in the vertex shader, only the frustum's azimuth sector drawn) writes depth, and scene.wgsl shades each pixel once from that depth. Forward-shading the mesh was tried first and cost 4.6 ms: sub-pixel triangles near the horizon shade 2x2 quads each, so the same fragment work ran several times per pixel |

| 4 cloud ghosting: parallax reprojection + fast refresh after changes | 2.5 | 1.7 | 0.39 | 1.63 | 1.2 | 0.23 | on battery. The cloud history now stores the mean cloud depth (second attachment) and reprojects through the world point at that depth, so altitude changes no longer smear; any change of sun, haze, altitude or cloud parameters switches the next two frames to a checkerboard refresh (one texel in two, full blend), which reads as a 33 ms crossfade instead of a one-second ghost. The checkerboard frames cost 1.42 ms against 1.17 at rest |

| 5 compact cloud march + proper march noise | 1.6 | 0.87 | 0.39 | 1.33 | 0.86 | 0.23 | on battery. The live texels are marched packed into a viewport of the compact size (w/4 x h/4 at rest, w/2 x h in the fast mode) and a resolve pass scatters them into the history and reprojects the rest. The march start and light-sample jitter moved from an integer hash that was a linear ramp along rows (0.014 per pixel: every row shared one offset, so the step quantisation drew horizontal bands on cloud undersides at sunset) to interleaved gradient noise animated per frame while accumulating. The fast mode now selects a real checkerboard (the first half of the Bayer ranks); the previous rank % 2 test picked row pairs, which showed as horizontal stripes while the altitude changed |

Finding from step 4, revised by step 5: marching every cloud texel cost 1.68 ms against 1.17 for one in sixteen,
and packing the live texels only brought the pass to 0.86 ms. The pass is latency-bound, not throughput-bound: a
long ray through thick cloud is a chain of ~160 steps each with several dependent texture fetches plus a six-sample
light march, and with only 19k texels in flight the GPU cannot hide that chain. Further cloud savings have to shorten
the per-ray chain (fewer steps, cheaper density far away, a shorter light march), not the texel count.

After steps 1 to 5 the live frame is about 1.3 ms of GPU work at 60 fps against 8.7 ms at 120 fps before: about a
thirteenth of the GPU load. The cloud march is now 70 % of the frame; the terrain (depth prepass plus shading) 14 %.
The depth prepass is also the entry point for rasterized models: anything that writes into it inherits the aerial
perspective, the cloud occlusion and the sky compositing.
