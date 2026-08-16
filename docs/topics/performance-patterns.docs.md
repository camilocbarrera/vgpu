# Performance patterns

This is the quick index. Open `performance-playbook` for copy-paste before/after snippets.

## Static scene

Use `bundle(gpu, { target }, recorder)`, materialize it once with `await prepare(gpu, [{ bundle }])`, and replay with `p.bundles(bundle)`.

## First-frame stability

Use `await prepare(gpu, [{ draw, target }])` so pipeline creation happens before the transition frame; pass a canvas `surface` or an offscreen `target` explicitly (a `Surface` is legal outside `frame()`). Under the default `pendingPipelines: "throw"` a missed combination throws instead of hitching, so first-frame stalls become deterministic errors rather than rare spikes.

## Animated uniforms

Create the draw/effect once and call `.set(binding, value)` for what changed — bytes only, no bind-group rebuild, and a struct partial collapses into one buffer write. Do not allocate a new pass or uniform buffer every frame.

## Many objects

Use `instances` when geometry and material are shared. Use `draw.group()` with a claimed bind group plus dynamic `offsets` when each object needs a different uniform block. Skip draws hidden behind occluders with `visibility(gpu)` proxy queries. When a compute pass decides the counts, draw with `indirect` arguments instead of reading them back to the CPU.

## Shared globals

Use one `uniform(gpu, { time, mouse, camera })` object, declare it in `bindings` of every shader that needs the same struct, and update it with `globals.set({ … })`: one write, every pipeline sees it.

## Iterative effects

Use `pingPong(gpu)` for targets or `pingPongStorage(gpu)` for compute, and `.bind()` the swapped halves. Do not allocate temporary targets/storage in the loop.

## 3D targets

Create targets (and surfaces) with `depth: true` and `sampleCount: 4` when needed; those two plus the color formats **are** the pipeline signature, so warm each signature with `await prepare(gpu, [{ draw, target }])`. A resize that keeps the signature invalidates nothing.
