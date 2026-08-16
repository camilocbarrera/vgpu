# T04-19 — `prepare()` insertion: the decision behind every example

This ticket is **semi-automatic by design**. `scripts/codemods/prepare-insertion.mjs` does the
mechanical half (find constructions by type, pair each encode site with the target expression it
renders into, locate the setup boundary, classify what it cannot answer). *What* to prepare, *how to
group it* and *where the await lands* is criterion, and criterion has to be written down. This file
is that writing.

Reproduce the machine half with:

```sh
node scripts/codemods/prepare-insertion.mjs --dry-run   # the report attached to the PR
node scripts/codemods/prepare-insertion.mjs --verify     # exits non-zero on an uncovered combination
```

## Completeness criterion (and how it was actually checked)

The insertion is **semantically inert today**: the `pendingPipelines` default is still `"sync"`, so
the corpus behaves identically with or without it. That is the safety property of the additive
phase — and it is also what makes a missing `prepare()` invisible. "The tests still pass" proves
nothing here.

So the criterion is the one that will matter: **under the default T04-21 will ship (`"throw"`),
every example runs without `VGPU-PIPELINE-PENDING`.**

`packages/vgpu-api/tests/prepare-corpus-throw.test.ts` executes exactly that. It mocks the single
constant T04-21 will edit (`DEFAULT_PENDING_PIPELINES`) to `"throw"` and then runs the **real,
shipped** example modules against it — imported through the same `vgpu/node` alias production uses,
not re-implemented. It also asserts the mock is in effect, so it cannot pass for the wrong reason.

That test found one real gap during development (`s09-bundles` had no `prepare()` and threw
`VGPU-PIPELINE-PENDING` at `p.bundles(staticScene)`). That is the whole argument for building it.

## Falsifications of the ticket (verified against the tree, not assumed)

| Ticket said | Reality | Evidence |
|---|---|---|
| "19 of 20 `apps/docs/examples` have a real renderer; confirm which one does not" | **20 of 20.** There is no example directory without a vgpu renderer. `air-painting` and `mnist-classifier` were the candidates (they hide their vgpu use in `visual-pipeline.ts` / `renderer.ts` rather than a file named `renderer.ts` importing `vgpu` at top level), and both render. | `prepare-insertion.mjs --dry-run` resolves constructions in all 20 |
| "the hero has **12** `Effect`s" | **13.** The ticket's own list enumerates thirteen names (`bake`, `refine`, `shade`, `bloomExtract`, `bloomBlurH0/V0/H1/V1/H2/V2` = 6, `bloomDown1/2` = 2, `composite`) while calling it twelve. | `hero/renderer.ts` `createEffects()`, 13 constructions |
| "the 15 examples of `examples/`" (as 15 renderers to prepare) | **12 renderers.** `examples/` does have 15 directories, but `nextjs-flare` contains **no TypeScript at all** (a PNG and a PPM), `next-wgsl` is a Turbopack loader demo with no vgpu runtime, and `fluid-validation` is tests only. Only the twelve `by-example-s02..s13` projects have a `src/example.ts` to prepare. | `git ls-files examples/<dir>` |
| "the `prepare()` insertion is the new work" | **Most of the docs corpus already prepared by hand**, with the legacy per-object `compile()` and a `Promise.all` fan-out — 15 files. The real work was *migrating* those to `prepare()` while preserving their conditions, not inventing preparation. | `grep -rn "\.compile("` across `apps/docs` |
| (not in the ticket) | The codemod's own scanner initially **under-reported**: `f.pass(target, renderable)` — the shorthand where the second argument is a renderable instead of a callback — was invisible to it. `fft-ocean-surface` uses it. Fixed, with a comment. | `prepare-insertion.mjs`, `pass` branch |

## `examples/` — 12 projects

| Example | Combinations prepared | Where the await went | Verified under `throw` |
|---|---|---|---|
| s02-fullscreen | `{draw: wave, target: colorTarget}` | after `wave.set()`, before `frame()` | ✅ executed |
| s03-sharing | `cube→colorTarget`, `floor→colorTarget` | after both `.set()`, before `frame()` | ✅ executed |
| s04-shared-uniforms | `wave→colorTarget`, `tint→colorTarget` | after `globals.set()`, before `frame()` | ✅ executed |
| s05-fixits | **none — documented skip** | — | ✅ executed (own assertion) |
| s06-scene | `cube→colorTarget` | after the three `.set()`, before `frame()` | ✅ executed |
| s07-hdr-post | `solid→scene`, `post→output` | after construction, before `frame()` | ✅ executed |
| s08-ping-pong | `fill→buf.write`, `copy→buf.read` | before the first `frame()` | ✅ executed |
| s09-bundles | `{bundle: staticScene}` | after `bundle()`, before the first `frame()` | ✅ executed |
| s10-group-claim | `drawable→colorTarget` | after `drawable.group()`, before `frame()` | ✅ executed |
| s11-compute | **none — `dispatchOnce` covers it** | — | ✅ executed |
| s12-scheduling-resize | `post→baked` | before the first `frame()` | ✅ executed |
| s13-headless | `p→colorTarget` | before the one-shot `p.draw({target})` | ✅ executed |

Criteria worth stating:

- **s05-fixits is a deliberate skip.** Its whole job is to *collect* fix-it error messages, so
  "it did not throw" is not evidence. Preparing `missing` would make `prepare()` itself reject and
  replace the fix-it the example exists to demonstrate. Instead it gets a dedicated assertion:
  under `"throw"` the collected messages must still be fix-its and must **not** contain
  `VGPU-PIPELINE-PENDING`. They do not — pipeline resolution is reached only after the binding
  validation the example triggers.
- **s11-compute is a skip by contract, not by omission.** `dispatchOnce()` always takes the async
  readiness path regardless of the default (contract #20), so a `prepare()` here would be the
  redundant insertion the ticket's Prohibitions forbid.
- **s08-ping-pong: `prepare()` takes the target to derive a *signature*, never to pin a resource.**
  This matters because T04-17 shipped a bug of exactly the opposite shape (a ping-pong half frozen
  at construction). Warming a half cannot freeze it. Both halves share one signature anyway; naming
  the object each effect really writes into is for the reader.
- **s12: one prepare survives `baked.resize([8, 8])`** because a target signature is
  `{colors, depth, sampleCount}` — size is not in it.
- **s13 has no `frame()`.** The one-shot `p.draw({target})` *is* the first encode, and it resolves
  the policy the same way a frame draw does (`draw.ts` `encode()` → `#pipelineForEncode`). The
  setup boundary is the line before it, not before a loop that does not exist.

## `apps/docs` — 20 examples + hero

`✅ throw` = covered by the mechanical `--verify` gate and by typecheck; these are browser renderers
driven by canvases, `ResizeObserver` and ML runtimes, so they are **not** executed under a forced
`throw` default the way `examples/` are. That is the honest limit of this ticket's verification and
the stated focus for QA.

| Example | Combinations prepared | Where / criterion |
|---|---|---|
| gradient | `shader→canvasSurface`; `shader→target` (thumb) | two sites; the live one goes after `measure()`, before `frameLoop` |
| mnist-classifier | `effect→output` | **new `Visualizer.prepare()`**: `render()` is sync, called from ORT completion; prepared where the output is born |
| depth-estimation | `effect→output` | **new `SideBySidePipeline.prepare()`**, same reason. `reducer` excluded: legacy `dispatch()` is lazy-sync and ignores the policy |
| air-painting | `frostH→frostA`, `frostV→frostB`, `composite→output` | **new `prepareVisualFrame()`**; the 3 computes excluded for the same `dispatch()` reason |
| anti-aliasing | 6 (`scene`→output/msaa/ssaa/ldr, `resolve`→output, `fxaa`→output) | migrated `prewarm()`; covers all four AA modes |
| post-processing | 5 | migrated `prewarm()` |
| raymarched-fractal | 5 | migrated `prewarm()` |
| black-hole | 7 | migrated `prewarm()` |
| earth | 10 + 2 bake | migrated `prewarm()` + `bakeMaps()`; the 4 blur passes are written out by index because the source is, not looped |
| environment-map | 2 + 2 | migrated both inline `Promise.all`s |
| transmission | 5 + 2N (blur pyramid) + 2 (prefilter) | **dynamic**: the pyramid requests are built by `flatMap` over the same array the loop builds |
| fft-ocean | 10 + N (ifft stages) + 2N (bloom levels) + preview | **dynamic**: `map`/`flatMap` over `g.ifft` and `g.levels`, never flattened |
| fft-ocean-surface | `skydome→hdr`, `ocean→hdr`, `composite→surface` | **new**; `composite` is encoded through the shorthand `f.pass(target, effect)` form |
| fluid | `{bundle}` ×2 | prepared into a **local**, published to `fluid.bundles` only after the await. Subsumes the two display `compile()`s |
| batch-rendering | `{bundle}` + blit (both paths) | prepared **after** `bundle()` records — the only edge out of `pending-pipelines` |
| instanced-rendering | `{bundle}` + blit (both paths) | same |
| radiance-cascades | 4 + N jfa + N cascade | **dynamic**; keeps signature literals (see below) |
| agent-radiance-cascades | 4 + N jfa + N cascade | same |
| triangle-led-front | 4 draws + 3 bundles (scene) + 2 bundles (light-sources) | see "lazy birth" and the readiness gate below |
| nextjs-flare | 5, inside the preserved `needsCompile` guard | see below |
| **hero** | **13 effects, ONE `prepare()`** | see below |

### Criteria that are not mechanical

**`output` replaces `{ colors: [output.format] }`.** Nine substitutions across nine files — eight
under `apps/docs/examples` plus `apps/docs/components/hero/renderer.ts` — built a signature literal by hand
because 0.3.0 refused to derive a `Surface` signature outside `frame()`. That restriction is gone
(rev6.1 §0: a surface signature comes from its *configuration*). Naming the real destination is what
makes prepare and encode **provably** agree — the encode derives its signature from that same
object. Where the two would have differed, the legacy prewarm was warming a signature the encode
path never asked for; this migration cannot make that worse, only better.

**Signature literals that were KEPT, on purpose.** The radiance-cascades pair, the transmission blur
pyramid and the hero bloom chain keep `{ colors: [FMT] }`. In each, no single Target object is "the"
destination: the passes ping-pong between two recycled atlases, or walk a pyramid of different
sizes. One *format* is what they genuinely share, and a signature is exactly what the pipeline key
is built from.

**One of those literals is weaker than the others, and the QA was right to say so.** The
triangle-led-front floor draws have exactly **one** stable destination (`colorTarget`) — the theme
ternary picks between two *draws*, not two targets — so they could have been named. The literal
produces the correct key today only because that surface carries no depth and no MSAA, which is a
coincidence of the current configuration rather than a property of the code. Left as-is (naming it
is a behaviour-adjacent change past a FAIL-then-fix cycle) and recorded here as the one kept literal
that rests on a coincidence.

**Dynamic multi-pass is iterated, never flattened.** `transmission`, `fft-ocean`,
`radiance-cascades`, `agent-radiance-cascades` and `batch-rendering` build their passes with
`map`/`flatMap`/`for`. Their request lists are built by iterating *the same structure*. Add a
cascade level or a bloom stage and the preparation follows; a flattened list of N literal lines
would have frozen a count the scene decides at construction.

**Hero: 13 effects, one await.** The array form is the point — a post chain this size warms as a
batch and reports failures as a batch (`VGPU-PREPARE-FAILED` enumerates every one), instead of the
first rejection hiding the other twelve, which is what the `Promise.all` it replaces did.
`bake`/`refine`/`shade` name their real targets; the nine bloom passes share one signature.

**nextjs-flare: the `needsCompile` guard is preserved verbatim.** `resize()` re-creates all four
intermediates on every size change, but only their *size* changes — formats are fixed, so every
resize after the first asks for an already-cached key. Dropping the guard while migrating the
spelling was the one thing this migration had to not do.

**triangle-led-front: one lazy birth hoisted, one left as a documented blocker.**
- The **floor bundles** were born inside the synchronous frame callback
  (`currentParts.floorBundles ?? recordFloorBundles(...)`). A bundle created there can never be
  prepared. `recordFloorBundles` memoizes, so its *birth* was moved to the async `prewarm()`; the
  `??` in the callback stays as the fallback it always was, and is now always a cache hit.
- The **clear bundle** in `light-sources-raw.ts` is **re-recorded** inside `encode()` whenever the
  bake key changes (occluder toggle / clip-inset tunable). Its birth cannot be hoisted: the key is a
  continuous tunable, so the variants are unbounded. `p.bundles()` takes no per-call
  `pendingPipelines` (`frame.ts` says so explicitly) and the callback cannot await.

## The QA cycle: four renderers were broken, and the skip is where they lived

The first submission declared the `apps/docs` renderers "covered by `--verify` and typecheck, not
executed under a forced throw" and called that an honest limit. **It was a hole, and four renderers
were in it.** Adversarial QA extended the harness (mock the adapter to `src/mock.ts`, keep
`prepare`/`frame`/`bundle`/`pipeline-store` REAL, fake canvas, pumped rAF) and got 17 of 21 docs
renderers executing under the real `"throw"` default. Four raised `VGPU-PIPELINE-PENDING`. A control
run with the throw mock neutralized was 21/21 green, so the failures were the code, not the harness.

**The lesson is the one worth carrying into T04-21: the skipped verification is exactly where the
bugs were.** Not one of the four was visible to typecheck, to `--verify`, or to the per-example
`renderer.test.ts` files — those stub `vgpu` wholesale with `prepare: () => Promise.resolve([])`, so
they can *never* catch this class. All twelve `examples/` projects, which were executed under a
forced throw from the start, were correct on the first pass. That is not a coincidence.

### Class A — the bundle was never prepared (batch-rendering, instanced-rendering)

The original comment asserted the draws had to be prepared *before* `bundle()` "because the
recording needs the pipelines". **That premise is false.** `bundle.ts`'s frozen transition table is
explicit: `bundle(gpu, {target}, rec)` **always** lands on `pending-pipelines` ("the native bundle
is not materialized at construction"), recording needs no pipelines at all, and the only edge to
`ready` is `prepare(gpu, [{ bundle }])` — which compiles the recorded draws **and** encodes the
native bundle.

So the request is the **bundle**, and it comes **after** the recording. The `{draw, target}`
requests were both unnecessary (subsumed) and insufficient (never moved the bundle off
`pending-pipelines`). The port was faithful to the legacy `.compile()` prewarm, which only ever
worked because `"sync"` materializes the bundle inline on first replay — the exact crutch T04-21
removes. `s09-bundles` got this right on the first pass *because the shipped test executed it*.

Fixing it revealed a **layered** failure: the live `createRenderer` path never prepared the
`*-blit` effect either (only `renderThumbnail` did). Both paths now do.

### Class B — a new bundle reaches a running loop before its prepare resolves

- **fluid — this branch introduced the window.** The legacy order was `await compile()` → create
  bundles, with nothing after the await. Moving the await below the construction (correct in
  itself) made it create → `await prepare()` → publish. On resize `output` is the *same* surface
  object, so `renderFluid`'s `fluid.output !== output` guard cannot fire and the loop encodes the
  new, unprepared bundle. Fixed by holding the pair in a **local** and assigning `fluid.bundles`
  only after the await: the loop keeps replaying the previous ready pair, so a resize is late, never
  broken.
- **triangle-led-front.** `rebuild()` and `setOutputTarget()` re-record the floor bundles
  synchronously on every resize — and `rebuild()` builds an entirely new `raycastBundle` — while the
  loop runs and `prewarm()` is fire-and-forget. The earlier claim that the memoized `??` is "always
  a cache hit" is true and **beside the point: a cache hit is not readiness.** Fixed with an
  explicit readiness generation — `rebuild()`/`setOutputTarget()` bump it, `prewarm()` opens the
  gate only if nothing re-recorded mid-flight, and `renderFrame()` skips until they match.
- **triangle-led-front, second bug found by re-running the harness after the gate landed:** the
  floor-noise bake ran in the *synchronous body* of `createHeroRenderer()` — a real encode before
  anything could possibly be prepared, since a constructor cannot await. Moved into `prewarm()`,
  after the prepare that already covered the combination.

### Class C — re-record inside the sync callback (unchanged, T04-21 input)

`apps/docs/examples/triangle-led-front/light-sources-raw.ts`: `clearBundle` is re-recorded inside
`encode()` whenever the bake key changes (occluder toggle / clip-inset tunable). Its birth cannot be
hoisted — the key is a continuous tunable, so the variants are unbounded — `p.bundles()` takes no
per-call `pendingPipelines`, and the callback cannot await.

QA scanned all 9 `bundle()` construction sites and confirmed this is the **only class-C site in the
corpus**. It is also the **least severe** of the three: it needs a bake-key change to fire, whereas
class A threw on frame 1. Under today's `"sync"` default it re-encodes inline exactly as it always
has — nothing is broken by this ticket. Under `"throw"` it will raise on the first bake-key change.
The fix is a frame-level policy or hoisting the re-record; both are *behaviour* changes to the
example, which a purely-additive ticket does not make.

**T04-21 inherits: one class-C site, and the standing instruction to run the extended docs harness
rather than trust `--verify` alone.**

## Verification, restated honestly

| Corpus | How verified | Result |
|---|---|---|
| `examples/` (12) | shipped `prepare-corpus-throw.test.ts`, real modules, forced `"throw"` | 13/13 |
| `apps/docs` (21 incl. hero) | extended QA harness, real prepare/frame/bundle, pumped rAF | 21/21 |
| control (mock neutralized) | same harness, `"sync"` default | 21/21 — failures were real |
| still not executed | `fft-ocean-surface` / `nextjs-flare` (need `document.createElement`), 3 ML renderers (no `createRenderer`) | `--verify` + typecheck only |

## What was deliberately NOT touched

- **`packages/*/tests/**`** — many of those files are the pinned *subject* of a readiness behaviour
  (they assert what happens *without* `prepare()`). Inserting there would delete the coverage
  T04-21's flip depends on.
- **`experiments/**`** — in no repo gate (T04-18's finding), so a change there is unverifiable by CI.
- **Legacy `dispatch()` computes** — `compute.ts` states that legacy `dispatch()` stays lazy-sync and
  never consults `pendingPipelines`. Preparing them would be dead weight today; migrating them off
  `dispatch()` belongs to the spelling retirement (T04-22).
