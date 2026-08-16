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
| fluid | `{bundle: bundles[0]}`, `{bundle: bundles[1]}` | the await **moved below** the bundle construction: the encode site is `p.bundles(...)`, and a bundle must exist to be prepared. Subsumes the two display `compile()`s |
| batch-rendering | 4 draws + blit | prepared **before** `bundle()` records them — recording needs the pipelines |
| instanced-rendering | drawable + blit | same ordering constraint |
| radiance-cascades | 4 + N jfa + N cascade | **dynamic**; keeps signature literals (see below) |
| agent-radiance-cascades | 4 + N jfa + N cascade | same |
| triangle-led-front | 4 draws + 3 bundles (scene) + 2 bundles (light-sources) | see "lazy birth" below |
| nextjs-flare | 5, inside the preserved `needsCompile` guard | see below |
| **hero** | **13 effects, ONE `prepare()`** | see below |

### Criteria that are not mechanical

**`output` replaces `{ colors: [output.format] }`.** Nine files built a signature literal by hand
because 0.3.0 refused to derive a `Surface` signature outside `frame()`. That restriction is gone
(rev6.1 §0: a surface signature comes from its *configuration*). Naming the real destination is what
makes prepare and encode **provably** agree — the encode derives its signature from that same
object. Where the two would have differed, the legacy prewarm was warming a signature the encode
path never asked for; this migration cannot make that worse, only better.

**Signature literals that were KEPT, on purpose.** The radiance-cascades pair, the transmission blur
pyramid, the hero bloom chain and the triangle-led-front floor draws keep `{ colors: [FMT] }`. In
each, no single Target object is "the" destination: the passes ping-pong between two recycled
atlases, or walk a pyramid of different sizes, or are chosen per frame by theme. One *format* is
what they genuinely share, and a signature is exactly what the pipeline key is built from. Naming
any one target would imply a specificity the key does not have.

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

## Known residual — a T04-21 blocker, not a T04-19 gap

`apps/docs/examples/triangle-led-front/light-sources-raw.ts`: the re-recorded `clearBundle`.

Under today's `"sync"` default it re-encodes inline exactly as it always has — **nothing is broken
by this ticket**. Under `"throw"` it will raise on the first bake-key change. The fix is either a
frame-level `pendingPipelines` policy for that renderer or hoisting the re-record out of the
callback; both are *behaviour* changes to the example, which a purely-additive ticket does not make.
Recorded here so T04-21 inherits a known item rather than a surprise.

## What was deliberately NOT touched

- **`packages/*/tests/**`** — many of those files are the pinned *subject* of a readiness behaviour
  (they assert what happens *without* `prepare()`). Inserting there would delete the coverage
  T04-21's flip depends on.
- **`experiments/**`** — in no repo gate (T04-18's finding), so a change there is unverifiable by CI.
- **Legacy `dispatch()` computes** — `compute.ts` states that legacy `dispatch()` stays lazy-sync and
  never consults `pendingPipelines`. Preparing them would be dead weight today; migrating them off
  `dispatch()` belongs to the spelling retirement (T04-22).
