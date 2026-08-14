/**
 * The one compilation policy shared by every synchronous encode context.
 *
 * A render pipeline cannot exist without a target signature, so readiness is a property of a
 * *combination* — `(renderable, target signature)` — never of an object. When a synchronous encode
 * meets a combination whose pipeline is not ready yet, this policy decides what happens:
 *
 * - `"throw"`: does **not** start compilation and throws `VGPU-PIPELINE-PENDING` immediately.
 * - `"skip"`: starts (or continues) async compilation in the background, omits the command this
 *   frame, and never throws per frame.
 * - `"sync"`: performs immediate pipeline creation inline — the only synchronous-compilation
 *   opt-in, and the only one that can stall.
 *
 * The value is resolved **call site → frame → gpu** (`DrawCallOptions.pendingPipelines` →
 * `frame(gpu, cb, { pendingPipelines })` / `FrameLoopOptions` → `init({ pendingPipelines })`).
 *
 * This module is types plus one constant on purpose: `kernel.ts` must not import a feature module,
 * and every consumer (draw, frame, pipeline store, prepare) needs the same vocabulary.
 */
export type PendingPipelines = "throw" | "skip" | "sync";

/**
 * Effective default of the whole train (T04-05, index invariant 2): **`"sync"`, not `"throw"`**.
 *
 * The frozen design (AC #3) mandates `"throw"` as the final default, and this branch implements the
 * complete mechanism — three values, full call site → frame → gpu resolution chain. What it does
 * NOT do is flip the default: zero call sites have migrated to `prepare()` yet, so `"throw"` would
 * break every existing program at once. `"sync"` is exactly the eager-compile behavior the corpus
 * has today, which keeps this wave purely additive. The flip to `"throw"` belongs to the cut wave,
 * together with the codemods that migrate the call sites.
 */
export const DEFAULT_PENDING_PIPELINES: PendingPipelines = "sync";
