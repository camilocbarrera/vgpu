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
 * The default of the library: **`"throw"`** (AC #3 of the frozen design, satisfied in value and no
 * longer only in mechanism).
 *
 * It is uniform — same value in the browser, in node, and over the mock adapter, with no dev/prod
 * divergence and no migration flag — because the alternative is the bug this whole design exists to
 * remove: a synchronous encode that silently stalls the frame on `createRenderPipeline` in
 * development and only shows up as a hitch on someone else's machine. Under `"throw"` an encode
 * that meets an unready combination does not start compilation and does not stall; it names the
 * combination and points at `prepare()`.
 *
 * Why it is safe *now*, where it was not during the additive phase (T04-05 shipped the mechanism
 * with a temporary `"sync"` default, index invariant 2): every encode site of the corpus reaches
 * its first frame through `prepare()` — `examples/*` and `apps/docs/examples/*` were migrated by
 * T04-19 and the last re-record inside a synchronous frame callback
 * (`triangle-led-front/light-sources-raw.ts`) was deleted by this ticket, which is all it ever
 * needed: that recording never depended on anything the frame changed, so the prepared bundle is
 * replayed instead of being rebuilt.
 * `packages/vgpu-api/tests/prepare-corpus-throw.test.ts` executes that claim against the shipped
 * example modules instead of asserting it.
 *
 * `"sync"` did not disappear: it is a permanent, explicit opt-in (`init({ pendingPipelines:
 * "sync" })`, `frame(gpu, cb, { pendingPipelines: "sync" })`, or per call site), for code that
 * genuinely prefers a stall to a throw. It just stopped being what you get by accident.
 */
export const DEFAULT_PENDING_PIPELINES: PendingPipelines = "throw";
