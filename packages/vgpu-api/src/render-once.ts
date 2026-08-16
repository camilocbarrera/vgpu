/**
 * `renderOnce()` — the render half of the one-shot paths (§7 of the frozen design).
 *
 * ```ts
 * await renderOnce(gpu, s, (p) => p.draw(quad));   // standalone render, own encoder + one submit
 * await sim.dispatchOnce(1024);                    // standalone compute, own encoder + one submit
 * ```
 *
 * `dispatchOnce()` is convenience for a compute operation that is independent of a frame loop:
 * initialization, preprocessing, tests and headless work. If the compute pipeline is not ready, it
 * awaits `createComputePipelineAsync()`, then creates its own command encoder and compute pass,
 * records one dispatch and submits exactly once. An already-prepared pipeline is reused without
 * compilation.
 *
 * ```ts
 * // Standalone work:
 * await sim.dispatchOnce(workgroups);
 *
 * // Work that must share ordering and one submission with other commands:
 * frame(gpu, (f) => {
 *   f.compute(sim, workgroups);
 *   f.pass(screen, (p) => p.draw(result));
 * });
 * ```
 *
 * `dispatchOnce()` is not the API for work that belongs inside `frame()`: its encoder and submit are
 * independent. Use `f.compute()` for that case. Both one-shot helpers resolve after pipeline
 * readiness and submission — **never after GPU completion**. Waiting for completion is explicit and
 * separate: `await gpu.settled()`. The `pendingPipelines` policy controls synchronous encode
 * contexts; one-shot helpers always use their async readiness path.
 */
import { encodeDraw, InternalDraw, type Draw, type DrawCallOptions } from "./draw.ts";
import { effectDraw, InternalEffect, type Effect } from "./effect.ts";
import { prepareFailedError, type PrepareFailure } from "./errors.ts";
import type { Gpu } from "./kernel.ts";
import { assertDeviceUsable } from "./lifecycle.ts";
import { liveKernel } from "./live-kernel.ts";
import { normalizeSignature, signatureKeyOf } from "./pipeline-store.ts";
import { enterFrame, leaveFrame } from "./surface.ts";
import type { Target } from "./target.ts";
import { BUILT_IN_CLEAR_COLOR } from "./target-utils.ts";

/**
 * What a `renderOnce()` body may record. Deliberately smaller than `FramePass`:
 * - `FramePass` takes a live `GPURenderPassEncoder` in its constructor (`frame.ts`), so it cannot
 *   exist during the collect phase, which by definition opens nothing native;
 * - `occlusion()` needs a `Visibility` attached through `FramePassOptions`, and `bundles()` needs
 *   the bundle readiness path — both are frame-scoped features, and neither appears in any §7
 *   example. `p.bundles()` inside `renderOnce` is **out of scope** for this first version (a bundle
 *   would need `prepareBundle()` chained into phase 2), not a silent omission: use
 *   `frame(gpu, (f) => f.pass(target, (p) => p.bundles(b)))`.
 */
export interface RenderOncePass {
  draw(drawable: Draw | Effect, opts?: DrawCallOptions): void;
}

/** One recorded `p.draw()`, resolved to the draw the encode phase will replay. */
interface RecordedDraw {
  readonly draw: InternalDraw;
  readonly opts: DrawCallOptions;
}

/**
 * Collect phase recorder. It resolves which `InternalDraw` each `p.draw()` names and remembers the
 * call, and does nothing else: no encoder is opened, no pipeline is compiled — the same "construction
 * records the logical command list without compiling" the bundle recorder implements in its
 * encoder-less mode.
 */
class CollectingPass implements RenderOncePass {
  readonly commands: RecordedDraw[] = [];
  draw(drawable: Draw | Effect, opts: DrawCallOptions = {}): void {
    const draw = drawable instanceof InternalEffect ? effectDraw(drawable) : drawable as InternalDraw;
    // The options bag is snapshotted (shallow, like the bundle replay's own `{ ...opts }`) because
    // the encode happens after an await: a caller that reuses one options object across draws must
    // not have every recorded call collapse onto its last mutation.
    this.commands.push({ draw, opts: { ...opts } });
  }
}

/**
 * Standalone render: compiles what the body names through the **async** readiness path, then owns
 * one command encoder, one render pass and exactly one `queue.submit()`.
 *
 * ```ts
 * await renderOnce(gpu, screen, (p) => p.draw(quad));
 * ```
 *
 * Three phases, so that no `await` ever happens inside an open render pass (WebGPU has no such
 * thing) and no pipeline is ever compiled inline:
 * 1. **collect** — the body runs once against a recorder that only resolves the draws it names;
 * 2. **prepare** — every distinct draw is compiled in parallel through `pipelineForAsync()`
 *    (`createRenderPipelineAsync`), never through the synchronous `pendingPipelines` path. A failure
 *    rejects with `VGPU-PREPARE-FAILED` enumerating every failed combination — the very error
 *    `prepare()` raises, because this is the same "a batch of combinations fails as a batch" report,
 *    and a second code for it would be a second vocabulary for one thing;
 * 3. **encode + submit** — the recorded command list is encoded into a private encoder with every
 *    pipeline already warm, and submitted once. The returned promise resolves right after that
 *    submit; it **never** awaits `onSubmittedWorkDone` (contract #20). `await gpu.settled()` is the
 *    explicit spelling for GPU completion.
 *
 * **The body runs exactly once.** It is a naming pass, not a replay: what is retained between the
 * collect phase and the encode phase is the resolved command list, not the closure (which is what
 * `bundle()`'s recording is, and why a bundle's `record` callback can run more than once). Side
 * effects in the body therefore happen once, in call order, *before* anything is compiled or
 * encoded.
 *
 * **The encode reads the current state of each draw**, not a snapshot taken while the body ran: a
 * `.set()` or `.bind()` that lands between the body and the submit is reflected by this render —
 * the same rule a bundle materialization follows.
 *
 * The pass clears with the target's own `clearColor` (or the built-in `[0, 0, 0, 1]`). A one-shot
 * render has no pass options: `preserve`, viewport/scissor, depth-read-only, timers and occlusion
 * are `frame()`'s `f.pass(options, body)` surface, and a one-shot that grew them would be a second
 * spelling of it.
 */
export async function renderOnce(gpu: Gpu, target: Target, body: (pass: RenderOncePass) => void): Promise<void> {
  // Same entry guard as every other free function: a disposed gpu fails here, not inside a driver call.
  const kernel = liveKernel(gpu, "renderOnce");
  const device = kernel.device;
  assertDeviceUsable(device, "renderOnce");

  // --- Phase 1: collect. Nothing native is touched, nothing is compiled.
  const recorder = new CollectingPass();
  body(recorder);
  const commands = recorder.commands;

  // --- Phase 2: async readiness for every distinct combination, in parallel.
  const draws = [...new Set(commands.map((command) => command.draw))];
  if (draws.length) {
    // Settled rather than raced, exactly like `prepare()`: a batch of combinations fails as a batch,
    // so every failure has to be reported, not whichever one rejected first.
    const readiness = Promise.allSettled(draws.map((draw) => draw.pipelineForAsync(target)));
    // A compile this call started is work vgpu started, so `gpu.settled()` must see it (issue #332
    // tracks the same gap on the compute side, where the one-shot has no kernel to register with).
    // `allSettled` never rejects, so tracking it cannot turn a rejection into a console error — the
    // rejection is still delivered to this call's own caller, below.
    void kernel.trackDelivery(readiness);
    const settled = await readiness;
    // The device can be lost — or the gpu disposed — while a compile is in flight, and that is the
    // proximate cause of every rejection in `settled`: report it as itself instead of burying a
    // `VGPU-DEVICE-LOST` inside a compile-failure batch. Same re-check `dispatchOnce()` does after
    // its own await, one phase earlier than the guard before the encode.
    assertDeviceUsable(device, "renderOnce");
    const failures: PrepareFailure[] = [];
    settled.forEach((result, index) => {
      if (result.status === "rejected") failures.push({ label: draws[index]!.label ?? "draw", signature: signatureKeyFor(target), cause: result.reason });
    });
    if (failures.length) throw prepareFailedError(failures);
  }
  // The device can be lost — or the gpu disposed — while a compile is in flight: re-check before
  // opening anything, the same way `dispatchOnce()` re-checks after its own await.
  assertDeviceUsable(device, "renderOnce");

  // --- Phase 3: one encoder, one pass, one submit.
  // A `Surface` target resolves its presentation texture here, and `draw.encode()` refuses to encode
  // one outside a frame; a one-shot render IS a (very small) frame, so it opens the same scope
  // `frame()` opens, for the duration of this synchronous encode only.
  enterFrame();
  try {
    const descriptor = target.renderPassDescriptor({ clear: target.clearColor ?? BUILT_IN_CLEAR_COLOR });
    const encoder = device.gpu.createCommandEncoder({ label: "vgpu.renderOnce" });
    const pass = encoder.beginRenderPass(descriptor);
    // Every pipeline is warm, so this hits the store's ready path: no `createRenderPipeline` here,
    // whatever the resolved `pendingPipelines` default is.
    for (const command of commands) encodeDraw(command.draw, pass, target, command.opts);
    pass.end();
    device.gpu.queue.submit([encoder.finish()]);
  } finally {
    leaveFrame();
  }
}

/** Resolved signature key for the failure report only; a target that is itself the problem must not shadow the real failure. */
function signatureKeyFor(target: Target): string | undefined {
  try { return signatureKeyOf(normalizeSignature(target)); }
  catch { return undefined; }
}
