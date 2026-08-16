import { createRenderBundle } from "./core/render-bundle.ts";
import { InternalDraw, drawUsesBlendConstant, drawUsesStencilReference, encodeDraw, registerDrawBundle, unregisterDrawBundle, type BundleBackReference, type BundleStaleEvent, type Draw, type DrawCallOptions } from "./draw.ts";
import { InternalEffect, effectDraw, type Effect } from "./effect.ts";
import type { CompileTarget, Target, TargetSignature } from "./target.ts";
import { normalizeSignature, signatureKeyOf, validateTargetSignature } from "./pipeline-store.ts";
import { bundleBlendConstantError, bundleDisposedError, bundleStencilReferenceError, compileFailedError, pipelinePendingError, surfaceNotInFrameError, VGPUError } from "./errors.ts";
import { isFrameActive, isSurface } from "./surface.ts";
import { FRAME_BUNDLE, type FrameBundleProtocol } from "./frame-protocols.ts";
import { assertDeviceUsable } from "./lifecycle.ts";
import { liveKernel } from "./live-kernel.ts";
import type { PendingPipelines } from "./pending-pipelines.ts";
import type { Gpu } from "./kernel.ts";

/**
 * Records an explicit WebGPU render bundle: the draws in `record` are encoded once and replayed
 * with `pass.bundles(bundle)`.
 *
 * Recording needs only formats — never `getCurrentTexture()` — so `bundle(gpu, { target: surface },
 * …)` is legal outside `frame()`, exactly like `prepare()`. Construction captures the **logical**
 * command list and compiles nothing: the bundle is born `"pending-pipelines"` and
 * `prepare(gpu, [{ bundle }])` materializes it. Construction never throws for pending pipelines.
 *
 * A bundle freezes its commands, its bind groups and the target signature it was recorded for; the
 * recorded state is re-checked at replay and a target-signature mismatch throws
 * `VGPU-R3-BUNDLE-STALE` instead of drawing something stale.
 */
export function bundle(gpu: Gpu, opts: BundleOptions, record: (recorder: BundleRecorder) => void): Bundle {
  const kernel = liveKernel(gpu, "bundle");
  // Same wiring pattern the other kernel-backed features use (`render-service.ts:31-34`): the
  // feature never reaches for the `Gpu`, it takes exactly the three kernel capabilities it needs —
  // the device, the last link of the pendingPipelines chain, and the error channel.
  return createBundle({
    device: kernel.device,
    pendingPipelinesDefault: () => kernel.pendingPipelinesDefault(),
    reportError: (error) => { void kernel.reportError(error); },
    trackSettled: (promise) => { void kernel.trackDelivery(promise); },
    assertDeviceUsable: (where) => assertDeviceUsable(kernel.device, where),
  }, opts, record);
}

export interface BundleOptions {
  readonly target: CompileTarget;
  readonly label?: string;
}

export interface BundleRecorder {
  draw(drawable: Draw | Effect, opts?: DrawCallOptions): void;
}

/**
 * Observable state of a recorded bundle — a **frozen** union (contract #15).
 *
 * Semantics: `pending-pipelines` = pipelines still to compile (and the native bundle to encode);
 * `stale` = pipelines compiled, the native bundle must be re-encoded **from the retained logical
 * recording** — that re-encode is `prepare(gpu, [{ bundle }])` (or an explicit `"sync"` replay);
 * `rebuild()` is never required to leave `stale`. Every public transition:
 *
 * | From | Event | To |
 * |---|---|---|
 * | — | `bundle(gpu, { target }, rec)` | `pending-pipelines` (always: the native bundle is not materialized at construction) |
 * | `pending-pipelines` \| `stale` \| `failed` | `prepare(gpu, [{ bundle }])` succeeds | `ready` (compiles missing pipelines, then encodes the native bundle) |
 * | any non-`disposed` | `prepare()` fails | `failed`, retaining the error in `bundle.error` |
 * | `ready` | `.set()` byte update on a captured resource | `ready` (no transition) |
 * | `ready` | `.bind()` identity update, or a captured `Target` recreating its texture (resize) | `stale` |
 * | `pending-pipelines` \| `stale` \| `failed` | identity update | unchanged |
 * | any non-`disposed` | `rebuild(cb)` (synchronous) | Clear `bundle.error`, then move to `stale`, or `pending-pipelines` if the new recording introduces an uncompiled combination. `rebuild()` replaces the logical command list; it never compiles and never throws for readiness reasons |
 * | any | `dispose()` | `disposed` (terminal; releases the native bundle and the retained recording; a second `dispose()` is a no-op) |
 *
 * `prepare()` on a `ready` bundle is a no-op. **`prepare()` always retries a `failed` bundle** —
 * there is no "retryable failure" classification; during that retry `bundle.error` remains
 * available and the status stays `failed`, then success clears the error and moves to `ready`.
 * `rebuild()` clears the retained error immediately because it replaces the logical recording to
 * which that failure belonged. `bundle.error?: VGPUError` is defined only while status is `failed`.
 *
 * **Replay of a non-`ready` bundle** follows the one `pendingPipelines` chain (frame → gpu for
 * bundles; `p.bundles()` takes no per-call options):
 * - `"throw"`: `pending-pipelines` → `VGPU-PIPELINE-PENDING`; `stale` → `VGPU-R3-BUNDLE-STALE`
 *   (message names `prepare(gpu, { bundle })`; no `rebuild()` is required); `failed` → the retained
 *   error is rethrown (`cause` preserved).
 * - `"skip"`: the bundle is skipped this frame; `pending-pipelines` starts/continues async
 *   compilation; `failed` is reported once via `gpu.onError` and keeps being skipped; `stale` is
 *   skipped — never silently re-encoded.
 * - `"sync"`: do the pending work inline — `pending-pipelines`: compile + encode (the stall this
 *   feature exists to avoid); `stale`: re-encode from the retained recording (bounded: no
 *   compilation — this is the opt-in auto-heal); `failed`: does **not** retry, the retained error is
 *   thrown (retry is only through `prepare()`).
 * - `disposed` **always** throws `VGPU-BUNDLE-DISPOSED`, under every policy.
 */
export type BundleStatus = "pending-pipelines" | "ready" | "stale" | "failed" | "disposed";

export interface Bundle {
  readonly id: string;
  /**
   * The native bundle, once it exists. `undefined` until the first successful `prepare()` (or
   * `"sync"` replay) encodes it — same semantics as `Draw.gpu`, which is only defined once that
   * combination compiled. A `disposed` bundle drops it back to `undefined`.
   */
  readonly gpu: GPURenderBundle | undefined;
  /** See {@link BundleStatus} for the full transition table. */
  readonly status: BundleStatus;
  /** The retained failure of the last `prepare()`; defined **only** while `status === "failed"`. */
  readonly error: VGPUError | undefined;
  /**
   * Replaces the logical command list, synchronously. It clears `error` immediately (the failure
   * belonged to the recording being replaced), never compiles and never throws for readiness
   * reasons: the bundle lands on `stale`, or on `pending-pipelines` when the new recording
   * introduces a combination that is not compiled yet. `rebuild()` is **not** how a `stale` bundle
   * becomes `ready` — that is `prepare(gpu, [{ bundle }])`.
   */
  rebuild(record: (recorder: BundleRecorder) => void): void;
  /** Terminal: releases the native bundle and the retained recording. A second call is a no-op. */
  dispose(): void;
}

/**
 * The kernel capabilities a bundle needs. Given explicitly (instead of the `Gpu`) so `bundle.ts`
 * stays testable and never reaches back into the kernel module's public surface.
 *
 * @internal
 */
export interface BundleHost {
  readonly device: { readonly gpu: GPUDevice };
  /** Last link of the `pendingPipelines` chain, for a replay whose frame named no policy. */
  pendingPipelinesDefault?(): PendingPipelines;
  /** `gpu.onError` channel: where a `"skip"` replay reports a `failed` bundle, once. */
  reportError?(error: VGPUError): void;
  /** Joins `gpu.settled()`: background preparation started by a `"skip"` replay is observable. */
  trackSettled?(promise: Promise<unknown>): void;
  /**
   * Device-state guard of the owning gpu (contract #19): throws `VGPU-DEVICE-LOST` after a real loss
   * and `VGPU-DEVICE-DISPOSED` after `gpu.dispose()`. Optional like the rest of this host, so a bundle
   * built over a hand-made host (tests) keeps working — the guard is a capability, not a requirement.
   */
  assertDeviceUsable?(where: string): void;
}

let nextBundleId = 1;
let recordingDepth = 0;

/** Records explicit WebGPU render bundles and keeps the R3 stale signature checked at replay time. */
export function createBundle(host: BundleHost, opts: BundleOptions, record: (recorder: BundleRecorder) => void): Bundle {
  const id = opts.label ?? `bundle${nextBundleId++}`;
  if (isSurface(opts.target) && !isFrameActive()) throw surfaceNotInFrameError("bundle");
  const signature = normalizeBundleSignature(opts.target);
  const bundle = new RecordedBundle(host, id, signature);
  bundle.track(record);
  return bundle;
}

class RecordedBundle implements Bundle, BundleBackReference {
  #gpu?: GPURenderBundle;
  #status: BundleStatus = "pending-pipelines";
  #error?: VGPUError;
  #reportedFailure = false;
  #staleEvent?: BundleStaleEvent;
  /**
   * The logical recording, retained. `record()` used to take the closure and forget it, which made
   * re-encoding structurally impossible; keeping it is what lets `prepare(gpu, [{ bundle }])`
   * rebuild the native bundle from the same commands with the current resources.
   */
  #record?: (recorder: BundleRecorder) => void;
  /**
   * Bumped by every `rebuild()`. A `prepare()` in flight compares it after its `await`: if the
   * logical recording was replaced while pipelines were compiling, the prepare re-runs the warm-up
   * for the NEW recording instead of encoding a command list nobody asked for any more.
   */
  #generation = 0;
  /** The in-flight `prepare()`, so the same bundle requested twice in one batch does the work once. */
  #preparing?: Promise<GPURenderBundle>;
  readonly #signatureKey: string;
  #draws = new Set<InternalDraw>();

  constructor(private readonly host: BundleHost, readonly id: string, readonly signature: TargetSignature) {
    this.#signatureKey = signatureKeyOf(signature);
  }

  get gpu(): GPURenderBundle | undefined { return this.#gpu; }
  get status(): BundleStatus { return this.#status; }
  get error(): VGPUError | undefined { return this.#status === "failed" ? this.#error : undefined; }

  /**
   * Frame bundle protocol: `pass.bundles()` replays through this, so `frame.ts` never imports
   * bundle.ts nor learns what a `BundleStatus` is.
   */
  get [FRAME_BUNDLE](): FrameBundleProtocol { return this; }

  /**
   * Construction (and `rebuild()`): run the closure against a tracking-only recorder to learn which
   * draws the recording names, without opening a native encoder and without compiling anything.
   *
   * @internal
   */
  track(record: (recorder: BundleRecorder) => void): void {
    this.#record = record;
    // A brand new recording starts from an empty draw set, and a draw the previous recording named
    // and this one does not is unregistered from its back-reference registry: forgetting it here is
    // not enough, because the registration is what makes the draw call markStale() on this bundle.
    // Leaving it in place stales this bundle forever from a draw it no longer encodes (and keeps a
    // dropped draw holding the bundle alive).
    const previous = this.#draws;
    this.#draws = new Set();
    this.#recordCommands(record);
    for (const draw of this.#draws) registerDrawBundle(draw, this);
    for (const draw of previous) if (!this.#draws.has(draw)) unregisterDrawBundle(draw, this);
  }

  /**
   * Async readiness for `prepare(gpu, [{ bundle }])`: pre-warms the pipelines of every recorded draw
   * through the draw's own `pipelineForAsync` — one async compile per missing combination, never a
   * synchronous `createRenderPipeline` — and only then encodes the native bundle.
   *
   * @internal
   */
  prepareCombination(): Promise<GPURenderBundle> {
    if (this.#status === "disposed") return Promise.reject(bundleDisposedError(this.id, "prepare"));
    // A lost/disposed device beats every readiness state, including the ready fast path below: there is
    // nothing to compile against and nothing to replay. Rejected, not thrown, so the async entry point
    // stays async for its callers.
    try { this.host.assertDeviceUsable?.(`bundle '${this.id}' prepare`); }
    catch (error) { return Promise.reject(error); }
    // prepare() on a ready bundle is a no-op: no new pipeline, no re-encode (contract #7, bundle half).
    if (this.#status === "ready" && this.#gpu) return Promise.resolve(this.#gpu);
    if (this.#preparing) return this.#preparing;
    const preparing = this.#prepareNow();
    this.#preparing = preparing;
    // Clearing the in-flight slot must not turn a handled rejection into an unhandled one, so both
    // outcomes are observed here while the original promise is what callers await.
    preparing.then(() => { if (this.#preparing === preparing) this.#preparing = undefined; }, () => { if (this.#preparing === preparing) this.#preparing = undefined; });
    return preparing;
  }

  async #prepareNow(): Promise<GPURenderBundle> {
    for (;;) {
      const generation = this.#generation;
      const settled = await Promise.allSettled([...this.#draws].map((draw) => draw.pipelineForAsync(this.signature)));
      // dispose() is terminal, including for work that was already in flight when it landed.
      if (this.#status === "disposed") throw bundleDisposedError(this.id, "prepare");
      // So is a device loss that landed mid-compile: encoding the bundle now would hand back a handle
      // for a dead device. Thrown raw (not through #fail): the bundle did not fail, the device did.
      this.host.assertDeviceUsable?.(`bundle '${this.id}' prepare`);
      // rebuild() replaced the recording mid-flight: prepare() prepares whatever recording is
      // current when it finishes, so warm the new one instead of encoding the old command list.
      if (this.#generation !== generation) continue;
      const failure = settled.find((result): result is PromiseRejectedResult => result.status === "rejected");
      if (failure) throw this.#fail(failure.reason);
      // Every pipeline is warm now, so this encode hits the store's ready fast path: zero new
      // createRenderPipeline calls, which is what makes re-preparing a ready bundle free.
      try { return this.#materialize("sync"); }
      catch (cause) { throw this.#fail(cause); }
    }
  }

  /**
   * Replay resolution for `FramePass.bundles()`: the whole `BundleStatus` × `pendingPipelines`
   * table lives here, so the frame only ever sees "this is the native bundle to execute" or
   * "nothing to execute this frame".
   *
   * @internal
   */
  resolveForReplay(target: Target, policy?: PendingPipelines): GPURenderBundle | undefined {
    // Terminal beats policy: a disposed bundle has no native bundle to replay under any of them.
    if (this.#status === "disposed") throw bundleDisposedError(this.id, "replay");
    // Same for the device behind it — a lost device cannot execute even an already-encoded bundle.
    this.host.assertDeviceUsable?.(`bundle '${this.id}' replay`);
    // A different target signature is a usage error, not a readiness state: it is checked before the
    // policy and no policy can turn it into a skip or an inline re-encode.
    this.#assertSignature(target);
    if (this.#status === "ready" && this.#gpu) return this.#gpu;
    const resolved = policy ?? this.host.pendingPipelinesDefault?.() ?? "sync";
    if (this.#status === "failed") {
      // "skip" reports it once and keeps skipping; "throw"/"sync" rethrow. Retry is only prepare().
      if (resolved === "skip") { this.#reportFailureOnce(); return undefined; }
      throw this.#error;
    }
    if (resolved === "sync") return this.#materialize("sync");
    if (resolved === "skip") {
      // A stale bundle is never silently re-encoded; a pending one continues compiling in background.
      if (this.#status === "pending-pipelines") this.#startBackgroundPrepare();
      return undefined;
    }
    if (this.#status === "stale") throw bundleStaleError(this.id, staleEventMessage(this.id, this.#staleEvent));
    throw pipelinePendingError(`bundle '${this.id}' replay`, this.id, this.#signatureKey);
  }

  rebuild(record: (recorder: BundleRecorder) => void): void {
    if (this.#status === "disposed") throw bundleDisposedError(this.id, "rebuild");
    this.host.assertDeviceUsable?.(`bundle '${this.id}' rebuild`);
    this.#generation += 1;
    // The retained error belonged to the recording being replaced, so it goes away with it.
    this.#error = undefined;
    this.#reportedFailure = false;
    this.#staleEvent = undefined;
    this.track(record);
    // Never compiles, never encodes: the last native bundle stays the last valid one until a
    // prepare() (or a "sync" replay) re-encodes the new command list.
    this.#status = [...this.#draws].every((draw) => draw.readyForSignature(this.signature)) ? "stale" : "pending-pipelines";
  }

  dispose(): void {
    if (this.#status === "disposed") return;
    this.#status = "disposed";
    // WebGPU has no GPURenderBundle.destroy(): releasing the native bundle IS dropping the last
    // reference to it, together with the retained recording and the tracked draws.
    this.#gpu = undefined;
    this.#record = undefined;
    this.#error = undefined;
    this.#staleEvent = undefined;
    // Every draw drops its back-reference to this bundle: a disposed bundle must not be reachable
    // from a live draw (that was a leak, and markStale() traffic for nothing).
    for (const draw of this.#draws) unregisterDrawBundle(draw, this);
    this.#draws = new Set();
  }

  markStale(event: BundleStaleEvent): void {
    if (recordingDepth > 0) return;
    // Only a ready bundle can go stale: on pending-pipelines/stale/failed the status is unchanged,
    // because the encode that is still owed will read the current resources anyway.
    if (this.#status !== "ready") return;
    this.#staleEvent = event;
    this.#status = "stale";
  }

  /** @internal Populated by the recorders: the draws the current logical recording names. */
  remember(draw: InternalDraw): void {
    this.#draws.add(draw);
  }

  #assertSignature(target: Target): void {
    const actual = normalizeBundleSignature(target);
    const actualKey = signatureKeyOf(actual);
    if (this.#signatureKey !== actualKey) throw bundleStaleError(this.id, targetSignatureStaleMessage(this.id, this.#signatureKey, actualKey));
  }

  /**
   * Runs the retained recording against a real `GPURenderBundleEncoder`. `policy` fills in for the
   * recorded draws that named none, so a `"sync"` replay may compile inline while the prepare path
   * (whose pipelines are already warm) never does.
   *
   * Encoding is synchronous from end to end, so no identity change can interleave with it: an event
   * that arrived while a `prepare()` was awaiting its pipelines is already reflected by the bind
   * groups this encode reads, which is why clearing `#staleEvent` here cannot lose an update.
   */
  #materialize(policy?: PendingPipelines): GPURenderBundle {
    const record = this.#record;
    if (!record) throw bundleDisposedError(this.id, "replay");
    const gpu = createRenderBundle(this.host.device, {
      label: this.id,
      colorFormats: this.signature.colors,
      depthStencilFormat: this.signature.depth,
      sampleCount: this.signature.sampleCount ?? 1,
      record: (recorder) => this.#recordCommands(record, recorder.gpu as unknown as GPURenderPassEncoder, policy),
    });
    this.#gpu = gpu;
    this.#staleEvent = undefined;
    this.#error = undefined;
    this.#reportedFailure = false;
    this.#status = "ready";
    return gpu;
  }

  #recordCommands(record: (recorder: BundleRecorder) => void, encoder?: GPURenderPassEncoder, policy?: PendingPipelines): void {
    recordingDepth += 1;
    try { record(new BundleRecording(this, encoder, policy)); }
    finally { recordingDepth -= 1; }
  }

  #fail(cause: unknown): VGPUError {
    const error = compileFailedError(`bundle '${this.id}' prepare`, cause, this.#signatureKey);
    this.#status = "failed";
    this.#error = error;
    this.#reportedFailure = false;
    return error;
  }

  /**
   * Background preparation for a `"skip"` replay. Local to this module on purpose: the skip path
   * needs nothing new from the pipeline store, only the same async warm-up `prepare()` runs.
   */
  #startBackgroundPrepare(): void {
    if (this.#preparing) return;
    const promise = this.prepareCombination().then(() => undefined, () => { this.#reportFailureOnce(); });
    this.host.trackSettled?.(promise);
  }

  #reportFailureOnce(): void {
    if (this.#reportedFailure) return;
    const error = this.#error;
    if (!error) return;
    this.#reportedFailure = true;
    this.host.reportError?.(error);
  }
}

/**
 * The one recorder, in its two modes. Without an `encoder` it is the **tracking** pass construction
 * and `rebuild()` run: it resolves which `InternalDraw`s the closure names and nothing else — no
 * native encoder is opened, no command is encoded, no pipeline is compiled, which is what
 * "construction records the logical command list without compiling" means. With an encoder it is the
 * real recording the materialization performs. Both modes reject the two commands a render bundle
 * encoder cannot express, so a user's mistake is still reported where the user wrote it.
 */
class BundleRecording implements BundleRecorder {
  constructor(private readonly bundle: RecordedBundle, private readonly encoder?: GPURenderPassEncoder, private readonly policy?: PendingPipelines) {}

  draw(drawable: Draw | Effect, opts: DrawCallOptions = {}): void {
    // Blend/writeMask are constructor-only draw pipeline state. If they ever become mutable or per-call,
    // bundles need a new staleness dimension beyond the target signature checked at replay.
    const draw = drawable instanceof InternalEffect ? effectDraw(drawable) : drawable as InternalDraw;
    // The blend constant is render-pass state; GPURenderBundleEncoder has no setBlendConstant, so reject at recording.
    if (drawUsesBlendConstant(draw)) throw bundleBlendConstantError(this.bundle.id, draw.label);
    // Likewise the stencil reference: GPURenderBundleEncoder has no setStencilReference. Stencil pipeline state without ref records fine.
    if (drawUsesStencilReference(draw)) throw bundleStencilReferenceError(this.bundle.id, draw.label);
    this.bundle.remember(draw);
    if (!this.encoder) return;
    // The replay's resolved policy fills in for a recorded draw that named none — the same collapse
    // of the chain `FramePass.draw()` does, one link later.
    const resolved = opts.pendingPipelines === undefined && this.policy !== undefined ? { ...opts, pendingPipelines: this.policy } : opts;
    encodeDraw(draw, this.encoder, this.bundle.signature, resolved);
  }
}

/**
 * Compiles the pipelines of the recorded draws, encodes the native bundle and reports the handle
 * `prepare()` hands back. Genuinely async: this is the one spelling for bundle readiness.
 *
 * @internal
 */
export async function prepareBundle(bundle: Bundle): Promise<{ readonly signature: TargetSignature; readonly gpu: GPURenderBundle }> {
  if (!(bundle instanceof RecordedBundle)) throw new VGPUError({
    code: "VGPU-BUNDLE-FOREIGN",
    message: "prepare({ bundle }) received an object this library did not record.",
    fix: "Pass the bundle returned by bundle(gpu, { target }, record).",
    where: "prepare",
  });
  const gpu = await bundle.prepareCombination();
  return { signature: bundle.signature, gpu };
}

function normalizeBundleSignature(target: CompileTarget): TargetSignature {
  const signature = normalizeSignature(target);
  validateTargetSignature(signature, "bundle");
  return signature;
}

function targetSignatureStaleMessage(id: string, recordedKey: string, actualKey: string): string {
  return `bundle '${id}' is stale: the replay target signature does not match the recorded signature. Bundles freeze format/depth/sampleCount and bind groups.\n  Recorded signature: ${recordedKey}\n  Actual signature: ${actualKey}\n  Fix: re-record the bundle for this target → ${id} = bundle(gpu, { target: scene }, ...)\n  (re-recording is always your responsibility; the library only detects this).`;
}

/**
 * An identity change only made the native bundle stale: the logical recording is still valid, so the
 * fix names `prepare()` (the re-encode), never `rebuild()` (which replaces the commands).
 */
function staleEventMessage(id: string, event: BundleStaleEvent | undefined): string {
  const fix = `  Fix: re-encode the retained recording → await prepare(gpu, { bundle: ${id} })\n  (or pendingPipelines: "sync" to re-encode inline; rebuild() only replaces the recorded commands).`;
  if (!event) return `bundle '${id}' is stale: its native bundle must be re-encoded from the retained recording.\n${fix}`;
  if (event.kind === "group-claim") {
    return `bundle '${id}' is stale: group ${event.group} of draw\n  '${event.drawLabel}' changed bind group after recording. Bundles freeze commands and bind groups.\n${fix}`;
  }
  return `bundle '${id}' is stale: binding \`${event.bindingName}\` (@group(${event.group}) @binding(${event.binding})) of draw\n  '${event.drawLabel}' changed resource after recording. Bundles freeze commands and bind groups.\n${fix}`;
}

function bundleStaleError(id: string, message: string): VGPUError {
  return new VGPUError({ code: "VGPU-R3-BUNDLE-STALE", message, where: `bundle '${id}' replay` });
}
