/**
 * `prepare()` — the one spelling for readiness.
 *
 * **Readiness is a property of a combination, never of an object.** The unit of preparation is
 * `(renderable, target signature)`: a `Draw` can be ready for the screen, uncompiled for an HDR
 * target and failed for a `depth24plus-stencil8` target at the same time. No `Draw`/`Effect`/
 * `Compute` carries `ready`/`pending`/`failed` fields, and no per-target status map is exposed:
 * `prepare()` **is** the query — idempotent, O(1) on an already-prepared combination, and it
 * rejects with the failure. Every other pipeline-affecting input (geometry vertex layout,
 * topology/strip format, cull/front face, depth/stencil/multisample/blend state, pipeline
 * constants, entry points) is fixed at construction of the renderable, so `(renderable, signature)`
 * is a complete key — that is why `prepare()` needs no `geometry:` or state axis. The internal
 * pipeline cache key is **not** public API (it embeds process-local module/layout ids).
 *
 * **`prepare()` request keys mirror the encode call they warm.** `p.draw(Draw | Effect)` →
 * `{ draw, target }`; `f.compute(c)` → `{ compute }`; `p.bundles(b)` → `{ bundle }`. There is no
 * `{ effect }` branch (the `draw:` key takes `Draw | Effect`, exactly like `p.draw()`) and no
 * `render:` spelling (no encode API is called `render` at object granularity). A `{ bundle }`
 * request carries no `target` — a bundle froze its target signature at construction.
 */
import { prepareBundle } from "./bundle.ts";
import { ComputePipeline } from "./compute.ts";
import type { Draw, InternalDraw } from "./draw.ts";
import { effectDraw, InternalEffect, type Effect } from "./effect.ts";
import type { Bundle } from "./bundle.ts";
import type { Compute } from "./api-types.ts";
import type { Gpu } from "./kernel.ts";
import type { CompileTarget, TargetSignature } from "./target.ts";
import { normalizeSignature, signatureKeyOf } from "./pipeline-store.ts";
import { prepareFailedError, unsupportedError, type PrepareFailure } from "./errors.ts";
import { liveKernel } from "./live-kernel.ts";

export type PrepareRequest =
  | { readonly draw: Draw | Effect; readonly target: CompileTarget }
  | { readonly compute: Compute }
  | { readonly bundle: Bundle };

export type PreparedDraw = { readonly draw: Draw | Effect; readonly signature: TargetSignature; readonly gpu: GPURenderPipeline };
export type PreparedCompute = { readonly compute: Compute; readonly gpu: GPUComputePipeline };
export type PreparedBundle = { readonly bundle: Bundle; readonly signature: TargetSignature; readonly gpu: GPURenderBundle };

export type PreparedFor<R> =
  R extends { compute: Compute } ? PreparedCompute :
  R extends { bundle: Bundle } ? PreparedBundle : PreparedDraw;

/**
 * Compiles every requested combination and hands back one handle per request.
 *
 * ```ts
 * const [cube, sim] = await prepare(gpu, [
 *   { draw: cubeDraw, target: screen },
 *   { compute: simulation },
 * ]);
 *
 * cube.draw;       // the renderable, echoed back — the handle identifies the combination
 * cube.signature;  // resolved TargetSignature { colors, depth, sampleCount }
 * cube.gpu;        // GPURenderPipeline — the same-agent low-level escape hatch
 * sim.gpu;         // GPUComputePipeline (a compute request has no target signature)
 * ```
 *
 * Rules:
 * - **Handles are data, not state.** There is no `prepared.status`: `prepare()` **rejects** if any
 *   requested combination fails, with `VGPU-PREPARE-FAILED` enumerating every failure (renderable
 *   label + resolved signature + `cause`). Combinations that did compile stay cached, so
 *   re-preparing the good subset is free. Ignoring the result stays valid — the happy path is
 *   unchanged.
 * - The array form preserves request order and tuple inference; the single-request form returns one
 *   handle.
 * - `prepared.gpu` is the low-level pipeline escape hatch, and it is same-agent only — like every
 *   device-local object.
 *
 * A `{ draw, target }` request against a `Surface` is legal **outside** `frame()`: a surface
 * signature comes from its configuration, not from the current presentation texture, so the
 * signature resolved here is the very one the encode path uses inside a frame.
 */
// Overload order is public API surface (see the same note on `effect`/`compute`): the single-object
// form is declared first so it is the one call sites read first; an array literal never matches it,
// so call resolution is unambiguous either way.
export function prepare<const R extends PrepareRequest>(gpu: Gpu, request: R): Promise<PreparedFor<R>>;
export function prepare<const R extends readonly PrepareRequest[]>(gpu: Gpu, requests: R): Promise<{ [K in keyof R]: PreparedFor<R[K]> }>;
export async function prepare(gpu: Gpu, request: PrepareRequest | readonly PrepareRequest[]): Promise<unknown> {
  // Same entry guard as every other factory: a disposed gpu fails here, not deep inside a driver call.
  liveKernel(gpu, "prepare");
  const requests: readonly PrepareRequest[] = Array.isArray(request) ? request : [request as PrepareRequest];
  // Parallel, and settled rather than raced: a batch of combinations fails as a batch, so every
  // failure must be reported — not just whichever one happened to reject first.
  const settled = await Promise.allSettled(requests.map((entry) => prepareOne(entry)));
  const failures: PrepareFailure[] = [];
  settled.forEach((result, index) => {
    if (result.status === "rejected") failures.push({ label: labelOf(requests[index]!), signature: signatureKeyFor(requests[index]!), cause: result.reason });
  });
  if (failures.length) throw prepareFailedError(failures);
  const handles = settled.map((result) => (result as PromiseFulfilledResult<unknown>).value);
  return Array.isArray(request) ? handles : handles[0];
}

async function prepareOne(request: PrepareRequest): Promise<PreparedDraw | PreparedCompute | PreparedBundle> {
  if ("draw" in request) {
    const drawable = request.draw;
    const internal = drawable instanceof InternalEffect ? effectDraw(drawable) : drawable as InternalDraw;
    // Deliberately the draw's own async resolution path — same signature normalization, same
    // pipeline key, same store dedupe. `prepare()` is a second door to it, never a second compile.
    const gpu = await internal.pipelineForAsync(request.target);
    return { draw: drawable, signature: normalizeSignature(request.target), gpu };
  }
  if ("compute" in request) {
    return { compute: request.compute, gpu: await asComputePipeline(request.compute).prepareCombination() };
  }
  if ("bundle" in request) {
    // Genuinely awaited: preparing a bundle pre-warms the pipelines of every draw it recorded
    // through the async path, and only then encodes the native bundle.
    const { signature, gpu } = await prepareBundle(request.bundle);
    return { bundle: request.bundle, signature, gpu };
  }
  throw unsupportedError("prepare", "a prepare() request must be { draw, target }, { compute } or { bundle }.");
}

function asComputePipeline(compute: Compute): ComputePipeline {
  if (compute instanceof ComputePipeline) return compute;
  throw unsupportedError("prepare", "prepare({ compute }) received an object this library did not create; pass the compute returned by compute(gpu, ...).");
}

/** Best-effort label for the failure list; never the reason a report is lost. */
function labelOf(request: PrepareRequest): string {
  if ("draw" in request) {
    const drawable = request.draw;
    return (drawable instanceof InternalEffect ? effectDraw(drawable) : drawable as InternalDraw).label ?? "draw";
  }
  if ("compute" in request) return (request.compute as Partial<ComputePipeline>).label ?? "compute";
  if ("bundle" in request) return request.bundle?.id ?? "bundle";
  return "request";
}

/**
 * Resolved signature key of a failed combination, for the report only. A request whose target is
 * itself the problem cannot name a signature, and that must not shadow the real failure.
 */
function signatureKeyFor(request: PrepareRequest): string | undefined {
  if ("draw" in request) {
    try { return signatureKeyOf(normalizeSignature(request.target)); }
    catch { return undefined; }
  }
  if ("bundle" in request) return undefined;
  // A compute pipeline has no target signature — asymmetry by platform fact, not inconsistency.
  return undefined;
}
