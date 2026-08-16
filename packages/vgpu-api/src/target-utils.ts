import { clearColorInvalidError, surfaceSampleCountError, targetSampleCountConflictError, targetSampleCountError, targetSizeRequiredError, targetStencilOnlyDepthError, unsupportedError } from "./errors.ts";
import type { Target, TargetOptions, TargetTextureOptions } from "./target.ts";

export const DEFAULT_FORMAT: GPUTextureFormat = "rgba8unorm";
export type ClearColor = GPUColor | readonly [number, number, number, number];

/** Clear color used by a pass when neither the pass nor the target chose one. */
export const BUILT_IN_CLEAR_COLOR: ClearColor = Object.freeze([0, 0, 0, 1]);

/**
 * Validates a clear color on assignment (`target.clearColor`, `surface(gpu, canvas, { clearColor })`),
 * so a typo lands on the line that wrote it instead of inside a render pass descriptor.
 */
export function validateClearColor(value: ClearColor, where: string): ClearColor {
  const object = value as { r?: unknown; g?: unknown; b?: unknown; a?: unknown };
  const components = Array.isArray(value) ? value : [object?.r, object?.g, object?.b, object?.a];
  if (components.length !== 4 || !components.every((component) => typeof component === "number" && Number.isFinite(component))) throw clearColorInvalidError(where);
  return copyClearColor(value);
}

/** Defensive copy: clear colors are mutable WebGPU dictionaries/arrays at runtime. */
export function copyClearColor(value: ClearColor): ClearColor {
  const object = value as { r?: number; g?: number; b?: number; a?: number };
  return Array.isArray(value) ? [value[0], value[1], value[2], value[3]] : { r: object.r!, g: object.g!, b: object.b!, a: object.a! };
}

export interface TargetDeviceCaps {
  readonly isCompatibilityMode?: boolean;
}

export function colorSpecsFor(options: TargetTextureOptions): readonly { readonly format: GPUTextureFormat }[] {
  return options.colors ?? [{ format: options.format ?? DEFAULT_FORMAT }];
}

export function depthFormatFor(options: TargetTextureOptions): GPUTextureFormat | undefined {
  return options.depth === true ? "depth24plus" : options.depth || undefined;
}

/**
 * Sample count of a target. `sampleCount: 1 | 4` is the WebGPU platform spelling — the same vocabulary
 * `surface()` already uses — and `msaa: boolean | 4` is its legacy 0.3.0 spelling, kept alive so the
 * corpus that spells `msaa: true` keeps compiling and rendering identically. They describe one option,
 * so supplying both with different meanings is a conflict rather than a precedence puzzle.
 */
export function sampleCountFor(options: TargetTextureOptions): 1 | 4 {
  const msaa = options.msaa as unknown;
  const sampleCount = (options as { readonly sampleCount?: unknown }).sampleCount;
  if (sampleCount !== undefined) {
    const resolved = targetSampleCountFor(sampleCount);
    if (msaa !== undefined && msaaSampleCountFor(msaa) !== resolved) throw targetSampleCountConflictError(msaa, sampleCount);
    return resolved;
  }
  return msaaSampleCountFor(msaa);
}

function targetSampleCountFor(sampleCount: unknown): 1 | 4 {
  if (sampleCount === 4) return 4;
  if (sampleCount === 1) return 1;
  throw targetSampleCountError(sampleCount);
}

function msaaSampleCountFor(msaa: unknown): 1 | 4 {
  if (msaa === true || msaa === 4) return 4;
  if (msaa === undefined || msaa === false) return 1;
  const e = targetSizeRequiredError();
  (e as { code: string }).code = "VGPU-TARGET-MSAA-INVALID";
  e.message = `msaa received ${msaa}; WebGPU 1|4; use true`;
  throw e;
}

/**
 * Sample count of a surface, from its own `sampleCount` option — WebGPU platform vocabulary, so a
 * surface spells the count itself (`1 | 4`) instead of `Target`'s boolean `msaa`. The two spellings
 * meet again in `TargetSignature.sampleCount`, which is what pipelines are keyed by.
 */
export function surfaceSampleCountFor(options: { readonly sampleCount?: 1 | 4 }): 1 | 4 {
  const sampleCount = options.sampleCount as unknown;
  if (sampleCount === 4) return 4;
  if (sampleCount === undefined || sampleCount === 1) return 1;
  throw surfaceSampleCountError(sampleCount);
}

/**
 * Attachment description of a surface derived from its configuration alone — the depth format the
 * surface owns and the sample count it renders at. Shared by the surface itself and by anything that
 * needs its pipeline signature without touching `getCurrentTexture()`.
 */
export function surfaceAttachmentsFor(options: { readonly depth?: boolean | GPUTextureFormat; readonly sampleCount?: 1 | 4 }): { readonly depth: GPUTextureFormat | undefined; readonly sampleCount: 1 | 4 } {
  const depth = depthFormatFor({ depth: options.depth });
  // Same rule as validateTargetOptions: the default depth state (depthWriteEnabled: true) cannot compile
  // against a stencil-only format, which has no depth aspect.
  if (depth === "stencil8") throw targetStencilOnlyDepthError(depth);
  return { depth, sampleCount: surfaceSampleCountFor(options) };
}

export function validateTargetOptions(options: Partial<TargetOptions> | undefined, caps: TargetDeviceCaps): void {
  if (!options?.size) throw targetSizeRequiredError();
  const depthFormat = depthFormatFor(options);
  // Stencil-only formats have no depth aspect, so the default depth state (depthWriteEnabled: true) cannot compile against them.
  if (depthFormat === "stencil8") throw targetStencilOnlyDepthError(depthFormat);
  if (sampleCountFor(options) !== 4) return;
  for (const spec of colorSpecsFor(options)) validateMsaaFormat(spec.format, caps);
}

function validateMsaaFormat(format: GPUTextureFormat, caps: TargetDeviceCaps): void {
  if (!(caps.isCompatibilityMode && format === "rgba16float")) return;
  throw unsupportedError(
    "target",
    "Dawn compatibility mode does not support rgba16float+msaa.",
    "Use rgba8unorm for MSAA here, or disable msaa.",
  );
}

/**
 * **Transient attachments are automatic, not a flag.** No public `transient` option: a transient
 * attachment cannot be sampled or copied, and a public flag invites exactly that misuse.
 * `surface()`/`target()` may allocate an internally-managed attachment with `TRANSIENT_ATTACHMENT`
 * (feature-detected) **only when the attachment satisfies all four conditions at allocation time**:
 * (1) vgpu allocated it and it is **unreachable through every public accessor** (`.color`,
 * `.colors`, `.depth`, `read()`, `readFloats()`, and any binding path); (2) it is used
 * **exclusively** as a render attachment — never sampled, copied, or bound; (3) **every** pass that
 * can write it uses `storeOp: "discard"` / `depthStoreOp: "discard"`; (4) **no public option can
 * cause it to be loaded** (`preserve` / `loadOp: "load"` / `depthReadOnly`). The predicate must be
 * decidable when the texture is created — usage flags are immutable, so "nobody will preserve it
 * later" is not something the implementation may assume.
 *
 * Under this tree's shape the only candidate is the **intermediate multisample color attachment**
 * (`view: msaa.createView()` below, `storeOp: "discard"` unconditionally; `target-offscreen.ts`'s
 * `#createMsaaColors()` and `surface.ts`'s `#msaaColor`, both allocated `render_attachment`-only and
 * exposed by no getter). Conditions 1, 2 and 3 hold for it — **condition 4 does not**: `preserve`
 * puts `loadOp: "load"` on that very `view` (and `depthLoadOp: "load"` on a multisample depth aspect
 * in `depthAttachment()` below), i.e. a public option loads an attachment whose `storeOp` was
 * `"discard"`. That is bug **#323**, which the authoritative design leaves explicitly out of scope
 * ("this design only prevents the class going forward, through the transient predicate's four
 * decidable conditions"). So **this tree applies `TRANSIENT_ATTACHMENT` nowhere**: applying it while
 * `preserve` can still reach the candidate would turn a silent contents bug into native validation
 * failure or worse, and the design does not require the optimization — contract #22 is negative
 * ("Applying the flag is **not** a required behavior — the required behavior is that it never
 * appears on an observable attachment"), and it is verified as such in
 * `tests/transient-attachment.test.ts`.
 *
 * Explicitly **not** transient regardless of #323: a `depth: true` attachment (single-sample OR
 * multisample) fails condition 1 — `target.depth` is public and allocated `texture_binding` — and a
 * target's `.color`/`.colors` and a surface's canvas texture fail conditions 1 and 2.
 */
export function colorAttachment(resolved: { createView(): GPUTextureView }, msaa: { createView(): GPUTextureView } | undefined, clear: ClearColor, preserve?: boolean): GPURenderPassColorAttachment {
  const attachment: GPURenderPassColorAttachment = {
    view: (msaa ?? resolved).createView(),
    resolveTarget: msaa ? resolved.createView() : undefined,
    loadOp: preserve ? "load" : "clear",
    storeOp: msaa ? "discard" : "store",
  };
  if (!preserve) attachment.clearValue = colorValue(clear);
  return attachment;
}

export function depthAttachment(depth: { createView(): GPUTextureView; readonly sampleCount?: number; readonly format?: GPUTextureFormat }, preserve?: boolean, clearDepth?: number, clearStencil?: number, readOnly?: boolean): GPURenderPassDepthStencilAttachment {
  if (readOnly) {
    // WebGPU requires the ops to be OMITTED for read-only aspects: "If format has a depth aspect and
    // this.depthReadOnly is false: this.depthLoadOp must be provided. this.depthStoreOp must be provided.
    // Otherwise: this.depthLoadOp must not be provided. this.depthStoreOp must not be provided." — and the
    // same for stencilLoadOp/stencilStoreOp with stencilReadOnly, so combined formats mark both aspects.
    const attachment: GPURenderPassDepthStencilAttachment = { view: depth.createView(), depthReadOnly: true };
    if (hasStencilAspect(depth.format)) attachment.stencilReadOnly = true;
    return attachment;
  }
  const attachment: GPURenderPassDepthStencilAttachment = { view: depth.createView(), depthLoadOp: preserve ? "load" : "clear", depthStoreOp: depth.sampleCount! > 1 ? "discard" : "store" };
  if (!preserve) attachment.depthClearValue = clearDepth ?? 1;
  // WebGPU requires stencilLoadOp/stencilStoreOp whenever the format has a stencil aspect and stencilReadOnly is not set.
  if (depth.format && hasStencilAspect(depth.format)) {
    attachment.stencilLoadOp = preserve ? "load" : "clear";
    attachment.stencilStoreOp = depth.sampleCount! > 1 ? "discard" : "store";
    if (!preserve) attachment.stencilClearValue = clearStencil ?? 0;
  }
  return attachment;
}

export function hasStencilAspect(format: GPUTextureFormat | undefined): boolean {
  return !!format && format.includes("stencil");
}

export function colorValue(clear: ClearColor): GPUColor {
  return Array.isArray(clear) ? { r: clear[0], g: clear[1], b: clear[2], a: clear[3] } : clear;
}

export function sameSize(a: readonly [number, number], b: readonly [number, number]): boolean {
  return a[0] === b[0] && a[1] === b[1];
}


/** @internal Internal normalization guard: `renderPassDescriptor` is required on Target and never on options bags. */
export function isTarget(value: unknown): value is Target {
  return typeof value === "object" && value !== null
    && typeof (value as Target).renderPassDescriptor === "function";
}
