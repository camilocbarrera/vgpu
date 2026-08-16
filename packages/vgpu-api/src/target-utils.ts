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
