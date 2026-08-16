import type { ClearColor } from "./target-utils.ts";
import type { Texture, ResourceDestroyCallback, ResourceIdentity, UnsubscribeResourceDestroy } from "@vgpu/core";

export interface TargetTextureOptions {
  readonly format?: GPUTextureFormat;
  /**
   * Default clear color of this target, used by passes that clear without naming a color.
   * Defaults to `[0, 0, 0, 1]`; mutable at runtime through `target.clearColor`.
   */
  readonly clearColor?: ClearColor;
  readonly colors?: readonly { readonly format: GPUTextureFormat }[];
  readonly depth?: boolean | GPUTextureFormat;
  /**
   * Sample count of the render pass this target opens: `4` renders into internal multisample color
   * attachments and resolves into the sampleable ones, `1` (the default) renders straight into them.
   * This is the WebGPU platform vocabulary `surface()` already uses — the one spelling of multisampling.
   */
  readonly sampleCount?: 1 | 4;
  /**
   * Legacy 0.3.0 spelling of {@link TargetTextureOptions.sampleCount} — `msaa: true` means
   * `sampleCount: 4`. Still honored so existing code keeps working; prefer `sampleCount`. Supplying both
   * with different meanings throws `VGPU-TARGET-SAMPLE-COUNT-CONFLICT`.
   */
  readonly msaa?: boolean | 4;
  readonly label?: string;
}

export interface TargetOptions extends TargetTextureOptions {
  readonly size: readonly [number, number];
}

export interface TargetSignature {
  readonly colors: readonly GPUTextureFormat[];
  readonly depth?: GPUTextureFormat;
  readonly sampleCount?: 1 | 4;
}

export type CompileTarget = RenderDestination | TargetSignature;

/** Options bag for `Target.renderPassDescriptor()`. `Frame.pass` supplies these from `FramePassOptions`. */
export interface RenderPassDescriptorOptions {
  /** Clear color for all color attachments unless `preserve` is true. Defaults to `[0, 0, 0, 1]`. */
  readonly clear?: ClearColor;
  /** When true, color and depth attachments load existing contents and omit clear values. */
  readonly preserve?: boolean;
  /** Depth clear value used when the pass clears. Defaults to `1`. */
  readonly clearDepth?: number;
  /** Stencil clear value used when the pass clears. Defaults to `0`. */
  readonly clearStencil?: number;
  /** Builds the depth-stencil attachment read-only, omitting its load/store ops (stencil aspect included). */
  readonly depthReadOnly?: boolean;
}

/**
 * What `Target` and `Surface` share: a place a render pass can draw into.
 *
 * **Normative distinction** (design §4c). Both are *render destinations*: accepted by `f.pass()` and by
 * `prepare()` as a target signature. `RenderDestination` is the shared supertype — size, formats, sample
 * count, clear color, resize, render-pass descriptor.
 * - **{@link Target}** additionally has **persistent resource identity**: it exposes `.color`/`.colors`/
 *   `.depth`, may be used as a texture binding, and auto-heals bindings across resize/recreation.
 * - **`Surface` is presentation-only and is NOT a texture binding.** `Surface` does **not** extend
 *   `Target`; passing one as a binding fails with `VGPU-SURFACE-NOT-BINDABLE`.
 * - **Resize invalidation is by signature, not by identity.** A resize that preserves color format, depth
 *   format and sample count does **not** invalidate a prepared pipeline or a render bundle — only a change
 *   to that render-pass signature invalidates the corresponding prepared combinations.
 *
 * The design rule that follows from it: **bind the `Target`, not its texture snapshot.**
 */
export interface RenderDestination {
  readonly gpu: unknown;
  readonly size: readonly [number, number];
  readonly texelSize: readonly [number, number];
  readonly format: GPUTextureFormat;
  readonly sampleCount: 1 | 4;
  /**
   * Default clear color of this target: the color a pass uses when it clears without naming one
   * (`pass(target, body)` or `clear: true`). Writable at runtime, validated on assignment.
   * Precedence: pass `clear` color > `target.clearColor` > the built-in `[0, 0, 0, 1]`.
   */
  clearColor: ClearColor;
  readonly resourceIdentity: ResourceIdentity;
  resize(size: readonly [number, number]): void;
  /** Raw unpadded bytes of `color` in the target's own format (`bgra*` swizzled to RGBA). */
  read(): Promise<Uint8Array>;
  /** Components of `color` decoded to f32 — the HDR-friendly readback for `rgba16float`/`rgba32float` targets. */
  readFloats(): Promise<Float32Array>;
  onDestroy(cb: ResourceDestroyCallback<Target>): UnsubscribeResourceDestroy;
  renderPassDescriptor(opts?: RenderPassDescriptorOptions): GPURenderPassDescriptor;
}

/**
 * A render destination with **persistent resource identity**: its attachments are textures the target
 * owns, so it can be bound (`bindings: { src: scene }`) and vgpu re-binds them for you when a resize
 * recreates them. This is the half of the destination pair a `Surface` deliberately is not — see
 * {@link RenderDestination} for the normative distinction.
 */
export interface Target extends RenderDestination {
  readonly color: Texture;
  readonly colors: readonly [Texture, ...Texture[]];
  readonly depth?: Texture;
}

export { OffscreenTarget } from "./target-offscreen.ts";
export type { Surface, SurfaceOptions, SurfaceResizeEvent } from "./surface.ts";
