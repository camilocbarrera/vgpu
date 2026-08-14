import { Texture, createResourceIdentity, DestroySignal, type Device, type ResourceDestroyCallback, type ResourceIdentity, type UnsubscribeResourceDestroy } from "@vgpu/core";
import { BUILT_IN_CLEAR_COLOR, colorAttachment, copyClearColor, depthAttachment, sameSize, surfaceAttachmentsFor, validateClearColor, type ClearColor } from "./target-utils.ts";
import type { RenderPassDescriptorOptions, Target, TargetSignature } from "./target.ts";
import {
  surfaceAutoResizeUnsupportedError,
  surfaceContextError,
  surfaceDisposedError,
  surfaceDuplicateError,
  surfaceResizeReentrantError,
} from "./errors.ts";
import { frameState } from "./frame-state.ts";
import { liveKernel } from "./live-kernel.ts";
import { serviceToken, type Gpu, type Kernel } from "./kernel.ts";

export interface SurfaceOptions {
  readonly autoResize?: boolean;
  /**
   * Default clear color of this surface, used by passes that clear without naming a color.
   * Defaults to `[0, 0, 0, 1]`; mutable at runtime through `surface.clearColor`.
   */
  readonly clearColor?: ClearColor;
  readonly dpr?: number | readonly [number, number];
  readonly size?: readonly [number, number];
  readonly format?: GPUTextureFormat;
  /**
   * Depth attachment this surface owns: `true` for the shared default (`depthFormatFor` → `depth24plus`),
   * or an explicit depth format. The surface recreates it on every resize and reports it as `surface.depth`.
   */
  readonly depth?: boolean | GPUTextureFormat;
  /**
   * Sample count of the render pass this surface opens. `4` renders into an internal multisample color
   * attachment and resolves into the presentation texture; `1` (the default) renders straight into it.
   */
  readonly sampleCount?: 1 | 4;
  readonly alphaMode?: GPUCanvasAlphaMode;
  readonly colorSpace?: PredefinedColorSpace;
  readonly label?: string;
}

export interface SurfaceResizeEvent {
  readonly width: number;
  readonly height: number;
  readonly dpr: number;
  readonly surface: Surface;
}

export interface Surface extends Target {
  readonly canvas: HTMLCanvasElement | OffscreenCanvas;
  readonly context: GPUCanvasContext;
  readonly autoResize: boolean;
  readonly layoutBacked: boolean;
  readonly dpr: number;
  readonly disposed: boolean;
  onResize(cb: (event: SurfaceResizeEvent) => void): () => void;
  dispose(): void;
}

export type SurfaceCanvas = HTMLCanvasElement | OffscreenCanvas;

/**
 * Canvas render target of this gpu: configures the canvas context and keeps it sized.
 *
 * One live surface per canvas — a second `surface(gpu, canvas)` on the same canvas throws
 * `VGPU-SURFACE-DUPLICATE`, because reconfiguring a context out from under a live surface silently
 * invalidates its textures. Disposing the surface frees the canvas for a new one.
 *
 * Lifecycle: the surface resizes itself right after the frame clock advances (auto-resize is a
 * frame-state hook, so no rAF of its own), and it goes down with the gpu in the `resource` phase —
 * after the loops stopped, before the caches and the device.
 */
export function surface(gpu: Gpu, canvas: SurfaceCanvas, opts: SurfaceOptions = {}): Surface {
  const kernel = liveKernel(gpu, "surface");
  const open = openSurfaces(kernel);
  const existing = open.get(canvas);
  if (existing && !existing.disposed) throw surfaceDuplicateError(existing.label);
  const created = new CanvasSurface(kernel.device, canvas, opts, (disposed) => {
    if (open.get(disposed.canvas) === disposed) open.delete(disposed.canvas);
    releaseAutoResize();
    releaseOwnership();
  });
  const releaseAutoResize = frameState(kernel).onAdvance(() => created.applyAutoResize());
  const releaseOwnership = kernel.own("resource", () => created.dispose());
  open.set(canvas, created);
  return created;
}

/** Live surfaces of a gpu, keyed by canvas: the duplicate-configure guard, created on first surface. */
const openSurfacesToken = serviceToken<Map<SurfaceCanvas, CanvasSurface>>("surfaces");
function openSurfaces(kernel: Kernel): Map<SurfaceCanvas, CanvasSurface> {
  return kernel.service(openSurfacesToken, () => new Map<SurfaceCanvas, CanvasSurface>());
}

let resizeCallbackDepth = 0;
let frameDepth = 0;
export function isSurfaceResizeCallbackActive(): boolean { return resizeCallbackDepth > 0; }
export function isFrameActive(): boolean { return frameDepth > 0; }
export function enterFrame(): void { frameDepth += 1; }
export function leaveFrame(): void { frameDepth -= 1; }
export function isSurface(target: unknown): target is CanvasSurface { return target instanceof CanvasSurface; }

export class CanvasSurface implements Surface {
  readonly resourceIdentity = createResourceIdentity("render-target");
  readonly label: string | undefined;
  readonly context: GPUCanvasContext;
  readonly autoResize: boolean;
  readonly layoutBacked: boolean;
  readonly format: GPUTextureFormat;
  /** Depth format this surface owns, resolved once from its configuration (`depthFormatFor`). */
  readonly depthFormat: GPUTextureFormat | undefined;
  readonly #sampleCount: 1 | 4;
  readonly #destroySignal = new DestroySignal<Target>();
  readonly #callbacks = new Set<(event: SurfaceResizeEvent) => void>();
  readonly #texturesRecreatedCallbacks = new Set<() => void>();
  #depthTexture?: Texture;
  #msaaColor?: Texture;
  #attachmentSize: readonly [number, number] = [0, 0];
  #currentDpr: number;
  #clearColor: ClearColor;
  #isDisposed = false;
  #notifying = false;

  constructor(
    private readonly device: Device,
    readonly canvas: SurfaceCanvas,
    private readonly options: SurfaceOptions,
    private readonly unregister: (surface: CanvasSurface) => void,
  ) {
    this.label = options.label;
    this.#clearColor = options.clearColor === undefined ? BUILT_IN_CLEAR_COLOR : validateClearColor(options.clearColor, "surface.clearColor");
    const context = canvas.getContext("webgpu") as GPUCanvasContext | null;
    if (!context) throw surfaceContextError();
    this.context = context;
    this.layoutBacked = isLayoutBacked(canvas);
    if (options.autoResize === true && !this.layoutBacked) throw surfaceAutoResizeUnsupportedError();
    this.autoResize = options.autoResize ?? (options.size ? false : this.layoutBacked);
    this.#currentDpr = effectiveDpr(options.dpr);
    this.format = options.format ?? preferredCanvasFormat();
    // Depth format and sample count come from the options bag alone, so the pipeline signature of this
    // surface is fixed here, at construction, and never depends on the current presentation texture.
    const attachments = surfaceAttachmentsFor(options);
    this.depthFormat = attachments.depth;
    this.#sampleCount = attachments.sampleCount;
    const initialSize = initialCanvasSize(canvas, options, this.layoutBacked, this.#currentDpr);
    if (options.size || this.layoutBacked) setCanvasSize(canvas, initialSize);
    context.configure({
      device: device.gpu,
      format: this.format,
      alphaMode: options.alphaMode ?? "premultiplied",
      colorSpace: options.colorSpace ?? "srgb",
      usage: canvasTextureUsage(),
    });
    this.#createAttachments();
  }

  get gpu(): unknown { return this.context; }
  get size(): readonly [number, number] { this.#assertLive(); return canvasSize(this.canvas); }
  get texelSize(): readonly [number, number] { const size = this.size; return [1 / size[0], 1 / size[1]]; }
  get color(): Texture {
    this.#assertLive();
    return new Texture(this.device, this.context.getCurrentTexture(), {
      size: this.size,
      format: this.format,
      usage: ["render_attachment", "texture_binding", "copy_src"],
      label: this.options.label ? `${this.options.label}.color` : "surface.color",
    }, "external");
  }
  get colors(): readonly [Texture, ...Texture[]] { return [this.color]; }
  get depth(): Texture | undefined { this.#assertLive(); this.#syncAttachments(); return this.#depthTexture; }
  get sampleCount(): 1 | 4 { this.#assertLive(); return this.#sampleCount; }
  /**
   * Pipeline signature of this surface, derived from its configuration alone: the format fixed by
   * `context.configure()`, the depth format of `SurfaceOptions.depth`, and `SurfaceOptions.sampleCount`.
   * Frame-independent by construction — it never touches `getCurrentTexture()`, so compiling against a
   * surface outside `frame()` is legal and yields exactly the signature the encode path uses inside one.
   */
  get pipelineSignature(): TargetSignature {
    // Frame-independent, not lifetime-independent: a disposed surface rejects here exactly like every
    // other getter, so compiling against a stale surface still fails where the mistake is, instead of
    // silently warming the pipeline cache.
    this.#assertLive();
    return { colors: [this.format], depth: this.depthFormat, sampleCount: this.#sampleCount };
  }
  get dpr(): number { return this.#currentDpr; }
  /** Default clear color of this surface; passes that clear without naming a color use it. */
  get clearColor(): ClearColor { return copyClearColor(this.#clearColor); }
  set clearColor(value: ClearColor) { this.#clearColor = validateClearColor(value, "surface.clearColor"); }
  get disposed(): boolean { return this.#isDisposed; }

  resize(size: readonly [number, number]): void {
    this.#assertLive();
    if (this.#notifying) throw surfaceResizeReentrantError(this.options.label);
    this.#applyResize(sanitizeSize(size), this.#currentDpr, true);
  }

  applyAutoResize(): void {
    if (this.#isDisposed || !this.autoResize || !this.layoutBacked) return;
    const nextDpr = effectiveDpr(this.options.dpr);
    const nextSize = layoutCanvasSize(this.canvas, nextDpr);
    this.#applyResize(nextSize, nextDpr, true);
  }

  onResize(cb: (event: SurfaceResizeEvent) => void): () => void {
    this.#assertLive();
    this.#callbacks.add(cb);
    this.#notifying = true;
    resizeCallbackDepth += 1;
    try { cb(this.#event()); }
    finally { resizeCallbackDepth -= 1; this.#notifying = false; }
    return () => { this.#callbacks.delete(cb); };
  }

  async read(): Promise<Uint8Array> { this.#assertLive(); return this.color.read(); }
  async readFloats(): Promise<Float32Array> { this.#assertLive(); return this.color.readFloats(); }
  onDestroy(cb: ResourceDestroyCallback<Target>): UnsubscribeResourceDestroy { this.#assertLive(); return this.#destroySignal.onDestroy(this, cb); }
  onTexturesRecreated(cb: () => void): () => void { this.#assertLive(); this.#texturesRecreatedCallbacks.add(cb); return () => { this.#texturesRecreatedCallbacks.delete(cb); }; }

  renderPassDescriptor(opts: RenderPassDescriptorOptions = {}): GPURenderPassDescriptor {
    const { clear = [0, 0, 0, 1], preserve, clearDepth, clearStencil, depthReadOnly } = opts;
    this.#assertLive();
    // Encoding is the only path that needs the current texture — and the only one that needs the
    // internal attachments to match its size.
    this.#syncAttachments();
    const presentation = this.context.getCurrentTexture();
    return {
      // MSAA renders into the internal multisample color and resolves into the presentation texture.
      colorAttachments: [colorAttachment(presentation, this.#msaaColor, clear, preserve)],
      depthStencilAttachment: this.#depthTexture ? depthAttachment(this.#depthTexture, preserve, clearDepth, clearStencil, depthReadOnly) : undefined,
    };
  }

  dispose(): void {
    if (this.#isDisposed) return;
    this.#isDisposed = true;
    try { this.context.unconfigure?.(); } catch { /* ignore native cleanup failures */ }
    this.#destroyAttachments();
    this.unregister(this);
    this.#callbacks.clear();
    this.#texturesRecreatedCallbacks.clear();
    this.#destroySignal.emit(this);
  }

  #applyResize(size: readonly [number, number], dpr: number, notify: boolean): void {
    const changed = !sameSize(canvasSize(this.canvas), size);
    this.#currentDpr = dpr;
    if (!changed) return;
    setCanvasSize(this.canvas, size);
    // The presentation texture follows the canvas by itself; the attachments this surface owns do not.
    // #recreateAttachments emits texturesRecreated, so a surface without attachments still notifies.
    this.#recreateAttachments();
    if (notify) this.#notify();
  }

  /**
   * Keeps the internal attachments the same size as the canvas, including when the canvas was resized
   * behind this surface's back (a direct `canvas.width = …`), which never goes through `#applyResize`.
   * A resize only ever changes their size — never their format or sample count — so the pipeline
   * signature of this surface survives it untouched.
   */
  #syncAttachments(): void {
    if (!this.#depthTexture && !this.#msaaColor) return;
    if (sameSize(this.#attachmentSize, this.#attachmentTargetSize())) return;
    this.#recreateAttachments();
  }

  #recreateAttachments(): void {
    this.#destroyAttachments();
    this.#createAttachments();
    // Emitted here, not only from #applyResize: the canvas-drifted path recreates the very textures
    // consumers cache views and bind groups over, so it has to announce them too. A surface without
    // attachments still reaches this from #applyResize, which is how a plain surface keeps notifying.
    this.#emitTexturesRecreated();
  }

  /** Attachment size for the canvas as it is right now, sanitized to WebGPU's 1×1 minimum like every other size this surface computes. */
  #attachmentTargetSize(): readonly [number, number] {
    return sanitizeSize(canvasSize(this.canvas));
  }

  #createAttachments(): void {
    const size = this.#attachmentTargetSize();
    this.#attachmentSize = size;
    // Not a transient attachment: TRANSIENT_ATTACHMENT is an optimization of its own, not a surface concern.
    this.#msaaColor = this.#sampleCount === 4 ? this.device.createTexture({
      size,
      format: this.format,
      usage: ["render_attachment"],
      sampleCount: 4,
      label: this.options.label ? `${this.options.label}.color.msaa` : "surface.color.msaa",
    }) : undefined;
    // texture_binding matches target(): read-only depth passes can bind `surface.depth` as a sampled texture.
    this.#depthTexture = this.depthFormat ? this.device.createTexture({
      size,
      format: this.depthFormat,
      usage: ["render_attachment", "texture_binding"],
      sampleCount: this.#sampleCount,
      label: this.options.label ? `${this.options.label}.depth` : "surface.depth",
    }) : undefined;
  }

  #destroyAttachments(): void {
    this.#msaaColor?.destroy();
    this.#depthTexture?.destroy();
    this.#msaaColor = undefined;
    this.#depthTexture = undefined;
  }

  #emitTexturesRecreated(): void {
    for (const cb of [...this.#texturesRecreatedCallbacks]) cb();
  }

  #notify(): void {
    this.#notifying = true;
    resizeCallbackDepth += 1;
    try {
      const event = this.#event();
      for (const cb of [...this.#callbacks]) cb(event);
    } finally {
      resizeCallbackDepth -= 1;
      this.#notifying = false;
    }
  }

  #event(): SurfaceResizeEvent {
    const size = canvasSize(this.canvas);
    return { width: size[0], height: size[1], dpr: this.#currentDpr, surface: this };
  }

  #assertLive(): void {
    if (this.#isDisposed) throw surfaceDisposedError(this.options.label);
  }
}

export function isLayoutBacked(canvas: unknown): boolean {
  return typeof (canvas as { clientWidth?: unknown }).clientWidth === "number";
}

function initialCanvasSize(canvas: SurfaceCanvas, options: SurfaceOptions, layoutBacked: boolean, dpr: number): readonly [number, number] {
  if (options.size) return sanitizeSize(options.size);
  if (layoutBacked) return layoutCanvasSize(canvas, dpr);
  return sanitizeSize(canvasSize(canvas));
}

function layoutCanvasSize(canvasLike: unknown, dpr: number): readonly [number, number] {
  const canvas = canvasLike as { clientWidth: number; clientHeight: number };
  return sanitizeSize([Math.round(canvas.clientWidth * dpr), Math.round(canvas.clientHeight * dpr)]);
}

function canvasSize(canvasLike: unknown): readonly [number, number] {
  const canvas = canvasLike as { width: number; height: number };
  return [canvas.width, canvas.height];
}

function setCanvasSize(canvasLike: unknown, size: readonly [number, number]): void {
  const canvas = canvasLike as { width: number; height: number };
  canvas.width = size[0];
  canvas.height = size[1];
}

function sanitizeSize(size: readonly [number, number]): readonly [number, number] {
  return [Math.max(1, Math.floor(size[0])), Math.max(1, Math.floor(size[1]))];
}

function effectiveDpr(dpr: SurfaceOptions["dpr"]): number {
  const raw = globalThis.devicePixelRatio ?? 1;
  if (Array.isArray(dpr)) return Math.min(dpr[1], Math.max(dpr[0], raw));
  if (typeof dpr === "number") return dpr;
  return raw;
}

function preferredCanvasFormat(): GPUTextureFormat {
  return (globalThis.navigator as (Navigator & { gpu?: GPU }) | undefined)?.gpu?.getPreferredCanvasFormat?.() ?? "bgra8unorm";
}

function canvasTextureUsage(): GPUTextureUsageFlags | undefined {
  const usage = (globalThis as { GPUTextureUsage?: typeof GPUTextureUsage }).GPUTextureUsage;
  return usage ? usage.RENDER_ATTACHMENT | usage.TEXTURE_BINDING | usage.COPY_SRC : undefined;
}
