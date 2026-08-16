/**
 * Minimal `Gpu` core plus its private kernel.
 *
 * The kernel is the only shared, cross-feature runtime: device handle, error delivery,
 * `settled()` bookkeeping, ownership/teardown ordering and a lazy service registry.
 * It MUST NOT import a feature module (frame, draw, effect, surface, timer, geometry, ...):
 * every service is registered by the feature that owns it, through a token + factory, so a
 * program that never imports the feature never pays for it. See T202-06 negative metafile
 * assertions and the `init-only` bundle budget.
 */
import { Device, validateRequiredFeatures, type RequiredDeviceLimits, type VGPUAdapter } from "@vgpu/core";
import type { GpuErrorListener } from "./api-types.ts";
import { DEFAULT_PENDING_PIPELINES, type PendingPipelines } from "./pending-pipelines.ts";
import { submittedWorkDone } from "./submitted-work-done.ts";
import { unsupportedError, VGPUError } from "./errors.ts";

/**
 * Options for the device vgpu creates and owns; it destroys that device on `dispose()`.
 *
 * To adopt a device another library owns, call `initFromDevice(device)` instead. `device` is
 * declared here only so that passing one lands on the compile-time signpost rather than being
 * silently ignored.
 */
export interface InitOptions {
  readonly adapter?: VGPUAdapter;
  /**
   * Never set: adoption lives in `initFromDevice(device)`. Declared so passing a device here is a
   * type error instead of a silently ignored option. There is deliberately no runtime check: one
   * does not fit the `init-only` budget, which is the budget this split exists to protect.
   */
  readonly device?: never;
  readonly powerPreference?: GPUPowerPreference;
  readonly requiredFeatures?: readonly GPUFeatureName[];
  readonly requiredLimits?: RequiredDeviceLimits;
  readonly label?: string;
  /**
   * Gpu-wide default of the `pendingPipelines` policy: what a synchronous encode does when it meets
   * a `(renderable, target signature)` combination whose pipeline is not ready yet. The policy is
   * resolved call site → frame → here.
   *
   * Omitted, it resolves to {@link DEFAULT_PENDING_PIPELINES} — `"sync"` during the 0.4 train
   * (T04-05, index invariant 2), which is the eager-compile behavior every existing program already
   * relies on. The frozen design's `"throw"` default lands in the cut wave, with the codemods.
   */
  readonly pendingPipelines?: PendingPipelines;
}

export type AdapterFactory = () => VGPUAdapter;
export type EntryKind = "browser" | "node" | "mock";

/**
 * Ring-1 core shared by browser, node, and mock entrypoints: a device handle, an error
 * channel and a lifetime. Everything else is a free function that takes this object.
 */
export interface Gpu {
  readonly device: Device;
  readonly gpu: GPUDevice;
  /** True once `dispose()` ran. Reads stay legal; new work does not. */
  readonly disposed: boolean;
  /**
   * Resolves **only** when this device is really lost (design §9, contract #19) — a driver reset, a
   * TDR, a tab evicted from the GPU process — with the native `GPUDeviceLostInfo` when the platform
   * gives one. A deliberate `dispose()` leaves it pending forever: disposal is observed through
   * `disposed`, loss through this promise, and there is deliberately no `onLost()` callback (one
   * spelling per semantic).
   *
   * A real loss is terminal and graph-wide: by the time this resolves, every frame loop this gpu owns
   * is stopped, and every operation on any object of this graph throws `VGPU-DEVICE-LOST`. vgpu never
   * restores, reattaches or re-points anything at a replacement device — recovery is
   * `init()` again, from scratch.
   */
  readonly lost: Promise<GPUDeviceLostInfo | undefined>;
  onError(cb: GpuErrorListener): () => void;
  settled(): Promise<void>;
  dispose(): void;
}

/** Contributes promises to `gpu.settled()`; queried (not retained) at each call, so it is a snapshot source. */
export type SettledSource = () => readonly Promise<unknown>[];
/** Undo function returned by every kernel registration. Idempotent by contract. */
export type Release = () => void;
export type Disposer = () => void;

/**
 * Teardown order of `gpu.dispose()`. Phases run in declaration order; within a phase,
 * registration order.
 *
 * - `scheduler`: rAF/timer loops. Stopped first — a tick landing mid-teardown would encode
 *   against a dying device.
 * - `resource`: surfaces, query rings, views and anything holding GPU objects.
 * - `service`: caches and stores shared by features (pipelines, bind groups, samplers).
 *
 * The device is disposed after every phase.
 */
export type OwnershipPhase = "scheduler" | "resource" | "service";
const PHASES: readonly OwnershipPhase[] = ["scheduler", "resource", "service"];

declare const SERVICE_TYPE: unique symbol;
/** Nominal handle for a lazily created shared service. Created by the feature that owns the service. */
export interface ServiceToken<T> {
  readonly name: string;
  /** Phantom: carries `T` for inference, never present at runtime. */
  readonly [SERVICE_TYPE]?: T;
}

export function serviceToken<T>(name: string): ServiceToken<T> {
  return { name };
}

/** Private per-`Gpu` runtime. Not exported from any entrypoint; reachable only through `kernelOf`. */
export interface Kernel {
  readonly device: Device;
  readonly disposed: boolean;
  /**
   * Gpu-wide `pendingPipelines` default resolved at `init()`; the last link of the
   * call site → frame → gpu chain. `"sync"` unless `init({ pendingPipelines })` said otherwise.
   */
  pendingPipelinesDefault(): PendingPipelines;
  /**
   * Lazily creates (and memoizes) the service behind `token`. The factory lives in the calling
   * feature module, so the kernel never references it statically.
   */
  service<T>(token: ServiceToken<T>, factory: (kernel: Kernel) => T): T;
  /** The instance if it was already created, else undefined. Never runs the factory. */
  peekService<T>(token: ServiceToken<T>): T | undefined;
  /** Registers a teardown callback in `phase`; returns the release that unregisters it. */
  own(phase: OwnershipPhase, disposer: Disposer): Release;
  /**
   * A real device loss landed: stop the `scheduler` phase (the rAF/timer loops this gpu owns) and
   * nothing else. Resources and services are deliberately left alone — a lost device has nothing to
   * release, and every object of the graph is already terminal through `Device.assertUsable`.
   * `disposed` stays `false`: loss is not disposal.
   */
  notifyDeviceLost(): void;
  addErrorListener(cb: GpuErrorListener): Release;
  /** Delivers asynchronously to the listeners (or `console.error`). Never rejects, never throws. */
  reportError(error: VGPUError): Promise<void>;
  /** Adds a promise to the `settled()` set; swallows rejections into `console.error`. */
  trackDelivery(promise: Promise<unknown>): Promise<void>;
  registerSettledSource(source: SettledSource): Release;
  settled(): Promise<void>;
  dispose(): void;
}

const kernels = new WeakMap<Gpu, Kernel>();

/** Internal accessor for the kernel of a `Gpu`. Throws for objects this library did not create. */
export function kernelOf(gpu: Gpu): Kernel {
  const kernel = kernels.get(gpu);
  if (!kernel) {
    throw new VGPUError({
      code: "VGPU-GPU-FOREIGN",
      message: "This object was not created by init(); it has no vgpu kernel.",
      fix: "Pass the gpu returned by init() from vgpu, vgpu/node or vgpu/mock.",
      where: "gpu",
    });
  }
  return kernel;
}

class KernelImpl implements Kernel {
  readonly #services = new Map<ServiceToken<unknown>, unknown>();
  readonly #owners: ReadonlyMap<OwnershipPhase, Set<Disposer>> = new Map(PHASES.map((phase) => [phase, new Set<Disposer>()]));
  readonly #errorListeners = new Set<GpuErrorListener>();
  readonly #pendingDeliveries = new Set<Promise<void>>();
  readonly #settledSources = new Set<SettledSource>();
  #disposed = false;

  constructor(readonly device: Device, private readonly pendingPipelines: PendingPipelines = DEFAULT_PENDING_PIPELINES) {}

  get disposed(): boolean { return this.#disposed; }

  pendingPipelinesDefault(): PendingPipelines { return this.pendingPipelines; }

  service<T>(token: ServiceToken<T>, factory: (kernel: Kernel) => T): T {
    const existing = this.#services.get(token as ServiceToken<unknown>);
    if (existing !== undefined) return existing as T;
    const created = factory(this);
    this.#services.set(token as ServiceToken<unknown>, created);
    return created;
  }

  peekService<T>(token: ServiceToken<T>): T | undefined {
    return this.#services.get(token as ServiceToken<unknown>) as T | undefined;
  }

  own(phase: OwnershipPhase, disposer: Disposer): Release {
    const set = this.#owners.get(phase)!;
    set.add(disposer);
    return () => { set.delete(disposer); };
  }

  addErrorListener(cb: GpuErrorListener): Release {
    this.#errorListeners.add(cb);
    return () => { this.#errorListeners.delete(cb); };
  }

  reportError(error: VGPUError): Promise<void> {
    if (this.#disposed) return Promise.resolve();
    const delivery = Promise.resolve().then(() => {
      const listeners = [...this.#errorListeners];
      if (!listeners.length) {
        console.error(error);
        return;
      }
      for (const listener of listeners) {
        try { listener(error); }
        catch (listenerError) { console.error(listenerError); }
      }
    });
    return this.trackDelivery(delivery);
  }

  trackDelivery(promise: Promise<unknown>): Promise<void> {
    const tracked = Promise.resolve(promise).then(() => undefined, (error) => { console.error(error); });
    this.#pendingDeliveries.add(tracked);
    void tracked.finally(() => this.#pendingDeliveries.delete(tracked));
    return tracked;
  }

  registerSettledSource(source: SettledSource): Release {
    this.#settledSources.add(source);
    return () => { this.#settledSources.delete(source); };
  }

  async settled(): Promise<void> {
    const snapshot = [
      submittedWorkDone(this.device),
      ...this.#pendingDeliveries,
      ...[...this.#settledSources].flatMap((source) => source()),
    ];
    await Promise.allSettled(snapshot);
  }

  notifyDeviceLost(): void {
    this.#runPhase("scheduler");
  }

  #runPhase(phase: OwnershipPhase): void {
    const set = this.#owners.get(phase)!;
    // Copy: a disposer usually calls its own release(), mutating the set while we walk it.
    for (const disposer of [...set]) disposer();
    set.clear();
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    for (const phase of PHASES) this.#runPhase(phase);
    this.#services.clear();
    this.#settledSources.clear();
    this.#errorListeners.clear();
    this.device.dispose();
  }
}

/** Builds the minimal `Gpu` for an already-created device and registers its kernel. */
export function attachKernel(device: Device, opts: Pick<InitOptions, "pendingPipelines"> = {}): Gpu {
  const kernel = new KernelImpl(device, opts.pendingPipelines ?? DEFAULT_PENDING_PIPELINES);
  // One reaction to the loss, wired here and not per consumer: the loops are stopped inside this
  // handler, so anyone awaiting `gpu.lost` already observes a graph that stopped ticking. `device.lost`
  // stays pending after a deliberate `dispose()` (core suppresses it), which is exactly why no extra
  // "was it disposed?" check is needed here.
  const lost = device.lost.then((info) => { kernel.notifyDeviceLost(); return info; });
  const gpu: Gpu = {
    device,
    gpu: device.gpu,
    get disposed(): boolean { return kernel.disposed; },
    lost,
    onError: (cb: GpuErrorListener) => kernel.addErrorListener(cb),
    settled: () => kernel.settled(),
    dispose: () => { kernel.dispose(); },
  };
  kernels.set(gpu, kernel);
  return gpu;
}

/** Entry-agnostic core constructor: resolve a device, wrap it in the minimal `Gpu`. */
export async function createCoreGpu(entry: EntryKind, opts: InitOptions = {}, adapterFactory?: AdapterFactory): Promise<Gpu> {
  return attachKernel(await createDevice(entry, opts, adapterFactory), opts);
}

export async function createDevice(entry: EntryKind, opts: InitOptions, adapterFactory?: AdapterFactory): Promise<Device> {
  if (opts.adapter || adapterFactory) return (opts.adapter ?? adapterFactory!()).requestDevice(opts);
  if (entry === "browser") return requestBrowserDevice(opts);
  throw unsupportedError("init", `init(${entry}) requires adapterFactory.`);
}

async function requestBrowserDevice(opts: InitOptions): Promise<Device> {
  const nav = globalThis.navigator as Navigator & { gpu?: GPU };
  const adapter = await nav.gpu?.requestAdapter({ powerPreference: opts.powerPreference });
  if (!adapter) throw unsupportedError("init", "navigator.gpu.requestAdapter() returned null.");
  validateRequiredFeatures(adapter.features, opts.requiredFeatures);
  const gpuDevice = await adapter.requestDevice({ requiredFeatures: opts.requiredFeatures, requiredLimits: opts.requiredLimits });
  return new Device(gpuDevice, adapter.info ?? null);
}
