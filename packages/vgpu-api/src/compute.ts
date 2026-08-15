import type { Device } from "@vgpu/core";
import type { ShaderSource } from "@vgpu/wgsl";
import { reflectSource, type BindingInfo, type EntryPointInfo, type Reflection } from "@vgpu/wgsl/reflect-source";
import { entryMetadata } from "./entry-metadata.ts";
import { createBindGroupCache, identityKey, type BindGroupCache, type BindGroupIdentityPart } from "./bind-cache.ts";
import { createSetCore, bindGroupLayoutsForReflection, pipelineLayoutFor, type SetBag, type SetCore } from "./set-core.ts";
import { visibilityForEntries } from "./set-layouts.ts";
import type { Compute, ComputeOptions, DispatchOptions } from "./api-types.ts";
import { normalizeConstantsOptions, selectEntryPoint } from "./pipeline-store.ts";
import { dispatchCountInvalidError, indirectInvalidError, pipelinePendingError, unsupportedError, writableStorageAliasingError, type VGPUError } from "./errors.ts";
import { FRAME_COMPUTABLE, type FrameComputableProtocol } from "./frame-protocols.ts";
import { assertDeviceUsable } from "./lifecycle.ts";
import { DEFAULT_PENDING_PIPELINES, type PendingPipelines } from "./pending-pipelines.ts";
import type { Gpu } from "./kernel.ts";
import { liveKernel } from "./live-kernel.ts";
import { renderService } from "./render-service.ts";
import { resolveShaderInput, toWgsl } from "./shader-source.ts";
import { resolveIndirect } from "./indirect.ts";

/**
 * Compute pipeline for this gpu, ready to `set()` bindings and `dispatch()`.
 *
 * Each compute owns its own pipeline (no shared pipeline store), but it resolves the gpu's single
 * lazy bind group cache through the kernel, so a bind group built for a draw and one built here are
 * the same object when the resources match — and the cache is torn down once, in the service phase.
 */
// Overload order is public API surface: `Parameters<typeof compute>[1]` resolves to the LAST
// declared overload, so the single-object form must come first — otherwise consumers that derive
// types from `Parameters<>`/`ReturnType<>` (see apps/docs/examples/fft-ocean-surface/scene.ts) see
// `ComputeOptions & { shader }` instead of `string | ShaderSource` for that position. Call resolution
// itself is unaffected by this order: the single-object form requires `shader`, which neither a
// string nor a ShaderSource artifact has, so it never matches those calls.
/** Single-object form: exactly two arguments. A third `opts` argument here is silently ignored — put every option in `input`. */
export function compute(gpu: Gpu, input: ComputeOptions & { readonly shader: string | ShaderSource }): Compute;
export function compute(gpu: Gpu, input: string | ShaderSource): Compute;
export function compute(gpu: Gpu, source: string | ShaderSource, opts: ComputeOptions): Compute;
export function compute(gpu: Gpu, input: string | ShaderSource | ComputeOptions, opts: ComputeOptions = {}): Compute {
  const [source, resolvedOpts] = resolveShaderInput("compute", input, opts);
  const kernel = liveKernel(gpu, "compute");
  // Same wiring render-service.ts gives createPipelineStore: the gpu-wide pendingPipelines default
  // (last link of the call site → frame → gpu chain) and the gpu's error sink.
  return new ComputePipeline(kernel.device, toWgsl(source), resolvedOpts, renderService(kernel).binds, kernel.pendingPipelinesDefault(), (error) => { void kernel.reportError(error); });
}

let nextComputeId = 1;

/**
 * Internal Ring-1 compute implementation behind `compute(gpu, source, opts)`.
 *
 * @internal
 */
export class ComputePipeline implements Compute {
  readonly id = nextComputeId++;
  readonly label: string;
  readonly reflection: Reflection;
  readonly entryPoint: string;
  readonly setCore: SetCore;
  readonly bindGroupLayouts: ReadonlyMap<number, GPUBindGroupLayout>;
  readonly pipelineLayout: GPUPipelineLayout;
  readonly shaderModule: GPUShaderModule;
  readonly #storageBindings: readonly BindingInfo[];
  readonly #pipelineDescriptor: GPUComputePipelineDescriptor;
  #pipeline?: GPUComputePipeline;
  #pipelinePending?: Promise<GPUComputePipeline>;
  /**
   * `"skip"` never throws per frame, so a background compile failure has exactly one channel:
   * `gpu.onError`. Reported once per instance — a frame loop would otherwise flood the sink with the
   * same rejection every tick.
   */
  #skipFailureReported = false;

  constructor(
    private readonly device: Device,
    readonly source: string,
    readonly opts: ComputeOptions = {},
    private readonly cache: BindGroupCache = createBindGroupCache(),
    /**
     * Last link of the pendingPipelines chain (call site → frame → gpu) for `f.compute()`: the
     * gpu-wide default from `init()`. Defaults to the train-wide default so the legacy
     * `new ComputePipeline(device, source)` shape keeps working unchanged.
     */
    private readonly pendingPipelinesDefault: PendingPipelines = DEFAULT_PENDING_PIPELINES,
    /** Error sink of the owning gpu, for the one channel a `"skip"` background compile can report on. */
    private readonly reportError: (error: VGPUError) => void = () => undefined,
  ) {
    assertDeviceUsable(device, "Compute.constructor");
    this.label = opts.label ?? "compute";
    this.reflection = reflectSource(source, `${this.label}.wgsl`);
    // Entry selection runs before everything derived from the selected entry — binding visibility, bind group
    // layouts, and the active-binding set for storage aliasing all reflect the chosen variant.
    const entry = computeEntryPoint(this.reflection, this.label, opts.entry);
    this.entryPoint = entry.name;
    const { constants } = normalizeConstantsOptions(this.label, opts.constants, this.reflection.overrides, "compute");
    this.bindGroupLayouts = bindGroupLayoutsForReflection(device, this.label, this.reflection, visibilityForEntries(this.reflection.bindings, [entry]));
    this.pipelineLayout = pipelineLayoutFor(device, this.bindGroupLayouts);
    this.shaderModule = device.gpu.createShaderModule({ label: `${this.label}.shader`, code: source });
    // Each Compute owns its pipeline (no shared store), so constants join the descriptor directly; the record is
    // omitted when the option is absent to keep the descriptor byte-identical to before. The pipeline itself is
    // NOT compiled here — construction stays free of createComputePipeline(Async) calls; #ensurePipeline()/
    // #ensurePipelineAsync() compile it lazily the first time dispatch()/dispatchOnce() needs it.
    this.#pipelineDescriptor = {
      label: `${this.label}.pipeline`,
      layout: this.pipelineLayout,
      compute: { module: this.shaderModule, entryPoint: this.entryPoint, ...(constants ? { constants } : {}) },
    };
    // Ownership is fixed here, before any set(): the engine applies `bindings` (external) and
    // `values` (instance-owned) while it builds the binding state machine.
    this.setCore = createSetCore({ device, label: this.label, drawId: this.id, reflection: this.reflection, bindGroupLayouts: this.bindGroupLayouts, cache: this.cache, values: opts.values, bindings: opts.bindings });
    const active = new Set(entryMetadata(entry, "bindings", this.label).map((binding) => `${binding.group}:${binding.binding}`));
    this.#storageBindings = this.reflection.bindings.filter((binding) => binding.kind === "buffer" && binding.addressSpace === "storage" && active.has(`${binding.group}:${binding.binding}`));
    if (opts.set) this.set(opts.set);
  }

  set(binding: string, value: unknown): this;
  set(values: SetBag): this;
  set(bindingOrValues: string | SetBag, value?: unknown): this {
    assertDeviceUsable(this.device, `${this.label}.set`);
    if (typeof bindingOrValues === "string") this.setCore.setScoped(bindingOrValues, value);
    else this.setCore.set(bindingOrValues);
    return this;
  }

  bind(binding: string, resource: unknown): this {
    assertDeviceUsable(this.device, `${this.label}.bind`);
    this.setCore.bind(binding, resource);
    return this;
  }

  dispatch(x: number, y?: number, z?: number): void;
  dispatch(opts: DispatchOptions): void;
  dispatch(x: number | DispatchOptions, y?: number, z?: number): void {
    const where = `${this.label}.dispatch`;
    assertDeviceUsable(this.device, where);
    const indirect = typeof x === "object" && x !== null ? this.#resolveIndirectDispatch(x, y, z, where) : this.#validateCounts(x as number, y, z, where) && undefined;
    this.#preflightAliasing(where);
    // Legacy dispatch() stays lazy-sync: it compiles inline the first time it is called (not at construction),
    // reusing the draws' pipelineFor/getSync pattern, but synchronously — createComputePipeline(), not Async.
    const pipeline = this.#ensurePipeline();
    const encoder = this.device.gpu.createCommandEncoder({ label: `${this.label}.encoder` });
    const pass = encoder.beginComputePass({ label: `${this.label}.pass` });
    pass.setPipeline(pipeline);
    for (const binding of this.setCore.bindGroups()) pass.setBindGroup(binding.group, binding.bindGroup, binding.offsets);
    if (indirect) pass.dispatchWorkgroupsIndirect(indirect.buffer, indirect.offset);
    else pass.dispatchWorkgroups(x as number, y ?? 1, z ?? 1);
    pass.end();
    this.device.gpu.queue.submit([encoder.finish()]);
  }

  /**
   * Async twin of `dispatch()`: compiles the pipeline through `createComputePipelineAsync()` when it is not
   * ready yet (reusing it unchanged otherwise, whether a prior `dispatch()` or `dispatchOnce()` created it),
   * then owns its own encoder and submits exactly once. Resolves right after the submit — it never awaits
   * `onSubmittedWorkDone`.
   */
  dispatchOnce(x: number, y?: number, z?: number): Promise<void>;
  dispatchOnce(opts: DispatchOptions): Promise<void>;
  async dispatchOnce(x: number | DispatchOptions, y?: number, z?: number): Promise<void> {
    const where = `${this.label}.dispatchOnce`;
    assertDeviceUsable(this.device, where);
    const indirect = typeof x === "object" && x !== null ? this.#resolveIndirectDispatch(x, y, z, where) : this.#validateCounts(x as number, y, z, where) && undefined;
    this.#preflightAliasing(where);
    const pipeline = await this.#ensurePipelineAsync();
    // The device may have been disposed while the pipeline compile was in flight — re-check before touching it.
    assertDeviceUsable(this.device, where);
    // Bindings are not snapshotted at call time either (consistent with `set()` never snapshotting — the
    // last `set()` wins): a `set()` that introduces writable-storage aliasing while the compile was in
    // flight must still be caught here, before any encoding happens, exactly like the sync dispatch() path.
    this.#preflightAliasing(where);
    const encoder = this.device.gpu.createCommandEncoder({ label: `${this.label}.encoder` });
    const pass = encoder.beginComputePass({ label: `${this.label}.pass` });
    pass.setPipeline(pipeline);
    for (const binding of this.setCore.bindGroups()) pass.setBindGroup(binding.group, binding.bindGroup, binding.offsets);
    if (indirect) pass.dispatchWorkgroupsIndirect(indirect.buffer, indirect.offset);
    else pass.dispatchWorkgroups(x as number, y ?? 1, z ?? 1);
    pass.end();
    this.device.gpu.queue.submit([encoder.finish()]);
  }

  /**
   * Nominal `FrameComputable` protocol, so `frame.ts` never imports this module: a program that only
   * uses `frame()` / `f.pass()` must not pay for the whole compute package (pay-for-what-you-import).
   *
   * @internal
   */
  get [FRAME_COMPUTABLE](): FrameComputableProtocol {
    return this;
  }

  /**
   * Third encoding path of this class, for `f.compute()`: it records one compute pass into an encoder
   * it does **not** own — the frame's — so it never calls `encoder.finish()` nor `queue.submit()`.
   * That is what puts the dispatch inside the frame's single submit, in program order (contract #1),
   * which `dispatch()`/`dispatchOnce()` structurally cannot do: both open their own encoder.
   *
   * Validation is the very same private code the legacy paths use — `#validateCounts` (contract #17)
   * and `#preflightAliasing` (writable-storage double-bind guard) — never a reimplementation.
   *
   * @internal
   */
  encodeForFrame(encoder: GPUCommandEncoder, x: number, y: number | undefined, z: number | undefined, policy: PendingPipelines | undefined): void {
    const where = `${this.label}.compute`;
    assertDeviceUsable(this.device, where);
    this.#validateCounts(x, y, z, where);
    this.#preflightAliasing(where);
    const resolved = policy ?? this.pendingPipelinesDefault;
    let pipeline = this.#pipeline;
    if (!pipeline) {
      // "throw": does not start compilation and throws immediately. `signature` is undefined because a
      // compute pipeline has no target signature — the same asymmetry prepare()'s signatureKeyFor has.
      if (resolved === "throw") throw pipelinePendingError(where, this.label, undefined);
      // "skip": start (or continue) the async compile in the background, omit the dispatch this frame,
      // and never throw per frame. Deliberately the same #ensurePipelineAsync() dispatchOnce()/
      // prepare() use — same dedupe, same anti-poisoning — so the compile it starts is the one a later
      // dispatchOnce()/prepare() awaits, never a second one.
      if (resolved === "skip") {
        void this.#ensurePipelineAsync().catch((error: unknown) => {
          if (this.#skipFailureReported) return;
          this.#skipFailureReported = true;
          this.reportError(error as VGPUError);
        });
        return;
      }
      // "sync": immediate createComputePipeline(), exactly like legacy dispatch().
      pipeline = this.#ensurePipeline();
    }
    const pass = encoder.beginComputePass({ label: `${this.label}.pass` });
    pass.setPipeline(pipeline);
    for (const binding of this.setCore.bindGroups()) pass.setBindGroup(binding.group, binding.bindGroup, binding.offsets);
    pass.dispatchWorkgroups(x, y ?? 1, z ?? 1);
    pass.end();
  }

  /**
   * Async readiness for one compute kernel, for `prepare(gpu, [{ compute }])`.
   *
   * Deliberately the very same `#ensurePipelineAsync()` `dispatchOnce()` uses — same dedupe of
   * concurrent callers, same reuse of a pipeline a previous `dispatch()` already compiled, same
   * anti-poisoning on failure. `prepare()` is a second door to that door, never a second compile.
   *
   * @internal
   */
  prepareCombination(): Promise<GPUComputePipeline> {
    assertDeviceUsable(this.device, `${this.label}.prepare`);
    return this.#ensurePipelineAsync();
  }

  /** Compiles the pipeline synchronously the first time it is needed, then memoizes it for every later call. */
  #ensurePipeline(): GPUComputePipeline {
    if (!this.#pipeline) this.#pipeline = this.device.gpu.createComputePipeline(this.#pipelineDescriptor);
    return this.#pipeline;
  }

  /**
   * Compiles the pipeline through `createComputePipelineAsync()` the first time it is needed, sharing one
   * in-flight promise across concurrent callers, and reuses whatever `#ensurePipeline()` already compiled.
   *
   * `#ensurePipeline()` (the sync path used by legacy `dispatch()`) does not know about an in-flight async
   * compile, so a `dispatch()` call racing an un-awaited `dispatchOnce()` on the same instance can still
   * compile a second `GPUComputePipeline` — `??=` below keeps whichever one lands first as the memoized
   * pipeline instead of letting the async result silently clobber a sync one that already resolved (and
   * that `dispatch()` may already have bound to a pass), which would otherwise churn pipeline identity.
   *
   * A rejected compile must not poison the instance forever: both branches clear `#pipelinePending` once
   * settled, but only if it is still *this* attempt's promise (an in-flight retry started after a prior
   * settle must not have its own pending slot yanked out from under it), so a later `dispatchOnce()` call
   * starts a fresh `createComputePipelineAsync()` instead of re-awaiting (or being poisoned by) a stale
   * rejection.
   */
  #ensurePipelineAsync(): Promise<GPUComputePipeline> {
    if (this.#pipeline) return Promise.resolve(this.#pipeline);
    if (!this.#pipelinePending) {
      const attempt: Promise<GPUComputePipeline> = this.device.gpu.createComputePipelineAsync(this.#pipelineDescriptor).then(
        (pipeline) => {
          this.#pipeline ??= pipeline;
          if (this.#pipelinePending === attempt) this.#pipelinePending = undefined;
          return this.#pipeline;
        },
        (error: unknown) => {
          if (this.#pipelinePending === attempt) this.#pipelinePending = undefined;
          throw error;
        },
      );
      this.#pipelinePending = attempt;
    }
    return this.#pipelinePending;
  }

  /** The GPU reads the workgroup counts from the buffer, so explicit counts alongside indirect are dead options and throw. */
  #resolveIndirectDispatch(opts: DispatchOptions, y?: number, z?: number, where = `${this.label}.dispatch`): { readonly buffer: GPUBuffer; readonly offset: number } {
    if (y !== undefined || z !== undefined) throw indirectInvalidError(this.label, `indirect cannot be combined with explicit workgroup counts in the same call; the GPU reads the counts from the buffer, so the CPU-side values would be ignored.`, where);
    return resolveIndirect(this.label, where, opts.indirect, "dispatchWorkgroupsIndirect");
  }

  /** Contract #17: dispatch()/dispatchOnce() workgroup counts must be integers >= 0 — no NaN/negative/fractional counts reach WebGPU. */
  #validateCounts(x: number, y: number | undefined, z: number | undefined, where: string): true {
    validateDispatchCount(this.label, "x", x, where);
    if (y !== undefined) validateDispatchCount(this.label, "y", y, where);
    if (z !== undefined) validateDispatchCount(this.label, "z", z, where);
    return true;
  }

  #preflightAliasing(where = `${this.label}.dispatch`): void {
    if (!this.#storageBindings.length) return;
    const buckets = new Map<string, { identity: BindGroupIdentityPart; writable: boolean }[]>();
    for (const binding of this.#storageBindings) {
      const state = this.setCore.bindingState(binding.name);
      if (!state) continue;
      const key = identityKey(state.identity);
      if (!buckets.has(key)) buckets.set(key, []);
      buckets.get(key)!.push({ identity: state.identity, writable: binding.access !== "read" });
    }
    for (const bucket of buckets.values()) {
      if (bucket.length < 2) continue;
      if (!bucket.some((entry) => entry.writable)) continue;
      throw writableStorageAliasingError(where);
    }
  }
}

function computeEntryPoint(reflection: Reflection, label: string, name?: string): EntryPointInfo {
  // A named entry validates existence and stage inside selectEntryPoint (VGPU-ENTRY-INVALID); only the
  // no-name case can come back undefined, keeping today's error for a shader without any @compute entry.
  const entry = selectEntryPoint(label, reflection.entryPoints, "compute", name, "compute");
  if (!entry) throw unsupportedError(`${label}.compute`, "The compute shader requires a @compute entry point.");
  return entry;
}

/** Contract #17, shared by dispatch()/dispatchOnce(): workgroup counts must be integers >= 0. */
function validateDispatchCount(label: string, field: string, value: number, where: string): void {
  if (Number.isInteger(value) && value >= 0) return;
  throw dispatchCountInvalidError(label, field, value, where);
}
