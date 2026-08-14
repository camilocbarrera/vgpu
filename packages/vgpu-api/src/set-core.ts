import { bindGroupLayoutMetadata, bindGroupMetadataFor, type Buffer, type Device, type UnsubscribeResourceDestroy } from "@vgpu/core";
import type { BindingInfo, Reflection } from "@vgpu/wgsl/reflect-source";
import { identityKey, type BindGroupCache, type BindGroupIdentityPart } from "./bind-cache.ts";
import { entryMetadata } from "./entry-metadata.ts";
import { claimedGroupIncompatibleError, claimedGroupSetError, externalBindingError, neverSetError, ownershipFlipError, unsupportedError } from "./errors.ts";
import { bindGroupLayoutEntriesForGroup, bindGroupLayoutsForReflection, pipelineLayoutFor } from "./set-layouts.ts";
import { isPlainObject, isPlainValue, normalizeResource } from "./set-resources.ts";
import { writeLayoutValue } from "./set-packing.ts";

export type SetBag = Record<string, unknown>;
export type BindingOwnership = "lib" | "user";

export interface SetCoreOptions {
  readonly device: Device;
  readonly label: string;
  readonly drawId: number;
  readonly reflection: Reflection;
  readonly bindGroupLayouts: ReadonlyMap<number, GPUBindGroupLayout>;
  readonly cache: BindGroupCache;
  readonly onIdentityChange?: (change: BindingIdentityChange) => void;
  /**
   * Initial values for instance-owned (value-owned) bindings, keyed by WGSL binding name.
   * Declaring a binding here pins it value-owned at construction — it never latches by call order.
   */
  readonly values?: Readonly<Record<string, unknown>>;
  /**
   * Externally-owned resources, keyed by WGSL binding name. A binding declared here is external
   * from construction: `.set()` on it fails with VGPU-R1-EXTERNAL-BINDING, and only `.bind()`
   * swaps its identity.
   */
  readonly bindings?: Readonly<Record<string, unknown>>;
}

export interface BindingIdentityChange {
  readonly group: number;
  readonly binding: number;
  readonly bindingName: string;
  readonly bindingKind: string;
  readonly previousIdentity?: string;
  readonly newIdentity: string;
}

/** Ring-1 set() engine: latches ownership, validates completeness, and returns cached bind groups. */
export interface SetCore {
  readonly groups: readonly number[];
  set(values: SetBag): readonly BindingIdentityChange[];
  /**
   * Binding-scoped write: `binding` names a complete WGSL binding (never a struct member) owned by
   * this instance, `value` is a partial for a struct binding and the complete value otherwise.
   * One call is one buffer write.
   */
  setScoped(binding: string, value: unknown): readonly BindingIdentityChange[];
  /**
   * Identity swap for an externally-owned binding. Dedupes by resource identity — rebinding the same
   * resource is free and rebuilds nothing.
   */
  bind(binding: string, resource: unknown): readonly BindingIdentityChange[];
  claimGroup(group: number, bindGroup: GPUBindGroup, expectedLayout: GPUBindGroupLayout): string | undefined;
  layout(group: number): GPUBindGroupLayout;
  bindGroups(): readonly { readonly group: number; readonly bindGroup: GPUBindGroup; readonly offsets: readonly number[]; readonly claimValidation?: { readonly label: string; readonly group: number } }[];
  bindingState(name: string): BindingState | undefined;
}

export interface BindingState {
  readonly info: BindingInfo;
  readonly ownership: BindingOwnership;
  readonly resource: GPUBindingResource;
  readonly identity: BindGroupIdentityPart;
}

type MutableBindingState = {
  readonly info: BindingInfo;
  ownership?: BindingOwnership;
  /**
   * Construction-time (or `.bind()`-time) declaration that the resource behind this binding belongs
   * to someone else. Distinct from `ownership === "user"`, which the legacy flat bag latches by call
   * order: `external` is never inferred from a value, so VGPU-R1-EXTERNAL-BINDING can only fire for
   * the `{ bindings }` / `.bind()` forms.
   */
  external?: boolean;
  /** Last resource passed to `.bind()`, kept for the free identity dedup of a repeated rebind. */
  boundValue?: unknown;
  readonly memberOwnership: Map<string, BindingOwnership>;
  buffer?: Buffer;
  bytes?: ArrayBuffer;
  libValue?: unknown;
  resource?: GPUBindingResource;
  identity?: BindGroupIdentityPart;
  unsubscribe?: UnsubscribeResourceDestroy;
  unsubscribeRecreate?: () => void;
};

/** Creates the per-Draw binding state machine used by Effect/Draw.set(). */
export function createSetCore(options: SetCoreOptions): SetCore {
  const bindings = initializeBindings(options.reflection);
  const groups = [...options.bindGroupLayouts.keys()].sort((a, b) => a - b);
  const claimedGroups = new Map<number, GPUBindGroup>();

  function set(values: SetBag): readonly BindingIdentityChange[] {
    const changes: BindingIdentityChange[] = [];
    for (const [name, value] of Object.entries(values)) changes.push(...setNamedValue(name, value));
    return changes;
  }

  /**
   * Construction-time ownership. `bindings` entries become external immediately (no latch, so
   * VGPU-R1-OWNERSHIP-FLIP is unreachable for them) and `values` entries become value-owned with
   * their storage created and written right here.
   */
  function declareOwnership(): void {
    const declaredValues = options.values ?? {};
    for (const [name, resource] of Object.entries(options.bindings ?? {})) {
      if (Object.hasOwn(declaredValues, name)) throw unsupportedError(options.label, `Binding '${name}' is declared in both values and bindings of '${options.label}'.`, "Ownership is fixed at construction: declare it in exactly one of them.");
      bindExternal(requiredBindingByName(name, "bindings"), resource);
    }
    for (const [name, value] of Object.entries(declaredValues)) setScoped(name, value);
  }

  function bindExternal(state: MutableBindingState, resource: unknown): void {
    state.external = true;
    state.ownership = "user";
    state.boundValue = resource;
    setUserOwned(state, resource);
  }

  /**
   * Binding-scoped write. Resolves `name` as a complete binding only — the member-name shortcut of
   * the flat bag does not exist here (design §6: "no member-name shortcut"), and a member name gets
   * an error that names the binding to use instead.
   */
  function setScoped(name: string, value: unknown): readonly BindingIdentityChange[] {
    const state = requiredBindingByName(name, "set");
    // The external check comes first: naming an externally-bound resource is the interesting
    // mistake, and its fix-it is more useful than "this is not a JS value".
    ensureNotExternal(state, name);
    if (ownershipFor(state.info, value) !== "lib") {
      throw unsupportedError(`${options.label}.set`, `Binding '${name}' in '${options.label}' received a resource; set() writes bytes.`, `Swap resource identity with ${options.label}.bind("${name}", resource).`);
    }
    // Same write path as the flat bag once the name is resolved: a struct merges the partial on the
    // CPU and rewrites completely (one buffer write), a non-struct value replaces wholesale.
    return setBinding(state, name, value);
  }

  /**
   * Identity swap. Dedupes on the resource reference first (the per-frame rebind of the same object
   * is free) and then on the normalized resource identity, so a rebind that resolves to the same
   * GPU resource reports no change and the bind group cache keeps its entry.
   */
  function bind(name: string, resource: unknown): readonly BindingIdentityChange[] {
    const state = requiredBindingByName(name, "bind");
    ensureGroupSettable(state.info.group);
    // Owned -> external transitions are not supported (design §6): recreate the instance instead.
    if (!state.external && state.ownership === "lib") throw ownershipFlipError(name, "lib");
    if (state.resource && Object.is(state.boundValue, resource)) return [];
    const before = identityString(state.identity);
    bindExternal(state, resource);
    return bindingIsActive(state) ? identityChangeFor(state, before) : [];
  }

  /**
   * Resolves a complete binding by name. A struct member name is not a binding here (design §6: no
   * member-name shortcut), so it gets the fix-it that names the binding to use instead — reusing the
   * flat bag's member lookup, which already reports an ambiguous member name.
   */
  function requiredBindingByName(name: string, verb: string): MutableBindingState {
    const state = bindings.get(name);
    if (state) return state;
    const owner = findMemberBinding(name, bindings, options.label);
    const fix = owner ? `'${name}' is a member of binding '${owner.info.name}': ${options.label}.set("${owner.info.name}", { ${name}: value }).` : undefined;
    throw unsupportedError(`${options.label}.${verb}`, `Binding '${name}' does not exist in '${options.label}'.`, fix);
  }

  function bindingIsActive(state: MutableBindingState): boolean {
    const layout = options.bindGroupLayouts.get(state.info.group);
    return !!layout && !!bindGroupLayoutMetadata(layout)?.entries.some((entry) => entry.binding === state.info.binding);
  }

  function setNamedValue(name: string, value: unknown): readonly BindingIdentityChange[] {
    const direct = bindings.get(name);
    if (direct) return setBinding(direct, name, value);
    const member = findMemberBinding(name, bindings, options.label);
    if (!member) throw unsupportedError(`${options.label}.set`, `Binding '${name}' does not exist in '${options.label}'.`);
    return setBindingMember(member, name, value);
  }

  function setBinding(state: MutableBindingState, name: string, value: unknown): readonly BindingIdentityChange[] {
    ensureGroupSettable(state.info.group);
    ensureNotExternal(state, name);
    const ownership = ownershipFor(state.info, value);
    latchBindingOwnership(state, name, ownership);
    const before = identityString(state.identity);
    if (ownership === "lib") setLibOwned(state, mergeLibValue(state.libValue, value));
    else setUserOwned(state, value);
    return bindingIsActive(state) ? identityChangeFor(state, before) : [];
  }

  function setBindingMember(state: MutableBindingState, memberName: string, value: unknown): readonly BindingIdentityChange[] {
    ensureGroupSettable(state.info.group);
    ensureNotExternal(state, state.info.name);
    const ownership = ownershipFor(state.info, value);
    latchBindingOwnership(state, memberName, ownership);
    latchMemberOwnership(state, memberName, ownership);
    if (ownership !== "lib") throw unsupportedError(`${options.label}.set`, `Member '${memberName}' needs a JS value; set resource '${state.info.name}' instead.`);
    const before = identityString(state.identity);
    setLibOwned(state, { ...objectValue(state.libValue), [memberName]: value });
    return bindingIsActive(state) ? identityChangeFor(state, before) : [];
  }

  function setLibOwned(state: MutableBindingState, value: unknown): void {
    const layout = requiredLibLayout(state);
    state.libValue = value;
    const bytes = writeLayoutValue(layout, value);
    if (!state.buffer) createLibBuffer(state, layout.size);
    state.bytes = bytes;
    state.buffer!.write(bytes, 0);
  }

  function resourceContext(binding: BindingInfo) {
    const entry = bindGroupLayoutMetadata(options.bindGroupLayouts.get(binding.group)!)?.entries.find((item) => item.binding === binding.binding);
    const pair = options.reflection.entryPoints.flatMap((item) => entryMetadata(item, "samplingPairs", options.label)).find((item) => item.mode === "filtering" && item.texture.group === binding.group && item.texture.binding === binding.binding);
    const pairedSampler = pair && options.reflection.bindings.find((item) => item.group === pair.sampler.group && item.binding === pair.sampler.binding);
    return { sourceHint: options.label, filterableTexture: entry?.texture?.sampleType === "float", float32Filterable: options.device.features.has("float32-filterable"), pairedSampler };
  }

  function setUserOwned(state: MutableBindingState, value: unknown): void {
    const normalized = normalizeResource(state.info, value, resourceContext(state.info));
    state.unsubscribe?.();
    state.unsubscribeRecreate?.();
    state.resource = normalized.resource;
    state.identity = normalized.identity;
    state.unsubscribe = normalized.unsubscribe?.(() => { if (state.identity) options.cache.evictIdentity(state.identity); });
    state.unsubscribeRecreate = normalized.onRecreate?.(() => rebindRecreatedResource(state, value));
  }

  function rebindRecreatedResource(state: MutableBindingState, value: unknown): void {
    const beforeIdentity = identityString(state.identity);
    if (state.identity) options.cache.evictIdentity(state.identity);
    const normalized = normalizeResource(state.info, value, resourceContext(state.info));
    state.unsubscribe?.();
    state.unsubscribeRecreate?.();
    state.resource = normalized.resource;
    state.identity = normalized.identity;
    state.unsubscribe = normalized.unsubscribe?.(() => { if (state.identity) options.cache.evictIdentity(state.identity); });
    // Refresh the recreation subscription on every re-normalization so the lifecycle
    // stays explicit even if a future target signal implementation becomes one-shot.
    state.unsubscribeRecreate = normalized.onRecreate?.(() => rebindRecreatedResource(state, value));
    if (bindingIsActive(state)) for (const change of identityChangeFor(state, beforeIdentity)) options.onIdentityChange?.(change);
  }

  function claimGroup(group: number, bindGroup: GPUBindGroup, expectedLayout: GPUBindGroupLayout): string | undefined {
    layout(group);
    validateClaimedGroup(options.label, group, bindGroup, expectedLayout);
    const previousIdentity = claimedGroups.has(group) ? `claimed-group:${group}` : undefined;
    claimedGroups.set(group, bindGroup);
    return previousIdentity;
  }

  function layout(group: number): GPUBindGroupLayout {
    const bgl = options.bindGroupLayouts.get(group);
    if (!bgl) throw unsupportedError(`${options.label}.layout`, `@group(${group}) does not exist in '${options.label}'.`);
    return bgl;
  }

  function bindGroups(): readonly { readonly group: number; readonly bindGroup: GPUBindGroup; readonly offsets: readonly number[]; readonly claimValidation?: { readonly label: string; readonly group: number } }[] {
    return groups.map(bindGroupFor);
  }

  function bindGroupFor(group: number): { readonly group: number; readonly bindGroup: GPUBindGroup; readonly offsets: readonly number[]; readonly claimValidation?: { readonly label: string; readonly group: number } } {
    const claimed = claimedGroups.get(group);
    if (claimed) return { group, bindGroup: claimed, offsets: [], claimValidation: rawClaimValidation(claimed, group) };
    const active = new Set(bindGroupLayoutMetadata(layout(group))?.entries.map((entry) => entry.binding));
    const groupBindings = options.reflection.bindings.filter((binding) => binding.group === group && active.has(binding.binding));
    const entries = bindGroupEntries(groupBindings);
    const identities = identitiesFor(groupBindings);
    const bindGroup = options.cache.getOrCreate(options.drawId, group, identities, () => options.device.gpu.createBindGroup({
      label: `${options.label}.group${group}`,
      layout: layout(group),
      entries,
    }));
    return { group, bindGroup, offsets: [] };
  }

  function rawClaimValidation(bindGroup: GPUBindGroup, group: number): { readonly label: string; readonly group: number } | undefined {
    return bindGroupMetadataFor(bindGroup) ? undefined : { label: options.label, group };
  }

  function bindGroupEntries(groupBindings: readonly BindingInfo[]): GPUBindGroupEntry[] {
    return groupBindings.map((binding) => {
      const state = requiredState(binding);
      return { binding: binding.binding, resource: state.resource! };
    });
  }

  function identitiesFor(groupBindings: readonly BindingInfo[]): BindGroupIdentityPart[] {
    return groupBindings.map((binding) => requiredState(binding).identity!);
  }

  function requiredState(binding: BindingInfo): MutableBindingState {
    const state = bindings.get(binding.name);
    if (!state?.resource || !state.identity) throw neverSetError(options.label, binding);
    return state;
  }

  function ensureGroupSettable(group: number): void {
    if (claimedGroups.has(group)) throw claimedGroupSetError(options.label, group);
  }

  function ensureNotExternal(state: MutableBindingState, name: string): void {
    if (state.external) throw externalBindingError(`${options.label}.set`, options.label, name);
  }

  function createLibBuffer(state: MutableBindingState, size: number): void {
    state.buffer = options.device.createBuffer({ size, usage: ["uniform", "copy_dst"], label: `${options.label}.${state.info.name}` });
    state.resource = { buffer: state.buffer.gpu, offset: 0, size };
    state.identity = state.buffer.resourceIdentity;
    state.unsubscribe = state.buffer.onDestroy(() => options.cache.evictIdentity(state.buffer!.resourceIdentity));
  }

  function requiredLibLayout(state: MutableBindingState): NonNullable<BindingInfo["layout"]> & { readonly size: number } {
    if (state.info.kind !== "buffer" || !state.info.layout?.size) throw unsupportedError(`${options.label}.set`, `Binding '${state.info.name}' needs a compatible resource, not JS.`);
    return state.info.layout as NonNullable<BindingInfo["layout"]> & { readonly size: number };
  }

  declareOwnership();

  return {
    get groups() { return groups; },
    set,
    setScoped,
    bind,
    claimGroup,
    layout,
    bindGroups,
    bindingState(name) {
      const state = bindings.get(name);
      if (!state?.ownership || !state.resource || !state.identity) return undefined;
      return { info: state.info, ownership: state.ownership, resource: state.resource, identity: state.identity };
    },
  };
}

function initializeBindings(reflection: Reflection): Map<string, MutableBindingState> {
  return new Map(reflection.bindings.map((binding) => [binding.name, { info: binding, memberOwnership: new Map() }]));
}

function reflectedGroups(reflection: Reflection): readonly number[] {
  return [...new Set(reflection.bindings.map((binding) => binding.group))].sort((a, b) => a - b);
}

function findMemberBinding(memberName: string, bindings: ReadonlyMap<string, MutableBindingState>, label: string): MutableBindingState | undefined {
  let match: MutableBindingState | undefined;
  for (const state of bindings.values()) {
    if (!state.info.layout?.members?.some((member) => member.name === memberName)) continue;
    if (match) throw unsupportedError(`${label}.set`, `Binding member '${memberName}' is ambiguous in '${label}'; set the complete binding.`);
    match = state;
  }
  return match;
}

function ownershipFor(binding: BindingInfo, value: unknown): BindingOwnership {
  return binding.bindingLayout?.kind === "buffer" && isPlainValue(value) ? "lib" : "user";
}

function latchBindingOwnership(state: MutableBindingState, name: string, ownership: BindingOwnership): void {
  if (state.ownership && state.ownership !== ownership) throw ownershipFlipError(name, state.ownership);
  state.ownership ??= ownership;
}

function latchMemberOwnership(state: MutableBindingState, memberName: string, ownership: BindingOwnership): void {
  const previous = state.memberOwnership.get(memberName);
  if (previous && previous !== ownership) throw ownershipFlipError(memberName, previous);
  state.memberOwnership.set(memberName, ownership);
}

function validateClaimedGroup(label: string, group: number, bindGroup: GPUBindGroup, expectedLayout: GPUBindGroupLayout): void {
  const claimedMetadata = bindGroupMetadataFor(bindGroup);
  if (!claimedMetadata) return;
  const expectedMetadata = bindGroupLayoutMetadata(expectedLayout);
  if (!expectedMetadata) return;
  const reason = layoutMismatchReason(expectedMetadata.entries, claimedMetadata.layout.entries);
  if (reason) throw claimedGroupIncompatibleError(label, group, reason);
}

function layoutMismatchReason(expected: readonly GPUBindGroupLayoutEntry[], claimed: readonly GPUBindGroupLayoutEntry[]): string | undefined {
  if (expected.length !== claimed.length) return `expected ${expected.length} bindings and received ${claimed.length}`;
  const expectedByBinding = entriesByBinding(expected);
  const claimedByBinding = entriesByBinding(claimed);
  for (const [binding, entry] of expectedByBinding) {
    const claimedEntry = claimedByBinding.get(binding);
    if (!claimedEntry) return `missing @binding(${binding})`;
    if (entrySignature(entry) !== entrySignature(claimedEntry)) return `@binding(${binding}) does not match the reflected layout`;
  }
  return undefined;
}

function entriesByBinding(entries: readonly GPUBindGroupLayoutEntry[]): ReadonlyMap<number, GPUBindGroupLayoutEntry> {
  return new Map(entries.map((entry) => [entry.binding, entry]));
}

function entrySignature(entry: GPUBindGroupLayoutEntry): string {
  return JSON.stringify({
    binding: entry.binding,
    visibility: entry.visibility,
    buffer: entry.buffer,
    sampler: entry.sampler,
    texture: entry.texture,
    storageTexture: entry.storageTexture,
    externalTexture: entry.externalTexture ? {} : undefined,
  });
}

function identityChangeFor(state: MutableBindingState, previousIdentity: string | undefined): readonly BindingIdentityChange[] {
  const nextIdentity = identityString(state.identity);
  if (!nextIdentity || previousIdentity === nextIdentity) return [];
  return [{
    group: state.info.group,
    binding: state.info.binding,
    bindingName: state.info.name,
    bindingKind: state.info.kind,
    previousIdentity,
    newIdentity: nextIdentity,
  }];
}

function identityString(identity: BindGroupIdentityPart | undefined): string | undefined {
  return identity === undefined ? undefined : identityKey(identity);
}

function mergeLibValue(previous: unknown, value: unknown): unknown {
  return isPlainObject(previous) && isPlainObject(value) ? { ...previous, ...value } : value;
}

function objectValue(value: unknown): Record<string, unknown> {
  return isPlainObject(value) ? value : {};
}

export { bindGroupLayoutEntriesForGroup, bindGroupLayoutsForReflection, pipelineLayoutFor } from "./set-layouts.ts";
export { writeLayoutValue } from "./set-packing.ts";
