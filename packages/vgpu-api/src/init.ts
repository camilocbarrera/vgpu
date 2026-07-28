export { createGpu, type Gpu } from "./gpu.ts";
export type { AdapterFactory, InitOptions } from "./kernel.ts";
export type { Compute, ComputeOptions, DispatchOptions, GpuErrorListener, PingPongStorage, PingPongTargets, SharedUniforms, StorageAccess, StorageBuffer, StorageOptions } from "./api-types.ts";
export type { ClearColor } from "./target-utils.ts";
export type { Timer, TimerSpan } from "./timer.ts";
export type { Visibility, VisibilityOptions, VisibilityQuery } from "./visibility.ts";
export type { Bundle, BundleOptions, BundleRecorder } from "./bundle.ts";
export type { Surface, SurfaceOptions, SurfaceResizeEvent } from "./surface.ts";

import { createGpu } from "./gpu.ts";
import type { AdapterFactory, EntryKind, InitOptions } from "./kernel.ts";

export function initWithAdapter(entry: EntryKind, adapterFactory?: AdapterFactory, options?: InitOptions) {
  return createGpu(entry, options, adapterFactory);
}
