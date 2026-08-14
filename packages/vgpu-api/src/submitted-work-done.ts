import type { Device } from "@vgpu/core";

/**
 * Resolves after the device queue reports submitted work completion, when supported.
 *
 * This is feature-guarded because the mock and some compatibility environments
 * may omit `onSubmittedWorkDone`.
 *
 * Deliberately a standalone leaf module (no feature-module imports, no other exports): both
 * `kernel.ts` (the `init`-only Ring-1 core, budget-constrained) and `claim-validation.ts` (render
 * infra) depend on this single function without either depending on the other's surface.
 */
export function submittedWorkDone(device: Device): Promise<void> {
  return device.gpu.queue.onSubmittedWorkDone?.() ?? Promise.resolve();
}
