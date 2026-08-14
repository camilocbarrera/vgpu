import { compute, effect } from "../../src/index.ts";
import type { Gpu } from "../../src/index.ts";

declare const gpu: Gpu;
declare const wgsl: string;

// Two-argument form still type-checks (aditive overloads must not break the existing call sites).
effect(gpu, wgsl, { blend: "additive" });
compute(gpu, wgsl, { entry: "main" });

// Single-object form type-checks when `shader` is present.
effect(gpu, { shader: wgsl, blend: "additive" });
compute(gpu, { shader: wgsl, entry: "main" });

// Shorthand (no opts) still type-checks.
effect(gpu, wgsl);
compute(gpu, wgsl);

// @ts-expect-error an options object without `shader` is not a valid single-argument call.
effect(gpu, {});
// @ts-expect-error an options object without `shader` is not a valid single-argument call.
compute(gpu, {});
