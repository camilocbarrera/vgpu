import type { Compute, Gpu } from "../../src/index.ts";
import { compute } from "../../src/index.ts";

declare const gpu: Gpu;

const c: Compute = compute(gpu, "@compute @workgroup_size(1) fn main() {}");
c.set({});
c.dispatch(1);
void c.dispatchOnce(1);

// @ts-expect-error Compute exposes no `pipeline` field — the low-level escape hatch is `prepared.gpu`,
// added by prepare() in a later ticket, not a public field on the Compute returned by compute(gpu, source).
c.pipeline;
