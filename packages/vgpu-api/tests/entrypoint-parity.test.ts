import { expect, test } from "vitest";
import * as index from "../src/index.ts";
import * as mock from "../src/mock.ts";
import * as node from "../src/node.ts";

// `vgpu/mock` is the documented testing path and `vgpu/node` the documented headless one, so a
// value export that only leaves the root entrypoint is broken for both without anything failing at
// build time. That is how `uniform()` shipped bound to `.` alone while its sibling `uniforms()`
// worked everywhere, and it is a mistake every new API of the train can repeat for free.
//
// The check is one-directional on purpose: `./mock` and `./node` legitimately add their own exports
// (createMockAdapter, initFromDevice), they just may never be MISSING one.
for (const [name, entrypoint] of [["mock", mock] as const, ["node", node] as const]) {
  test(`./${name} re-exports every value export of the root entrypoint`, () => {
    const missing = Object.keys(index).filter((exported) => !(exported in entrypoint));
    expect(missing, `missing from ./${name}: ${missing.join(", ")}`).toEqual([]);
  });
}
