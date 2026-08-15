import { readFileSync } from "node:fs";
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

// The runtime half above cannot see a TYPE export: types are erased, so `import * as node` has no
// key for one. A type that only leaves the root entrypoint is just as broken for a `vgpu/node` user
// (`BundleStatus` shipped exactly like that, and `InitOptions` had been missing from ./node for
// longer), and nothing in the build fails. Reading the export lists as text is crude but it is the
// only check that runs without a second typecheck project, and it is the one that would have caught
// both. The same one-directional rule applies: an entrypoint may add types, never miss one.
function typeExportsOf(entrypoint: string): string[] {
  const source = readFileSync(new URL(`../src/${entrypoint}`, import.meta.url), "utf8");
  return [...source.matchAll(/export\s+type\s*\{([^}]*)\}/g)]
    .flatMap(([, group]) => group.split(","))
    .map((entry) => entry.trim().split(/\s+as\s+/).pop()?.trim() ?? "")
    .filter((name) => name.length > 0);
}

for (const name of ["mock", "node"]) {
  test(`./${name} re-exports every type export of the root entrypoint`, () => {
    const exported = new Set(typeExportsOf(`${name}.ts`));
    const missing = typeExportsOf("index.ts").filter((type) => !exported.has(type));
    expect(missing, `missing from ./${name}: ${missing.join(", ")}`).toEqual([]);
  });
}
