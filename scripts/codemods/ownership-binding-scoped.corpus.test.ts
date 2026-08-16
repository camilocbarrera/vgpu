// A standing invariant over the MIGRATED corpus, not a unit test of a transform.
//
// `ownership-binding-scoped.mjs` moves a resource into a construction-time `bindings` bag only when
// the value cannot change between the constructor and the `.set()` it replaced. Its first version
// decided that by counting syntactic assignment sites, which is a question about text, and pinned
// two ping-pong halves that a `swap()` was flipping underneath. The unit tests and the mutants pin
// the RULE; this file pins the RESULT: after the migration, no construction site in the whole corpus
// may pin a value it reads through something mutable.
//
// It matters that this runs over the real tree rather than fixtures. The sites it protects are
// asserted only by `tests/gpu/*.test.ts`, which are `describe.skipIf(VGPU_DOCKER_TEST !== "1")` — so
// on a normal run nothing else in the suite would notice the regression. It also keeps the guard
// honest across the tickets still to come: T04-18 and T04-19 rewrite these same construction calls.
import path from "node:path";
import { describe, expect, it } from "vitest";
import ts from "typescript";
import { createCorpusProgram } from "./lib/corpus-program.mjs";
import { instanceFactoryName } from "./lib/wgsl-oracle.mjs";
import { readsThroughMutatedObject } from "./ownership-binding-scoped.mjs";

const REPO_ROOT = path.resolve(import.meta.dirname, "../..");

/**
 * The one file allowed to pin a mutable resource, because it holds the deliberate pinned
 * counter-example this invariant describes: `ping-pong-binding-identity.test.ts` constructs
 * `bindings: { src: buf.read }` on purpose, next to a `buf.swap()`, to prove that form samples the
 * wrong half. The rule reads text, not reachability, so it cannot tell that arm from a real site —
 * and it should not try. Naming the file is the honest fix; hoisting the expression to hide it from
 * the rule would make the counter-example pass by accident of shape and quietly stop being one.
 */
const ALLOWLIST = new Set(["packages/vgpu-api/tests/ping-pong-binding-identity.test.ts"]);

/** The construction call a `bindings:` bag belongs to, or null if it is some other object. */
function owningConstruction(bindingsProp: ts.Node): ts.CallExpression | null {
  const bag = bindingsProp.parent;
  if (!bag || !ts.isObjectLiteralExpression(bag)) return null;
  const call = bag.parent;
  if (!call || !ts.isCallExpression(call)) return null;
  return instanceFactoryName(call) ? call : null;
}

describe("the migrated corpus", () => {
  it("never pins a resource it reads through a mutated object into `bindings`", () => {
    const { program, checker, corpus } = createCorpusProgram(REPO_ROOT);
    const offenders: string[] = [];

    for (const rel of corpus) {
      if (ALLOWLIST.has(rel)) continue;
      const sf = program.getSourceFile(path.join(REPO_ROOT, rel));
      if (!sf) continue;
      const visit = (node: ts.Node): void => {
        if (
          ts.isPropertyAssignment(node)
          && !ts.isComputedPropertyName(node.name)
          && node.name.getText(sf) === "bindings"
          && ts.isObjectLiteralExpression(node.initializer)
        ) {
          const ctorCall = owningConstruction(node);
          if (ctorCall) {
            for (const entry of node.initializer.properties) {
              if (!ts.isPropertyAssignment(entry)) continue;
              if (!readsThroughMutatedObject(entry.initializer, ctorCall, checker)) continue;
              const line = sf.getLineAndCharacterOfPosition(entry.getStart(sf)).line + 1;
              offenders.push(`${rel}:${line}  ${entry.getText(sf)}`);
            }
          }
        }
        ts.forEachChild(node, visit);
      };
      visit(sf);
    }

    // Named rather than counted: a regression should say which site and why, not just "1 != 0".
    expect(offenders).toEqual([]);
  });

  it("keeps the allowlist honest", () => {
    // An allowlist that outlives its reason is an exemption nobody re-reads. Both halves are pinned:
    // the file must still be IN the corpus (or the entry is dead and hiding nothing), and it must
    // still CONTAIN the pinned counter-example (or the exemption is now blanket cover for whatever
    // that file grows into next).
    const { program, corpus } = createCorpusProgram(REPO_ROOT);
    for (const rel of ALLOWLIST) {
      expect(corpus, `${rel} is allowlisted but not in the corpus`).toContain(rel);
      const sf = program.getSourceFile(path.join(REPO_ROOT, rel));
      expect(sf?.text ?? "", `${rel} no longer holds a pinned counter-example`).toMatch(/bindings:\s*\{\s*src:\s*buf\.read\s*\}/u);
    }
  });

  it("still sees the corpus it is supposed to be checking", () => {
    // Without this the test above passes vacuously the day the corpus enumeration breaks — the
    // silent-zero failure mode #342's B1 finding was about. 33 files carry a `bindings:` bag today.
    const { program, corpus } = createCorpusProgram(REPO_ROOT);
    let bags = 0;
    for (const rel of corpus) {
      const sf = program.getSourceFile(path.join(REPO_ROOT, rel));
      if (!sf) continue;
      const visit = (node: ts.Node): void => {
        if (
          ts.isPropertyAssignment(node)
          && !ts.isComputedPropertyName(node.name)
          && node.name.getText(sf) === "bindings"
          && ts.isObjectLiteralExpression(node.initializer)
          && owningConstruction(node)
        ) bags += 1;
        ts.forEachChild(node, visit);
      };
      visit(sf);
    }
    expect(bags).toBeGreaterThan(20);
  });
});
