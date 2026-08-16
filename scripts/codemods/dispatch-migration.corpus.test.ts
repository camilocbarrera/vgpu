// A standing invariant over the MIGRATED corpus, not a unit test of a transform.
//
// `dispatch-migration.test.ts` pins the RULE against in-memory fixtures; this file pins the RESULT
// against the real tree: after T04-18, no `.dispatch()` call anywhere in the corpus may still sit in
// a context the codemod would have migrated on its own. That is the check that keeps the migration
// from silently regressing — a new example added later, in an `async` function, with a legacy
// `dispatch()` in it, fails here instead of surfacing as a build break in T04-22 when the method is
// deleted.
//
// It matters that this runs over the real tree rather than fixtures, and that it is STRUCTURAL. The
// behaviour the migrated sites have is only asserted by suites gated on real hardware
// (`describe.skipIf(process.env.VGPU_DOCKER_TEST !== '1')` in `examples/fluid-validation`, and
// `tests/gpu/*`), so on a normal CI run nothing else in the suite would notice a regression at these
// call sites at all. Same reasoning, and same shape, as T04-17's
// `ownership-binding-scoped.corpus.test.ts`.
import path from "node:path";
import { describe, expect, it } from "vitest";
import { createCorpusProgram } from "./lib/corpus-program.mjs";
import { planProgram, LEGACY_DISPATCH_TEST_SUBJECTS } from "./dispatch-migration.mjs";

const REPO_ROOT = path.resolve(import.meta.dirname, "../..");

/**
 * The `.dispatch()` sites T04-18 deliberately left behind, and why. Every one is a call whose
 * containing function is synchronous and whose `void` return is part of a signature the module
 * publishes, so `await dispatchOnce()` cannot be inserted without changing that contract, and no
 * `Frame` is lexically in scope to offer `f.compute()` instead.
 *
 * This table is DEBT, pinned so it cannot grow by accident: **T04-22 cannot delete `dispatch()`
 * until it is empty.** Each of these needs a hand decision that is a design change, not a rewrite —
 * either thread a `Frame` into the function (which is what the rev6.1 design actually wants: it puts
 * these dispatches in the same submit as the render that consumes them) or make the function async
 * and propagate that to its callers.
 *
 * Counted per file rather than pinned per line so that unrelated edits above a site do not fail
 * this test, while adding or removing a site does.
 */
const AMBIGUOUS_DEBT: ReadonlyArray<readonly [file: string, sites: number, reason: string]> = [
  ["apps/docs/examples/air-painting/visual-pipeline.ts", 3, "`dispatchCrop` is module-local but its two callers `cropDetectorInput()`/`cropLandmarkInput()` and `consumeHandLandmarks()` are all declared `: void` on the exported `VisualPipeline` interface, and `ort-runtime.ts` calls them from sync paths."],
  ["apps/docs/examples/depth-estimation/renderer.ts", 1, "`SideBySidePipeline.draw()` is declared `: void`. This one has a `runFrame()` five lines below it in the SAME method — the natural migration is to move the reduce inside that frame callback as `f.compute(reducer, 1)`, which is code motion, not a rewrite the codemod may do."],
  ["apps/docs/examples/fft-ocean-surface/scene.ts", 4, "`OceanScene.simulate()`/`rebuildSpectrum()` are declared `: void`. `renderer.ts` calls `scene.simulate(dt)` from INSIDE a `frameLoop((currentFrame) => …)` callback — dynamically in a frame, but not lexically, so there is no `f` at the dispatch. Threading the `Frame` into `simulate()` is the migration the design wants and it changes the example's published interface."],
  ["apps/docs/examples/fluid/simulation.ts", 7, "`stepFluid()` is an exported `: void` function called in tight loops (up to 5,000 iterations in `validation.ts`); 7 awaits per step is a different program, not the same one. This one wants a `Frame` threaded through, not async propagation."],
];

function planCorpus() {
  const ctx = createCorpusProgram(REPO_ROOT);
  return { ...ctx, ...planProgram(ctx) };
}

describe("the migrated corpus", () => {
  it("has no `.dispatch()` left that the codemod could have migrated on its own", () => {
    const { entries, fileTexts } = planCorpus();
    // Named rather than counted: a regression should say which site and why, not just "1 != 0".
    const migratable = entries
      .filter((e: any) => String(e.classification).startsWith("auto-"))
      .map((e: any) => `${e.file}:${e.line}  ${e.classification}  ${e.before}`);
    expect(migratable).toEqual([]);
    // The same claim from the other side: re-running the codemod for real would write nothing.
    expect([...fileTexts.keys()]).toEqual([]);
  });

  it("pins the ambiguous debt T04-22 has to clear by hand", () => {
    const { entries } = planCorpus();
    const byFile = new Map<string, number>();
    for (const e of entries as any[]) {
      if (!String(e.classification).startsWith("ambiguous")) continue;
      byFile.set(e.file, (byFile.get(e.file) ?? 0) + 1);
    }
    expect([...byFile].sort()).toEqual(AMBIGUOUS_DEBT.map(([file, n]) => [file, n]).sort());
    // Every ambiguous site carries a machine-readable reason and a human-readable note — "classified,
    // never silent" is the property, and an empty reason would satisfy the count check above.
    for (const e of entries as any[]) {
      if (!String(e.classification).startsWith("ambiguous")) continue;
      expect(e.reason, `${e.file}:${e.line} has no reason`).toBeTruthy();
      expect(e.note, `${e.file}:${e.line} has no note`).toBeTruthy();
    }
  });

  it("keeps LEGACY_DISPATCH_TEST_SUBJECTS honest", () => {
    // An exclusion that outlives its reason is an exemption nobody re-reads. Both halves are pinned:
    // the file must still be in the corpus AND still contain a `.dispatch()` site, or the entry is
    // dead and hiding nothing.
    const { entries, corpus } = planCorpus();
    for (const [rel, reason] of LEGACY_DISPATCH_TEST_SUBJECTS) {
      expect(corpus, `${rel} is excluded but not in the corpus`).toContain(rel);
      expect(
        (entries as any[]).some((e) => e.file === rel && e.classification === "excluded-test-subject"),
        `${rel} no longer has a \`.dispatch()\` site — drop the exclusion`,
      ).toBe(true);
      expect(reason.length, `${rel} has no stated reason`).toBeGreaterThan(40);
    }
  });

  it("still sees the corpus it is supposed to be checking", () => {
    // Without this the tests above pass vacuously the day the corpus enumeration breaks — the
    // silent-zero failure mode #342's B1 finding was about. Measured on this branch: the tree-wide
    // universe was 74 sites in 21 files immediately before the migration was applied, and 68 sites
    // in 18 files after it (the three migrated files no longer contain a legacy `.dispatch()` at
    // all). The assertions are floors, not equalities, so an unrelated new call site does not fail
    // here — the first test in this file catches the ones that matter.
    const { entries, corpus, sourceFileFor } = planCorpus();
    expect(entries.length).toBeGreaterThanOrEqual(60);
    expect(new Set((entries as any[]).map((e) => e.file)).size).toBeGreaterThanOrEqual(17);
    // The three files T04-18 migrated are the other half of the same claim: they must still be
    // ENUMERATED (or "nothing left to migrate" is true because the codemod stopped looking) and they
    // must actually hold the migrated spelling.
    for (const rel of [
      "examples/by-example-s11-compute/src/example.ts",
      "examples/fluid-validation/src/fluid-gpu.test.ts",
      "experiments/ort-init-device/shared/pipeline.ts",
    ]) {
      expect(corpus, `${rel} vanished from the corpus enumeration`).toContain(rel);
      expect(sourceFileFor(rel)?.text ?? "", `${rel} lost its migrated call`).toMatch(/await\s+[^;]*\.dispatchOnce\(/u);
    }
  });
});
