# `scripts/codemods/` — shared harness for the 0.4 cut codemods (T04-15)

This directory holds the AST/text-splicing infrastructure shared by the four migration codemods of
the additive phase of the 0.4 cut (T04-16 unified-signature, T04-17 ownership binding-scoped,
T04-18 dispatch migration, T04-19 prepare-insertion). It does **not** contain any migration logic
itself — T04-15 is purely the reusable library; the actual `*.mjs` codemod entrypoints are added by
their own tickets alongside this README.

## Why this shape

This replicates the pattern that already worked in the previous train
(`~/codemod-t202/codemod.mjs`, 235 lines): use the TS Compiler API (`ts.createSourceFile`) **only**
to find real token boundaries — never to do a full AST rewrite — combined with targeted regexes and
text splicing by absolute offset. That keeps every codemod's diff small, auditable, and reviewable
by a human scanning the raw text diff, instead of hiding the transformation inside a printer/AST
round-trip that can reformat unrelated code. T202 migrated 2,760 call sites this way without
breaking anything; this train reuses the same approach for the cut's 3 remaining codemods (a 4th,
prepare-insertion, is semi-automated and uses the same primitives for its automatic slice).

## Modules

- **`lib/token-scan.mjs`** — `tokenStarts(text, fileName)` returns the `Set` of offsets where a
  real identifier/`this` token starts, so a regex match that lands inside a string, a template
  chunk, or a comment can be discarded instead of corrupting that text.
- **`lib/splice.mjs`** — `applyEdits(text, edits)` applies a list of `{start, end, replacement}`
  edits (offsets against the **original** text, never recomputed) back-to-front. Throws instead of
  silently corrupting output if two edits overlap; adjacent edits (one's `end` equal to the next's
  `start`) are fine.
- **`lib/glob-corpus.mjs`** — `getCorpusFiles(repoRoot)` enumerates the real corpus files this
  train's codemods operate on: `apps/docs/examples/**/*.ts(x)`, `apps/docs/components/hero/**/*.ts`,
  `apps/docs/app/**/*.ts(x)` (filtered to the ones that actually import `vgpu`),
  `examples/**/src/**/*.ts`, `packages/*/tests/**/*.ts(x)`, and `experiments/**/*.ts`. The last two
  zones exist because T04-16/T04-17 both cite incidental usage inside `packages/vgpu-api/tests/**`
  and `experiments/ort-init-device/shared/pipeline.ts` as part of their verified counts — without
  them, ~35–60% of those tickets' target sites are invisible here even though the report still
  *looks* complete (an adversarial QA pass caught this before push; see
  `lib/glob-corpus.test.ts`'s two coverage tests). Explicitly excludes `**/*.generated.*`,
  `**/dist/**`, `**/node_modules/**`, and `apps/docs/generated/**` (derived content, regenerates
  itself — see T04-20). Each codemod imports this list and filters it further by its own pattern of
  interest; none of them re-derive the corpus independently. Which files inside
  `packages/*/tests/**` are the legacy-form SUBJECT of a test (must stay on the old signature) vs.
  incidental usage (must migrate) is each downstream codemod's own classification job — this
  module only guarantees the file is visible.
  **Known limitation:** this is built on `git ls-files`, so it only sees *tracked* files — a new
  example file not yet `git add`ed is silently invisible until staged.
- **`lib/report.mjs`** — the standard `--dry-run` contract: `isDryRun(argv)`, `reportEntry(...)` /
  `formatReport(entries)` / `printReport(entries)` for the `{file, line, before, after,
  classification}` JSON report, and `writeUnlessDryRun({dryRun, file, text})` as the single choke
  point every codemod must funnel writes through.

## The hard rule

**Every codemod in this train runs with `--dry-run` first.** Its JSON report (from
`printReport()`) gets attached to the PR description. The diff produced by the real
(non-dry-run) run must correspond 1:1 to that report — this is what lets adversarial QA diff
"expected" against "actual" before a single byte changes on disk. **No downstream codemod PR
(T04-16/17/18/19) is approved without that report attached.**

## Usage sketch for a downstream codemod

```js
#!/usr/bin/env node
import { readFileSync } from "node:fs";
import { getCorpusFiles } from "./lib/glob-corpus.mjs";
import { tokenStarts } from "./lib/token-scan.mjs";
import { applyEdits } from "./lib/splice.mjs";
import { isDryRun, reportEntry, printReport, writeUnlessDryRun } from "./lib/report.mjs";

const dryRun = isDryRun();
const files = getCorpusFiles(process.cwd()).filter((f) => /* this codemod's own pattern */ true);

const report = [];
for (const file of files) {
  const text = readFileSync(file, "utf8");
  const starts = tokenStarts(text, file);
  // ... find matches, skip any whose offset is not in `starts` ...
  const edits = [/* {start, end, replacement} */];
  if (edits.length === 0) continue;
  const next = applyEdits(text, edits);
  report.push(reportEntry({ file, line: 1, before: text, after: next, classification: "auto" }));
  writeUnlessDryRun({ dryRun, file, text: next });
}
printReport(report);
```

## Tests

`lib/*.test.ts` run via the repo's root `vitest` (same mechanism as `scripts/bundle-budgets.test.ts`
— a `.test.ts` file importing directly from the sibling `.mjs` module; `vitest.config.ts`'s
`scripts/**/*.test.ts` include picks it up with no extra wiring). Run with:

```bash
pnpm vitest run scripts/codemods
```

### Mutation-testing note (post adversarial-QA pass)

A pre-push adversarial QA pass mutation-tested `token-scan.mjs` (10 hand-written mutants) and
`splice.mjs` (12 mutants) against this test suite. After the fixes and test cases added in that
pass, **19/22 mutants are killed**. The 3 that survive are genuine *equivalent mutants* — verified
by hand (see the git history of this file / the QA report for the repro) to produce byte-identical
output to the un-mutated code for every reachable, valid input, not gaps in test coverage:

- **`S-sort-by-end`** (`splice.mjs` sorts by `end` instead of `start`): for any *valid* (accepted,
  non-overlapping, non-duplicate) edit set, start-order and end-order are provably the same
  ordering — two disjoint ranges `[s1,e1)` and `[s2,e2)` with `s1 < s2` always also have `e1 <= s2`,
  so `e1 <= e2`. The only place the two orderings could differ is a tie on `start`, and the
  tie-break (`|| a.end - b.end`) sorts by `end` too — so this mutant is indistinguishable from the
  real code on any input `applyEdits` actually accepts.
- **`T-setParentNodes-false`** (`token-scan.mjs`'s `ts.createSourceFile` call): `getStart(source)`
  does not depend on `.parent` when the `sourceFile` is passed explicitly (as this module always
  does), so `setParentNodes` cannot change any offset this module reports.
  **Audited (do not re-litigate):** an independent adversarial QA pass re-checked this claim
  against 44 synthetic constructs (JSDoc `@example`/`@type`, decorators, static blocks,
  `#priv in obj`, `accessor`, `satisfies`, `using`/`await using`, numeric separators, BigInt,
  `?.`/`??`/`||=`, top-level `await`, `import ... with { type: "json" }`, regexp `v`/`d` flags,
  enum/namespace/`declare module`, type-only imports — each in both `.ts` and `.tsx`) *and* a
  differential run over the full real corpus (462 files at the time of the audit): **0
  divergences** in either check. This one is confirmed equivalent for real.

`T-target-ES5` (an old `ts.ScriptTarget`) is **not** equivalent, despite an earlier draft of this
README claiming otherwise — see `lib/token-scan.test.ts`'s
`"identifiers using ES2015+ ID_Start characters are still real token starts"` test and the note
below for why, and why the fix is "keep parsing at `ESNext`", not "add a caveat".
