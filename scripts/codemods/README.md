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
  `apps/docs/app/**/*.ts(x)` (filtered to the ones that actually import `vgpu`), and
  `examples/**/src/**/*.ts`. Explicitly excludes `**/*.generated.*`, `**/dist/**`,
  `**/node_modules/**`, and `apps/docs/generated/**` (derived content, regenerates itself — see
  T04-20). Each codemod imports this list and filters it further by its own pattern of interest;
  none of them re-derive the corpus independently.
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
