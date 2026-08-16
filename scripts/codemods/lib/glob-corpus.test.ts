import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "vitest";
import { corpusGlobs, getCorpusFiles } from "./glob-corpus.mjs";
import { tokenStarts } from "./token-scan.mjs";

const repoRoot = execFileSync("git", ["rev-parse", "--show-toplevel"], { encoding: "utf8" }).trim();
void fileURLToPath; // keep import graph explicit if this file is ever run standalone

test("getCorpusFiles returns real, existing, non-empty corpus paths", () => {
  const files = getCorpusFiles(repoRoot);
  expect(files.length).toBeGreaterThan(0);
  for (const f of files.slice(0, 20)) {
    expect(f).not.toMatch(/node_modules\//u);
    expect(f).not.toMatch(/\/dist\//u);
    expect(f).not.toMatch(/\.generated\./u);
    expect(f).not.toMatch(/^apps\/docs\/generated\//u);
  }
});

test("getCorpusFiles includes known by-example directories under examples/**/src", () => {
  const files = getCorpusFiles(repoRoot);
  expect(files.some((f) => f.startsWith("examples/by-example-s02-fullscreen/src/"))).toBe(true);
});

test("getCorpusFiles includes apps/docs/examples files", () => {
  const files = getCorpusFiles(repoRoot);
  expect(files.some((f) => f.startsWith("apps/docs/examples/"))).toBe(true);
});

test("getCorpusFiles de-duplicates and sorts its output", () => {
  const files = getCorpusFiles(repoRoot);
  expect(files).toEqual([...new Set(files)].sort());
});

test("corpusGlobs exposes the raw pattern lists for introspection", () => {
  const globs = corpusGlobs();
  expect(globs.direct).toContain("apps/docs/examples/**/*.ts");
  expect(globs.docsApp).toContain("apps/docs/app/**/*.ts");
});

// --- adversarial QA pre-push additions (T04-15, B1) --------------------------------------------
// This is the acceptance test the QA report calls out explicitly: getCorpusFiles() must be a
// superset of the exact file sets the downstream tickets (T04-16/T04-18) verified against the
// real tree — a codemod that follows this module's corpus would otherwise under-migrate ~35-60%
// of its target sites while still printing a --dry-run report that *looks* complete.

test("getCorpusFiles covers T04-18's 6 legacy .dispatch( files exactly", () => {
  const files = getCorpusFiles(repoRoot);
  const t18 = [
    "apps/docs/examples/air-painting/visual-pipeline.ts",
    "apps/docs/examples/depth-estimation/renderer.ts",
    "apps/docs/examples/fft-ocean-surface/scene.ts",
    "apps/docs/examples/fluid/simulation.ts",
    "examples/by-example-s11-compute/src/example.ts",
    "examples/fluid-validation/src/fluid-gpu.test.ts",
  ];
  for (const f of t18) expect(files).toContain(f);
});

test("getCorpusFiles covers every real (token-verified) unified-signature call site — none of T04-16's target files are invisible to it", () => {
  // Independently re-derive "which .ts/.tsx files actually contain a real effect(gpu, x, {}) /
  // compute(gpu, x, {}) call site" via git ls-files + token-scan (the same mechanism T04-16 must
  // use), then assert getCorpusFiles() is a superset — this is what actually caught the
  // packages/*/tests/** and experiments/** gap (39 files) that a hand-picked file-count assertion
  // would not have.
  const re = /\b(effect|compute)\(\s*gpu\s*,\s*[A-Za-z_$][\w.]*\s*,\s*\{/g;
  const allTsFiles = execFileSync(
    "git",
    [
      "ls-files",
      "--",
      ":(glob)**/*.ts",
      ":(glob)**/*.tsx",
      ":(glob,exclude)**/node_modules/**",
      ":(glob,exclude)**/dist/**",
      ":(glob,exclude)**/*.generated.*",
    ],
    { cwd: repoRoot, encoding: "utf8" },
  )
    .split("\n")
    .filter(Boolean);

  const corpus = new Set(getCorpusFiles(repoRoot));
  const missing = [];
  for (const rel of allTsFiles) {
    const text = readFileSync(path.join(repoRoot, rel), "utf8");
    re.lastIndex = 0;
    if (!re.test(text)) continue;
    re.lastIndex = 0;
    const starts = tokenStarts(text, rel);
    let hasRealSite = false;
    let m;
    while ((m = re.exec(text)) !== null) {
      if (starts.has(m.index)) { hasRealSite = true; break; }
    }
    if (hasRealSite && !corpus.has(rel)) missing.push(rel);
  }
  expect(missing).toEqual([]);
});
