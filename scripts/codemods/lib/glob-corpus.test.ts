import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { expect, test } from "vitest";
import { corpusGlobs, getCorpusFiles } from "./glob-corpus.mjs";

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
