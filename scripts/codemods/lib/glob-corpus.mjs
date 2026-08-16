// Shared corpus enumeration for the 04/codemod-tooling harness (T04-15).
//
// Every codemod downstream (T04-16..19) migrates the SAME set of real-world files. Rather than
// re-deriving the file list per codemod, this module is the single source of truth for "which
// files count as corpus" — each codemod imports `getCorpusFiles()` and then filters the result by
// its own pattern of interest (e.g. `\beffect\(` for unified-signature).
//
// Uses `git ls-files` (already the mechanism `scripts/verify-docs-snippets.mjs` uses) instead of a
// glob package, so this module adds no new dependency — `:(glob)` pathspec magic gives real `**`
// semantics (plain fnmatch pathspecs do not always expand `**` across directory boundaries).
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";

const CORPUS_GLOBS = [
  "apps/docs/examples/**/*.ts",
  "apps/docs/examples/**/*.tsx",
  "apps/docs/components/hero/**/*.ts",
  "examples/**/src/**/*.ts",
  // packages/*/tests/** and experiments/** carry real, migratable call sites too (T04-16/17
  // explicitly enumerate incidental usage inside `packages/vgpu-api/tests/**` and
  // `experiments/ort-init-device/shared/pipeline.ts` as part of their verified counts) —
  // without these, getCorpusFiles() silently hides ~35-60% of the sites those two codemods must
  // touch. Which files inside `packages/*/tests/**` are the legacy-form SUBJECT of a test (must
  // stay on the old signature) vs incidental usage (must migrate) is each codemod's own
  // classification job — this module's only job is to make sure the file is visible at all.
  "packages/*/tests/**/*.ts",
  "packages/*/tests/**/*.tsx",
  "experiments/**/*.ts",
];

// `apps/docs/app/**/*.ts(x)` is scanned separately: only files that actually import `vgpu` belong
// to the corpus (most of `apps/docs/app/**` is site plumbing — routing, layout, etc — with no vgpu
// call sites at all).
const DOCS_APP_GLOBS = ["apps/docs/app/**/*.ts", "apps/docs/app/**/*.tsx"];

// Explicitly excluded: derived/generated content that regenerates itself (T04-20's concern, not
// this train's codemods) and build output.
const EXCLUDE_RE = /(^|\/)(node_modules|dist)\//u;
const GENERATED_RE = /\.generated\./u;
const DOCS_GENERATED_RE = /^apps\/docs\/generated\//u;

const VGPU_IMPORT_RE = /\bfrom\s+["']vgpu(?:\/(?:node|mock|scene|core|client))?["']|\brequire\(\s*["']vgpu/u;

// LIMITATION (documented, not fixed — this is `git ls-files`'s contract, not a bug here): this
// only sees TRACKED files. A brand-new example file created mid-branch and not yet `git add`ed is
// invisible to getCorpusFiles() until it is staged, with no warning. A codemod run should
// `git add -A` (or otherwise stage) new corpus files before running the harness against them.
function gitLsFiles(repoRoot, globs) {
  const args = ["ls-files", "--", ...globs.map((g) => `:(glob)${g}`)];
  const out = execFileSync("git", args, { cwd: repoRoot, encoding: "utf8" });
  return out.split("\n").filter(Boolean);
}

function isExcluded(relPath) {
  return EXCLUDE_RE.test(relPath) || GENERATED_RE.test(relPath) || DOCS_GENERATED_RE.test(relPath);
}

function importsVgpu(repoRoot, relPath) {
  const text = readFileSync(path.join(repoRoot, relPath), "utf8");
  return VGPU_IMPORT_RE.test(text);
}

/**
 * Returns the sorted, de-duplicated list of repo-relative paths that make up the codemod corpus
 * for this train (T04-16..19). `repoRoot` defaults to `process.cwd()`, but callers running from
 * outside the repo (per the T202 tooling-lives-outside-the-repo precedent) must pass it
 * explicitly.
 */
export function getCorpusFiles(repoRoot = process.cwd()) {
  const direct = gitLsFiles(repoRoot, CORPUS_GLOBS);
  const docsAppCandidates = gitLsFiles(repoRoot, DOCS_APP_GLOBS);
  const docsApp = docsAppCandidates.filter((f) => importsVgpu(repoRoot, f));

  const all = [...direct, ...docsApp].filter((f) => !isExcluded(f));
  return [...new Set(all)].sort();
}

/** Exposed for callers/tests that want to reason about the raw glob list without running git. */
export function corpusGlobs() {
  return { direct: [...CORPUS_GLOBS], docsApp: [...DOCS_APP_GLOBS] };
}
