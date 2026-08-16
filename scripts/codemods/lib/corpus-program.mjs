// Shared `ts.Program` + `ts.TypeChecker` over the codemod corpus (T04-15 harness, added by T04-17).
//
// T04-16 could discover its sites from AST *shape* alone (3-argument `effect(...)`), so it never
// needed a checker. T04-17 cannot: the ticket's hard rule is "classify by TYPE, never by variable
// name" — `globals.set({ time: t })` on a `SharedUniforms` must not be touched while
// `fx.set({ time: t })` on an `Effect` must, and the two are indistinguishable syntactically. Same
// for telling a texture apart from a vec2 in a property value. T04-18/T04-19 need the same object,
// so building it lives here rather than in one codemod.
//
// Why one hand-rolled option set instead of reading a `tsconfig.json`: the corpus spans FOUR
// programs with incompatible settings (`packages/*` are `composite` NodeNext projects that import
// `./foo.ts` with explicit extensions; `apps/docs` is a `bundler`-resolution Next.js app with an
// `@/*` path alias and JSX). No single project file covers them, and `tsc -b`'s per-project graph
// cannot answer "the type of this expression" across the whole corpus at once. The options below
// are the union that makes every corpus file resolve: NodeNext resolution (so `packages/*`'s
// `./x.ts` specifiers work) + `allowImportingTsExtensions` + the docs `@/*` paths + JSX.
//
// Correctness of this choice is not asserted, it is CHECKED: `assertCorpusProgram()` verifies that
// every corpus file produced a `SourceFile`, and each codemod additionally refuses to act on a
// receiver whose type it could not resolve (an unresolved receiver is a report bucket, never a
// silent skip). A resolution regression therefore shows up as files moving into that bucket.
import path from "node:path";
import ts from "typescript";
import { getCorpusFiles } from "./glob-corpus.mjs";

/** The compiler options used for the corpus program. Exported so tests can assert on them. */
export function corpusCompilerOptions(repoRoot) {
  return {
    target: ts.ScriptTarget.ESNext,
    module: ts.ModuleKind.NodeNext,
    moduleResolution: ts.ModuleResolutionKind.NodeNext,
    allowImportingTsExtensions: true,
    rewriteRelativeImportExtensions: true,
    resolveJsonModule: true,
    jsx: ts.JsxEmit.ReactJSX,
    strict: true,
    skipLibCheck: true,
    noEmit: true,
    allowJs: true,
    baseUrl: path.join(repoRoot, "apps/docs"),
    paths: { "@/*": ["./*"], "@/.source": [".source"] },
    lib: ["lib.esnext.d.ts", "lib.dom.d.ts", "lib.dom.iterable.d.ts"],
    types: ["node", "@webgpu/types"],
  };
}

/**
 * Builds the corpus program. Returns `{ program, checker, corpus, sourceFileFor }` where
 * `sourceFileFor(relPath)` is the repo-relative accessor the codemods use.
 */
export function createCorpusProgram(repoRoot, { corpus = getCorpusFiles(repoRoot) } = {}) {
  const program = ts.createProgram(corpus.map((f) => path.join(repoRoot, f)), corpusCompilerOptions(repoRoot));
  const checker = program.getTypeChecker();
  const sourceFileFor = (relPath) => program.getSourceFile(path.join(repoRoot, relPath));
  return { program, checker, corpus, sourceFileFor, repoRoot };
}

/**
 * Fails loudly if any corpus file is missing from the program. A missing SourceFile means the
 * codemod would report "0 sites" for a file it cannot see — exactly the silent under-migration
 * failure mode #342's B1 finding was about, one level up (corpus visible, program blind).
 */
export function assertCorpusProgram({ corpus, sourceFileFor }) {
  const missing = corpus.filter((f) => !sourceFileFor(f));
  if (missing.length > 0) {
    throw new Error(
      `corpus-program: ${missing.length} corpus file(s) produced no SourceFile — the codemod would `
        + `silently report zero sites for them: ${missing.slice(0, 10).join(", ")}`
        + `${missing.length > 10 ? `, +${missing.length - 10} more` : ""}`,
    );
  }
}

/** Strips `as T` / `satisfies T` / `(...)` / `!` wrappers to reach the expression underneath. */
export function unwrap(node) {
  let cur = node;
  for (;;) {
    if (
      ts.isParenthesizedExpression(cur)
      || ts.isAsExpression(cur)
      || ts.isSatisfiesExpression(cur)
      || ts.isNonNullExpression(cur)
    ) {
      cur = cur.expression;
      continue;
    }
    return cur;
  }
}
