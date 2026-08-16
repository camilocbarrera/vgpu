#!/usr/bin/env node
// T04-16 — codemod 1 of the 0.4 cut: migrate every real call site off the positional
// `effect(gpu, source, opts)` / `compute(gpu, source, opts)` form onto the single-object form
// `effect(gpu, { shader: source, ...opts })`.
//
// ADDITIVE PHASE. This codemod does not touch `effect.ts`/`compute.ts` — all four overloads stay
// alive until T04-22. It only migrates *callers*, so the corpus is green before and after.
//
//   effect(gpu, SRC, { blend: "additive" })  ->  effect(gpu, { shader: SRC, blend: "additive" })
//   effect(gpu, SRC, opts)                   ->  effect(gpu, { shader: SRC, ...opts })
//   effect(gpu, SRC)                         ->  untouched (the 1-arg shorthand survives the cut)
//
// ## Discovery is by AST SHAPE, not by regex
//
// The plan's discovery regex (`(effect|compute)\(\s*gpu\s*,\s*[A-Za-z_$][\w.]*\s*,\s*\{`) has
// documented false negatives: it only accepts an identifier/dotted-path as the source argument and
// only an object literal as the options argument. On this corpus it finds 269 of the 279 real sites
// and misses 10 (source is a call — `withTopLeftFullscreen(logoWgsl)`; a ShaderSource artifact
// object literal — `{ version: 1, wgsl: FRAGMENT }`; or a template literal). So the matcher here is
// the real thing: walk every `ts.CallExpression` whose callee is (or ends in) `effect`/`compute`
// and which has exactly 3 arguments. `tokenStarts()` from the shared harness is still consulted, as
// a redundant cross-check that the offset this codemod is about to splice at really is a live token
// start (an independent implementation agreeing with our own node offsets), and
// `parseDiagnosticsCount()` gates the whole file: a file we could not parse cleanly is refused
// loudly instead of silently under-migrated.
//
// ## Splicing strategy: minimal, format-preserving text edits
//
// Per the harness README, this is never an AST reprint — that would reformat unrelated code and
// make the diff unreviewable. For the common case (options is an object literal) the transformation
// is two surgical edits:
//
//   1. delete the source argument (from the end of the preceding comma through its own comma), and
//   2. insert `shader: <source>,` as the FIRST property inside the existing options braces.
//
// which is why `{ blend: "additive" }` becomes `{ shader: SRC, blend: "additive" }` and not
// `{ shader: SRC, ...{ blend: "additive" } }`, and why a multi-line options bag keeps its exact
// formatting, its comments, and its trailing comma. Only when the options argument is *not* an
// object literal (an identifier, a property access, a call, …) does a real `...spread` get emitted.
//
// Source text is spliced VERBATIM and never re-indented — the source argument is frequently a
// template literal holding WGSL, where re-indenting would change the string's contents.
//
// Usage:
//   node scripts/codemods/unified-signature.mjs --dry-run        # report only, touches no files
//   node scripts/codemods/unified-signature.mjs                  # apply
//   node scripts/codemods/unified-signature.mjs --dry-run a.ts   # restrict to given corpus paths
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";
import { getCorpusFiles } from "./lib/glob-corpus.mjs";
import { parseDiagnosticsCount, tokenStarts } from "./lib/token-scan.mjs";
import { applyEdits } from "./lib/splice.mjs";
import { isDryRun, positionalArgs, reportEntry, printReport, writeUnlessDryRun } from "./lib/report.mjs";

const TARGET_CALLEES = new Set(["effect", "compute"]);

/**
 * Files whose SUBJECT is the legacy positional form itself: migrating their call sites would make
 * the assertion vacuous (or delete the only proof the retired overload still resolves). The ticket
 * requires these to stay on the old signature here and be removed by T04-22 (retiro), and to be
 * listed explicitly in the PR — hence a table with a written reason per file rather than a name
 * pattern.
 *
 * NOT in this table, deliberately (see the PR body): `packages/vgpu-api/tests/compile-api.test.ts`.
 * The ticket guessed it by filename, but its only positional site is
 * `effect(gpu, WGSL, { label: "fx" })` inside "Effect compile delegates to Draw, fixes gpu getter,
 * and shares the device store" — incidental usage; the "signature" that file is about is
 * `compile()`'s (`VGPU-COMPILE-SIGNATURE-INVALID`), not `effect()`'s argument form. Since T04-22
 * will not delete that test, leaving it on the retired form would break T04-22 instead of
 * protecting it.
 */
const LEGACY_FORM_TEST_SUBJECTS = new Map([
  [
    "packages/vgpu-api/tests/unified-signature.test.ts",
    "the whole file exists to prove effect(gpu, source, opts) and effect(gpu, { shader, ...opts }) "
      + "produce the same pipeline; its `fromTwoArgs` bindings ARE the legacy form under test, so "
      + "migrating them would compare the new form against itself. Deleted by T04-22.",
  ],
  [
    "packages/vgpu-api/tests/uniforms/unified-signature-types.ts",
    "type-level fixture whose own comment reads \"Two-argument form still type-checks (aditive "
      + "overloads must not break the existing call sites)\" — the positional overload IS the "
      + "assertion. Deleted by T04-22.",
  ],
]);

function scriptKindFor(fileName) {
  if (fileName.endsWith(".tsx")) return ts.ScriptKind.TSX;
  if (fileName.endsWith(".jsx")) return ts.ScriptKind.JSX;
  return ts.ScriptKind.TS;
}

/**
 * The `effect`/`compute` identifier node of a callee, for both a bare call (`effect(...)`) and a
 * namespace/alias call (`vgpu.effect(...)`, `api.compute(...)`) — the ticket calls out
 * `fft-ocean-surface/scene.ts`'s `api.effect` alias explicitly.
 */
function calleeNameNode(expr) {
  if (ts.isIdentifier(expr)) return expr;
  if (ts.isPropertyAccessExpression(expr) && ts.isIdentifier(expr.name)) return expr.name;
  return null;
}

/** Strips `as T` / `satisfies T` / `(...)` / `!` wrappers to reach the expression underneath. */
function unwrap(node) {
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

/**
 * Can `node` plausibly BE an options bag (i.e. is this really the 3-argument `effect`/`compute`
 * constructor form)? A numeric literal cannot, which is how the positional dispatch API
 * (`framePass.compute(job, x, y, z)` — same method name, entirely different function) is rejected
 * without needing a type checker.
 */
function isOptionsBagShaped(node) {
  const inner = unwrap(node);
  switch (inner.kind) {
    case ts.SyntaxKind.ObjectLiteralExpression:
    case ts.SyntaxKind.Identifier:
    case ts.SyntaxKind.PropertyAccessExpression:
    case ts.SyntaxKind.ElementAccessExpression:
    case ts.SyntaxKind.CallExpression:
    case ts.SyntaxKind.ConditionalExpression:
    case ts.SyntaxKind.AwaitExpression:
      return true;
    default:
      return false;
  }
}

/** Does this object literal already carry a `shader` key? */
function hasShaderProperty(objectLiteral) {
  return objectLiteral.properties.some((prop) => {
    const name = prop.name;
    if (!name) return false;
    if (ts.isIdentifier(name) || ts.isStringLiteral(name)) return name.text === "shader";
    return false;
  });
}

/**
 * `...expr` in an object literal parses an AssignmentExpression, so most shapes need no parens —
 * but wrap anything that is not a plain reference/call so the emitted spread cannot change meaning
 * (`...a as T` etc. stays unambiguous, and a future exotic shape degrades to "extra parens", never
 * to "wrong code").
 */
function spreadText(text, node, sourceFile) {
  const raw = text.slice(node.getStart(sourceFile), node.end);
  switch (node.kind) {
    case ts.SyntaxKind.Identifier:
    case ts.SyntaxKind.PropertyAccessExpression:
    case ts.SyntaxKind.ElementAccessExpression:
    case ts.SyntaxKind.CallExpression:
    case ts.SyntaxKind.NonNullExpression:
    case ts.SyntaxKind.ParenthesizedExpression:
    case ts.SyntaxKind.ObjectLiteralExpression:
      return `...${raw}`;
    default:
      return `...(${raw})`;
  }
}

/**
 * A property value also parses an AssignmentExpression, so the ONLY shape that has to be
 * parenthesized when it moves into `shader: <here>` is a comma/sequence expression, whose comma
 * would otherwise read as the property separator.
 */
function propertyValueText(text, node, sourceFile) {
  const raw = text.slice(node.getStart(sourceFile), node.end);
  if (ts.isBinaryExpression(node) && node.operatorToken.kind === ts.SyntaxKind.CommaToken) {
    return `(${raw})`;
  }
  return raw;
}

/** Offset just past the `,` that follows `from`, skipping whitespace. `-1` if there is none. */
function commaEndAfter(text, from) {
  let i = from;
  while (i < text.length && /\s/u.test(text[i])) i++;
  return text[i] === "," ? i + 1 : -1;
}

/** The leading whitespace of the line `offset` sits on, or `null` if anything else precedes it. */
function lineIndentAt(text, offset) {
  const lineStart = text.lastIndexOf("\n", offset - 1) + 1;
  const prefix = text.slice(lineStart, offset);
  return /^[ \t]*$/u.test(prefix) ? prefix : null;
}

/**
 * Builds the edits for one site, or a reason to skip it.
 * @returns {{edits: {start: number, end: number, replacement: string}[], classification: string}
 *   | {skip: string}}
 */
function planSite(text, sourceFile, call) {
  const [, sourceArg, optionsArg] = call.arguments;

  // An already-migrated call that additionally passes a (silently ignored) third argument must not
  // be "migrated" again — `shader` would end up nested inside itself.
  const innerSource = unwrap(sourceArg);
  if (ts.isObjectLiteralExpression(innerSource) && hasShaderProperty(innerSource)) {
    return { skip: "skipped-already-single-object" };
  }

  if (!isOptionsBagShaped(optionsArg)) {
    return { skip: "skipped-not-options-bag" };
  }

  // Everything between the previous comma and the start of the options argument is about to be
  // deleted. If a human left a comment in there, splicing would silently eat it — refuse instead.
  const removalStart = sourceArg.getFullStart();
  const interArgText = text.slice(removalStart, optionsArg.getStart(sourceFile));
  if (interArgText.includes("//") || interArgText.includes("/*")) {
    return { skip: "skipped-manual-review-comment-between-args" };
  }

  const sourceText = propertyValueText(text, sourceArg, sourceFile);
  const options = unwrap(optionsArg);

  // --- options is NOT an object literal: emit a real spread, replacing both arguments at once.
  if (!ts.isObjectLiteralExpression(options)) {
    const start = sourceArg.getStart(sourceFile);
    return {
      classification: "auto-spread-opts",
      edits: [{
        start,
        end: optionsArg.end,
        replacement: `{ shader: ${sourceText}, ${spreadText(text, optionsArg, sourceFile)} }`,
      }],
    };
  }

  if (hasShaderProperty(options)) {
    return { skip: "skipped-manual-review-shader-key-conflict" };
  }

  const commaEnd = commaEndAfter(text, sourceArg.end);
  if (commaEnd === -1 || commaEnd > optionsArg.getStart(sourceFile)) {
    return { skip: "skipped-manual-review-unexpected-argument-separator" };
  }
  const removeSourceArg = { start: removalStart, end: commaEnd, replacement: "" };

  // --- empty options bag: no spread of nothing, just the shader key. Replacing the braces
  // wholesale (rather than inserting into them) keeps `{}` from becoming `{ shader: X  }`.
  if (options.properties.length === 0) {
    return {
      classification: "auto-inline-opts-empty",
      edits: [
        removeSourceArg,
        {
          start: options.getStart(sourceFile),
          end: options.end,
          replacement: `{ shader: ${sourceText} }`,
        },
      ],
    };
  }

  // --- options bag with properties: splice `shader: <source>,` in as the first property, matching
  // the bag's own layout so the diff is one added line (or one widened line) and nothing else.
  const insertAt = options.getStart(sourceFile) + 1;
  const firstProperty = options.properties[0];
  const gap = text.slice(insertAt, firstProperty.getStart(sourceFile));
  let insertion;
  if (gap.includes("\n")) {
    const indent = lineIndentAt(text, firstProperty.getStart(sourceFile))
      ?? `${lineIndentAt(text, options.getStart(sourceFile)) ?? ""}  `;
    insertion = `\n${indent}shader: ${sourceText},`;
  } else if (gap === "") {
    insertion = `shader: ${sourceText}, `;
  } else {
    insertion = ` shader: ${sourceText},`;
  }

  return {
    classification: unwrap(optionsArg) === optionsArg ? "auto-inline-opts" : "auto-inline-opts-asserted",
    edits: [removeSourceArg, { start: insertAt, end: insertAt, replacement: insertion }],
  };
}

/**
 * The whole transformation for ONE file, as a pure function of its text — no fs, no corpus, no
 * argv. This is the unit under test (`unified-signature.test.ts`) and the unit the DoD's mutation
 * pass targets; `main()` below is only corpus iteration + IO around it.
 *
 * @param {string} text - the file's source text.
 * @param {string} relPath - repo-relative path (drives `.tsx` script-kind selection and reporting).
 * @param {{legacySubjectReason?: string}} [options] - when `legacySubjectReason` is set, every site
 *   in the file is reported as `excluded-test-subject` and the text is returned unchanged.
 * @returns {{text: string, entries: {file: string, line: number, before: string, after: string,
 *   classification: string}[]}}
 */
export function transformSource(text, relPath, options = {}) {
  const diagnostics = parseDiagnosticsCount(text, relPath);
  if (diagnostics > 0) {
    throw new Error(
      `unified-signature: ${relPath} has ${diagnostics} parse diagnostics — refusing to trust `
        + `its token offsets. Fix the file (or its extension) and re-run; a partial token set `
        + `would silently under-migrate it.`,
    );
  }

  const { sourceFile, sites } = collectSites(text, relPath);
  if (sites.length === 0) return { text, entries: [] };

  const liveTokens = tokenStarts(text, relPath);
  const entries = [];
  const edits = [];

  for (const { call, nameNode } of sites) {
    const callStart = call.getStart(sourceFile);
    const line = sourceFile.getLineAndCharacterOfPosition(callStart).line + 1;
    const before = text.slice(callStart, call.end);

    // Redundant-by-design cross-check: an independent implementation must agree that the callee we
    // matched is a live token and not text inside a string/template/comment.
    if (!liveTokens.has(nameNode.getStart(sourceFile))) {
      throw new Error(
        `unified-signature: ${relPath}:${line} — AST found a call site whose callee offset is not `
          + `a live token start according to token-scan. Refusing to splice.`,
      );
    }

    const planned = options.legacySubjectReason
      ? { skip: "excluded-test-subject" }
      : planSite(text, sourceFile, call);

    if (planned.skip) {
      entries.push(reportEntry({ file: relPath, line, before, after: before, classification: planned.skip }));
      continue;
    }

    // The reported `after` is produced by the SAME edits that will hit disk, re-based onto the call
    // slice — so a report entry can never claim a rewrite the real run does not perform.
    const after = applyEdits(
      before,
      planned.edits.map((e) => ({ ...e, start: e.start - callStart, end: e.end - callStart })),
    );
    entries.push(reportEntry({ file: relPath, line, before, after, classification: planned.classification }));
    edits.push(...planned.edits);
  }

  if (edits.length === 0) return { text, entries };
  const next = applyEdits(text, edits);
  if (next === text) {
    throw new Error(`unified-signature: ${relPath} produced ${edits.length} edits but no change`);
  }
  return { text: next, entries };
}

function collectSites(text, relPath) {
  const sourceFile = ts.createSourceFile(
    relPath,
    text,
    ts.ScriptTarget.ESNext,
    true,
    scriptKindFor(relPath),
  );
  const sites = [];
  const visit = (node) => {
    if (ts.isCallExpression(node) && node.arguments.length === 3) {
      const nameNode = calleeNameNode(node.expression);
      if (nameNode && TARGET_CALLEES.has(nameNode.text)) sites.push({ call: node, nameNode });
    }
    ts.forEachChild(node, visit);
  };
  visit(sourceFile);
  sites.sort((a, b) => a.call.getStart(sourceFile) - b.call.getStart(sourceFile));
  return { sourceFile, sites };
}

function main() {
  const dryRun = isDryRun();
  const only = new Set(positionalArgs());
  const repoRoot = process.cwd();

  const corpus = getCorpusFiles(repoRoot);
  const files = only.size > 0 ? corpus.filter((f) => only.has(f)) : corpus;
  if (only.size > 0 && files.length !== only.size) {
    throw new Error(
      `unified-signature: these paths are not in the codemod corpus: `
        + `${[...only].filter((f) => !files.includes(f)).join(", ")}`,
    );
  }

  const report = [];
  const tally = new Map();
  const bump = (key) => tally.set(key, (tally.get(key) ?? 0) + 1);
  const touchedFiles = [];

  for (const relPath of files) {
    const absPath = path.join(repoRoot, relPath);
    const text = readFileSync(absPath, "utf8");
    if (!/\b(effect|compute)\s*\(/u.test(text)) continue;

    const legacySubjectReason = LEGACY_FORM_TEST_SUBJECTS.get(relPath);
    const { text: next, entries } = transformSource(text, relPath, { legacySubjectReason });
    for (const entry of entries) {
      report.push(entry);
      bump(entry.classification);
    }

    if (next === text) continue;
    touchedFiles.push(relPath);
    writeUnlessDryRun({ dryRun, file: absPath, text: next });
  }

  const stale = [...LEGACY_FORM_TEST_SUBJECTS.keys()].filter(
    (f) => !report.some((entry) => entry.file === f),
  );
  if (stale.length > 0 && only.size === 0) {
    throw new Error(
      `unified-signature: LEGACY_FORM_TEST_SUBJECTS is stale — no call site found in `
        + `${stale.join(", ")}. Drop the entry (or fix the path) instead of carrying a dead `
        + `exclusion that hides future sites.`,
    );
  }

  printReport(report);

  const migrated = [...tally].filter(([k]) => k.startsWith("auto-")).reduce((n, [, v]) => n + v, 0);
  process.stderr.write(
    `\n${dryRun ? "[dry-run] " : ""}unified-signature: ${report.length} call sites examined in `
      + `${new Set(report.map((e) => e.file)).size} files; ${migrated} migrated across `
      + `${touchedFiles.length} files.\n`
      + [...tally].sort().map(([k, v]) => `  ${String(v).padStart(4)}  ${k}\n`).join("")
      + (dryRun ? "  (no files written)\n" : ""),
  );

  for (const [file, reason] of LEGACY_FORM_TEST_SUBJECTS) {
    process.stderr.write(`  excluded on purpose: ${file}\n    ${reason}\n`);
  }
}

// Only drive the corpus when invoked as a script — `unified-signature.test.ts` imports
// `transformSource` from here and must not trigger a repo-wide run (let alone a write) on import.
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
