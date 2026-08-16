#!/usr/bin/env node
// T04-18 — codemod 3 of the 0.4 cut: retire the legacy lazy-sync `Compute.dispatch()` at the call
// sites, onto the two paths the rev6.1 design defines:
//
//   frame(gpu, (f) => { sim.dispatch(n) })   ->  frame(gpu, (f) => { f.compute(sim, n) })
//   sim.dispatch(n)            (standalone)  ->  await sim.dispatchOnce(n)
//
// ADDITIVE PHASE. `compute.ts` is untouched — `dispatch()` the METHOD stays alive until T04-22.
// This codemod only migrates callers.
//
// ## Why this is the most delicate transform of the chain, despite being the smallest
//
// `dispatchOnce()` is **async**. `dispatch()` is not. Inserting an `await` is therefore not a
// rename: it splits the containing function at that point, and every caller of that function now
// observes work that used to be finished on return still in flight. The transform is only sound
// where the `await` costs nothing observable — where the enclosing function is ALREADY async (its
// callers already await it, its declared return type is already a promise), or at the top level of
// an ES module (top-level await). Anywhere else — a sync callback handed to a third party, a method
// whose `: void` return type is fixed by an interface the module exports, a `expect(() => …)` arm —
// forcing async would change the module's contract, so the site goes to a NAMED bucket with the
// reason, and a human decides. `dispatch()` cannot be removed in T04-22 until those buckets are
// empty; leaving them silent would let T04-22 discover them as build breaks instead.
//
// `f.compute()` has the opposite property: it is sync, so it is always safe where it applies. What
// limits it is scope — it needs the frame's `f` in hand. "Inside a frame" here means LEXICALLY
// inside the callback and directly in it, not merely reached from one at runtime: a `f.compute()`
// nested inside `f.pass(…)`/`f.raw(…)` throws VGPU-FRAME-ENCODER-LOCKED, and a dispatch that a
// frame callback reaches through a helper in another module has no `f` to name. (The corpus's
// motivating example, `fft-ocean-surface`, is exactly that second case — see the report.)
//
// ## Oracles, no heuristics
//
//  1. **Type checker** (`lib/corpus-program.mjs`) decides that a `.dispatch()` receiver really is
//     one of our `Compute` instances — structurally, by carrying both `dispatch` and `dispatchOnce`
//     — never by variable name, and never by the method name alone (any object may own a
//     `.dispatch()`). An unresolvable receiver is a bucket, never a silent skip.
//  2. **Type checker** again for the ARGUMENTS: `dispatch(x, y?, z?)` and `dispatch(opts)` are two
//     overloads and only the first has an `f.compute()` twin (`f.compute` takes no `indirect`).
//     Number-like arguments pick the counts overload; anything else is the options overload.
//  3. **AST scope** for the context: the nearest enclosing function-like node, whether it is async,
//     and whether it is the frame callback itself.
//
// Usage:
//   node scripts/codemods/dispatch-migration.mjs --dry-run          # report only, no writes
//   node scripts/codemods/dispatch-migration.mjs                    # apply
//   node scripts/codemods/dispatch-migration.mjs --dry-run a.ts     # restrict to paths
import path from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";
import { createCorpusProgram, assertCorpusProgram, unwrap } from "./lib/corpus-program.mjs";
import { getCorpusFiles } from "./lib/glob-corpus.mjs";
import { tokenStarts, parseDiagnosticsCount } from "./lib/token-scan.mjs";
import { applyEdits } from "./lib/splice.mjs";
import { isDryRun, positionalArgs, reportEntry, printReport, writeUnlessDryRun } from "./lib/report.mjs";

/**
 * Files whose SUBJECT is `dispatch()` itself — the lazy-sync legacy path, its overloads, its
 * validation, and its interaction with the paths that replace it. Their calls are the assertion,
 * not a way of getting work onto the GPU: migrating them would delete the proof that the behaviour
 * T04-22 is about to remove still behaves as documented while it is still shipped. T04-22 owns
 * their deletion/rewrite, together with the method.
 *
 * Every entry is checked against the real corpus on each run (see `main()`): an entry that no
 * longer matches a site is a stale exemption and fails the run rather than quietly covering
 * nothing. The context each excluded site WOULD have been classified as is still computed and
 * reported (`contextWouldBe`), so the exclusion never hides an inconvenient site.
 */
export const LEGACY_DISPATCH_TEST_SUBJECTS = new Map([
  ["packages/vgpu-api/tests/compute/dispatch-once.test.ts", "T04-04's acceptance suite: it pins dispatch() vs dispatchOnce() side by side (pipeline reuse across the two, the sync-compile-wins race, shared VGPU-R1-DISPATCH-COUNT validation, VGPU-DEVICE-DISPOSED). Every assertion needs a live legacy dispatch() to make."],
  ["packages/vgpu-api/tests/compute/aliasing.test.ts", "The writable-storage aliasing preflight is asserted THROUGH dispatch() (`expect(() => sim.dispatch(1)).toThrowError(...)`): the call is the trigger under test, and its synchronous throw is what the assertion shape depends on."],
  ["packages/vgpu-api/tests/compute/compute-pipeline-types.ts", "A `tsc`-only type fixture for the Compute surface: it exists to prove the legacy `dispatch()` overloads still type-check. It is never executed."],
  ["packages/vgpu-api/tests/indirect.test.ts", "Subject is the `dispatch(opts: DispatchOptions)` overload (VGPU-INDIRECT-INVALID for size/alignment, buffer+offset form). f.compute() has no indirect overload at all, so these sites cannot move until T04-22 decides indirect's final home."],
  ["packages/vgpu-api/tests/settled-queue.test.ts", "T04-03 pins that legacy dispatch() returns `undefined` synchronously (`const result = sim.dispatch(1)`) while the queue settles separately — the sync return IS the assertion."],
  ["packages/vgpu-api/tests/frame-unified.test.ts", "T04-08 compares f.compute()'s error codes against legacy dispatch()'s on the same input (`caught(() => sim.dispatch(1))?.code`): the legacy call is the reference side of the comparison."],
  ["packages/vgpu-api/tests/unified-signature.test.ts", "T04-01 pins that a Compute built from either signature spelling is dispatchable on the legacy path (`expect(() => x.dispatch(1)).not.toThrow()`) — the legacy path is half of the parity claim."],
  ["packages/vgpu-api/tests/prepare.test.ts", "T04-05 pins the interaction between prepare()/pendingPipelines and the lazy-sync compile that only dispatch() performs: it is the un-prepared control arm."],
  ["packages/vgpu-api/tests/external-device-init.test.ts", "Pins VGPU-DEVICE-DISPOSED / VGPU-DEVICE-LOST on the synchronous path; an awaited twin would report through a rejected promise instead and stop testing the sync guard."],
  ["packages/vgpu-api/tests/ownership.test.ts", "T04-06's acceptance suite (already the excluded subject of T04-17): its dispatch() calls are what force the bind-group build whose ownership latching the file asserts."],
  ["packages/vgpu-api/tests/override-constants.test.ts", "The override constants only reach the GPU through the pipeline dispatch() compiles inline, lazily, at the call — the lazy-sync compile is the mechanism under test."],
  ["packages/vgpu-api/tests/entry-selection.test.ts", "Same mechanism: entry-point selection is observed through the pipeline that the lazy-sync dispatch() compiles."],
  ["packages/vgpu-api/tests/dispatch-order-motivation.test.ts", "T04-18's own motivation-#1 recording: it runs legacy dispatch() inside a frame callback ON PURPOSE, next to the f.compute() arm, to record that the legacy form costs an extra submit per dispatch and inverts execution order when it follows a pass. Migrating those calls would delete the control arm and leave two identical recordings. (This file was caught by dispatch-migration.corpus.test.ts on its first full run — the invariant works.)"],
  ["packages/vgpu-api/tests/gpu/compute.test.ts", "The real-device (`VGPU_DOCKER_TEST`) arm for the legacy dispatch path; it is the only place its end-to-end behaviour on hardware is pinned."],
]);

// --------------------------------------------------------------------------------------------------
// oracle 1 — is the receiver one of our Compute instances?
// --------------------------------------------------------------------------------------------------

/**
 * `"compute" | "foreign" | "unresolved"` for a `.dispatch()` receiver.
 *
 * Structural, never nominal: a `Compute` is the thing that carries BOTH `dispatch` and
 * `dispatchOnce`. `dispatch` alone is far too common (schedulers, stores, event buses) and the
 * corpus's own `renderer.ts` has an unrelated `.draw()`-shaped object; keying on the method name
 * would migrate somebody else's API.
 */
export function classifyReceiver(expr, checker) {
  const type = checker.getTypeAtLocation(expr);
  if (!type) return { kind: "unresolved", typeText: "<no type>" };
  const parts = type.isUnion() ? type.types : [type];
  let compute = false;
  let foreign = false;
  let unresolved = false;
  for (const t of parts) {
    if (t.flags & (ts.TypeFlags.Undefined | ts.TypeFlags.Null)) continue;
    if (t.flags & (ts.TypeFlags.Any | ts.TypeFlags.Unknown)) { unresolved = true; continue; }
    if (t.getProperty("dispatch") && t.getProperty("dispatchOnce")) { compute = true; continue; }
    foreign = true;
  }
  const typeText = checker.typeToString(type);
  if (unresolved) return { kind: "unresolved", typeText };
  // A union of "ours" and "not ours" is not migratable either way.
  if (compute && foreign) return { kind: "unresolved", typeText };
  if (compute) return { kind: "compute", typeText };
  return { kind: "foreign", typeText };
}

// --------------------------------------------------------------------------------------------------
// oracle 2 — which `dispatch()` overload is this call?
// --------------------------------------------------------------------------------------------------

/**
 * `"counts" | "options" | "unresolved"`.
 *
 * `dispatch(x, y?, z?)` maps onto `f.compute(inst, x, y?, z?)` and onto `dispatchOnce(x, y?, z?)`.
 * `dispatch({ indirect })` maps onto `dispatchOnce({ indirect })` but has NO `f.compute()` twin
 * (`FrameComputeOptions` carries no `indirect`), so the two destinations disagree and the overload
 * has to be known before a destination is chosen.
 */
export function classifyOverload(call, checker) {
  if (call.arguments.length === 0) return "unresolved";
  if (call.arguments.length > 3) return "unresolved";
  let sawNumber = false;
  let sawOther = false;
  for (const arg of call.arguments) {
    if (ts.isSpreadElement(arg)) return "unresolved";
    const inner = unwrap(arg);
    if (ts.isObjectLiteralExpression(inner)) { sawOther = true; continue; }
    const type = checker.getTypeAtLocation(inner);
    const parts = type?.isUnion() ? type.types : [type];
    let numeric = parts != null && parts.length > 0;
    for (const t of parts ?? []) {
      if (t.flags & (ts.TypeFlags.Undefined | ts.TypeFlags.Null)) continue;
      if (!(t.flags & ts.TypeFlags.NumberLike)) { numeric = false; break; }
    }
    if (numeric) sawNumber = true;
    else sawOther = true;
  }
  if (sawNumber && !sawOther) return "counts";
  if (sawOther && !sawNumber && call.arguments.length === 1) return "options";
  return "unresolved";
}

// --------------------------------------------------------------------------------------------------
// oracle 3 — lexical context
// --------------------------------------------------------------------------------------------------

const FUNCTION_LIKE = new Set([
  ts.SyntaxKind.FunctionDeclaration,
  ts.SyntaxKind.FunctionExpression,
  ts.SyntaxKind.ArrowFunction,
  ts.SyntaxKind.MethodDeclaration,
  ts.SyntaxKind.GetAccessor,
  ts.SyntaxKind.SetAccessor,
  ts.SyntaxKind.Constructor,
  ts.SyntaxKind.ClassStaticBlockDeclaration,
]);

/** The nearest enclosing function-like node, or `null` when the node sits at module top level. */
export function enclosingFunction(node) {
  for (let cur = node.parent; cur; cur = cur.parent) {
    if (FUNCTION_LIKE.has(cur.kind)) return cur;
    if (ts.isSourceFile(cur)) return null;
  }
  return null;
}

const isAsyncFunction = (fn) =>
  fn != null && ts.canHaveModifiers(fn) && (ts.getModifiers(fn) ?? []).some((m) => m.kind === ts.SyntaxKind.AsyncKeyword);

const isGeneratorFunction = (fn) => fn != null && "asteriskToken" in fn && fn.asteriskToken != null;

/**
 * If `fn` is the callback of a `frame(...)`/`frameLoop(...)`, returns the identifier the frame is
 * bound to; otherwise `null`.
 *
 * Decided by TYPE, not by callee name: the callback's first parameter is typed `Frame` by
 * `FrameCallback<R>`, so the checker answers this for `frame`, `frameLoop`, an aliased import
 * (`frame as runFrame`, which two corpus files use) and a re-export alike. A destructured or
 * missing parameter yields `{ frame: true, param: null }` — a real frame callback with no name to
 * write, which is a bucket rather than a skip.
 */
export function frameCallbackParam(fn, checker) {
  if (fn == null) return null;
  if (!ts.isArrowFunction(fn) && !ts.isFunctionExpression(fn)) return null;
  const param = fn.parameters[0];
  if (!param) return null;
  const type = checker.getTypeAtLocation(param);
  const name = (type?.getSymbol() ?? type?.aliasSymbol)?.getName();
  // Structural fallback keeps this working if `Frame` is ever renamed or intersected: the frame is
  // the object that owns `compute`, `pass` and `copyBuffer` together.
  const structural = type != null && type.getProperty("compute") != null && type.getProperty("pass") != null && type.getProperty("copyBuffer") != null;
  if (name !== "Frame" && !structural) return null;
  return { param: ts.isIdentifier(param.name) ? param.name.text : null };
}

/** Names the enclosing function for the report, falling back to its syntactic role. */
function describeFunction(fn, sf) {
  if (fn == null) return "<module top level>";
  if (ts.isFunctionDeclaration(fn) || ts.isMethodDeclaration(fn)) {
    return fn.name ? fn.name.getText(sf) : "<anonymous>";
  }
  if (ts.isConstructorDeclaration(fn)) return "constructor";
  if (ts.isClassStaticBlockDeclaration(fn)) return "static {}";
  const parent = fn.parent;
  if (parent && ts.isVariableDeclaration(parent) && ts.isIdentifier(parent.name)) return parent.name.text;
  if (parent && ts.isPropertyAssignment(parent) && !ts.isComputedPropertyName(parent.name)) return parent.name.getText(sf);
  if (parent && ts.isCallExpression(parent)) return `<callback of ${parent.expression.getText(sf)}()>`;
  return "<anonymous>";
}

/**
 * Evidence that the enclosing function's `void` return is part of a contract this module publishes,
 * i.e. that making it `async` would change an exported signature rather than a local detail.
 *
 * Two shapes, both mechanical: an `export`ed declaration, and a member of an object literal whose
 * CONTEXTUAL type declares that member (the `return { simulate(dt) {…} }` shape every example in
 * this corpus uses to implement its published interface). Reported, never acted on — the codemod
 * refuses these sites either way; this only tells the human reading the bucket which of the two
 * kinds of change they would be signing up for.
 */
export function publicSignatureEvidence(fn, checker, sf) {
  if (fn == null) return null;
  if (ts.canHaveModifiers(fn) && (ts.getModifiers(fn) ?? []).some((m) => m.kind === ts.SyntaxKind.ExportKeyword)) {
    return { kind: "exported-declaration", detail: describeFunction(fn, sf) };
  }
  const parent = fn.parent;
  const member = parent && (ts.isPropertyAssignment(parent) || ts.isMethodDeclaration(parent) ? parent : null);
  const owner = ts.isMethodDeclaration(fn) ? fn.parent : member?.parent;
  if (!owner || !ts.isObjectLiteralExpression(owner)) return null;
  const nameNode = ts.isMethodDeclaration(fn) ? fn.name : member?.name;
  if (!nameNode || ts.isComputedPropertyName(nameNode)) return null;
  const contextual = checker.getContextualType(owner);
  if (!contextual) return null;
  const prop = contextual.getProperty(nameNode.getText(sf));
  if (!prop) return null;
  return {
    kind: "declared-interface-member",
    detail: `${checker.typeToString(contextual)}.${nameNode.getText(sf)}`,
  };
}

// --------------------------------------------------------------------------------------------------
// site collection
// --------------------------------------------------------------------------------------------------

/**
 * Every `<expr>.dispatch(...)` call in the file, with the token-scan guard applied so a `.dispatch(`
 * that lives inside a string/template/comment can never reach the planner.
 */
export function collectSites(sf, checker, relPath) {
  if (parseDiagnosticsCount(sf.text, relPath) > 0) {
    return { parseError: true, sites: [] };
  }
  const starts = tokenStarts(sf.text, relPath);
  const sites = [];
  const visit = (node) => {
    if (
      ts.isCallExpression(node)
      && ts.isPropertyAccessExpression(node.expression)
      && node.expression.name.text === "dispatch"
      && starts.has(node.expression.name.getStart(sf))
    ) {
      sites.push({
        file: relPath,
        sourceFile: sf,
        call: node,
        nameNode: node.expression.name,
        receiverExpr: node.expression.expression,
        line: sf.getLineAndCharacterOfPosition(node.getStart(sf)).line + 1,
      });
    }
    ts.forEachChild(node, visit);
  };
  visit(sf);
  return { parseError: false, sites };
}

// --------------------------------------------------------------------------------------------------
// planning
// --------------------------------------------------------------------------------------------------

/** `true` if the raw text of `[start, end)` contains a comment opener we would delete by splicing. */
function hasCommentBetween(text, start, end) {
  const slice = text.slice(start, end);
  return slice.includes("//") || slice.includes("/*");
}

/** Offset just past the call's `(`, i.e. where the first argument's text begins. */
function argumentListStart(call, sf) {
  const first = call.arguments[0];
  const open = call.expression.end;
  // `first.getFullStart()` includes the argument's leading trivia, which is exactly what must be
  // preserved when the head `.dispatch(` is replaced by `, `.
  return first ? first.getFullStart() : open;
}

/**
 * Decides the destination of one site. Returns `{ classification, reason?, note?, frameParam?,
 * publicSignature? }`. Never returns undefined for a site — "no bucket" is the failure mode this
 * whole chain exists to avoid.
 */
export function planSite(site, checker) {
  const sf = site.sourceFile;
  const receiver = classifyReceiver(site.receiverExpr, checker);
  if (receiver.kind === "unresolved") {
    return { classification: "unresolved-receiver", note: `receiver type: ${receiver.typeText}` };
  }
  if (receiver.kind === "foreign") {
    return { classification: "not-a-compute", note: `receiver type: ${receiver.typeText}` };
  }

  const overload = classifyOverload(site.call, checker);
  const fn = enclosingFunction(site.call);
  const frameCb = frameCallbackParam(fn, checker);
  const fnName = describeFunction(fn, sf);
  const publicSignature = publicSignatureEvidence(fn, checker, sf);

  // ---- path A: lexically the frame callback's own body -> f.compute(), sync, same encoder.
  if (frameCb) {
    if (frameCb.param == null) {
      return {
        classification: "ambiguous-sync-context",
        reason: "frame-callback-without-named-parameter",
        note: "the enclosing frame callback destructures (or omits) its Frame parameter, so there is no `f` to write",
      };
    }
    if (overload === "options") {
      return {
        classification: "ambiguous-indirect-options",
        reason: "no-f-compute-indirect-overload",
        note: `f.compute() takes no \`indirect\`; \`${fnName}\` is a frame callback so dispatchOnce() would re-introduce the extra submit f.compute() exists to remove`,
      };
    }
    if (overload === "unresolved") {
      return { classification: "ambiguous-arguments", reason: "overload-unresolved", note: `${site.call.arguments.length} argument(s), not all number-like` };
    }
    if (hasCommentBetween(sf.text, site.receiverExpr.end, argumentListStart(site.call, sf))) {
      return { classification: "ambiguous-arguments", reason: "comment-inside-call-head", note: "a comment between the receiver and the first argument would be deleted by the splice" };
    }
    return { classification: "auto-f-compute", frameParam: frameCb.param };
  }

  // ---- path B: standalone -> await dispatchOnce(). Only where the `await` is free.
  const inEsModule = ts.isExternalModule(sf);
  const context = fn == null
    ? (inEsModule ? "top-level-module" : "top-level-script")
    : isGeneratorFunction(fn) ? "generator"
      : isAsyncFunction(fn) ? "async-function" : "sync-function";

  if (context === "async-function" || context === "top-level-module") {
    if (overload === "unresolved") {
      return { classification: "ambiguous-arguments", reason: "overload-unresolved", note: `${site.call.arguments.length} argument(s), not all number-like` };
    }
    // `await x` is an expression, so it is only a drop-in where the call's VALUE is discarded and
    // no enclosing operator changes meaning around it. Every other position (an argument, an
    // arrow body, an operand) would need parentheses and/or would change what the surrounding
    // expression evaluates to, so it is reported instead of guessed at.
    if (!ts.isExpressionStatement(site.call.parent)) {
      return {
        classification: "ambiguous-await-position",
        reason: "call-is-not-an-expression-statement",
        note: `the call's value is consumed by a ${ts.SyntaxKind[site.call.parent.kind]}; inserting \`await\` there needs parentheses and changes the surrounding expression`,
      };
    }
    return { classification: "auto-dispatch-once", context };
  }

  const reason = fn == null
    ? "top-level-of-a-non-module-script"
    : context === "generator"
      ? "enclosing-generator-cannot-await"
      : ts.isArrowFunction(fn) && fn.parent && ts.isCallExpression(fn.parent)
        ? "sync-callback-argument"
        : publicSignature?.kind === "declared-interface-member"
          ? "sync-member-of-a-declared-interface"
          : publicSignature?.kind === "exported-declaration"
            ? "sync-exported-function"
            : "sync-local-function";
  return {
    classification: "ambiguous-sync-context",
    reason,
    note: `\`${fnName}\` is synchronous and is not a frame callback; \`await dispatchOnce()\` would make it async`
      + (publicSignature ? ` and change its published signature (${publicSignature.kind}: ${publicSignature.detail})` : "")
      + ". Migrating it needs a human decision (thread a `Frame` through, or propagate async to its callers).",
  };
}

// --------------------------------------------------------------------------------------------------
// emission
// --------------------------------------------------------------------------------------------------

/**
 * The edits for one planned site. Both transforms are expressed as the SMALLEST pair of splices
 * that produce the new text, so the resulting diff shows only the head of the call and leaves the
 * receiver and every argument byte-for-byte untouched (including multi-line bags and their
 * indentation).
 */
export function emitEdits(site, plan) {
  const sf = site.sourceFile;
  const callStart = site.call.getStart(sf);
  if (plan.classification === "auto-f-compute") {
    return [
      { start: callStart, end: callStart, replacement: `${plan.frameParam}.compute(` },
      { start: site.receiverExpr.end, end: argumentListStart(site.call, sf), replacement: "," },
    ];
  }
  if (plan.classification === "auto-dispatch-once") {
    return [
      { start: callStart, end: callStart, replacement: "await " },
      { start: site.nameNode.getStart(sf), end: site.nameNode.end, replacement: "dispatchOnce" },
    ];
  }
  return [];
}

// --------------------------------------------------------------------------------------------------
// driver
// --------------------------------------------------------------------------------------------------

/**
 * Plans (and materializes the new text for) the migration over the whole corpus. Returns
 * `{ entries, fileTexts }`; `fileTexts` holds only the files whose text changed.
 */
export function planProgram({ checker, corpus, sourceFileFor, only = null }) {
  const files = only ? corpus.filter((f) => only.has(f)) : corpus;
  const entries = [];
  const fileTexts = new Map();

  for (const relPath of files) {
    const sf = sourceFileFor(relPath);
    if (!sf) continue;
    // Cheap pre-filter; the AST walk below is the authority. `.dispatch(` cannot appear as a call
    // without these bytes appearing first.
    if (!sf.text.includes(".dispatch")) continue;
    const { parseError, sites } = collectSites(sf, checker, relPath);
    if (parseError) {
      entries.push(reportEntry({ file: relPath, line: 1, before: "", after: "", classification: "skipped-parse-error" }));
      continue;
    }
    if (sites.length === 0) continue;

    const legacyReason = LEGACY_DISPATCH_TEST_SUBJECTS.get(relPath);
    const edits = [];
    for (const site of sites) {
      const plan = planSite(site, checker);
      const entry = reportEntry({
        file: relPath,
        line: site.line,
        before: sf.text.slice(site.call.getStart(sf), site.call.end),
        after: sf.text.slice(site.call.getStart(sf), site.call.end),
        classification: plan.classification,
      });
      if (plan.reason) entry.reason = plan.reason;
      if (plan.note) entry.note = plan.note;
      if (plan.context) entry.context = plan.context;

      if (legacyReason) {
        // Excluded, but never silent: the context the site WOULD have been given is still on the
        // record, so "this exclusion is hiding an easy migration" stays a checkable claim.
        entry.contextWouldBe = plan.classification;
        entry.classification = "excluded-test-subject";
        entries.push(entry);
        continue;
      }

      const siteEdits = emitEdits(site, plan);
      if (siteEdits.length > 0) {
        const spanStart = Math.min(site.call.getStart(sf), ...siteEdits.map((e) => e.start));
        const spanEnd = Math.max(site.call.end, ...siteEdits.map((e) => e.end));
        entry.before = sf.text.slice(spanStart, spanEnd);
        entry.after = applyEdits(entry.before, siteEdits.map((e) => ({ ...e, start: e.start - spanStart, end: e.end - spanStart })));
        edits.push(...siteEdits);
      }
      entries.push(entry);
    }
    if (edits.length > 0) {
      const next = applyEdits(sf.text, edits);
      if (next !== sf.text) fileTexts.set(relPath, next);
    }
  }

  entries.sort((a, b) => (a.file === b.file ? a.line - b.line : (a.file < b.file ? -1 : 1)));
  return { entries, fileTexts };
}

function main() {
  const dryRun = isDryRun();
  const only = new Set(positionalArgs());
  const repoRoot = process.cwd();

  const corpus = getCorpusFiles(repoRoot);
  if (only.size > 0) {
    const unknown = [...only].filter((f) => !corpus.includes(f));
    if (unknown.length > 0) throw new Error(`dispatch-migration: not in the codemod corpus: ${unknown.join(", ")}`);
  }
  const ctx = createCorpusProgram(repoRoot, { corpus });
  assertCorpusProgram(ctx);

  const { entries, fileTexts } = planProgram({ ...ctx, only: only.size > 0 ? only : null });

  for (const [relPath, text] of fileTexts) {
    writeUnlessDryRun({ dryRun, file: path.join(repoRoot, relPath), text });
  }

  if (only.size === 0) {
    const stale = [...LEGACY_DISPATCH_TEST_SUBJECTS.keys()].filter((f) => !entries.some((e) => e.file === f));
    if (stale.length > 0) {
      throw new Error(
        `dispatch-migration: LEGACY_DISPATCH_TEST_SUBJECTS is stale — no \`.dispatch()\` site found in `
          + `${stale.join(", ")}. Drop the entry instead of carrying a dead exclusion.`,
      );
    }
  }

  printReport(entries);

  const tally = new Map();
  for (const e of entries) tally.set(e.classification, (tally.get(e.classification) ?? 0) + 1);
  const migrated = [...tally].filter(([k]) => k.startsWith("auto-")).reduce((n, [, v]) => n + v, 0);
  process.stderr.write(
    `\n${dryRun ? "[dry-run] " : ""}dispatch-migration: ${entries.length} \`.dispatch()\` sites examined in `
      + `${new Set(entries.map((e) => e.file)).size} files; ${migrated} migrated across ${fileTexts.size} files.\n`
      + [...tally].sort(([a], [b]) => (a < b ? -1 : 1)).map(([k, v]) => `  ${String(v).padStart(4)}  ${k}\n`).join("")
      + (dryRun ? "  (no files written)\n" : ""),
  );
  for (const [file, reason] of LEGACY_DISPATCH_TEST_SUBJECTS) {
    process.stderr.write(`  excluded on purpose: ${file}\n    ${reason}\n`);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
