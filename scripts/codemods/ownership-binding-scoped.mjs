#!/usr/bin/env node
// T04-17 — codemod 2 of the 0.4 cut: migrate the flat bag `instance.set({ a: 1, tex: t })` onto the
// binding-scoped form of the final design (§6):
//
//   fx.set({ params: { time: 1 } })          ->  fx.set("params", { time: 1 })
//   fx.set({ time: 1, speed: 2 })            ->  fx.set("params", { time: 1, speed: 2 })   // members
//   fx.set({ src: tex })                     ->  effect(gpu, { shader: S, bindings: { src: tex } })
//   fx.set({ src: other })  (rebind)         ->  fx.bind("src", other)
//
// ADDITIVE PHASE. `set-core.ts` / `set-resources.ts` / `set-layouts.ts` are untouched; the flat bag
// overload stays alive until T04-22. This codemod only migrates *callers*.
//
// ## Why this is not a rename
//
// T04-06 fixes ownership at CONSTRUCTION. `bindings` entries are `external` from birth (`.set()` on
// them throws VGPU-R1-EXTERNAL-BINDING, `.bind()` swaps their identity); everything else is
// value-owned. The flat bag instead latches `ownership = "user"` by call order and never sets
// `external`, so a resource that arrives through `.set({...})` today is user-owned but NOT external
// — and `.bind()` on it throws VGPU-R1-OWNERSHIP-FLIP (`set-core.ts`: `if (!state.external) throw
// ownershipFlipError(...)`). A codemod that only rewrote the call would therefore leave every
// resource binding unusable by `.bind()` after the cut. Resources must MOVE to the constructor's
// `bindings`, which is why this codemod edits construction sites too.
//
// ## Three oracles, no heuristics
//
//  1. **Type checker** (`lib/corpus-program.mjs`) decides whether a `.set()` receiver is one of ours
//     (`Draw`/`Effect`/`Compute` — structurally: it has both `set` and `bind`) or a foreign `.set()`
//     that must not be touched: `SharedUniforms.set(values)` keeps its one-argument form forever
//     (design §5 — the receiver IS the binding), and the corpus also calls `.set()` on
//     `PerspectiveCamera`/`OrbitControls`/`SceneNode`/`Map`. Never by variable name.
//  2. **Type checker** again for each property VALUE: bytes (number/array/plain object) vs resource
//     (`Texture`/`Target`/`GPUSampler`/`StorageBuffer`/`Buffer`/`SharedUniforms`/…).
//  3. **WGSL reflection** (`lib/wgsl-oracle.mjs`) for each property KEY, because the flat bag accepts
//     both a binding name and a struct member name and the binding-scoped form accepts only the
//     former. Without reflection `.set({ time: 1 })` cannot be told apart from `.set({ params: ... })`
//     and a guess produces code that throws.
//
// Anything an oracle cannot answer becomes a NAMED report bucket. A site is migrated only when every
// one of its properties is decided; partial migration of a bag is never emitted, because a half-moved
// bag changes ownership latching in a way that is worse than not migrating.
//
// Usage:
//   node scripts/codemods/ownership-binding-scoped.mjs --dry-run          # report only, no writes
//   node scripts/codemods/ownership-binding-scoped.mjs                    # apply
//   node scripts/codemods/ownership-binding-scoped.mjs --dry-run a.ts     # restrict to paths
import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";
import { createCorpusProgram, assertCorpusProgram, unwrap } from "./lib/corpus-program.mjs";
import { getCorpusFiles } from "./lib/glob-corpus.mjs";
import { applyEdits } from "./lib/splice.mjs";
import { isDryRun, positionalArgs, reportEntry, printReport, writeUnlessDryRun } from "./lib/report.mjs";
import {
  instanceFactoryName, optionsBagOf, propertyNamed, shaderExpressionOf, resolveWgslText,
  loadReflection, reflectionFor, reflectionKey, resolveKey,
} from "./lib/wgsl-oracle.mjs";

/**
 * Files whose SUBJECT is the flat bag itself. `ownership.test.ts` is the acceptance suite T04-06
 * shipped with: it asserts the *latching* semantics of `.set({...})` (VGPU-R1-OWNERSHIP-FLIP fires
 * when a flat-bag value flips a binding's ownership, `.bind()` on a non-external binding is refused,
 * a flat-bag resource does NOT become external, …). Every one of those assertions needs a live
 * flat-bag call to make; migrating them would delete the proof that the guards this codemod has to
 * satisfy still fire. It is also where all 23 already-binding-scoped `.set("name", …)` calls in the
 * corpus live — the file deliberately exercises both spellings side by side. T04-22 owns its
 * rewrite.
 */
const LEGACY_FORM_TEST_SUBJECTS = new Map([
  [
    "packages/vgpu-api/tests/ownership.test.ts",
    "T04-06's ownership acceptance suite: its flat-bag calls ARE the subject (ownership latching by "
      + "call order, VGPU-R1-OWNERSHIP-FLIP, VGPU-R1-EXTERNAL-BINDING, `.bind()` refused on a "
      + "non-external binding). Migrating them would remove the proof that the guards this codemod "
      + "must satisfy still fire. Rewritten by T04-22 when the flat bag is retired.",
  ],
]);

/** Type names the checker may report for a value that is a GPU RESOURCE, not bytes. */
const RESOURCE_TYPE_NAMES = new Set([
  "Texture", "Target", "Surface", "CanvasSurface", "OffscreenTarget", "Draw", "Effect", "Compute",
  "StorageBuffer", "SharedUniforms", "Buffer", "Uniform", "GPUSampler", "GPUTexture", "GPUTextureView",
  "GPUBuffer", "GPUExternalTexture", "PingPongTargets", "PingPongStorage",
]);

// --------------------------------------------------------------------------------------------------
// value classification (oracle 2)
// --------------------------------------------------------------------------------------------------

/**
 * `"bytes" | "resource" | "unknown"` for one property value, mirroring the runtime's own rule
 * (`set-core.ts`'s `ownershipFor()` → `isPlainValue()` in `set-resources.ts`): a value is bytes when
 * it is a primitive, an array/typed array, or a plain object with no resource shape; it is a resource
 * when it carries `resourceIdentity`/`createView`/`gpu`.
 */
export function classifyValueType(expr, checker) {
  const inner = unwrap(expr);
  // Shapes no type can contradict. Kept ahead of the checker so a literal in a `.d.ts`-less file
  // still classifies, and so `[w, h]` never has to survive an array-type lookup.
  if (
    ts.isNumericLiteral(inner) || ts.isStringLiteral(inner) || ts.isNoSubstitutionTemplateLiteral(inner)
    || ts.isTemplateExpression(inner) || ts.isArrayLiteralExpression(inner)
    || inner.kind === ts.SyntaxKind.TrueKeyword || inner.kind === ts.SyntaxKind.FalseKeyword
    || ts.isPrefixUnaryExpression(inner) || ts.isBinaryExpression(inner)
  ) {
    return "bytes";
  }
  const type = checker.getTypeAtLocation(inner);
  const parts = type.isUnion() ? type.types : [type];
  let sawResource = false;
  let sawBytes = false;
  let sawUnknown = false;
  for (const t of parts) {
    if (t.flags & (ts.TypeFlags.Undefined | ts.TypeFlags.Null)) continue;
    if (t.flags & (ts.TypeFlags.Any | ts.TypeFlags.Unknown)) { sawUnknown = true; continue; }
    if (t.flags & (ts.TypeFlags.NumberLike | ts.TypeFlags.StringLike | ts.TypeFlags.BooleanLike | ts.TypeFlags.BigIntLike)) { sawBytes = true; continue; }
    const name = (t.getSymbol() ?? t.aliasSymbol)?.getName();
    if (name && RESOURCE_TYPE_NAMES.has(name)) { sawResource = true; continue; }
    // Structural fallback, same shapes `hasAnyResourceShape()` looks for at runtime.
    if (t.getProperty("resourceIdentity") || t.getProperty("createView")
      || (t.getProperty("gpu") && (t.getProperty("size") || t.getProperty("read")))) { sawResource = true; continue; }
    if (t.getProperty("byteLength") || t.flags & ts.TypeFlags.Object) { sawBytes = true; continue; }
    sawUnknown = true;
  }
  // A union that mixes the two (or leaves anything unresolved) is not decidable: report, never guess.
  if (sawResource && (sawBytes || sawUnknown)) return "unknown";
  if (sawResource) return "resource";
  if (sawUnknown) return "unknown";
  return sawBytes ? "bytes" : "unknown";
}

/** Is this receiver one of ours (`Draw`/`Effect`/`Compute`), a foreign `.set()`, or unresolved? */
export function classifyReceiver(expr, checker) {
  const type = checker.getTypeAtLocation(expr);
  if (!type) return { kind: "unresolved" };
  const parts = type.isUnion() ? type.types : [type];
  let ours = false;
  let sharedUniforms = false;
  let foreign = false;
  let unresolved = false;
  for (const t of parts) {
    if (t.flags & (ts.TypeFlags.Undefined | ts.TypeFlags.Null)) continue;
    if (t.flags & (ts.TypeFlags.Any | ts.TypeFlags.Unknown)) { unresolved = true; continue; }
    if (t.getProperty("set") && t.getProperty("bind")) { ours = true; continue; }
    const name = (t.getSymbol() ?? t.aliasSymbol)?.getName();
    if (name === "SharedUniforms" || name === "UniformSlot") sharedUniforms = true;
    else foreign = true;
  }
  if (unresolved) return { kind: "unresolved", typeText: checker.typeToString(type) };
  // A union of "ours" and "not ours" cannot be migrated safely either way.
  if (ours && (sharedUniforms || foreign)) return { kind: "unresolved", typeText: checker.typeToString(type) };
  if (ours) return { kind: "ours" };
  if (sharedUniforms) return { kind: "shared-uniforms", typeText: checker.typeToString(type) };
  return { kind: "foreign", typeText: checker.typeToString(type) };
}

// --------------------------------------------------------------------------------------------------
// receiver -> construction site
// --------------------------------------------------------------------------------------------------

/**
 * Every object literal in the corpus, indexed by the declaration of its CONTEXTUAL type. Built once,
 * lazily, because it is the only way to answer "where is `effects.bloomExtract` actually built?" when
 * `effects` is typed by an interface: the interface's `PropertySignature` is where symbol resolution
 * stops, and the construction lives in whatever object literal satisfies the interface (for
 * `components/hero` that is `createEffects()`'s `return { bake: vgpu.effect(...), ... }`, 117 of this
 * corpus's sites). One corpus-wide walk, not one per interface.
 */
function objectLiteralsByContextualType(ctx) {
  if (ctx.objectLiteralIndex) return ctx.objectLiteralIndex;
  const index = new Map();
  for (const relPath of ctx.corpus) {
    const sf = ctx.sourceFileFor(relPath);
    if (!sf) continue;
    const visit = (node) => {
      if (ts.isObjectLiteralExpression(node)) {
        const contextual = ctx.checker.getContextualType(node);
        for (const d of contextual?.getSymbol()?.getDeclarations() ?? []) {
          if (!index.has(d)) index.set(d, []);
          index.get(d).push(node);
        }
      }
      ts.forEachChild(node, visit);
    };
    visit(sf);
  }
  ctx.objectLiteralIndex = index;
  return index;
}

/**
 * Walks back from a `.set()` receiver to the `effect()`/`compute()`/`draw()` call that produced it,
 * following `const fx = effect(...)`, object-literal properties (`{ bake: effect(...) }`), shorthand
 * properties, class fields, and interface members via the object literal that satisfies the
 * interface. Returns `{ ok: true, call }` or `{ ok: false, why }` with a bucket name — an unreachable
 * constructor is reported, never guessed at.
 */
export function resolveConstruction(recvExpr, checker, ctx = null) {
  const seen = new Set();
  let cur = unwrap(recvExpr);
  for (let hops = 0; hops < 6; hops++) {
    const raw = checker.getSymbolAtLocation(cur);
    if (!raw) return { ok: false, why: "ctor-receiver-has-no-symbol" };
    const sym = (raw.flags & ts.SymbolFlags.Alias) ? checker.getAliasedSymbol(raw) : raw;
    const decls = sym.getDeclarations() ?? [];
    if (decls.length === 0) return { ok: false, why: "ctor-receiver-has-no-declaration" };
    if (decls.length > 1) return { ok: false, why: "ctor-receiver-has-multiple-declarations" };
    const d = decls[0];
    if (seen.has(d)) return { ok: false, why: "ctor-receiver-declaration-cycle" };
    seen.add(d);

    let init = null;
    if (ts.isVariableDeclaration(d) || ts.isPropertyAssignment(d) || ts.isPropertyDeclaration(d)) {
      init = d.initializer ? unwrap(d.initializer) : null;
    } else if (ts.isShorthandPropertyAssignment(d)) {
      const vd = checker.getShorthandAssignmentValueSymbol(d)?.getDeclarations()?.[0];
      init = vd && ts.isVariableDeclaration(vd) && vd.initializer ? unwrap(vd.initializer) : null;
    } else if (ts.isParameter(d)) {
      return { ok: false, why: "ctor-receiver-is-a-parameter" };
    } else if (ts.isPropertySignature(d)) {
      // `effects.bloomExtract` where `effects: Effects` is an interface: symbol resolution stops at
      // the `PropertySignature`, and the construction lives in whatever object literal satisfies the
      // interface. Recover it from the corpus-wide index — but only when the interface is satisfied
      // in exactly ONE place, otherwise "which `effect()` call is this receiver" has more than one
      // answer and the honest outcome is a report entry.
      if (!ctx) return { ok: false, why: "ctor-receiver-is-an-interface-member" };
      const literals = objectLiteralsByContextualType(ctx).get(d.parent) ?? [];
      if (literals.length === 0) return { ok: false, why: "ctor-receiver-is-an-interface-member-never-built-by-an-object-literal" };
      if (literals.length > 1) return { ok: false, why: "ctor-receiver-is-an-interface-member-built-in-several-places" };
      const prop = propertyNamed(literals[0], d.name && (ts.isIdentifier(d.name) || ts.isStringLiteral(d.name)) ? d.name.text : "\u0000");
      if (!prop) return { ok: false, why: "ctor-receiver-is-an-interface-member-absent-from-the-object-literal" };
      cur = ts.isShorthandPropertyAssignment(prop) ? prop.name : prop.initializer;
      const inner = unwrap(cur);
      if (instanceFactoryName(inner)) return { ok: true, call: inner, decl: prop };
      if (!ts.isIdentifier(inner) && !ts.isPropertyAccessExpression(inner)) {
        return { ok: false, why: `ctor-receiver-interface-member-initialized-by-${ts.SyntaxKind[inner.kind]}` };
      }
      cur = inner;
      continue;
    } else if (ts.isMethodSignature(d)) {
      return { ok: false, why: "ctor-receiver-is-a-method-signature" };
    } else if (ts.isBindingElement(d)) {
      return { ok: false, why: "ctor-receiver-is-a-destructured-binding" };
    } else {
      return { ok: false, why: `ctor-receiver-declared-as-${ts.SyntaxKind[d.kind]}` };
    }

    if (init && instanceFactoryName(init)) return { ok: true, call: init, decl: d };
    if (init && (ts.isIdentifier(init) || ts.isPropertyAccessExpression(init))) { cur = init; continue; }
    if (init && ts.isCallExpression(init)) return { ok: false, why: "ctor-receiver-comes-from-a-helper-call" };
    if (init && ts.isElementAccessExpression(init)) return { ok: false, why: "ctor-receiver-comes-from-an-element-access" };
    if (!init) return { ok: false, why: "ctor-receiver-has-no-initializer" };
    return { ok: false, why: `ctor-receiver-initialized-by-${ts.SyntaxKind[init.kind]}` };
  }
  return { ok: false, why: "ctor-receiver-hop-limit" };
}

// --------------------------------------------------------------------------------------------------
// text utilities
// --------------------------------------------------------------------------------------------------

/**
 * Are there real comment tokens inside `[start, end)`? Scanner-based on purpose: the naive
 * `text.includes("//")` guard the T04-16 QA pass flagged also fires on a `//` *inside a string
 * literal* (`.set({ url: "http://x" })`), which is fail-safe but silently lowers the migration rate.
 * The scanner sees tokens, so a `//` in a string is a string, and a real comment is a real comment.
 */
export function hasCommentIn(text, start, end, languageVersion = ts.ScriptTarget.ESNext) {
  const scanner = ts.createScanner(languageVersion, /* skipTrivia */ false, ts.LanguageVariant.Standard, text, undefined, start, end - start);
  for (;;) {
    const kind = scanner.scan();
    if (kind === ts.SyntaxKind.EndOfFileToken) return false;
    if (scanner.getTokenStart() >= end) return false;
    if (kind === ts.SyntaxKind.SingleLineCommentTrivia || kind === ts.SyntaxKind.MultiLineCommentTrivia) return true;
  }
}

/** A reference expression that is safe to repeat (no calls, no side effects). */
export function isPureReference(node) {
  const n = unwrap(node);
  if (ts.isIdentifier(n) || n.kind === ts.SyntaxKind.ThisKeyword) return true;
  if (ts.isPropertyAccessExpression(n)) return isPureReference(n.expression);
  if (ts.isElementAccessExpression(n)) {
    const arg = n.argumentExpression && unwrap(n.argumentExpression);
    return isPureReference(n.expression) && !!arg && (ts.isNumericLiteral(arg) || ts.isStringLiteral(arg));
  }
  return false;
}

/** Can this expression be re-evaluated at an EARLIER point (hoisted into the constructor)? */
function isHoistableExpression(node) {
  let ok = true;
  const visit = (n) => {
    if (!ok) return;
    if (
      ts.isCallExpression(n) || ts.isNewExpression(n) || ts.isAwaitExpression(n)
      || ts.isYieldExpression(n) || ts.isTaggedTemplateExpression(n)
      || (ts.isBinaryExpression(n) && ts.isAssignmentExpression(n))
      || ts.isDeleteExpression(n) || ts.isPostfixUnaryExpression(n)
      || (ts.isPrefixUnaryExpression(n)
        && (n.operator === ts.SyntaxKind.PlusPlusToken || n.operator === ts.SyntaxKind.MinusMinusToken))
    ) { ok = false; return; }
    ts.forEachChild(n, visit);
  };
  visit(unwrap(node));
  return ok;
}

/**
 * Re-indents a multi-line value that lost one level of nesting.
 *
 * `fx.set({\n  params: {\n    a: 1,\n  },\n})` nests the value TWO levels deep; after the rewrite
 * (`fx.set("params", { … })`) it is only one. Splicing the value's text verbatim — which is what keeps
 * every expression inside it byte-identical — would therefore leave the whole block over-indented by
 * one level, and these files are ingested verbatim into the docs site, so that is a visible defect and
 * not just a cosmetic one. The shift is derived from the value's OWN closing line rather than assumed:
 * that line's indentation is where the value's block level actually sits, and the target is the
 * statement's indentation.
 */
export function reindentValue(valueText, statementIndent) {
  if (!valueText.includes("\n")) return valueText;
  const lines = valueText.split("\n");
  const closingIndent = /^[ \t]*/u.exec(lines[lines.length - 1])[0];
  const shift = closingIndent.length - statementIndent.length;
  if (shift <= 0) return valueText;
  return lines.map((line, i) => {
    if (i === 0) return line;
    const indent = /^[ \t]*/u.exec(line)[0];
    // Never eat non-whitespace: a line indented less than `shift` (a template literal's contents, say)
    // keeps whatever it has.
    return line.slice(Math.min(shift, indent.length));
  }).join("\n");
}

/** The leading whitespace of the line `offset` sits on, or `null` if anything else precedes it. */
function lineIndentAt(text, offset) {
  const lineStart = text.lastIndexOf("\n", offset - 1) + 1;
  const prefix = text.slice(lineStart, offset);
  return /^[ \t]*$/u.test(prefix) ? prefix : null;
}

/** The nearest scope node that a declaration is visible throughout. */
function declarationScope(decl) {
  let cur = decl.parent;
  while (cur) {
    if (
      ts.isBlock(cur) || ts.isSourceFile(cur) || ts.isModuleBlock(cur) || ts.isCaseClause(cur)
      || ts.isDefaultClause(cur) || ts.isForStatement(cur) || ts.isForOfStatement(cur)
      || ts.isForInStatement(cur) || ts.isClassDeclaration(cur) || ts.isClassExpression(cur)
      || ts.isArrowFunction(cur) || ts.isFunctionDeclaration(cur) || ts.isFunctionExpression(cur)
      || ts.isMethodDeclaration(cur) || ts.isConstructorDeclaration(cur)
    ) return cur;
    cur = cur.parent;
  }
  return null;
}

/** The identifier a member chain reads through: `buf` in `buf.read.texelSize`. */
function rootIdentifierOf(expr) {
  let n = unwrap(expr);
  while (ts.isPropertyAccessExpression(n) || ts.isElementAccessExpression(n) || ts.isNonNullExpression(n)) {
    n = unwrap(n.expression);
  }
  return ts.isIdentifier(n) ? n : null;
}

/**
 * Does the `.set()` run at a different POINT IN TIME, or a different NUMBER OF TIMES, than the
 * constructor? True when a loop or a function boundary sits between them — the constructor runs once,
 * at its own moment; a `.set()` inside `for (…)` or inside a `frame(gpu, () => …)` callback does not.
 */
function crossesLoopOrCallback(site, ctorCall) {
  const ctorStart = ctorCall.getStart(ctorCall.getSourceFile());
  let cur = site.parent;
  while (cur) {
    const barrier = ts.isArrowFunction(cur) || ts.isFunctionExpression(cur) || ts.isFunctionDeclaration(cur)
      || ts.isMethodDeclaration(cur) || ts.isConstructorDeclaration(cur)
      || ts.isGetAccessorDeclaration(cur) || ts.isSetAccessorDeclaration(cur)
      || ts.isForStatement(cur) || ts.isForOfStatement(cur) || ts.isForInStatement(cur)
      || ts.isWhileStatement(cur) || ts.isDoStatement(cur);
    // A barrier that also encloses the constructor is shared by both — it is not a boundary BETWEEN
    // them, so it does not desynchronise anything.
    if (barrier && !(cur.getStart(cur.getSourceFile()) <= ctorStart && ctorCall.end <= cur.end)) return true;
    cur = cur.parent;
  }
  return false;
}

/**
 * Is the object the resource expression reads THROUGH mutated anywhere in the file? `buf.read` is a
 * getter over a parity flag (`ping-pong.ts`: `get read() { return this.halves[this.#parity] }`), and
 * `buf.swap()` flips it. Any method call on that object, or any assignment to it or through it, means
 * the value the expression yields is a function of mutable state and cannot be pinned at construction.
 *
 * Deliberately whole-file and deliberately blunt: proving a method does NOT mutate needs an effect
 * analysis this codemod has no business attempting, and the cost of a false positive is one site left
 * on the flat bag, while the cost of a false negative is a silently wrong render.
 */
function readsThroughMutatedObject(expr, ctorCall, checker) {
  const root = rootIdentifierOf(expr);
  // A bare identifier (`const tex = texture(...); fx.set({ src: tex })`) reads through nothing.
  if (!root || unwrap(expr) === root) return false;
  const sym = checker.getSymbolAtLocation(root);
  if (!sym) return true;
  const sameSymbol = (node) => {
    const r = rootIdentifierOf(node);
    return !!r && checker.getSymbolAtLocation(r) === sym;
  };
  let mutated = false;
  const visit = (n) => {
    if (mutated) return;
    if (ts.isCallExpression(n)) {
      const callee = unwrap(n.expression);
      if (ts.isPropertyAccessExpression(callee) && sameSymbol(callee.expression)) { mutated = true; return; }
    }
    if (ts.isBinaryExpression(n) && ts.isAssignmentExpression(n) && sameSymbol(n.left)) { mutated = true; return; }
    if ((ts.isPostfixUnaryExpression(n) || ts.isPrefixUnaryExpression(n)) && sameSymbol(n.operand)) { mutated = true; return; }
    ts.forEachChild(n, visit);
  };
  visit(ctorCall.getSourceFile());
  return mutated;
}

/**
 * Does `expr` denote the SAME resource evaluated at the constructor as it does at the `.set()`?
 *
 * `isMovableTo` only proves the expression can be WRITTEN there (names in scope, declared earlier,
 * no side effects). That is not the same question. The constructor runs once; the `.set()` may run
 * later, or repeatedly, or after something changed what the expression reads. Both failures are real
 * in this corpus and both were shipped by the first version of this codemod:
 *
 *   (a) `pingPong()` hands back `.read`/`.write` as GETTERS over a parity flag that `swap()` flips.
 *       `bindings: { src: buf.read }` freezes the half that was current at construction, so the
 *       effect samples the target it is drawing into and the feedback loop never accumulates.
 *   (b) `feedback.set({ src: pair.read })` inside `for (…) { frame(…); pair.swap(); }` is ONE
 *       syntactic site and FOUR evaluations. Counting assignment sites cannot tell those apart.
 *
 * This asks about execution, not about text, which is why counting sites (the rule this replaced)
 * could not see either case.
 */
function isStableBetween(expr, ctorCall, site, checker) {
  if (crossesLoopOrCallback(site, ctorCall)) {
    return { ok: false, why: "resource-set-runs-in-a-loop-or-callback-the-constructor-does-not" };
  }
  if (readsThroughMutatedObject(expr, ctorCall, checker)) {
    return { ok: false, why: "resource-value-reads-through-a-mutated-object" };
  }
  return { ok: true };
}

/**
 * Would `expr` still mean the same thing if it were evaluated at `target`? Requires every free
 * identifier to be declared before `target`, in a scope that encloses `target`, in the same file.
 * `this` is only allowed when `target` sits inside the same class.
 */
function isMovableTo(expr, target, checker) {
  if (!isHoistableExpression(expr)) return { ok: false, why: "resource-value-is-not-hoistable" };
  const targetStart = target.getStart(target.getSourceFile());
  const targetFile = target.getSourceFile().fileName;
  let verdict = { ok: true };
  const visit = (n) => {
    if (!verdict.ok) return;
    if (n.kind === ts.SyntaxKind.ThisKeyword) {
      let cls = target.parent;
      while (cls && !ts.isClassLike(cls)) cls = cls.parent;
      if (!cls) verdict = { ok: false, why: "resource-value-uses-this-outside-the-constructor-class" };
      return;
    }
    if (ts.isIdentifier(n)) {
      // Property names and the `b` of `a.b` are not free identifiers.
      if (ts.isPropertyAccessExpression(n.parent) && n.parent.name === n) return;
      if (ts.isPropertyAssignment(n.parent) && n.parent.name === n) return;
      const sym = checker.getSymbolAtLocation(n);
      if (!sym) { verdict = { ok: false, why: "resource-value-references-an-unresolvable-name" }; return; }
      const target2 = (sym.flags & ts.SymbolFlags.Alias) ? checker.getAliasedSymbol(sym) : sym;
      const decls = target2.getDeclarations() ?? [];
      if (decls.length === 0) return; // ambient/global (`Math`, `window`) — always in scope.
      for (const d of decls) {
        const dFile = d.getSourceFile();
        // An import/module-level declaration in another file is in scope everywhere in this one.
        if (dFile.fileName !== targetFile) continue;
        if (d.end > targetStart) { verdict = { ok: false, why: "resource-value-is-declared-after-the-constructor" }; return; }
        const scope = declarationScope(d);
        if (scope && !(scope.getStart(scope.getSourceFile()) <= targetStart && targetStart < scope.end)) {
          verdict = { ok: false, why: "resource-value-is-out-of-scope-at-the-constructor" };
          return;
        }
      }
      return;
    }
    ts.forEachChild(n, visit);
  };
  visit(unwrap(expr));
  return verdict;
}

// --------------------------------------------------------------------------------------------------
// site collection
// --------------------------------------------------------------------------------------------------

/**
 * Every `x.set(<object literal>)` call in one file, with the oracles' verdicts attached. Discovery is
 * by AST shape + type, never by regex: `tokenStarts()` is consulted afterwards purely as an
 * independent cross-check of the offsets we are about to splice at.
 */
function collectSites(sf, checker, repoRoot, relPath) {
  const text = sf.text;
  const out = [];
  const visit = (node) => {
    if (
      ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression)
      && ts.isIdentifier(node.expression.name) && node.expression.name.text === "set"
      && node.arguments.length === 1 && ts.isObjectLiteralExpression(unwrap(node.arguments[0]))
    ) {
      out.push({
        file: relPath,
        call: node,
        sourceFile: sf,
        line: sf.getLineAndCharacterOfPosition(node.getStart(sf)).line + 1,
        before: text.slice(node.getStart(sf), node.end),
        receiverExpr: node.expression.expression,
        bag: unwrap(node.arguments[0]),
        receiver: classifyReceiver(node.expression.expression, checker),
      });
    }
    ts.forEachChild(node, visit);
  };
  visit(sf);
  out.sort((a, b) => a.call.getStart(sf) - b.call.getStart(sf));
  return out;
}

/** Reads the bag's properties into `{ key, keyKind, valueExpr, valueText }`, or a bucket name. */
function readBagProperties(site) {
  const sf = site.sourceFile;
  const text = sf.text;
  const props = [];
  for (const p of site.bag.properties) {
    if (ts.isSpreadAssignment(p)) return { error: "ambiguous-spread-in-bag" };
    if (ts.isMethodDeclaration(p) || ts.isGetAccessorDeclaration(p) || ts.isSetAccessorDeclaration(p)) {
      return { error: "ambiguous-method-in-bag" };
    }
    const name = p.name;
    if (!name) return { error: "ambiguous-property-without-name" };
    if (ts.isComputedPropertyName(name)) {
      // `{ [textureName]: source }` — the key is only known at runtime, so no oracle can resolve it
      // to a binding. Reported, never migrated (a QA finding on #344 asked for exactly this).
      return { error: "ambiguous-computed-key" };
    }
    if (ts.isPrivateIdentifier(name) || ts.isNumericLiteral(name)) return { error: "ambiguous-non-string-key" };
    if (!ts.isIdentifier(name) && !ts.isStringLiteral(name) && !ts.isNoSubstitutionTemplateLiteral(name)) {
      return { error: "ambiguous-non-string-key" };
    }
    const valueExpr = ts.isShorthandPropertyAssignment(p) ? p.name : p.initializer;
    props.push({
      key: name.text,
      quoted: !ts.isIdentifier(name),
      shorthand: ts.isShorthandPropertyAssignment(p),
      valueExpr,
      valueText: text.slice(valueExpr.getStart(sf), valueExpr.end),
    });
  }
  return { props };
}

// --------------------------------------------------------------------------------------------------
// planning
// --------------------------------------------------------------------------------------------------

/** Emits `.set("b", v)` / `.bind("b", r)` text for one planned call. */
const callText = (c) => `.${c.method}(${JSON.stringify(c.binding)}, ${c.valueText})`;

/**
 * Builds the value text for a bytes group: a single whole-binding property keeps its value verbatim,
 * struct members are re-assembled into one object literal so ONE call writes the binding once
 * (design §6: "one call is one buffer write"; `setScoped` merges a partial and rewrites completely).
 */
function bytesValueText(group, statementIndent) {
  const whole = group.props.find((p) => p.resolved.kind === "binding");
  if (whole && group.props.length === 1) return reindentValue(whole.valueText, statementIndent);
  if (whole) return null; // binding AND members in one bag: the merge order would be guesswork.
  const parts = group.props.map((p) => (p.shorthand
    ? p.key
    : `${p.quoted ? JSON.stringify(p.key) : p.key}: ${reindentValue(p.valueText, statementIndent)}`));
  return `{ ${parts.join(", ")} }`;
}

/**
 * Decides what happens to one site. Pure w.r.t. the tree: the only cross-site input is
 * `resourceAssignments`, the per-(instance, binding) count computed by the caller in a prior pass.
 *
 * @returns {{classification: string}
 *   | {classification: string, calls: object[], ctorBindings: object[], ctorCall?: ts.Node}}
 */
function planSite(site, ctx) {
  const { checker, repoRoot, reflectSource, resourceAssignments } = ctx;
  const sf = site.sourceFile;
  const text = sf.text;

  if (site.receiver.kind === "shared-uniforms") return { classification: "excluded-uniform-1arg" };
  if (site.receiver.kind === "foreign") return { classification: "excluded-foreign-api" };
  if (site.receiver.kind === "unresolved") return { classification: "excluded-unresolved-receiver" };

  if (site.call.questionDotToken || ts.isPropertyAccessChain(site.call.expression)) {
    return { classification: "skipped-optional-chain-receiver" };
  }

  // Comments anywhere between the receiver and the end of the call would be dropped by the rewrite
  // (the bag stops existing). Scanner-checked, so a `//` inside a string value is not a false hit.
  const spliceStart = site.receiverExpr.end;
  if (hasCommentIn(text, spliceStart, site.call.end)) {
    return { classification: "skipped-comment-in-bag" };
  }

  if (site.bag.properties.length === 0) {
    // `.set({})` is a no-op today and has no binding-scoped spelling. T04-22 deletes the overload.
    return { classification: "skipped-empty-bag" };
  }

  const read = readBagProperties(site);
  if (read.error) return { classification: read.error };
  const props = read.props;

  // --- oracle 3: the shader behind this instance (resolved and reflected in the preload pass, since
  // a `.wgsl` entry has to go through the async `resolveShader()` to flatten its import graph)
  const resolved = site.shader;
  if (!resolved.ok) return { classification: `ambiguous-${resolved.why}` };
  const ctor = resolved.ctor;
  const ctorSf = ctor.call.getSourceFile();
  const reflection = reflectionFor(resolved);
  if (!reflection.ok) return { classification: `ambiguous-${reflection.why}` };

  // --- resolve every property against both oracles before deciding anything
  for (const p of props) {
    p.resolved = resolveKey(p.key, reflection);
    p.valueClass = classifyValueType(p.valueExpr, checker);
    if (p.resolved.kind === "ambiguous-member") return { classification: "ambiguous-member-owned-by-several-bindings" };
    if (p.resolved.kind === "unknown") return { classification: "ambiguous-key-absent-from-reflection" };
    if (p.valueClass === "unknown") return { classification: "ambiguous-value-type-undecidable" };
    if (p.valueClass === "resource" && p.resolved.kind === "member") {
      return { classification: "ambiguous-resource-assigned-to-a-struct-member" };
    }
  }

  // --- group by target binding
  const groups = new Map();
  for (const p of props) {
    if (!groups.has(p.resolved.binding)) groups.set(p.resolved.binding, { binding: p.resolved.binding, props: [], classes: new Set() });
    const g = groups.get(p.resolved.binding);
    g.props.push(p);
    g.classes.add(p.valueClass);
  }
  for (const g of groups.values()) {
    if (g.classes.size > 1) return { classification: "ambiguous-binding-receives-bytes-and-a-resource" };
    if (g.classes.has("resource") && g.props.length > 1) return { classification: "ambiguous-binding-receives-several-resources" };
  }

  const ctorBag = optionsBagOf(ctor.call);
  const declaredBindings = ctorBag ? propertyNamed(ctorBag, "bindings") : undefined;
  const declaredValues = ctorBag ? propertyNamed(ctorBag, "values") : undefined;
  const declaredSetBag = ctorBag ? propertyNamed(ctorBag, "set") : undefined;
  const declaredBindingNames = new Set();
  if (declaredBindings && ts.isPropertyAssignment(declaredBindings)) {
    const inner = unwrap(declaredBindings.initializer);
    if (!ts.isObjectLiteralExpression(inner)) return { classification: "ambiguous-constructor-bindings-is-not-an-object-literal" };
    for (const p of inner.properties) {
      if (p.name && (ts.isIdentifier(p.name) || ts.isStringLiteral(p.name))) declaredBindingNames.add(p.name.text);
      else return { classification: "ambiguous-constructor-bindings-has-an-undecidable-key" };
    }
  }

  // The indentation the rewritten call will sit at, for re-indenting multi-line values.
  const statementNode = ts.isExpressionStatement(site.call.parent) ? site.call.parent : site.call;
  const statementIndent = lineIndentAt(text, statementNode.getStart(sf)) ?? "";

  const calls = [];
  const ctorBindings = [];
  for (const g of groups.values()) {
    if (g.classes.has("bytes")) {
      if (declaredBindingNames.has(g.binding)) {
        // Declared external at construction: `.set()` on it throws VGPU-R1-EXTERNAL-BINDING. The
        // flat-bag call is writing bytes to a binding somebody already declared as a resource — a
        // pre-existing contradiction this codemod must not paper over.
        return { classification: "ambiguous-bytes-written-to-a-constructor-declared-binding" };
      }
      const valueText = bytesValueText(g, statementIndent);
      if (valueText === null) return { classification: "ambiguous-bag-mixes-a-binding-with-its-own-members" };
      calls.push({ method: "set", binding: g.binding, valueText, sortKey: g.props[0].valueExpr.getStart(sf) });
      continue;
    }

    // --- resource
    const p = g.props[0];
    if (declaredBindingNames.has(g.binding)) {
      // Already external from construction: this call is an identity update, which is exactly what
      // `.bind()` is for (design §6), and the R1 guard is satisfied.
      calls.push({ method: "bind", binding: g.binding, valueText: reindentValue(p.valueText, statementIndent), sortKey: p.valueExpr.getStart(sf) });
      continue;
    }
    if (ctorSf.fileName !== sf.fileName) return { classification: "ambiguous-constructor-in-another-file" };
    if (!ctorBag) return { classification: "ambiguous-constructor-has-no-options-bag" };
    if (declaredValues && ts.isPropertyAssignment(declaredValues)) {
      const inner = unwrap(declaredValues.initializer);
      if (ts.isObjectLiteralExpression(inner) && propertyNamed(inner, g.binding)) {
        return { classification: "ambiguous-binding-already-declared-value-owned-in-constructor" };
      }
    }
    if (declaredSetBag) return { classification: "ambiguous-constructor-passes-a-flat-set-bag" };

    const assignments = resourceAssignments.get(`${instanceKey(ctor.call)}\u0000${g.binding}`) ?? 0;
    if (assignments > 1) {
      // Ping-pong / resize rebinds: several call sites hand the same binding different resources and
      // source order is not evaluation order, so which one is "the constructor value" is a judgement
      // call, not a fact. Reported for human resolution (the ticket's `ambiguous-cross-file` bucket).
      return { classification: "ambiguous-resource-rebound-at-several-sites" };
    }
    const movable = isMovableTo(p.valueExpr, ctor.call, checker);
    if (!movable.ok) return { classification: `ambiguous-${movable.why}` };
    // Movable is not the same as stable: see isStableBetween().
    const stable = isStableBetween(p.valueExpr, ctor.call, site.call, checker);
    if (!stable.ok) return { classification: `ambiguous-${stable.why}` };
    ctorBindings.push({ binding: g.binding, valueText: p.valueText });
  }

  if (calls.length === 0 && ctorBindings.length === 0) return { classification: "skipped-nothing-to-do" };

  calls.sort((a, b) => a.sortKey - b.sortKey);
  return {
    classification: ctorBindings.length > 0
      ? "auto-resource-to-constructor"
      : (calls.some((c) => c.method === "bind") ? "auto-resource-rebind" : "auto-bytes"),
    calls,
    ctorBindings,
    ctorCall: ctor.call,
    ctorBag,
    declaredBindings,
  };
}

/** Stable identity for "the instance this construction call creates". */
function instanceKey(ctorCall) {
  const sf = ctorCall.getSourceFile();
  return `${sf.fileName}:${ctorCall.getStart(sf)}`;
}

// --------------------------------------------------------------------------------------------------
// edit emission
// --------------------------------------------------------------------------------------------------

/**
 * Turns a plan into text edits for the `.set()` call itself. Three shapes, chosen so the diff stays
 * reviewable:
 *   - one resulting call  -> splice `.set({...})` -> `.set("b", v)` in place;
 *   - nothing left (every property moved to the constructor) -> delete the whole statement;
 *   - several calls       -> extra statements after this one when the receiver is a pure reference
 *                            and the call is a standalone statement, otherwise a `.set().set()` chain.
 */
function emitCallEdits(site, plan) {
  const sf = site.sourceFile;
  const text = sf.text;
  const spliceStart = site.receiverExpr.end;
  const statement = ts.isExpressionStatement(site.call.parent) ? site.call.parent : null;

  if (plan.calls.length === 0) {
    if (!statement) return { error: "ambiguous-every-property-moved-but-the-call-value-is-used" };
    // Delete the statement and the line it occupies (leading indentation + trailing newline), so no
    // blank line is left behind.
    const indent = lineIndentAt(text, statement.getStart(sf));
    const from = indent === null ? statement.getStart(sf) : statement.getStart(sf) - indent.length;
    let to = statement.end;
    while (to < text.length && (text[to] === " " || text[to] === "\t")) to += 1;
    if (text[to] === "\r") to += 1;
    if (text[to] === "\n") to += 1;
    return { edits: [{ start: from, end: to, replacement: "" }] };
  }

  const chain = plan.calls.map(callText).join("");
  const oneLine = { start: spliceStart, end: site.call.end, replacement: chain };
  if (plan.calls.length === 1) return { edits: [oneLine] };

  const receiverText = text.slice(site.receiverExpr.getStart(sf), site.receiverExpr.end);
  const indent = statement ? lineIndentAt(text, statement.getStart(sf)) : null;
  const canSplit = statement !== null && indent !== null && isPureReference(site.receiverExpr)
    && !receiverText.includes("\n");
  const fitsOnOneLine = (lineIndentAt(text, site.call.getStart(sf)) ?? "").length
    + receiverText.length + chain.length <= 118 && !chain.includes("\n");
  if (!canSplit || fitsOnOneLine) return { edits: [oneLine] };

  // One statement per call, at the original statement's indentation.
  const first = plan.calls[0];
  const rest = plan.calls.slice(1).map((c) => `\n${indent}${receiverText}${callText(c)};`).join("");
  return {
    edits: [
      { start: spliceStart, end: site.call.end, replacement: callText(first) },
      { start: statement.end, end: statement.end, replacement: rest },
    ],
  };
}

/**
 * Edits that add `bindings: { … }` to a construction call, or extend the one it already has. One
 * edit per constructor even when several `.set()` sites contribute, because two insertions at the
 * same offset are rejected by `applyEdits` (#342's B3) — and rightly so: their order would decide
 * the output.
 */
function emitCtorEdits(ctorCall, ctorBag, declaredBindings, entries) {
  const sf = ctorCall.getSourceFile();
  const text = sf.text;
  const pairs = entries.map((e) => `${/^[A-Za-z_$][\w$]*$/u.test(e.binding) ? e.binding : JSON.stringify(e.binding)}: ${e.valueText}`);

  if (declaredBindings && ts.isPropertyAssignment(declaredBindings)) {
    const inner = unwrap(declaredBindings.initializer);
    const insertAt = inner.getStart(sf) + 1;
    if (inner.properties.length === 0) {
      return [{ start: inner.getStart(sf), end: inner.end, replacement: `{ ${pairs.join(", ")} }` }];
    }
    const gap = text.slice(insertAt, inner.properties[0].getStart(sf));
    if (gap.includes("\n")) {
      const indent = lineIndentAt(text, inner.properties[0].getStart(sf)) ?? "  ";
      return [{ start: insertAt, end: insertAt, replacement: pairs.map((p) => `\n${indent}${p},`).join("") }];
    }
    return [{ start: insertAt, end: insertAt, replacement: ` ${pairs.join(", ")},` }];
  }

  const property = `bindings: { ${pairs.join(", ")} }`;
  if (ctorBag.properties.length === 0) {
    return [{ start: ctorBag.getStart(sf), end: ctorBag.end, replacement: `{ ${property} }` }];
  }
  // Always AFTER the last property, single-line or not: `shader:` reads as the subject of the call and
  // #344 deliberately made it the first key, so inserting at the front would undo that.
  const last = ctorBag.properties[ctorBag.properties.length - 1];
  const multiline = text.slice(ctorBag.getStart(sf) + 1, ctorBag.properties[0].getStart(sf)).includes("\n");
  if (!multiline) return [{ start: last.end, end: last.end, replacement: `, ${property}` }];
  const indent = lineIndentAt(text, ctorBag.properties[0].getStart(sf)) ?? "  ";
  // Respect the bag's own trailing-comma style instead of imposing one.
  const hasTrailingComma = /^\s*,/u.test(text.slice(last.end, ctorBag.end));
  return hasTrailingComma
    ? [{ start: last.end + text.slice(last.end, ctorBag.end).indexOf(",") + 1, end: last.end + text.slice(last.end, ctorBag.end).indexOf(",") + 1, replacement: `\n${indent}${property},` }]
    : [{ start: last.end, end: last.end, replacement: `,\n${indent}${property}` }];
}

// --------------------------------------------------------------------------------------------------
// driver
// --------------------------------------------------------------------------------------------------

/**
 * Plans (and optionally applies) the migration over a whole program. Returns
 * `{ entries, fileTexts }` where `fileTexts` only contains files whose text changed.
 *
 * Two passes over the sites: the first counts, per (instance, binding), how many call sites hand it a
 * resource — a count of one means "this is the construction value, move it"; more than one means a
 * rebind sequence nobody can disambiguate statically. Then per-file edits are built, including the
 * constructor edits contributed by sites elsewhere in the same file.
 */
export async function planProgram({ checker, corpus, sourceFileFor, repoRoot, reflectSource, resolveShader, only = null }) {
  const files = only ? corpus.filter((f) => only.has(f)) : corpus;
  // One shared context: it memoizes the corpus-wide object-literal index, which is far too
  // expensive to rebuild per pass (and per interface).
  const ctx = { checker, corpus, sourceFileFor, repoRoot, reflectSource, resourceAssignments: new Map() };

  // ---- pass 0: collect
  const sitesByFile = new Map();
  for (const relPath of files) {
    const sf = sourceFileFor(relPath);
    if (!sf) continue;
    if (!sf.text.includes(".set(")) continue;
    const sites = collectSites(sf, checker, repoRoot, relPath);
    if (sites.length > 0) sitesByFile.set(relPath, sites);
  }

  // ---- pass 0.5: resolve constructor + shader per site, and preload every distinct reflection.
  // Split out because `resolveShader()` is async (it loads a `.wgsl` import graph off disk), while
  // planning has to stay synchronous to keep `planSite()` a plain, testable function.
  const allSites = [...sitesByFile.values()].flat();
  for (const site of allSites) {
    if (site.receiver.kind !== "ours") continue;
    const ctor = resolveConstruction(site.receiverExpr, checker, ctx);
    if (!ctor.ok) { site.shader = { ok: false, why: ctor.why }; continue; }
    const shaderExpr = shaderExpressionOf(ctor.call);
    if (!shaderExpr) { site.shader = { ok: false, why: "constructor-has-no-shader" }; continue; }
    const resolved = resolveWgslText(shaderExpr, { checker, repoRoot });
    site.shader = resolved.ok ? { ...resolved, ctor } : { ok: false, why: resolved.why };
  }
  const preloaded = new Set();
  for (const site of allSites) {
    if (!site.shader?.ok) continue;
    const key = reflectionKey(site.shader);
    if (preloaded.has(key)) continue;
    preloaded.add(key);
    await loadReflection(site.shader, { reflectSource, resolveShader, repoRoot });
  }

  // ---- pass 1: per-(instance, binding) resource assignment census
  const resourceAssignments = ctx.resourceAssignments;
  for (const site of allSites) {
    if (site.receiver.kind !== "ours" || !site.shader?.ok) continue;
    const read = readBagProperties(site);
    if (read.error) continue;
    const reflection = reflectionFor(site.shader);
    if (!reflection.ok) continue;
    for (const p of read.props) {
      if (classifyValueType(p.valueExpr, checker) !== "resource") continue;
      const resolved = resolveKey(p.key, reflection);
      if (!resolved.binding) continue;
      const key = `${instanceKey(site.shader.ctor.call)}\u0000${resolved.binding}`;
      resourceAssignments.set(key, (resourceAssignments.get(key) ?? 0) + 1);
    }
  }

  // ---- pass 2: plan every site
  const entries = [];
  const editsByFile = new Map();
  const ctorContributions = new Map(); // instanceKey -> { ctorCall, ctorBag, declaredBindings, entries[] }
  const plans = [];

  for (const [relPath, sites] of sitesByFile) {
    const legacyReason = LEGACY_FORM_TEST_SUBJECTS.get(relPath);
    for (const site of sites) {
      if (legacyReason && site.receiver.kind === "ours") {
        plans.push({ site, plan: { classification: "excluded-test-subject" } });
        continue;
      }
      plans.push({ site, plan: planSite(site, ctx) });
    }
  }

  for (const { site, plan } of plans) {
    if (!plan.calls && !plan.ctorBindings) continue;
    const emitted = emitCallEdits(site, plan);
    if (emitted.error) { plan.classification = emitted.error; plan.calls = undefined; plan.ctorBindings = undefined; continue; }
    plan.callEdits = emitted.edits;
    if (plan.ctorBindings.length > 0) {
      const key = instanceKey(plan.ctorCall);
      if (!ctorContributions.has(key)) {
        ctorContributions.set(key, {
          file: path.relative(repoRoot, plan.ctorCall.getSourceFile().fileName),
          ctorCall: plan.ctorCall, ctorBag: plan.ctorBag, declaredBindings: plan.declaredBindings, entries: [],
        });
      }
      ctorContributions.get(key).entries.push(...plan.ctorBindings);
    }
  }

  // A binding must not be contributed twice to one constructor (would emit a duplicate key).
  for (const c of ctorContributions.values()) {
    const seen = new Set();
    for (const e of c.entries) {
      if (seen.has(e.binding)) throw new Error(`ownership-binding-scoped: ${c.file} — binding '${e.binding}' contributed twice to one constructor; the per-instance census should have made this a rebind sequence`);
      seen.add(e.binding);
    }
  }

  // ---- pass 3: materialize edits and the report
  for (const { site, plan } of plans) {
    const sf = site.sourceFile;
    const callStart = site.call.getStart(sf);
    let before = site.before;
    let after = site.before;
    if (plan.callEdits) {
      // `before`/`after` are rendered over the AFFECTED SPAN — the union of the call and every edit
      // this site produces — not over the call alone. Some rewrites reach outside the call: deleting
      // the whole statement (when every property moved to the constructor) starts before the call and
      // ends past its semicolon, and splitting into several statements appends after it. Reporting
      // `after` over the narrow call span would have shown those sites UNCHANGED while the real run
      // rewrote them, silently breaking the "dry-run report == real diff" contract #342 pinned.
      const spanStart = Math.min(callStart, ...plan.callEdits.map((e) => e.start));
      const spanEnd = Math.max(site.call.end, ...plan.callEdits.map((e) => e.end));
      before = sf.text.slice(spanStart, spanEnd);
      after = applyEdits(before, plan.callEdits.map((e) => ({ ...e, start: e.start - spanStart, end: e.end - spanStart })));
      const list = editsByFile.get(site.file) ?? [];
      list.push(...plan.callEdits);
      editsByFile.set(site.file, list);
    }
    entries.push(reportEntry({
      file: site.file, line: site.line, before, after,
      classification: plan.classification,
    }));
    if (plan.ctorBindings?.length) {
      const c = ctorContributions.get(instanceKey(plan.ctorCall));
      entries[entries.length - 1].constructorEdit = {
        file: c.file,
        line: plan.ctorCall.getSourceFile().getLineAndCharacterOfPosition(plan.ctorCall.getStart(plan.ctorCall.getSourceFile())).line + 1,
        bindings: plan.ctorBindings.map((e) => e.binding),
      };
    }
  }

  for (const c of ctorContributions.values()) {
    const list = editsByFile.get(c.file) ?? [];
    list.push(...emitCtorEdits(c.ctorCall, c.ctorBag, c.declaredBindings, c.entries));
    editsByFile.set(c.file, list);
  }

  const fileTexts = new Map();
  for (const [relPath, edits] of editsByFile) {
    const sf = sourceFileFor(relPath);
    const next = applyEdits(sf.text, edits);
    if (next !== sf.text) fileTexts.set(relPath, next);
  }

  entries.sort((a, b) => (a.file === b.file ? a.line - b.line : (a.file < b.file ? -1 : 1)));
  return { entries, fileTexts };
}

async function main() {
  const dryRun = isDryRun();
  const only = new Set(positionalArgs());
  const repoRoot = process.cwd();
  const { reflectSource } = await import(path.join(repoRoot, "packages/wgsl/dist/runtime/reflect-source.js"));
  const { resolveShader } = await import(path.join(repoRoot, "packages/wgsl/dist/runtime/resolve-shader.js"));

  const corpus = getCorpusFiles(repoRoot);
  if (only.size > 0) {
    const unknown = [...only].filter((f) => !corpus.includes(f));
    if (unknown.length > 0) throw new Error(`ownership-binding-scoped: not in the codemod corpus: ${unknown.join(", ")}`);
  }
  const ctx = createCorpusProgram(repoRoot, { corpus });
  assertCorpusProgram(ctx);

  const { entries, fileTexts } = await planProgram({ ...ctx, reflectSource, resolveShader, only: only.size > 0 ? only : null });

  for (const [relPath, text] of fileTexts) {
    writeUnlessDryRun({ dryRun, file: path.join(repoRoot, relPath), text });
  }

  const stale = [...LEGACY_FORM_TEST_SUBJECTS.keys()].filter((f) => !entries.some((e) => e.file === f));
  if (stale.length > 0 && only.size === 0) {
    throw new Error(
      `ownership-binding-scoped: LEGACY_FORM_TEST_SUBJECTS is stale — no site found in `
        + `${stale.join(", ")}. Drop the entry instead of carrying a dead exclusion.`,
    );
  }

  printReport(entries);

  const tally = new Map();
  for (const e of entries) tally.set(e.classification, (tally.get(e.classification) ?? 0) + 1);
  const migrated = [...tally].filter(([k]) => k.startsWith("auto-")).reduce((n, [, v]) => n + v, 0);
  process.stderr.write(
    `\n${dryRun ? "[dry-run] " : ""}ownership-binding-scoped: ${entries.length} \`.set({…})\` sites `
      + `examined in ${new Set(entries.map((e) => e.file)).size} files; ${migrated} migrated across `
      + `${fileTexts.size} files.\n`
      + [...tally].sort(([a], [b]) => (a < b ? -1 : 1)).map(([k, v]) => `  ${String(v).padStart(4)}  ${k}\n`).join("")
      + (dryRun ? "  (no files written)\n" : ""),
  );
  for (const [file, reason] of LEGACY_FORM_TEST_SUBJECTS) {
    process.stderr.write(`  excluded on purpose: ${file}\n    ${reason}\n`);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
