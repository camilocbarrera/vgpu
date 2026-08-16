// The WGSL reflection oracle T04-17 classifies flat-bag keys with.
//
// ## Why a codemod needs reflection at all
//
// The flat bag accepts BOTH a complete binding name and a struct MEMBER name:
//
//   fx.set({ params: { time: 1 } })   // `params` is a binding
//   fx.set({ time: 1 })              // `time` is a MEMBER of binding `params`
//
// The binding-scoped form has no member shortcut (design §6, enforced by `setScoped()` in
// `set-core.ts`: a member name throws with a fix-it naming the owner binding). So the two shapes
// above migrate to *different* code — `.set("params", { time: 1 })` in both cases — and a codemod
// that blindly writes `.set("time", 1)` produces a call that throws at runtime. Nothing in the
// TypeScript types distinguishes them: `set(values: Record<string, unknown>)` accepts any key.
//
// The only source of truth is the shader, so this module resolves the WGSL text behind a
// construction site and runs the repo's own `reflectSource()` over it — the same reflection the
// runtime uses, not a re-implementation. `set-core.ts`'s `findMemberBinding()` is not importable
// (module-private, and closed over per-instance state), so its *resolution rule* is replicated here
// against the same `Reflection` object: a member name resolves to its owning binding, and a member
// name owned by more than one binding is ambiguous — byte-for-byte the runtime's own outcome
// (`Binding member '<x>' is ambiguous`), which is why an ambiguous key must be reported, never
// migrated.
import { existsSync } from "node:fs";
import path from "node:path";
import ts from "typescript";
import { unwrap } from "./corpus-program.mjs";

/** Factories whose result is a `Draw`/`Effect`/`Compute`, i.e. a receiver with WGSL behind it. */
export const INSTANCE_FACTORIES = new Set(["effect", "compute", "draw"]);

/** `effect`/`compute`/`draw` name of a call's callee, for bare and namespaced calls alike. */
export function instanceFactoryName(node) {
  if (!node || !ts.isCallExpression(node)) return null;
  const e = node.expression;
  const name = ts.isIdentifier(e)
    ? e.text
    : (ts.isPropertyAccessExpression(e) && ts.isIdentifier(e.name) ? e.name.text : null);
  return name && INSTANCE_FACTORIES.has(name) ? name : null;
}

/** The options object literal of a construction call, if it passes one. */
export function optionsBagOf(call) {
  for (const arg of call.arguments) {
    const inner = unwrap(arg);
    if (ts.isObjectLiteralExpression(inner)) return inner;
  }
  return null;
}

/** A named property of an object literal (identifier or string key), or `undefined`. */
export function propertyNamed(objectLiteral, name) {
  return objectLiteral.properties.find((p) =>
    (ts.isPropertyAssignment(p) || ts.isShorthandPropertyAssignment(p))
    && p.name && (ts.isIdentifier(p.name) || ts.isStringLiteral(p.name))
    && p.name.text === name);
}

/**
 * The expression that carries the shader for a construction call:
 *   - single-object form: the `shader:` property of the options bag (post-T04-16 this is the norm);
 *   - 2-argument shorthand `effect(gpu, SRC)`: the second argument (survives the cut, so it is not
 *     an error to meet one here — #344 explicitly left `effect(gpu, SRC)` untouched).
 */
export function shaderExpressionOf(call) {
  const bag = optionsBagOf(call);
  if (bag) {
    const prop = propertyNamed(bag, "shader");
    if (prop && ts.isPropertyAssignment(prop)) return prop.initializer;
    if (prop && ts.isShorthandPropertyAssignment(prop)) return prop.name;
  }
  // `effect(gpu, SRC)` / `compute(gpu, SRC)` / `draw(gpu, SRC)`: the shader is positional.
  if (call.arguments.length === 2) {
    const second = unwrap(call.arguments[1]);
    if (!ts.isObjectLiteralExpression(second)) return call.arguments[1];
  }
  return null;
}

/**
 * Resolves an expression to WGSL source text, following the shapes this corpus actually uses:
 * string/template literals, `import SRC from "./x.wgsl"` (read off disk — this is how
 * `components/hero` and most `apps/docs/examples` ship their shaders), identifiers pointing at
 * either of those, `+` concatenation, and the `{ version, wgsl }` ShaderSource artifact literal.
 *
 * Returns `{ ok: true, wgsl }` or `{ ok: false, why }`. `why` is propagated into the report so an
 * unresolved shader is a named bucket, never a silent skip.
 */
export function resolveWgslText(expr, { checker, repoRoot }, depth = 0, seen = new Set()) {
  if (depth > 8) return { ok: false, why: "shader-resolution-depth-limit" };
  const n = unwrap(expr);

  if (ts.isStringLiteral(n) || ts.isNoSubstitutionTemplateLiteral(n)) return { ok: true, wgsl: n.text };

  if (ts.isTemplateExpression(n)) {
    let out = n.head.text;
    for (const span of n.templateSpans) {
      const sub = resolveWgslText(span.expression, { checker, repoRoot }, depth + 1, seen);
      if (!sub.ok) return { ok: false, why: sub.why };
      out += sub.wgsl + span.literal.text;
    }
    return { ok: true, wgsl: out };
  }

  if (ts.isBinaryExpression(n) && n.operatorToken.kind === ts.SyntaxKind.PlusToken) {
    const a = resolveWgslText(n.left, { checker, repoRoot }, depth + 1, seen);
    if (!a.ok) return a;
    const b = resolveWgslText(n.right, { checker, repoRoot }, depth + 1, seen);
    if (!b.ok) return b;
    return { ok: true, wgsl: a.wgsl + b.wgsl };
  }

  if (ts.isObjectLiteralExpression(n)) {
    const w = propertyNamed(n, "wgsl");
    if (w && ts.isPropertyAssignment(w)) return resolveWgslText(w.initializer, { checker, repoRoot }, depth + 1, seen);
    return { ok: false, why: "shader-is-object-literal-without-wgsl" };
  }

  if (ts.isIdentifier(n) || ts.isPropertyAccessExpression(n)) {
    const raw = checker.getSymbolAtLocation(n);
    if (!raw) return { ok: false, why: "shader-identifier-has-no-symbol" };

    // `import bakeWgsl from './bake.wgsl'` — how `components/hero` and most `apps/docs/examples`
    // ship their shaders. This has to be tried on the UNALIASED symbol: the loader is typed by an
    // ambient `declare module '*.wgsl' { const src: string; export default src }` shim, so
    // `getAliasedSymbol()` lands on that shim's `const src: string` — a declaration with no
    // initializer, in a `.d.ts`, from which the real `.wgsl` file is unreachable. The import
    // declaration is only visible from the local (aliasing) symbol.
    //
    // The result is the FILE PATH, not its text: a `.wgsl` entry may `import` other `.wgsl` modules,
    // and `reflectSource()` refuses an import graph outright (VGPU-WGSL-REFLECT-SOURCE-IMPORT). Only
    // `resolveShader()` can flatten the graph, and it needs the entry path.
    for (const d of raw.getDeclarations() ?? []) {
      const entryPath = wgslFileForImport(d, repoRoot);
      if (entryPath !== null) return { ok: true, entryPath };
    }

    const sym = (raw.flags & ts.SymbolFlags.Alias) ? checker.getAliasedSymbol(raw) : raw;
    for (const d of sym.getDeclarations() ?? []) {
      if (seen.has(d)) continue;
      seen.add(d);
      if ((ts.isVariableDeclaration(d) || ts.isPropertyAssignment(d)) && d.initializer) {
        return resolveWgslText(d.initializer, { checker, repoRoot }, depth + 1, seen);
      }
      const entryPath = wgslFileForImport(d, repoRoot);
      if (entryPath !== null) return { ok: true, entryPath };
    }
    const t = checker.getTypeAtLocation(n);
    if (t.isStringLiteral()) return { ok: true, wgsl: t.value };
    return { ok: false, why: "shader-identifier-not-resolvable-to-text" };
  }

  if (ts.isCallExpression(n)) return { ok: false, why: "shader-is-a-call-expression" };
  return { ok: false, why: `shader-is-${ts.SyntaxKind[n.kind]}` };
}

/** If `decl` is an import of a `.wgsl` module, returns that file's absolute path. */
function wgslFileForImport(decl, repoRoot) {
  let cur = decl;
  while (cur && !ts.isImportDeclaration(cur)) cur = cur.parent;
  if (!cur || !ts.isStringLiteral(cur.moduleSpecifier)) return null;
  const spec = cur.moduleSpecifier.text;
  if (!spec.endsWith(".wgsl")) return null;
  const importerDir = path.dirname(cur.getSourceFile().fileName);
  const candidates = spec.startsWith(".")
    ? [path.resolve(importerDir, spec)]
    : [path.join(repoRoot, "apps/docs", spec.replace(/^@\//u, "")), path.join(repoRoot, spec)];
  for (const c of candidates) if (existsSync(c)) return c;
  return null;
}

/**
 * Reflection cache, addressed by `reflectionKey()` so an inline WGSL string and a `.wgsl` entry file
 * share one store. This corpus reuses the same shader across many instances (`bloomWgsl` backs six of
 * the hero's effects), so reflecting once per key rather than once per site matters.
 */
const reflectionCache = new Map();

/** Cache address for a resolved shader: its entry path, or the WGSL text itself. */
export function reflectionKey(resolved) {
  return resolved.entryPath ? `path:${resolved.entryPath}` : `text:${resolved.wgsl}`;
}

function toBindingIndex(reflection) {
  const bindings = new Map();
  const memberOwners = new Map();
  for (const b of reflection.bindings) {
    const members = b.layout?.members?.map((m) => m.name) ?? [];
    bindings.set(b.name, { kind: b.kind, bindingLayoutKind: b.bindingLayout?.kind, members });
    for (const m of members) {
      if (!memberOwners.has(m)) memberOwners.set(m, []);
      memberOwners.get(m).push(b.name);
    }
  }
  return { ok: true, bindings, memberOwners };
}

/**
 * Reflects one resolved shader into the cache. Async because a `.wgsl` entry has to go through
 * `resolveShader()` (which loads and flattens the import graph off disk); `validate: "off"` keeps a
 * WebGPU adapter out of a codemod's dependencies.
 */
export async function loadReflection(resolved, { reflectSource, resolveShader, repoRoot }) {
  const key = reflectionKey(resolved);
  const hit = reflectionCache.get(key);
  if (hit) return hit;
  let value;
  try {
    const reflection = resolved.entryPath
      ? (await resolveShader({ entry: resolved.entryPath, rootDir: repoRoot, validate: "off" })).reflection
      : reflectSource(resolved.wgsl, "<codemod>");
    value = toBindingIndex(reflection);
  } catch (error) {
    value = { ok: false, why: `wgsl-reflection-failed: ${String(error?.message ?? error).split("\n")[0].slice(0, 120)}` };
  }
  reflectionCache.set(key, value);
  return value;
}

/** Synchronous cache read, for the planning pass. Throws if `loadReflection()` was never awaited. */
export function reflectionFor(resolved) {
  const key = reflectionKey(resolved);
  const hit = reflectionCache.get(key);
  if (!hit) throw new Error(`wgsl-oracle: reflection for ${key.slice(0, 80)} was not preloaded`);
  return hit;
}

/** Only for tests: drops the memoized reflections so a fixture cannot leak into another case. */
export function clearReflectionCache() {
  reflectionCache.clear();
}

/**
 * Resolves one flat-bag key against a reflection, replicating `set-core.ts`'s resolution order:
 * a complete binding name wins, otherwise the key is looked up as a struct member and must have
 * exactly one owner.
 */
export function resolveKey(key, reflection) {
  const direct = reflection.bindings.get(key);
  if (direct) return { kind: "binding", binding: key, info: direct };
  const owners = reflection.memberOwners.get(key) ?? [];
  if (owners.length === 1) return { kind: "member", binding: owners[0], info: reflection.bindings.get(owners[0]) };
  if (owners.length > 1) return { kind: "ambiguous-member", owners };
  return { kind: "unknown" };
}
