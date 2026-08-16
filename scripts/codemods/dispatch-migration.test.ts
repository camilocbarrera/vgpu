// Unit + contract suite for T04-18's codemod (`dispatch-migration.mjs`).
//
// Like T04-17's, the unit under test is `planProgram()` driven over an IN-MEMORY program, because
// every decision this codemod makes needs a real `TypeChecker`: "is this receiver a `Compute` or
// somebody else's object that also owns a `.dispatch()`", "is this the counts overload or the
// indirect one", "is this callback parameter a `Frame`". A fixture-string test would be testing a
// different function than the one that ships.
//
// The suite is organised around the ONE property that makes this codemod dangerous: `dispatchOnce()`
// is async and `dispatch()` is not, so the `await`-insertion arm is pinned from both sides — the
// contexts where the `await` is free (already-async function, module top level) and the contexts
// where it would change a published signature and must therefore be reported instead.
import { describe, expect, it } from "vitest";
import path from "node:path";
import ts from "typescript";
import {
  planProgram, classifyReceiver, classifyOverload, enclosingFunction, frameCallbackParam,
} from "./dispatch-migration.mjs";

const REPO_ROOT = path.resolve(import.meta.dirname, "../..");

/**
 * Structural stand-ins for the real surface. `Compute` carries BOTH `dispatch` and `dispatchOnce`
 * (that pair is how the codemod recognises one), `Frame` carries `compute`/`pass`/`copyBuffer`, and
 * `Scheduler` is the adversarial neighbour: a foreign object that also owns a `.dispatch()` and must
 * never be touched.
 */
const AMBIENT = `
export interface Gpu { readonly __gpu: unique symbol }
export interface StorageBuffer { readonly gpu: unknown; readonly size: number }
export interface DispatchOptions { readonly indirect: StorageBuffer | { readonly buffer: StorageBuffer; readonly offset?: number } }
export interface Compute {
  set(values: Record<string, unknown>): this;
  bind(binding: string, resource: unknown): this;
  dispatch(x: number, y?: number, z?: number): void;
  dispatch(opts: DispatchOptions): void;
  dispatchOnce(x: number, y?: number, z?: number): Promise<void>;
  dispatchOnce(opts: DispatchOptions): Promise<void>;
}
export interface Pass { draw(d: unknown): void }
export interface Frame {
  compute(compute: Compute, x: number, y?: number, z?: number): void;
  pass(target: unknown, body: (p: Pass) => void): void;
  copyBuffer(opts: unknown): void;
  raw(body: (encoder: unknown) => void): void;
}
/** A foreign API that also has a \`.dispatch()\` — the reason receivers are typed, not name-matched. */
export interface Scheduler { dispatch(action: number): void }
export declare function compute(gpu: Gpu, opts: Record<string, unknown>): Compute;
export declare function storage(gpu: Gpu, n: number, opts?: Record<string, unknown>): StorageBuffer;
export declare function frame(gpu: Gpu, cb: (f: Frame) => void): void;
export declare function frameLoop(gpu: Gpu, cb: (f: Frame) => void): { stop(): void };
export declare const gpu: Gpu;
export declare const scheduler: Scheduler;
export declare const anything: any;
`;

const COMPILER_OPTIONS: ts.CompilerOptions = {
  target: ts.ScriptTarget.ESNext,
  module: ts.ModuleKind.ESNext,
  moduleResolution: ts.ModuleResolutionKind.Bundler,
  strict: true,
  skipLibCheck: true,
  noEmit: true,
  lib: ["lib.esnext.d.ts", "lib.dom.d.ts"],
  types: [],
};

function build(files: Record<string, string>) {
  const all: Record<string, string> = { "api.ts": AMBIENT, ...files };
  const host = ts.createCompilerHost(COMPILER_OPTIONS, true);
  const originalGetSourceFile = host.getSourceFile.bind(host);
  const originalReadFile = host.readFile.bind(host);
  const originalFileExists = host.fileExists.bind(host);
  const keyOf = (fileName: string) => fileName.replace(`${REPO_ROOT}/`, "");
  host.getSourceFile = (fileName, languageVersion, onError, shouldCreate) => {
    const key = keyOf(fileName);
    if (all[key] !== undefined) return ts.createSourceFile(fileName, all[key], languageVersion, true, ts.ScriptKind.TS);
    return originalGetSourceFile(fileName, languageVersion, onError, shouldCreate);
  };
  host.readFile = (fileName) => (all[keyOf(fileName)] !== undefined ? all[keyOf(fileName)] : originalReadFile(fileName));
  host.fileExists = (fileName) => (all[keyOf(fileName)] !== undefined ? true : originalFileExists(fileName));

  const corpus = Object.keys(all);
  const program = ts.createProgram(corpus.map((f) => path.join(REPO_ROOT, f)), COMPILER_OPTIONS, host);
  const checker = program.getTypeChecker();
  const sourceFileFor = (rel: string) => program.getSourceFile(path.join(REPO_ROOT, rel));
  return { all, corpus, program, checker, sourceFileFor };
}

/** Builds the program and runs the planner over it. */
function run(files: Record<string, string>) {
  const ctx = build(files);
  const { entries, fileTexts } = planProgram({ checker: ctx.checker, corpus: ctx.corpus, sourceFileFor: ctx.sourceFileFor });
  return { ...ctx, entries, fileTexts };
}

const IMPORTS = `import { compute, storage, frame, frameLoop, gpu, scheduler, anything, type Compute } from "./api";\n`;
const at = (entries: any[], line: number) => entries.find((e) => e.line === line);

// ---------------------------------------------------------------------------------------------
describe("f.compute() — the in-frame path", () => {
  it("rewrites a dispatch in a frame callback onto the frame's own encoder", () => {
    const { entries, fileTexts } = run({
      "a.ts": `${IMPORTS}const sim = compute(gpu, {});\nframe(gpu, (f) => {\n  sim.dispatch(2);\n});\n`,
    });
    expect(at(entries, 4).classification).toBe("auto-f-compute");
    expect(fileTexts.get("a.ts")).toContain("f.compute(sim,2);");
  });

  it("keeps every workgroup-count argument, in order", () => {
    const { fileTexts } = run({
      "a.ts": `${IMPORTS}const sim = compute(gpu, {});\nframe(gpu, (f) => {\n  sim.dispatch(2, 3, 4);\n});\n`,
    });
    expect(fileTexts.get("a.ts")).toContain("f.compute(sim,2, 3, 4);");
  });

  it("uses the callback's OWN parameter name, whatever it is called", () => {
    const { fileTexts } = run({
      "a.ts": `${IMPORTS}const sim = compute(gpu, {});\nframeLoop(gpu, (currentFrame) => {\n  sim.dispatch(1);\n});\n`,
    });
    expect(fileTexts.get("a.ts")).toContain("currentFrame.compute(sim,1);");
  });

  it("recognises the frame callback through an ALIASED import (typed, never name-matched)", () => {
    const { entries } = run({
      "a.ts": `import { compute, frame as runFrame, gpu } from "./api";\nconst sim = compute(gpu, {});\nrunFrame(gpu, (f) => {\n  sim.dispatch(1);\n});\n`,
    });
    expect(at(entries, 4).classification).toBe("auto-f-compute");
  });

  it("preserves a chained, multi-line receiver byte-for-byte", () => {
    const src = `${IMPORTS}const sim = compute(gpu, {});\nframe(gpu, (f) => {\n  sim.set({\n    a: 1,\n    b: 2,\n  }).dispatch(16, 9);\n});\n`;
    const { fileTexts } = run({ "a.ts": src });
    expect(fileTexts.get("a.ts")).toContain(`f.compute(sim.set({\n    a: 1,\n    b: 2,\n  }),16, 9);`);
  });

  it("refuses a dispatch nested inside f.pass() — f.compute() there throws VGPU-FRAME-ENCODER-LOCKED", () => {
    const { entries, fileTexts } = run({
      "a.ts": `${IMPORTS}const sim = compute(gpu, {});\nframe(gpu, (f) => {\n  f.pass(null, (p) => {\n    sim.dispatch(1);\n  });\n});\n`,
    });
    // The nearest enclosing function is the PASS callback, not the frame callback: a sync context
    // with no `f.compute()` available, so it is reported rather than moved onto a locked encoder.
    expect(at(entries, 5).classification).toBe("ambiguous-sync-context");
    expect(at(entries, 5).reason).toBe("sync-callback-argument");
    expect(fileTexts.has("a.ts")).toBe(false);
  });

  it("refuses the indirect overload in a frame callback — f.compute() has no indirect twin", () => {
    const { entries, fileTexts } = run({
      "a.ts": `${IMPORTS}const sim = compute(gpu, {});\nconst args = storage(gpu, 12, { indirect: true });\nframe(gpu, (f) => {\n  sim.dispatch({ indirect: args });\n});\n`,
    });
    expect(at(entries, 5).classification).toBe("ambiguous-indirect-options");
    expect(fileTexts.has("a.ts")).toBe(false);
  });

  it("refuses a frame callback that destructures its Frame — there is no `f` to write", () => {
    const { entries } = run({
      "a.ts": `${IMPORTS}const sim = compute(gpu, {});\nframe(gpu, ({ compute: c }) => {\n  sim.dispatch(1);\n});\n`,
    });
    expect(at(entries, 4).classification).toBe("ambiguous-sync-context");
    expect(at(entries, 4).reason).toBe("frame-callback-without-named-parameter");
  });
});

// ---------------------------------------------------------------------------------------------
describe("await dispatchOnce() — the standalone path, and the async boundary", () => {
  it("migrates inside a function that is ALREADY async (the await costs nothing observable)", () => {
    const { entries, fileTexts } = run({
      "a.ts": `${IMPORTS}const sim = compute(gpu, {});\nexport async function go() {\n  sim.dispatch(1);\n}\n`,
    });
    expect(at(entries, 4).classification).toBe("auto-dispatch-once");
    expect(at(entries, 4).context).toBe("async-function");
    expect(fileTexts.get("a.ts")).toContain("await sim.dispatchOnce(1);");
  });

  it("migrates at the top level of an ES module (top-level await)", () => {
    const { entries, fileTexts } = run({
      "a.ts": `${IMPORTS}const sim = compute(gpu, {});\nsim.dispatch(1);\n`,
    });
    expect(at(entries, 3).classification).toBe("auto-dispatch-once");
    expect(at(entries, 3).context).toBe("top-level-module");
    expect(fileTexts.get("a.ts")).toContain("await sim.dispatchOnce(1);");
  });

  it("keeps the indirect overload on the standalone path — dispatchOnce() DOES have that twin", () => {
    const { entries, fileTexts } = run({
      "a.ts": `${IMPORTS}const sim = compute(gpu, {});\nconst args = storage(gpu, 12, { indirect: true });\nexport async function go() {\n  sim.dispatch({ indirect: args });\n}\n`,
    });
    expect(at(entries, 5).classification).toBe("auto-dispatch-once");
    expect(fileTexts.get("a.ts")).toContain("await sim.dispatchOnce({ indirect: args });");
  });

  it("migrates inside a loop body of an async function without reordering anything", () => {
    const { fileTexts } = run({
      "a.ts": `${IMPORTS}const sim = compute(gpu, {});\nexport async function go() {\n  for (let i = 0; i < 8; i++) { sim.dispatch(2, 2); }\n}\n`,
    });
    expect(fileTexts.get("a.ts")).toContain("for (let i = 0; i < 8; i++) { await sim.dispatchOnce(2, 2); }");
  });

  // --- the refusals: every one of these would change an observable contract ---

  it("refuses a plain sync function (making it async is a caller-visible change)", () => {
    const { entries, fileTexts } = run({
      "a.ts": `${IMPORTS}const sim = compute(gpu, {});\nfunction step() {\n  sim.dispatch(1);\n}\nstep();\n`,
    });
    expect(at(entries, 4).classification).toBe("ambiguous-sync-context");
    expect(at(entries, 4).reason).toBe("sync-local-function");
    expect(fileTexts.has("a.ts")).toBe(false);
  });

  it("names an EXPORTED sync function as the published-signature case it is", () => {
    const { entries } = run({
      "a.ts": `${IMPORTS}const sim = compute(gpu, {});\nexport function step(): void {\n  sim.dispatch(1);\n}\n`,
    });
    expect(at(entries, 4).reason).toBe("sync-exported-function");
    expect(at(entries, 4).note).toContain("exported-declaration");
  });

  it("names a member of a DECLARED interface, the shape every example in the corpus uses", () => {
    const { entries } = run({
      "a.ts": `${IMPORTS}const sim = compute(gpu, {});\nexport interface Scene { simulate(dt: number): void }\nexport function make(): Scene {\n  return {\n    simulate(dt: number) {\n      sim.dispatch(1);\n    },\n  };\n}\n`,
    });
    expect(at(entries, 7).classification).toBe("ambiguous-sync-context");
    expect(at(entries, 7).reason).toBe("sync-member-of-a-declared-interface");
    expect(at(entries, 7).note).toContain("Scene.simulate");
  });

  it("refuses a sync callback handed to a third party, even inside an async function", () => {
    const { entries, fileTexts } = run({
      "a.ts": `${IMPORTS}const sim = compute(gpu, {});\nexport async function go() {\n  [1, 2].forEach(() => {\n    sim.dispatch(1);\n  });\n}\n`,
    });
    // The ENCLOSING function is the sync arrow, not `go`: awaiting there would resolve into a
    // callback whose return value `forEach` throws away, so the dispatch would escape the function.
    expect(at(entries, 5).classification).toBe("ambiguous-sync-context");
    expect(at(entries, 5).reason).toBe("sync-callback-argument");
    expect(fileTexts.has("a.ts")).toBe(false);
  });

  it("refuses a generator, which cannot await", () => {
    const { entries } = run({
      "a.ts": `${IMPORTS}const sim = compute(gpu, {});\nexport function* steps() {\n  sim.dispatch(1);\n  yield 1;\n}\n`,
    });
    expect(at(entries, 4).reason).toBe("enclosing-generator-cannot-await");
  });

  it("refuses a call whose VALUE is consumed — `await` there needs parentheses and changes meaning", () => {
    const { entries, fileTexts } = run({
      "a.ts": `${IMPORTS}const sim = compute(gpu, {});\nexport async function go() {\n  const done = [sim.dispatch(1)];\n  return done;\n}\n`,
    });
    expect(at(entries, 4).classification).toBe("ambiguous-await-position");
    expect(fileTexts.has("a.ts")).toBe(false);
  });
});

// ---------------------------------------------------------------------------------------------
describe("receiver and argument oracles", () => {
  it("never touches a foreign `.dispatch()` (a Scheduler is not a Compute)", () => {
    const { entries, fileTexts } = run({
      "a.ts": `${IMPORTS}export async function go() {\n  scheduler.dispatch(1);\n}\n`,
    });
    expect(at(entries, 3).classification).toBe("not-a-compute");
    expect(fileTexts.has("a.ts")).toBe(false);
  });

  it("reports an unresolved receiver as a bucket, never as a silent skip", () => {
    const { entries, fileTexts } = run({
      "a.ts": `${IMPORTS}export async function go() {\n  anything.dispatch(1);\n}\n`,
    });
    expect(at(entries, 3).classification).toBe("unresolved-receiver");
    expect(fileTexts.has("a.ts")).toBe(false);
  });

  it("ignores `.dispatch(` inside a string, a template chunk and a comment", () => {
    const { entries, fileTexts } = run({
      "a.ts": `${IMPORTS}const doc = "sim.dispatch(1)";\nconst tpl = \`call sim.dispatch(2) here\`;\n// sim.dispatch(3)\n/* sim.dispatch(4) */\nexport const n = doc.length + tpl.length;\n`,
    });
    expect(entries).toEqual([]);
    expect(fileTexts.size).toBe(0);
  });

  it("classifies the two overloads apart", () => {
    const ctx = build({
      "a.ts": `${IMPORTS}const sim = compute(gpu, {});\nconst args = storage(gpu, 12, { indirect: true });\nexport async function go() {\n  sim.dispatch(1, 2, 3);\n  sim.dispatch({ indirect: args });\n}\n`,
    });
    const sf = ctx.sourceFileFor("a.ts")!;
    const calls: ts.CallExpression[] = [];
    const visit = (n: ts.Node): void => {
      if (ts.isCallExpression(n) && ts.isPropertyAccessExpression(n.expression) && n.expression.name.text === "dispatch") calls.push(n);
      ts.forEachChild(n, visit);
    };
    visit(sf);
    expect(calls).toHaveLength(2);
    expect(classifyOverload(calls[0]!, ctx.checker)).toBe("counts");
    expect(classifyOverload(calls[1]!, ctx.checker)).toBe("options");
    expect(classifyReceiver(calls[0]!.expression.getChildAt(0), ctx.checker).kind).toBe("compute");
  });
});

// ---------------------------------------------------------------------------------------------
describe("harness contracts", () => {
  it("is idempotent: a second run over migrated text finds nothing left to do", () => {
    const first = run({
      "a.ts": `${IMPORTS}const sim = compute(gpu, {});\nframe(gpu, (f) => {\n  sim.dispatch(2);\n});\nexport async function go() {\n  sim.dispatch(3);\n}\n`,
    });
    const migrated = first.fileTexts.get("a.ts")!;
    expect(migrated).toContain("f.compute(sim,2);");
    expect(migrated).toContain("await sim.dispatchOnce(3);");
    const second = run({ "a.ts": migrated });
    expect(second.entries).toEqual([]);
    expect(second.fileTexts.size).toBe(0);
  });

  it("reports before/after spans that reproduce the real diff exactly (dry-run == apply)", () => {
    const src = `${IMPORTS}const sim = compute(gpu, {});\nframe(gpu, (f) => {\n  sim.dispatch(2);\n});\nexport async function go() {\n  sim.dispatch(3);\n}\n`;
    const { entries, fileTexts } = run({ "a.ts": src });
    let replayed = src;
    for (const entry of entries) {
      if (entry.before === entry.after) continue;
      expect(replayed).toContain(entry.before);
      replayed = replayed.replace(entry.before, entry.after);
    }
    expect(replayed).toBe(fileTexts.get("a.ts"));
  });

  it("excludes a legacy-form test subject, but still records the context it WOULD have had", () => {
    const rel = "packages/vgpu-api/tests/compute/aliasing.test.ts";
    const { entries, fileTexts } = run({
      [rel]: `${IMPORTS.replace("./api", "../../../../api")}const sim = compute(gpu, {});\nexport async function go() {\n  sim.dispatch(1);\n}\n`,
    });
    expect(at(entries, 4).classification).toBe("excluded-test-subject");
    // The exclusion is a decision, not a blind spot: the site it hides is one that would otherwise
    // have migrated cleanly, and the report says so.
    expect(at(entries, 4).contextWouldBe).toBe("auto-dispatch-once");
    expect(fileTexts.has(rel)).toBe(false);
  });

  it("gives every site a bucket — no site is ever dropped silently", () => {
    const { entries } = run({
      "a.ts": `${IMPORTS}const sim = compute(gpu, {});\nexport async function ok() { sim.dispatch(1); }\nexport function no() { sim.dispatch(2); }\nframe(gpu, (f) => { sim.dispatch(3); });\nexport async function foreign() { scheduler.dispatch(4); }\nexport async function un() { anything.dispatch(5); }\n`,
    });
    expect(entries.map((e) => e.classification)).toEqual([
      "auto-dispatch-once", "ambiguous-sync-context", "auto-f-compute", "not-a-compute", "unresolved-receiver",
    ]);
    for (const entry of entries) expect(typeof entry.classification).toBe("string");
  });
});

// ---------------------------------------------------------------------------------------------
describe("scope helpers", () => {
  it("enclosingFunction stops at the nearest function-like node, and returns null at top level", () => {
    const ctx = build({
      "a.ts": `${IMPORTS}const sim = compute(gpu, {});\nsim.dispatch(1);\nexport function outer() { function inner() { sim.dispatch(2); } inner(); }\n`,
    });
    const sf = ctx.sourceFileFor("a.ts")!;
    const calls: ts.CallExpression[] = [];
    const visit = (n: ts.Node): void => {
      if (ts.isCallExpression(n) && ts.isPropertyAccessExpression(n.expression) && n.expression.name.text === "dispatch") calls.push(n);
      ts.forEachChild(n, visit);
    };
    visit(sf);
    expect(enclosingFunction(calls[0]!)).toBeNull();
    const inner = enclosingFunction(calls[1]!) as ts.FunctionDeclaration;
    expect(inner.name?.text).toBe("inner");
  });

  it("frameCallbackParam answers only for a real Frame callback", () => {
    const ctx = build({
      "a.ts": `${IMPORTS}const sim = compute(gpu, {});\nframe(gpu, (f) => { sim.dispatch(1); });\n[1].forEach((n) => { sim.dispatch(n); });\n`,
    });
    const sf = ctx.sourceFileFor("a.ts")!;
    const arrows: ts.ArrowFunction[] = [];
    const visit = (n: ts.Node): void => { if (ts.isArrowFunction(n)) arrows.push(n); ts.forEachChild(n, visit); };
    visit(sf);
    expect(frameCallbackParam(arrows[0]!, ctx.checker)).toEqual({ param: "f" });
    expect(frameCallbackParam(arrows[1]!, ctx.checker)).toBeNull();
  });
});
