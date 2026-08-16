// Unit tests for the T04-19 discovery/verification codemod.
//
// This codemod has no `--apply` mode, so "did it rewrite correctly" is not the risk. The risk is
// the opposite one, and it is worse because it is silent: an UNDER-REPORT. A combination the
// scanner cannot see is a combination nobody prepares, and under T04-21's `"throw"` default that
// is a crash in an example the report swore was clean. Every test here pins a shape that actually
// under-reported at some point during this ticket, or a discrimination the scanner has to make to
// avoid the opposite failure (a false gap, which trains the reader to ignore the report).
import { expect, test } from "vitest";
import ts from "typescript";
import { classify, inScope, requestsOf, scanFile, COVERED_INDIRECTLY } from "./prepare-insertion.mjs";

/**
 * Builds a one-file program with a stub `vgpu` module. The scanner resolves factories by
 * DECLARATION FILE (`packages/vgpu-api/...`), so the stub has to live at such a path for the
 * symbol checks to behave the way they do against the real corpus.
 */
function scan(source: string) {
  const VGPU = "/repo/packages/vgpu-api/src/index.ts";
  const ENTRY = "/repo/examples/demo/src/example.ts";
  const lib = `
export declare class Draw { set(k: string, v: unknown): void }
export declare class Effect { set(k: string, v: unknown): void }
export declare class Compute {}
export declare class Bundle {}
export declare class Target { format: string }
export declare class FramePass {
  draw(d: Draw | Effect, opts?: object): void;
  bundles(...b: Bundle[]): void;
}
export declare class Frame {
  pass(opts: { target: Target } | Target, body: ((p: FramePass) => void) | Draw | Effect): void;
  compute(c: Compute, x?: number): void;
}
export declare function draw(gpu: unknown, o: unknown): Draw;
export declare function effect(gpu: unknown, o: unknown): Effect;
export declare function compute(gpu: unknown, o: unknown): Compute;
export declare function bundle(gpu: unknown, o: unknown, r: (b: FramePass) => void): Bundle;
export declare function target(gpu: unknown, o: unknown): Target;
export declare function frame(gpu: unknown, cb: (f: Frame) => void): void;
export declare function frameLoop(gpu: unknown, cb: (f: Frame) => void): { stop(): void };
export declare function prepare(gpu: unknown, r: unknown): Promise<unknown>;
`;
  const files = new Map([[VGPU, lib], [ENTRY, source]]);
  const host: ts.CompilerHost = {
    fileExists: (f) => files.has(f) || ts.sys.fileExists(f),
    readFile: (f) => files.get(f) ?? ts.sys.readFile(f),
    getSourceFile: (f, lang) => {
      const text = files.get(f) ?? ts.sys.readFile(f);
      return text === undefined ? undefined : ts.createSourceFile(f, text, lang, true);
    },
    getDefaultLibFileName: (o) => ts.getDefaultLibFilePath(o),
    writeFile: () => {},
    getCurrentDirectory: () => "/repo",
    getCanonicalFileName: (f) => f,
    useCaseSensitiveFileNames: () => true,
    getNewLine: () => "\n",
  };
  const program = ts.createProgram([ENTRY], { target: ts.ScriptTarget.ESNext, strict: false, skipLibCheck: true, noEmit: true }, host);
  const sourceFile = program.getSourceFile(ENTRY)!;
  return scanFile(program.getTypeChecker(), sourceFile, "examples/demo/src/example.ts");
}

const IMPORT = `import { draw, effect, compute, bundle, target, frame, frameLoop, prepare } from "/repo/packages/vgpu-api/src/index.ts";\n`;

test("pairs a p.draw() with the TARGET PROPERTY of the pass options, not the options object", () => {
  // The very first version of this scanner reported the whole `{ target: t, clear: CLEAR }` literal
  // as the target, which would have produced a `prepare()` request that does not typecheck.
  const s = scan(IMPORT + `
    const t = target(null, {});
    const e = effect(null, {});
    frame(null, (f) => f.pass({ target: t, clear: [0, 0, 0, 1] }, (p) => p.draw(e)));
  `);
  expect(requestsOf(s)).toEqual([
    expect.objectContaining({ kind: "draw", renderable: "e", target: "t" }),
  ]);
});

test("sees the SHORTHAND pass form f.pass(target, renderable)", () => {
  // Real under-report: `fft-ocean-surface` composites with this form, and the scanner walked right
  // past it because it only ever looked for `p.draw()` inside a callback.
  const s = scan(IMPORT + `
    const t = target(null, {});
    const e = effect(null, {});
    frame(null, (f) => f.pass({ target: t }, e));
  `);
  expect(requestsOf(s)).toEqual([
    expect.objectContaining({ kind: "draw", renderable: "e", target: "t" }),
  ]);
});

test("a bundle request carries NO target — the signature froze at construction", () => {
  const s = scan(IMPORT + `
    const t = target(null, {});
    const b = bundle(null, { target: t }, (r) => {});
    frame(null, (f) => f.pass({ target: t }, (p) => p.bundles(b)));
  `);
  const bundles = requestsOf(s).filter((r) => r.kind === "bundle");
  expect(bundles).toHaveLength(1);
  expect(bundles[0]!.target).toBeUndefined();
});

test("f.compute() is a compute combination; a same-named app helper is NOT", () => {
  // `fft-ocean-surface/scene.ts` calls `api.compute(gpu, …)` (the FACTORY) and the first scanner
  // reported it as an ENCODE site, inventing a combination that does not exist. Resolution is by
  // declaration, so the factory and the Frame method never collide.
  const s = scan(IMPORT + `
    const c = compute(null, {});
    const notAFrame = { compute(_x: unknown) {} };
    notAFrame.compute(c);
    frame(null, (f) => f.compute(c, 1));
  `);
  const computes = requestsOf(s).filter((r) => r.kind === "compute");
  expect(computes).toHaveLength(1);
  expect(computes[0]!.renderable).toBe("c");
});

test("resolves an ALIASED factory import by declaration, not by local spelling", () => {
  const s = scan(`import { effect as makeEffect, target, frame } from "/repo/packages/vgpu-api/src/index.ts";
    const t = target(null, {});
    const fx = makeEffect(null, {});
    frame(null, (f) => f.pass({ target: t }, (p) => p.draw(fx)));
  `);
  expect(s.constructions.map((c) => c.factory)).toContain("effect");
});

test("counts an existing prepare() even when it is imported under another name", () => {
  // A textual `prepare(` match reported three real files (air-painting, depth-estimation, mnist)
  // as unprepared. A false gap is as damaging as a missed one.
  const s = scan(`import { effect, target, prepare as warm } from "/repo/packages/vgpu-api/src/index.ts";
    const t = target(null, {});
    const e = effect(null, {});
    warm(null, [{ draw: e, target: t }]);
  `);
  expect(s.existingPrepares).toHaveLength(1);
});

test("flags a construction inside a loop as dynamic, so it is never auto-classified", () => {
  const s = scan(IMPORT + `
    const levels = [1, 2, 3].map(() => effect(null, {}));
    for (let i = 0; i < 3; i++) { effect(null, {}); }
  `);
  expect(s.constructions.filter((c) => c.dynamic === "iterated-callback")).toHaveLength(1);
  expect(s.constructions.filter((c) => c.dynamic === "loop")).toHaveLength(1);
  expect(classify(s).classification).toBe("manual-reviewed");
});

test("de-duplicates a renderable encoded twice into the same target, keeps it when the target differs", () => {
  const s = scan(IMPORT + `
    const a = target(null, {});
    const b = target(null, {});
    const e = effect(null, {});
    frame(null, (f) => {
      f.pass({ target: a }, (p) => { p.draw(e); p.draw(e); });
      f.pass({ target: b }, (p) => p.draw(e));
    });
  `);
  // Readiness is a property of a COMBINATION: (e, a) and (e, b) are two, (e, a) twice is one.
  expect(requestsOf(s)).toHaveLength(2);
});

test("per-call options objects are not mistaken for renderables", () => {
  const s = scan(IMPORT + `
    const t = target(null, {});
    const d = draw(null, {});
    frame(null, (f) => f.pass({ target: t }, (p) => p.draw(d, { pendingPipelines: "skip" })));
  `);
  expect(requestsOf(s)).toHaveLength(1);
});

test("scope excludes package tests and experiments, includes the three corpus roots", () => {
  expect(inScope("examples/by-example-s02-fullscreen/src/example.ts")).toBe(true);
  expect(inScope("apps/docs/examples/gradient/renderer.ts")).toBe(true);
  expect(inScope("apps/docs/components/hero/renderer.ts")).toBe(true);
  expect(inScope("packages/vgpu-api/tests/prepare.test.ts")).toBe(false);
  expect(inScope("experiments/ort-init-device/shared/pipeline.ts")).toBe(false);
  // A test file inside the corpus is still not corpus: it asserts behaviour, it does not ship it.
  expect(inScope("apps/docs/examples/gradient/renderer.test.ts")).toBe(false);
});

test("every indirect-coverage entry carries a reason, so the allowlist can never be a mute suppression", () => {
  const entries = Object.entries(COVERED_INDIRECTLY).flatMap(([file, byRenderable]) =>
    Object.entries(byRenderable).map(([renderable, reason]) => ({ file, renderable, reason })),
  );
  expect(entries.length).toBeGreaterThan(0);
  for (const entry of entries) {
    expect(typeof entry.reason, `${entry.file} / ${entry.renderable}`).toBe("string");
    // A reason has to name where the preparation actually happens; a one-word "covered" would make
    // the table unauditable.
    expect(entry.reason.length, `${entry.file} / ${entry.renderable}`).toBeGreaterThan(40);
  }
});
