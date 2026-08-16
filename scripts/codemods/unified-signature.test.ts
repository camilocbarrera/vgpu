import { describe, expect, it } from "vitest";
// @ts-expect-error - .mjs codemod module without type declarations, same pattern as lib/*.test.ts
import { transformSource } from "./unified-signature.mjs";

type Entry = { file: string; line: number; before: string; after: string; classification: string };
type Result = { text: string; entries: Entry[] };

const run = (text: string, relPath = "corpus/example.ts", options?: { legacySubjectReason?: string }): Result =>
  transformSource(text, relPath, options) as Result;

/** The transformed text of a one-call snippet, for terse assertions. */
const out = (text: string, relPath?: string): string => run(text, relPath).text;

describe("transformSource — the core rewrite", () => {
  it("moves the source argument into the existing options bag as `shader`, inline", () => {
    expect(out(`effect(gpu, SRC, { blend: "additive" });`))
      .toBe(`effect(gpu, { shader: SRC, blend: "additive" });`);
  });

  it("spreads the options bag's properties INLINE, never as a spread of an object literal", () => {
    // The whole point of the ticket's `{ shader: SRC, blend: "additive" }` requirement: a nested
    // `...{ blend }` would be legal JS but unreadable, and would defeat the type narrowing the
    // single-object overload relies on.
    const result = out(`effect(gpu, SRC, { blend: "additive" });`);
    expect(result).not.toContain("...");
  });

  it("handles compute() identically", () => {
    expect(out(`compute(gpu, SRC, { entry: "main" });`))
      .toBe(`compute(gpu, { shader: SRC, entry: "main" });`);
  });

  it("emits a real spread when the options argument is an identifier", () => {
    expect(out(`effect(gpu, SRC, opts);`)).toBe(`effect(gpu, { shader: SRC, ...opts });`);
  });

  it("emits a real spread for a property access / element access / call options argument", () => {
    expect(out(`effect(gpu, SRC, cfg.opts);`)).toBe(`effect(gpu, { shader: SRC, ...cfg.opts });`);
    expect(out(`effect(gpu, SRC, table["opts"]);`)).toBe(`effect(gpu, { shader: SRC, ...table["opts"] });`);
    expect(out(`effect(gpu, SRC, makeOpts());`)).toBe(`effect(gpu, { shader: SRC, ...makeOpts() });`);
  });

  it("parenthesizes a spread whose expression is not a plain reference or call", () => {
    expect(out(`effect(gpu, SRC, flag ? a : b);`))
      .toBe(`effect(gpu, { shader: SRC, ...(flag ? a : b) });`);
  });

  it("collapses an empty options bag instead of emitting a spread of nothing", () => {
    expect(out(`effect(gpu, SRC, {});`)).toBe(`effect(gpu, { shader: SRC });`);
    expect(run(`effect(gpu, SRC, {});`).entries[0]?.classification).toBe("auto-inline-opts-empty");
  });

  it("leaves the 1-arg shorthand and the already-migrated single-object form alone", () => {
    // D3: `effect(gpu, source)` survives the cut untouched, and so does `effect(gpu, { shader })`.
    const text = [
      `effect(gpu, SRC);`,
      `compute(gpu, SRC);`,
      `effect(gpu, { shader: SRC, blend: "additive" });`,
    ].join("\n");
    const result = run(text);
    expect(result.text).toBe(text);
    expect(result.entries).toEqual([]);
  });
});

describe("transformSource — formatting is preserved, not reprinted", () => {
  it("keeps a multi-line options bag multi-line and matches its indentation", () => {
    expect(out([
      `const fx = effect(gpu, SRC, {`,
      `  label: "fx",`,
      `  blend: "additive",`,
      `});`,
    ].join("\n"))).toBe([
      `const fx = effect(gpu, {`,
      `  shader: SRC,`,
      `  label: "fx",`,
      `  blend: "additive",`,
      `});`,
    ].join("\n"));
  });

  it("matches deeper indentation of a nested call", () => {
    expect(out([
      `const map = {`,
      `  logo: effect(gpu, withTopLeftFullscreen(logoWgsl), {`,
      `        label: 'logo',`,
      `      }),`,
      `};`,
    ].join("\n"))).toBe([
      `const map = {`,
      `  logo: effect(gpu, {`,
      `        shader: withTopLeftFullscreen(logoWgsl),`,
      `        label: 'logo',`,
      `      }),`,
      `};`,
    ].join("\n"));
  });

  it("keeps a trailing comma and inner comments in the options bag", () => {
    expect(out([
      `effect(gpu, SRC, {`,
      `  // additive so the passes accumulate`,
      `  blend: "additive",`,
      `});`,
    ].join("\n"))).toBe([
      `effect(gpu, {`,
      `  shader: SRC,`,
      `  // additive so the passes accumulate`,
      `  blend: "additive",`,
      `});`,
    ].join("\n"));
  });

  it("preserves a spaceless options bag's own style", () => {
    expect(out(`effect(gpu, SRC, {blend: "additive"});`))
      .toBe(`effect(gpu, {shader: SRC, blend: "additive"});`);
  });

  it("splices a multi-line template-literal source VERBATIM (re-indenting would change the WGSL)", () => {
    const text = [
      "const wave = effect(gpu, `",
      "    @fragment fn fs_main() -> @location(0) vec4f {",
      "      return vec4f(1.0);",
      "    }",
      "  `, { set: { params: { time: 0 } } });",
    ].join("\n");
    const result = out(text);
    expect(result).toBe([
      "const wave = effect(gpu, { shader: `",
      "    @fragment fn fs_main() -> @location(0) vec4f {",
      "      return vec4f(1.0);",
      "    }",
      "  `, set: { params: { time: 0 } } });",
    ].join("\n"));
    // The WGSL body itself must be byte-identical, whitespace included.
    expect(result).toContain("\n      return vec4f(1.0);\n");
  });

  it("only rewrites the call, leaving every other byte of the file untouched", () => {
    const text = [
      `// a comment mentioning effect(gpu, source, opts) in prose`,
      `const s = "effect(gpu, SRC, { blend: 'additive' })";`,
      `const fx = effect(gpu, SRC, { label: "fx" });`,
      `export default fx;`,
    ].join("\n");
    const result = run(text);
    expect(result.entries).toHaveLength(1);
    expect(result.text).toBe([
      `// a comment mentioning effect(gpu, source, opts) in prose`,
      `const s = "effect(gpu, SRC, { blend: 'additive' })";`,
      `const fx = effect(gpu, { shader: SRC, label: "fx" });`,
      `export default fx;`,
    ].join("\n"));
  });
});

describe("transformSource — source-argument shapes (the regex's false negatives)", () => {
  it("accepts a call expression as the source", () => {
    expect(out(`effect(gpu, withTopLeftFullscreen(logoWgsl), { label: 'logo' });`))
      .toBe(`effect(gpu, { shader: withTopLeftFullscreen(logoWgsl), label: 'logo' });`);
  });

  it("accepts a property access as the source (the `shaders.spectrumInit` shape)", () => {
    expect(out(`compute(gpu, shaders.spectrumInit, { label: "init" });`))
      .toBe(`compute(gpu, { shader: shaders.spectrumInit, label: "init" });`);
  });

  it("accepts a ShaderSource artifact object literal as the source", () => {
    expect(out(`effect(gpu, { version: 1, wgsl: FRAGMENT }, { label: "shader" });`))
      .toBe(`effect(gpu, { shader: { version: 1, wgsl: FRAGMENT }, label: "shader" });`);
  });

  it("parenthesizes a comma/sequence expression source so its comma cannot split the properties", () => {
    expect(out(`effect(gpu, (warmUp(), SRC), { label: "fx" });`))
      .toBe(`effect(gpu, { shader: (warmUp(), SRC), label: "fx" });`);
  });
});

describe("transformSource — callee shapes", () => {
  it("covers namespace/alias calls, not just bare imports", () => {
    // fft-ocean-surface/scene.ts uses `api.effect`/`api.compute`; the hero uses `vgpu.effect`.
    expect(out(`api.effect(gpu, SRC, { label: "a" });`))
      .toBe(`api.effect(gpu, { shader: SRC, label: "a" });`);
    expect(out(`vgpu.compute(gpu, SRC, { entry: "main" });`))
      .toBe(`vgpu.compute(gpu, { shader: SRC, entry: "main" });`);
  });

  it("does not touch a same-named method that is a different API (positional dispatch)", () => {
    // `framePass.compute(job, x, y, z)` shares the name but takes dispatch sizes: its 3rd argument
    // cannot be an options bag, which is how it is rejected without a type checker.
    const text = `frame(gpu, (f) => { f.compute(sim, 4, 5); f.compute(sim, 1, -2); });`;
    const result = run(text);
    expect(result.text).toBe(text);
    expect(result.entries.map((e) => e.classification))
      .toEqual(["skipped-not-options-bag", "skipped-not-options-bag"]);
  });

  it("does not touch an unrelated 3-argument function", () => {
    const text = `render(gpu, SRC, { label: "x" });`;
    expect(out(text)).toBe(text);
  });

  it("only ever matches EXACTLY 3 arguments, never 4 or more", () => {
    // Load-bearing, and the corpus proves it: `FramePass.compute` is called with 4 and 5 arguments
    // in frame-unified.test.ts, and in `f.compute(sim, 1, undefined, undefined, { … })` the THIRD
    // argument is the identifier `undefined` — which is options-bag-shaped. Matching `>= 3` would
    // therefore rewrite it to `f.compute(sim, { shader: 1, ...undefined }, undefined, { … })`, i.e.
    // corrupt a call to an entirely different API. A mutation pass caught this as an untested guard.
    const text = [
      `f.compute(sim, 1, 2, 3);`,
      `f.compute(sim, 1, undefined, undefined, { pendingPipelines: "skip" });`,
      `effect(gpu, SRC, { label: "a" }, extra);`,
    ].join("\n");
    const result = run(text);
    expect(result.text).toBe(text);
    expect(result.entries).toEqual([]);
  });

  it("ignores matching text inside strings, templates and comments", () => {
    const text = [
      `const doc = "effect(gpu, SRC, { blend: 'additive' })";`,
      "const tpl = `compute(gpu, SRC, { entry: 'main' })`;",
      `// effect(gpu, SRC, { label: "nope" })`,
      `/* compute(gpu, SRC, { label: "nope" }) */`,
    ].join("\n");
    const result = run(text);
    expect(result.text).toBe(text);
    expect(result.entries).toEqual([]);
  });
});

describe("transformSource — refusals (skip loudly, never mangle)", () => {
  it("refuses a site with a comment between the source and the options argument", () => {
    const text = `effect(gpu, SRC /* the shader */, { label: "fx" });`;
    const result = run(text);
    expect(result.text).toBe(text);
    expect(result.entries[0]?.classification).toBe("skipped-manual-review-comment-between-args");
  });

  it("refuses an options bag that already has a `shader` key rather than emitting a duplicate", () => {
    const text = `effect(gpu, SRC, { shader: OTHER, label: "fx" });`;
    const result = run(text);
    expect(result.text).toBe(text);
    expect(result.entries[0]?.classification).toBe("skipped-manual-review-shader-key-conflict");
  });

  it("refuses a single-object call that also passes a (silently ignored) third argument", () => {
    const text = `effect(gpu, { shader: SRC }, { label: "fx" });`;
    const result = run(text);
    expect(result.text).toBe(text);
    expect(result.entries[0]?.classification).toBe("skipped-already-single-object");
  });

  it("throws instead of splicing a file it could not parse cleanly", () => {
    // A partial token set under-migrates silently, which is worse than failing.
    expect(() => run(`effect(gpu, SRC, { label: "fx" });\nfunction (`))
      .toThrowError(/parse diagnostics — refusing to trust/u);
  });

  it("reports every site of a legacy-form test subject without changing a byte", () => {
    const text = `effect(gpu, SRC, { blend: "additive" });\ncompute(gpu, SRC, { entry: "main" });`;
    const result = run(text, "packages/vgpu-api/tests/unified-signature.test.ts", {
      legacySubjectReason: "the positional form is the subject under test",
    });
    expect(result.text).toBe(text);
    expect(result.entries.map((e) => e.classification))
      .toEqual(["excluded-test-subject", "excluded-test-subject"]);
  });
});

describe("transformSource — options bags behind a type assertion", () => {
  it("merges into the asserted object literal and keeps the assertion outside", () => {
    // scene.test.ts asserts `effect()` rejects `geometry`; the `as never` silences the (deliberate)
    // type error and must stay applied to the whole bag, or the test stops compiling.
    expect(out(`effect(gpu, SRC, { geometry: geo } as never);`))
      .toBe(`effect(gpu, { shader: SRC, geometry: geo } as never);`);
    expect(run(`effect(gpu, SRC, { geometry: geo } as never);`).entries[0]?.classification)
      .toBe("auto-inline-opts-asserted");
  });

  it("merges into a parenthesized object literal", () => {
    expect(out(`effect(gpu, SRC, ({ label: "fx" }));`))
      .toBe(`effect(gpu, ({ shader: SRC, label: "fx" }));`);
  });
});

describe("transformSource — report is 1:1 with the text it produces", () => {
  it("reports one entry per site, with before/after equal to the actual call slices", () => {
    const text = [
      `const a = effect(gpu, A, { label: "a" });`,
      `const b = compute(gpu, B, { entry: "main" });`,
    ].join("\n");
    const result = run(text);
    expect(result.entries).toHaveLength(2);
    expect(result.entries.map((e) => e.line)).toEqual([1, 2]);
    for (const entry of result.entries) {
      expect(text).toContain(entry.before);
      expect(result.text).toContain(entry.after);
      expect(entry.classification).toBe("auto-inline-opts");
    }
  });

  it("applying every reported before->after rewrite reproduces the file byte-for-byte", () => {
    // This is the harness's hard rule (dry-run report == real diff) at unit scale: the report is
    // not a second, hand-written description of the change.
    const text = [
      `const a = effect(gpu, A, { label: "a" });`,
      `const b = compute(gpu, shaders.b, opts);`,
      `const c = effect(gpu, C, {});`,
      `const d = effect(gpu, D);`,
    ].join("\n");
    const result = run(text);
    let rebuilt = text;
    for (const entry of result.entries) rebuilt = rebuilt.replace(entry.before, entry.after);
    expect(rebuilt).toBe(result.text);
    expect(result.entries.map((e) => e.classification))
      .toEqual(["auto-inline-opts", "auto-spread-opts", "auto-inline-opts-empty"]);
  });

  it("is idempotent: a second pass over its own output is a no-op", () => {
    const text = [
      `effect(gpu, A, { label: "a" });`,
      `compute(gpu, shaders.b, opts);`,
      `effect(gpu, C, {});`,
    ].join("\n");
    const once = run(text);
    const twice = run(once.text);
    expect(twice.text).toBe(once.text);
    expect(twice.entries).toEqual([]);
  });

  it("rewrites every site in a file with many, all at once", () => {
    const lines = Array.from({ length: 11 }, (_, i) => `const fx${i} = effect(gpu, SRC${i}, { label: "fx${i}" });`);
    const result = run(lines.join("\n"));
    expect(result.entries).toHaveLength(11);
    expect(result.text).toBe(
      Array.from({ length: 11 }, (_, i) => `const fx${i} = effect(gpu, { shader: SRC${i}, label: "fx${i}" });`).join("\n"),
    );
  });

  it("handles .tsx files (JSX must not be mis-tokenized as type assertions)", () => {
    const text = [
      `export function View() {`,
      `  const fx = effect(gpu, SRC, { label: "fx" });`,
      `  return <div data-a={a < b}>{fx.label}</div>;`,
      `}`,
    ].join("\n");
    const result = run(text, "corpus/view.tsx");
    expect(result.entries).toHaveLength(1);
    expect(result.text).toContain(`effect(gpu, { shader: SRC, label: "fx" })`);
    expect(result.text).toContain(`<div data-a={a < b}>`);
  });
});
