// Unit + contract suite for T04-17's codemod (`ownership-binding-scoped.mjs`).
//
// The codemod needs a `ts.Program` and a real `TypeChecker`, so the unit under test is
// `planProgram()` driven over an IN-MEMORY program built here — not a pure string transform like
// T04-16's. That is deliberate: the classification these tests are about (is this receiver a `Draw`
// or a `SharedUniforms`? is this value a texture or a vec2?) only exists once types are resolved, so a
// fixture-string test would be testing a different function than the one that ships.
//
// The WGSL oracle is NOT stubbed: `reflectSource()` is the real one from `@vgpu/wgsl`, so
// "`{ time: 1 }` means binding `params`, not binding `time`" is proved against the same reflection the
// runtime uses.
import { describe, expect, it, beforeEach } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";
import ts from "typescript";
import { reflectSource } from "@vgpu/wgsl/reflect-source";
import {
  planProgram, classifyValueType, classifyReceiver, hasCommentIn, isPureReference, reindentValue,
} from "./ownership-binding-scoped.mjs";
import { clearReflectionCache } from "./lib/wgsl-oracle.mjs";

const REPO_ROOT = path.resolve(import.meta.dirname, "../..");

/**
 * The ambient declarations every fixture gets: structural stand-ins for `Draw`/`Effect`/`Compute`
 * (both `set` and `bind`), `SharedUniforms` (only `set`), and the resource types. The codemod
 * classifies STRUCTURALLY, so these are faithful stand-ins rather than a mock of a different shape —
 * and using them keeps the suite from needing the built `vgpu` dist on disk.
 */
const AMBIENT = `
export interface Target { readonly resourceIdentity: string; readonly color: Texture; readonly texelSize: readonly [number, number]; readonly size: readonly [number, number]; onDestroy(cb: () => void): () => void }
export interface Texture { readonly resourceIdentity: string; createView(): unknown; readonly format: string }
export interface GPUSampler { readonly __sampler: unique symbol }
export interface StorageBuffer { readonly gpu: unknown; readonly size: number; read(): Promise<ArrayBuffer>; write(d: unknown): void }
export interface Effect { set(binding: string, value: unknown): this; set(values: Record<string, unknown>): this; bind(binding: string, resource: unknown): this }
export interface Draw { set(binding: string, value: unknown): this; set(values: Record<string, unknown>): this; bind(binding: string, resource: unknown): this }
export interface Compute { set(binding: string, value: unknown): this; set(values: Record<string, unknown>): this; bind(binding: string, resource: unknown): this; dispatch(x: number): void }
export interface SharedUniforms<T extends Record<string, unknown> = Record<string, unknown>> { set(values: Partial<T>): void }
export interface Gpu { readonly __gpu: unique symbol }
export declare function effect(gpu: Gpu, opts: Record<string, unknown>): Effect;
export declare function draw(gpu: Gpu, opts: Record<string, unknown>): Draw;
export declare function compute(gpu: Gpu, opts: Record<string, unknown>): Compute;
export declare function target(gpu: Gpu, opts: Record<string, unknown>): Target;
export declare function texture(gpu: Gpu, opts: Record<string, unknown>): Texture;
export declare function sampler(gpu: Gpu, opts?: Record<string, unknown>): GPUSampler;
export declare function storage(gpu: Gpu, n: number): StorageBuffer;
export declare function uniforms<T extends Record<string, unknown>>(gpu: Gpu, values: T): SharedUniforms<T>;
export declare const gpu: Gpu;
`;

/**
 * The WGSL fixtures are collapsed to a SINGLE LINE. WGSL is whitespace-insensitive so the reflection is
 * unchanged, but it keeps `const SRC = \`...\`;` one line long — otherwise every fixture's line numbers
 * shift by the height of its shader and the report lookups below silently miss.
 */
const oneLine = (wgsl: string) => wgsl.replace(/\s*\n\s*/gu, " ").trim();

/** A WGSL fixture with one struct-uniform binding (`params` with members `time`/`speed`). */
const WGSL_PARAMS = oneLine(`
struct Params { time: f32, speed: f32 }
@group(0) @binding(0) var<uniform> params: Params;
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
  return vec4f(uv, params.time * params.speed, 1);
}
`);

/** Texture + sampler + struct uniform: the shape that exercises the resource path. */
const WGSL_POST = oneLine(`
struct PostParams { texel: vec2f }
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> params: PostParams;
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
  return textureSampleLevel(src, samp, uv / params.texel, 0.0);
}
`);

/** Two struct bindings whose member names are DISJOINT — one bag can address both. */
const WGSL_TWO_BLOCKS = oneLine(`
struct Camera { viewProjection: mat4x4f }
struct Light { direction: vec3f, intensity: f32 }
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> light: Light;
@fragment fn main() -> @location(0) vec4f {
  return camera.viewProjection[0] * light.intensity + vec4f(light.direction, 1);
}
`);

/** Two struct bindings that SHARE a member name — `resolveKey` must call that ambiguous. */
const WGSL_COLLIDING_MEMBERS = oneLine(`
struct A { size: vec2f }
struct B { size: vec2f }
@group(0) @binding(0) var<uniform> a: A;
@group(0) @binding(1) var<uniform> b: B;
@fragment fn main() -> @location(0) vec4f { return vec4f(a.size + b.size, 0, 1); }
`);

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

/**
 * Builds an in-memory program over `files` (repo-relative-looking paths) plus the `api.ts` ambient
 * module, and runs the codemod's planner over it. Returns the report entries and the rewritten text
 * per file.
 */
async function run(files: Record<string, string>) {
  const all: Record<string, string> = { "api.ts": AMBIENT, ...files };
  const host = ts.createCompilerHost(COMPILER_OPTIONS, true);
  const originalGetSourceFile = host.getSourceFile.bind(host);
  const originalReadFile = host.readFile.bind(host);
  const originalFileExists = host.fileExists.bind(host);
  host.getSourceFile = (fileName, languageVersion, onError, shouldCreate) => {
    const key = fileName.replace(`${REPO_ROOT}/`, "");
    if (all[key] !== undefined) return ts.createSourceFile(fileName, all[key], languageVersion, true, ts.ScriptKind.TS);
    return originalGetSourceFile(fileName, languageVersion, onError, shouldCreate);
  };
  host.readFile = (fileName) => {
    const key = fileName.replace(`${REPO_ROOT}/`, "");
    return all[key] !== undefined ? all[key] : originalReadFile(fileName);
  };
  host.fileExists = (fileName) => {
    const key = fileName.replace(`${REPO_ROOT}/`, "");
    return all[key] !== undefined ? true : originalFileExists(fileName);
  };

  const corpus = Object.keys(all);
  const program = ts.createProgram(corpus.map((f) => path.join(REPO_ROOT, f)), COMPILER_OPTIONS, host);
  const checker = program.getTypeChecker();
  const sourceFileFor = (rel: string) => program.getSourceFile(path.join(REPO_ROOT, rel));
  const { entries, fileTexts } = await planProgram({
    checker, corpus, sourceFileFor, repoRoot: REPO_ROOT, reflectSource,
    // No fixture uses a `.wgsl` import, so a `resolveShader` that is never called is the honest stub.
    resolveShader: () => { throw new Error("resolveShader must not be called by these fixtures"); },
  });
  return { entries, fileTexts, program, checker, sourceFileFor };
}

const bucketOf = (entries: any[], line: number) => entries.find((e) => e.line === line)?.classification;

beforeEach(() => clearReflectionCache());

// ---------------------------------------------------------------------------------------------
describe("bytes bindings", () => {
  it("rewrites a whole-binding key to the binding-scoped form", async () => {
    const { entries, fileTexts } = await run({
      "a.ts": `import { effect, gpu } from "./api";\nconst SRC = \`${WGSL_PARAMS}\`;\nconst fx = effect(gpu, { shader: SRC });\nfx.set({ params: { time: 1, speed: 2 } });\n`,
    });
    expect(bucketOf(entries, 4)).toBe("auto-bytes");
    expect(fileTexts.get("a.ts")).toContain(`fx.set("params", { time: 1, speed: 2 });`);
  });

  it("groups STRUCT MEMBER keys under their owning binding — the reflection oracle's whole point", async () => {
    const { entries, fileTexts } = await run({
      "a.ts": `import { effect, gpu } from "./api";\nconst SRC = \`${WGSL_PARAMS}\`;\nconst fx = effect(gpu, { shader: SRC });\nfx.set({ time: 1, speed: 2 });\n`,
    });
    expect(bucketOf(entries, 4)).toBe("auto-bytes");
    // NOT `.set("time", 1).set("speed", 2)`: `time`/`speed` are members, and `setScoped()` rejects a
    // member name. One call per BINDING, so one buffer write.
    expect(fileTexts.get("a.ts")).toContain(`fx.set("params", { time: 1, speed: 2 });`);
  });

  it("preserves shorthand properties as shorthand", async () => {
    const { fileTexts } = await run({
      "a.ts": `import { effect, gpu } from "./api";\nconst SRC = \`${WGSL_PARAMS}\`;\nconst fx = effect(gpu, { shader: SRC });\nconst time = 1;\nfx.set({ time });\n`,
    });
    expect(fileTexts.get("a.ts")).toContain(`fx.set("params", { time });`);
  });

  it("emits one call per binding when members of two bindings share a bag", async () => {
    const { entries, fileTexts } = await run({
      "a.ts": `import { effect, gpu } from "./api";\nconst SRC = \`${WGSL_TWO_BLOCKS}\`;\nconst fx = effect(gpu, { shader: SRC });\nfx.set({ viewProjection: [1], intensity: 2 });\n`,
    });
    expect(bucketOf(entries, 4)).toBe("auto-bytes");
    // Two bindings -> two calls, chained because the result still fits on one line. One call per
    // BINDING (not per key) is what makes each binding exactly one buffer write.
    expect(fileTexts.get("a.ts"))
      .toContain(`fx.set("camera", { viewProjection: [1] }).set("light", { intensity: 2 });`);
  });

  it("splits into separate statements (not a chain) for a standalone statement with a pure receiver", async () => {
    const long = `fx.set({ viewProjection: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], direction: [1, 2, 3], intensity: 4 });`;
    const { fileTexts } = await run({
      "a.ts": `import { effect, gpu } from "./api";\nconst SRC = \`${WGSL_TWO_BLOCKS}\`;\nconst fx = effect(gpu, { shader: SRC });\n${long}\n`,
    });
    const out = fileTexts.get("a.ts")!;
    expect(out).toContain(`fx.set("camera",`);
    expect(out).toContain(`fx.set("light",`);
    expect(out.split("\n").filter((l) => l.includes("fx.set("))).toHaveLength(2);
  });

  it("chains instead of splitting when the call's value is used", async () => {
    const { fileTexts } = await run({
      "a.ts": `import { effect, gpu } from "./api";\nconst SRC = \`${WGSL_TWO_BLOCKS}\`;\nconst fx = effect(gpu, { shader: SRC });\nexport const out = fx.set({ viewProjection: [1], intensity: 2 });\n`,
    });
    expect(fileTexts.get("a.ts")).toContain(`fx.set("camera", { viewProjection: [1] }).set("light", { intensity: 2 })`);
  });

  it("re-indents a multi-line value that lost a nesting level", async () => {
    const { fileTexts } = await run({
      "a.ts": `import { effect, gpu } from "./api";\nconst SRC = \`${WGSL_PARAMS}\`;\nconst fx = effect(gpu, { shader: SRC });\nfunction go() {\n  fx.set({\n    params: {\n      time: 1,\n      speed: 2,\n    },\n  });\n}\n`,
    });
    expect(fileTexts.get("a.ts")).toContain(`  fx.set("params", {\n    time: 1,\n    speed: 2,\n  });`);
  });
});

// ---------------------------------------------------------------------------------------------
describe("resources move to the constructor", () => {
  it("moves a resource to `bindings` and deletes a statement that has nothing left", async () => {
    const { entries, fileTexts } = await run({
      "a.ts": `import { effect, target, gpu } from "./api";\nconst SRC = \`${WGSL_POST}\`;\nconst scene = target(gpu, { size: [8, 8] });\nconst fx = effect(gpu, { shader: SRC });\nfx.set({ src: scene.color });\n`,
    });
    expect(bucketOf(entries, 5)).toBe("auto-resource-to-constructor");
    const out = fileTexts.get("a.ts")!;
    expect(out).toContain(`effect(gpu, { shader: SRC, bindings: { src: scene.color } })`);
    expect(out).not.toContain("fx.set(");
  });

  it("keeps the bytes half of a mixed bag as a `.set()` call", async () => {
    const { entries, fileTexts } = await run({
      "a.ts": `import { effect, target, gpu } from "./api";\nconst SRC = \`${WGSL_POST}\`;\nconst scene = target(gpu, { size: [8, 8] });\nconst fx = effect(gpu, { shader: SRC });\nfx.set({ src: scene.color, texel: scene.texelSize });\n`,
    });
    expect(bucketOf(entries, 5)).toBe("auto-resource-to-constructor");
    const out = fileTexts.get("a.ts")!;
    expect(out).toContain(`bindings: { src: scene.color }`);
    expect(out).toContain(`fx.set("params", { texel: scene.texelSize });`);
  });

  it("appends `bindings` AFTER the last option so `shader` stays the first key", async () => {
    const { fileTexts } = await run({
      "a.ts": `import { effect, target, gpu } from "./api";\nconst SRC = \`${WGSL_POST}\`;\nconst scene = target(gpu, { size: [8, 8] });\nconst fx = effect(gpu, {\n  shader: SRC,\n  label: "fx",\n});\nfx.set({ src: scene.color });\n`,
    });
    const out = fileTexts.get("a.ts")!;
    expect(out.indexOf("shader: SRC")).toBeLessThan(out.indexOf("bindings:"));
    expect(out).toContain(`  label: "fx",\n  bindings: { src: scene.color },\n`);
  });

  it("extends a `bindings` bag the constructor already has", async () => {
    const { fileTexts } = await run({
      "a.ts": `import { effect, target, sampler, gpu } from "./api";\nconst SRC = \`${WGSL_POST}\`;\nconst scene = target(gpu, { size: [8, 8] });\nconst samp = sampler(gpu);\nconst fx = effect(gpu, { shader: SRC, bindings: { samp } });\nfx.set({ src: scene.color });\n`,
    });
    expect(fileTexts.get("a.ts")).toContain(`bindings: { src: scene.color, samp }`);
  });

  it("uses `.bind()` when the binding is ALREADY declared external in the constructor", async () => {
    // This is the R1 rule from #331: `.bind()` needs a binding declared external at construction, and
    // set-core's `bind()` throws VGPU-R1-OWNERSHIP-FLIP otherwise.
    const { entries, fileTexts } = await run({
      "a.ts": `import { effect, target, gpu } from "./api";\nconst SRC = \`${WGSL_POST}\`;\nconst a = target(gpu, { size: [8, 8] });\nconst b = target(gpu, { size: [8, 8] });\nconst fx = effect(gpu, { shader: SRC, bindings: { src: a.color } });\nfx.set({ src: b.color });\n`,
    });
    expect(bucketOf(entries, 6)).toBe("auto-resource-rebind");
    expect(fileTexts.get("a.ts")).toContain(`fx.bind("src", b.color);`);
  });

  it("refuses to move a resource declared AFTER the constructor", async () => {
    const { entries, fileTexts } = await run({
      "a.ts": `import { effect, target, gpu } from "./api";\nconst SRC = \`${WGSL_POST}\`;\nconst fx = effect(gpu, { shader: SRC });\nconst scene = target(gpu, { size: [8, 8] });\nfx.set({ src: scene.color });\n`,
    });
    expect(bucketOf(entries, 5)).toBe("ambiguous-resource-value-is-declared-after-the-constructor");
    expect(fileTexts.has("a.ts")).toBe(false);
  });

  it("refuses to move a resource out of scope at the constructor", async () => {
    const { entries } = await run({
      "a.ts": `import { effect, target, gpu } from "./api";\nconst SRC = \`${WGSL_POST}\`;\nconst fx = effect(gpu, { shader: SRC });\nexport function resize() {\n  const scene = target(gpu, { size: [8, 8] });\n  fx.set({ src: scene.color });\n}\n`,
    });
    // Declared after AND out of scope; either verdict is a refusal, and both are reported buckets.
    expect(bucketOf(entries, 6)).toMatch(/^ambiguous-resource-value-is-(declared-after-the-constructor|out-of-scope-at-the-constructor)$/u);
  });

  it("refuses to hoist a resource expression that calls something", async () => {
    const { entries } = await run({
      "a.ts": `import { effect, sampler, gpu } from "./api";\nconst SRC = \`${WGSL_POST}\`;\nconst fx = effect(gpu, { shader: SRC });\nfx.set({ samp: sampler(gpu, { minFilter: "linear" }) });\n`,
    });
    expect(bucketOf(entries, 4)).toBe("ambiguous-resource-value-is-not-hoistable");
  });

  it("reports a rebind SEQUENCE instead of guessing which site is the construction value", async () => {
    // Ping-pong: source order is not evaluation order, so no oracle can say which resource `src`
    // should be born with. The ticket's `ambiguous-cross-file` case, reported not guessed.
    const { entries, fileTexts } = await run({
      "a.ts": `import { effect, target, gpu } from "./api";\nconst SRC = \`${WGSL_POST}\`;\nconst a = target(gpu, { size: [8, 8] });\nconst b = target(gpu, { size: [8, 8] });\nconst fx = effect(gpu, { shader: SRC });\nfx.set({ src: a.color });\nfx.set({ src: b.color });\n`,
    });
    expect(bucketOf(entries, 6)).toBe("ambiguous-resource-rebound-at-several-sites");
    expect(bucketOf(entries, 7)).toBe("ambiguous-resource-rebound-at-several-sites");
    expect(fileTexts.has("a.ts")).toBe(false);
  });

  it("resolves a receiver reached through an interface member", async () => {
    const { entries, fileTexts } = await run({
      "a.ts": `import { effect, gpu, type Effect } from "./api";\nconst SRC = \`${WGSL_PARAMS}\`;\ninterface Fx { readonly one: Effect }\nfunction make(): Fx {\n  return { one: effect(gpu, { shader: SRC }) };\n}\nexport const fx: Fx = make();\nfx.one.set({ time: 1 });\n`,
    });
    expect(bucketOf(entries, 8)).toBe("auto-bytes");
    expect(fileTexts.get("a.ts")).toContain(`fx.one.set("params", { time: 1 });`);
  });

  it("reports an interface member built in several places rather than picking one", async () => {
    const { entries } = await run({
      "a.ts": `import { effect, gpu, type Effect } from "./api";\nconst SRC = \`${WGSL_PARAMS}\`;\ninterface Fx { readonly one: Effect }\nconst p: Fx = { one: effect(gpu, { shader: SRC }) };\nconst q: Fx = { one: effect(gpu, { shader: SRC }) };\nexport function go(f: Fx) { void p; void q; f.one.set({ time: 1 }); }\n`,
    });
    expect(bucketOf(entries, 6)).toBe("ambiguous-ctor-receiver-is-an-interface-member-built-in-several-places");
  });
});

// ---------------------------------------------------------------------------------------------
describe("exclusions decided by TYPE, never by name", () => {
  it("never touches a SharedUniforms receiver, even when it is called `fx`", async () => {
    const { entries, fileTexts } = await run({
      "a.ts": `import { uniforms, gpu } from "./api";\nconst fx = uniforms(gpu, { time: 0 });\nfx.set({ time: 1 });\n`,
    });
    expect(bucketOf(entries, 3)).toBe("excluded-uniform-1arg");
    expect(fileTexts.has("a.ts")).toBe(false);
  });

  it("never touches a foreign `.set()` (a Map), even next to real sites", async () => {
    const { entries } = await run({
      "a.ts": `const m = new Map<string, unknown>();\nm.set({ a: 1 } as never);\n`,
    });
    expect(bucketOf(entries, 2)).toBe("excluded-foreign-api");
  });

  it("migrates a receiver that is one of ours even when it is called `globals`", async () => {
    const { entries, fileTexts } = await run({
      "a.ts": `import { effect, gpu } from "./api";\nconst SRC = \`${WGSL_PARAMS}\`;\nconst globals = effect(gpu, { shader: SRC });\nglobals.set({ time: 1 });\n`,
    });
    expect(bucketOf(entries, 4)).toBe("auto-bytes");
    expect(fileTexts.get("a.ts")).toContain(`globals.set("params", { time: 1 });`);
  });
});

// ---------------------------------------------------------------------------------------------
describe("undecidable shapes are reported, never guessed", () => {
  it("reports a computed key", async () => {
    const { entries, fileTexts } = await run({
      "a.ts": `import { effect, gpu } from "./api";\nconst SRC = \`${WGSL_PARAMS}\`;\nconst fx = effect(gpu, { shader: SRC });\nconst k = "time";\nfx.set({ [k]: 1 });\n`,
    });
    expect(bucketOf(entries, 5)).toBe("ambiguous-computed-key");
    expect(fileTexts.has("a.ts")).toBe(false);
  });

  it("reports a spread in the bag", async () => {
    const { entries } = await run({
      "a.ts": `import { effect, gpu } from "./api";\nconst SRC = \`${WGSL_PARAMS}\`;\nconst BASE = { time: 0 };\nconst fx = effect(gpu, { shader: SRC });\nfx.set({ ...BASE, speed: 2 });\n`,
    });
    expect(bucketOf(entries, 5)).toBe("ambiguous-spread-in-bag");
  });

  it("reports a key the shader does not have", async () => {
    const { entries } = await run({
      "a.ts": `import { effect, gpu } from "./api";\nconst SRC = \`${WGSL_PARAMS}\`;\nconst fx = effect(gpu, { shader: SRC });\nfx.set({ nope: 1 });\n`,
    });
    expect(bucketOf(entries, 4)).toBe("ambiguous-key-absent-from-reflection");
  });

  it("reports a member name owned by two bindings, exactly as the runtime would", async () => {
    const { entries } = await run({
      "a.ts": `import { effect, gpu } from "./api";\nconst SRC = \`${WGSL_COLLIDING_MEMBERS}\`;\nconst fx = effect(gpu, { shader: SRC });\nfx.set({ size: [1, 2] });\n`,
    });
    expect(bucketOf(entries, 4)).toBe("ambiguous-member-owned-by-several-bindings");
  });

  it("reports a resource handed to a struct member", async () => {
    const { entries } = await run({
      "a.ts": `import { effect, target, gpu } from "./api";\nconst SRC = \`${WGSL_POST}\`;\nconst scene = target(gpu, { size: [8, 8] });\nconst fx = effect(gpu, { shader: SRC });\nfx.set({ texel: scene.color });\n`,
    });
    expect(bucketOf(entries, 5)).toBe("ambiguous-resource-assigned-to-a-struct-member");
  });

  it("reports one binding handed BOTH a resource and bytes", async () => {
    // `params` is addressed twice in one bag: once whole (a resource) and once through its member
    // `texel` (bytes). Migrating that would have to invent a merge order between a buffer write and an
    // identity swap on the same binding.
    const { entries, fileTexts } = await run({
      "a.ts": `import { effect, target, gpu } from "./api";\nconst SRC = \`${WGSL_POST}\`;\nconst scene = target(gpu, { size: [8, 8] });\nconst fx = effect(gpu, { shader: SRC });\nfx.set({ params: scene.color, texel: [1, 2] });\n`,
    });
    expect(bucketOf(entries, 5)).toBe("ambiguous-binding-receives-bytes-and-a-resource");
    expect(fileTexts.has("a.ts")).toBe(false);
  });

  it("reports a value whose type is a union of a resource AND bytes", async () => {
    // `Texture | number` is not decidable: picking "resource" would move a number into `bindings`,
    // picking "bytes" would write a texture as bytes. Both are wrong, so neither is chosen.
    const { entries, fileTexts } = await run({
      "a.ts": `import { effect, gpu, type Texture } from "./api";\nconst SRC = \`${WGSL_POST}\`;\ndeclare const either: Texture | number;\nconst fx = effect(gpu, { shader: SRC });\nfx.set({ src: either });\n`,
    });
    expect(bucketOf(entries, 5)).toBe("ambiguous-value-type-undecidable");
    expect(fileTexts.has("a.ts")).toBe(false);
  });

  it("leaves an empty bag alone", async () => {
    const { entries, fileTexts } = await run({
      "a.ts": `import { effect, gpu } from "./api";\nconst SRC = \`${WGSL_PARAMS}\`;\nconst fx = effect(gpu, { shader: SRC });\nfx.set({});\n`,
    });
    expect(bucketOf(entries, 4)).toBe("skipped-empty-bag");
    expect(fileTexts.has("a.ts")).toBe(false);
  });
});

// ---------------------------------------------------------------------------------------------
describe("comment handling (the T04-16 QA finding, fixed)", () => {
  it("refuses a bag that carries a real comment", async () => {
    const { entries, fileTexts } = await run({
      "a.ts": `import { effect, gpu } from "./api";\nconst SRC = \`${WGSL_PARAMS}\`;\nconst fx = effect(gpu, { shader: SRC });\nfx.set({ /* keep me */ time: 1 });\n`,
    });
    expect(bucketOf(entries, 4)).toBe("skipped-comment-in-bag");
    expect(fileTexts.has("a.ts")).toBe(false);
  });

  it("is NOT fooled by `//` inside a string value — the scanner sees tokens, not text", async () => {
    // T04-16's guard used `text.includes("//")` over a span that contained the argument itself, so a
    // URL in a value produced a false skip: fail-safe, but it silently lowered the migration rate.
    const { entries, fileTexts } = await run({
      "a.ts": `import { effect, gpu } from "./api";\nconst SRC = \`${WGSL_PARAMS}\`;\nconst fx = effect(gpu, { shader: SRC });\nconst u = "https://example.com";\nfx.set({ time: u.length });\n`,
    });
    expect(bucketOf(entries, 5)).toBe("auto-bytes");
    expect(fileTexts.get("a.ts")).toContain(`fx.set("params", { time: u.length });`);
  });

  it("hasCommentIn() sees a real comment and ignores a lookalike in a string", () => {
    const withComment = `f({ /* c */ a: 1 })`;
    expect(hasCommentIn(withComment, 1, withComment.length)).toBe(true);
    const withUrlInString = `f({ a: "http://x" })`;
    expect(hasCommentIn(withUrlInString, 1, withUrlInString.length)).toBe(false);
    const withLineComment = `f({ a: 1 }) // trailing`;
    expect(hasCommentIn(withLineComment, 1, 11)).toBe(false);
    expect(hasCommentIn(withLineComment, 1, withLineComment.length)).toBe(true);
  });
});

// ---------------------------------------------------------------------------------------------
describe("the dry-run contract", () => {
  it("reports an `after` that matches the text the run actually produces", async () => {
    const files = {
      "a.ts": `import { effect, target, gpu } from "./api";\nconst SRC = \`${WGSL_POST}\`;\nconst scene = target(gpu, { size: [8, 8] });\nconst fx = effect(gpu, { shader: SRC });\nfx.set({ src: scene.color, texel: scene.texelSize });\nfx.set({ texel: [1, 2] });\n`,
    };
    const { entries, fileTexts } = await run(files);
    const out = fileTexts.get("a.ts")!;
    for (const entry of entries.filter((e: any) => e.classification.startsWith("auto-"))) {
      // Every reported `after` must literally appear in the produced file, and the `before` must not.
      expect(out).toContain(entry.after.trimEnd());
      if (entry.before.trim() !== entry.after.trim()) expect(out).not.toContain(entry.before.trim());
    }
  });

  it("is idempotent: replanning the produced text migrates nothing more", async () => {
    const source = `import { effect, target, gpu } from "./api";\nconst SRC = \`${WGSL_POST}\`;\nconst scene = target(gpu, { size: [8, 8] });\nconst fx = effect(gpu, { shader: SRC });\nfx.set({ src: scene.color, texel: scene.texelSize });\n`;
    const first = await run({ "a.ts": source });
    const migrated = first.fileTexts.get("a.ts")!;
    clearReflectionCache();
    const second = await run({ "a.ts": migrated });
    expect(second.fileTexts.size).toBe(0);
    expect(second.entries.every((e: any) => !e.classification.startsWith("auto-"))).toBe(true);
  });

  it("reports every `.set({…})` site in the corpus, migrated or not", async () => {
    const { entries } = await run({
      "a.ts": `import { effect, uniforms, gpu } from "./api";\nconst SRC = \`${WGSL_PARAMS}\`;\nconst fx = effect(gpu, { shader: SRC });\nconst g = uniforms(gpu, { t: 0 });\nfx.set({ time: 1 });\ng.set({ t: 2 });\nfx.set({ nope: 1 });\n`,
    });
    expect(entries).toHaveLength(3);
    expect(new Set(entries.map((e: any) => e.classification)))
      .toEqual(new Set(["auto-bytes", "excluded-uniform-1arg", "ambiguous-key-absent-from-reflection"]));
  });

  it("widens `before`/`after` to the affected span when the rewrite deletes the statement", async () => {
    // INDENTED on purpose: the deletion edit starts before the call (it eats the line's indentation),
    // so a report span narrowed to the call would both mis-render this entry and hide the deletion.
    const { entries, fileTexts } = await run({
      "a.ts": `import { effect, target, gpu } from "./api";\nconst SRC = \`${WGSL_POST}\`;\nconst scene = target(gpu, { size: [8, 8] });\nexport function go() {\n  const fx = effect(gpu, { shader: SRC });\n  fx.set({ src: scene.color });\n  return fx;\n}\n`,
    });
    const entry = entries.find((e: any) => e.line === 6)!;
    expect(entry.after).toBe("");
    // The span starts at the line's indentation, not at `fx`.
    expect(entry.before).toBe("  fx.set({ src: scene.color });\n");
    expect(fileTexts.get("a.ts")).toBe(
      `import { effect, target, gpu } from "./api";\nconst SRC = \`${WGSL_POST}\`;\nconst scene = target(gpu, { size: [8, 8] });\nexport function go() {\n  const fx = effect(gpu, { shader: SRC, bindings: { src: scene.color } });\n  return fx;\n}\n`,
    );
  });
});

// ---------------------------------------------------------------------------------------------
describe("classifier units", () => {
  it("classifyValueType tells a resource from bytes", async () => {
    const { program, checker, sourceFileFor } = await run({
      "a.ts": `import { target, storage, sampler, gpu } from "./api";\nconst t = target(gpu, { size: [8, 8] });\nconst s = storage(gpu, 4);\nconst m = sampler(gpu);\nexport const values = [t, t.color, s, m, 1, [1, 2], { a: 1 }, "x", true];\n`,
    });
    void program;
    const sf = sourceFileFor("a.ts")!;
    const arr = (sf.statements.at(-1) as any).declarationList.declarations[0].initializer;
    const kinds = arr.elements.map((e: any) => classifyValueType(e, checker));
    expect(kinds).toEqual(["resource", "resource", "resource", "resource", "bytes", "bytes", "bytes", "bytes", "bytes"]);
  });

  it("classifyReceiver separates ours / shared-uniforms / foreign", async () => {
    const { checker, sourceFileFor } = await run({
      "a.ts": `import { effect, compute, uniforms, gpu } from "./api";\nconst SRC = \`${WGSL_PARAMS}\`;\nconst a = effect(gpu, { shader: SRC });\nconst b = compute(gpu, { shader: SRC });\nconst c = uniforms(gpu, { t: 0 });\nconst d = new Map<string, number>();\nexport const refs = [a, b, c, d];\n`,
    });
    const sf = sourceFileFor("a.ts")!;
    const arr = (sf.statements.at(-1) as any).declarationList.declarations[0].initializer;
    expect(arr.elements.map((e: any) => classifyReceiver(e, checker).kind))
      .toEqual(["ours", "ours", "shared-uniforms", "foreign"]);
  });

  it("isPureReference accepts references and rejects calls", async () => {
    const src = ts.createSourceFile("t.ts", `[a, a.b.c, a[0], a["k"], f(), a[i], this]`, ts.ScriptTarget.ESNext, true);
    const arr = (src.statements[0] as ts.ExpressionStatement).expression as ts.ArrayLiteralExpression;
    expect(arr.elements.map((e) => isPureReference(e))).toEqual([true, true, true, true, false, false, true]);
  });

  it("reindentValue shifts continuation lines by the value's own closing indent", () => {
    expect(reindentValue("{\n      a: 1,\n    }", "  ")).toBe("{\n    a: 1,\n  }");
    expect(reindentValue("{ a: 1 }", "  ")).toBe("{ a: 1 }");
    // Already at the statement's level: nothing to shift.
    expect(reindentValue("{\n  a: 1,\n}", "  ")).toBe("{\n  a: 1,\n}");
  });
});

// ---------------------------------------------------------------------------------------------
describe("the codemod is the only author of the migration", () => {
  it("exposes writeUnlessDryRun as the only fs writer in the codemod", () => {
    const source = readFileSync(path.join(REPO_ROOT, "scripts/codemods/ownership-binding-scoped.mjs"), "utf8");
    // `writeFileSync` may only be reached through the harness's single choke point.
    expect(source).not.toMatch(/\bwriteFileSync\s*\(/u);
    expect(source).toContain("writeUnlessDryRun");
  });
});
