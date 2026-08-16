import { expect, test } from "vitest";
import { isRealTokenStart, parseDiagnosticsCount, tokenStarts } from "./token-scan.mjs";

test("tokenStarts marks a real identifier reference", () => {
  const text = "const gpu = 1;\ngpu.effect(x);\n";
  const offset = text.indexOf("gpu.effect") + "gpu.".length; // start of `effect`
  expect(tokenStarts(text, "file.ts").has(offset)).toBe(true);
});

test("tokenStarts does NOT mark an occurrence inside a template string", () => {
  const text = "const msg = `call gpu.effect(x) please`;\n";
  const offset = text.indexOf("gpu.effect");
  expect(tokenStarts(text, "file.ts").has(offset)).toBe(false);
});

test("tokenStarts does NOT mark an occurrence inside a block comment", () => {
  const text = "/* gpu.effect(x) is the legacy 2-arg form */\nconst y = 1;\n";
  const offset = text.indexOf("gpu.effect");
  expect(tokenStarts(text, "file.ts").has(offset)).toBe(false);
});

test("tokenStarts does NOT mark an occurrence inside a line comment", () => {
  const text = "// gpu.effect(x) legacy\nconst y = 1;\n";
  const offset = text.indexOf("gpu.effect");
  expect(tokenStarts(text, "file.ts").has(offset)).toBe(false);
});

test("tokenStarts does NOT mark an occurrence inside a string literal", () => {
  const text = 'const s = "gpu.effect(x)";\n';
  const offset = text.indexOf("gpu.effect");
  expect(tokenStarts(text, "file.ts").has(offset)).toBe(false);
});

test("tokenStarts handles a substitution inside a template literal as real tokens", () => {
  const text = "const s = `value is ${gpu.effect(x)}`;\n";
  const offset = text.indexOf("gpu.effect");
  expect(tokenStarts(text, "file.ts").has(offset)).toBe(true);
});

test("tokenStarts parses .tsx with JSX without throwing", () => {
  const text = "const el = <div>{gpu.effect(x)}</div>;\n";
  expect(() => tokenStarts(text, "file.tsx")).not.toThrow();
  const offset = text.indexOf("gpu.effect");
  expect(tokenStarts(text, "file.tsx").has(offset)).toBe(true);
});

test("tokenStarts on an empty file returns an empty set", () => {
  expect(tokenStarts("", "file.ts").size).toBe(0);
});

test("isRealTokenStart is a thin convenience wrapper", () => {
  const text = "gpu.effect(x);\n";
  expect(isRealTokenStart(text, "file.ts", 0)).toBe(true);
  // Offset 1 lands mid-identifier (inside `gpu`), not at a token start.
  expect(isRealTokenStart(text, "file.ts", 1)).toBe(false);
});

// --- mutation-kill regression tests (adversarial QA pre-push, T04-15) -----------------------
// Each of these pins a behaviour that a plausible one-line mistake in token-scan.mjs would break
// silently (the committed suite above did not catch any of them before this pass).

test("tokenStarts marks `this` keyword usages, not just identifiers", () => {
  const text = "class C { m() { return this.value; } }\n";
  const offset = text.indexOf("this");
  expect(tokenStarts(text, "file.ts").has(offset)).toBe(true);
});

test("tokenStarts uses the token's exact start (getStart), not its leading-trivia start (pos)", () => {
  // A block comment sits between the previous token and `gpu`, so `gpu`'s AST node.pos (which
  // includes leading trivia) points at the comment, while node.getStart() points at the `g`.
  const text = "const a = 1;/* c */gpu.effect(x);\n";
  const realStart = text.indexOf("gpu");
  const triviaStart = text.indexOf("/* c */");
  const starts = tokenStarts(text, "file.ts");
  expect(starts.has(realStart)).toBe(true);
  // If token-scan used `node.pos` instead of `node.getStart(source)`, this offset (the comment's
  // start, not a real token) would be marked instead — shifting every offset after a comment.
  expect(starts.has(triviaStart)).toBe(false);
});

test("tokenStarts does NOT mark a string literal's own start as a token (only its interior text is prose)", () => {
  const text = 'const s = "gpu.effect(x)";\n';
  const stringStart = text.indexOf('"');
  expect(tokenStarts(text, "file.ts").has(stringStart)).toBe(false);
});

test("a .ts file is parsed with the TS ScriptKind: `<Foo>bar` is an angle-bracket type assertion, not JSX", () => {
  // Under TS ScriptKind, `<Foo>bar` is `(bar as Foo)` — both `Foo` and `bar` are identifiers.
  // Under TSX ScriptKind, the same text is an (invalid, error-recovered) JSX open tag, and `bar`
  // is JSX text, not an identifier — so this pins the file getting the TS (not TSX) ScriptKind.
  const text = "const v = <Foo>bar;\n";
  const starts = tokenStarts(text, "file.ts");
  expect(starts.has(text.indexOf("Foo"))).toBe(true);
  expect(starts.has(text.lastIndexOf("bar"))).toBe(true);
});

test("a .tsx file is parsed with the TSX ScriptKind: a JSX closing tag name is a real identifier", () => {
  // Under TS ScriptKind (wrong for .tsx), the closing `</div>` in this snippet fails to parse as
  // JSX and its `div` is not recovered as an identifier at all; under TSX it is.
  const text = "const el = <div>{gpu.effect(x)}</div>;\n";
  const starts = tokenStarts(text, "file.tsx");
  const closingDivOffset = text.lastIndexOf("div");
  expect(starts.has(closingDivOffset)).toBe(true);
});

// QA re-validation (T04-15) pinned this: the scanner's identifier tables are language-version
// dependent, so an old `ts.ScriptTarget` (e.g. ES5) drops or shifts offsets for identifiers whose
// leading character is only a valid ID_Start from ES2015+ (astral-plane math letters, some
// deprecated-script letters). A shifted offset is the dangerous case — a codemod would splice at
// the wrong position instead of merely skipping the site.
test("identifiers using ES2015+ ID_Start characters are still real token starts", () => {
  const src = "const \u{1D4CD} = effect(gpu, s, {}); \u{1D4CD}.run();";
  const starts = tokenStarts(src, "astral.ts");
  expect(starts.has(6)).toBe(true);
  expect(starts.has(src.lastIndexOf("\u{1D4CD}"))).toBe(true);
  expect(starts.has(src.indexOf("effect"))).toBe(true);
});

test("parseDiagnosticsCount is 0 for well-formed code and >0 for content that does not belong (JSX inside a .ts file)", () => {
  const good = "const a = gpu.effect(x);\n";
  expect(parseDiagnosticsCount(good, "file.ts")).toBe(0);
  const jsxInDotTs = "const el = <div>{gpu.effect(x)}</div>;\n";
  expect(parseDiagnosticsCount(jsxInDotTs, "file.ts")).toBeGreaterThan(0);
  // The same text, correctly labelled .tsx, parses clean — this is the escape hatch a codemod
  // should use before trusting tokenStarts() on a file whose extension it isn't sure about.
  expect(parseDiagnosticsCount(jsxInDotTs, "file.tsx")).toBe(0);
});
