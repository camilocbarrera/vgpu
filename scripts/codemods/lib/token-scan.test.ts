import { expect, test } from "vitest";
import { isRealTokenStart, tokenStarts } from "./token-scan.mjs";

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
