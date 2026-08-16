import { expect, test } from "vitest";
import { applyEdits } from "./splice.mjs";

test("applies a single edit", () => {
  expect(applyEdits("hello world", [{ start: 0, end: 5, replacement: "goodbye" }])).toBe(
    "goodbye world",
  );
});

test("returns the original text unchanged when there are no edits", () => {
  expect(applyEdits("hello world", [])).toBe("hello world");
});

test("applies multiple non-overlapping edits regardless of input order", () => {
  const text = "aaa bbb ccc";
  const edits = [
    { start: 8, end: 11, replacement: "ZZZ" },
    { start: 0, end: 3, replacement: "XXX" },
  ];
  expect(applyEdits(text, edits)).toBe("XXX bbb ZZZ");
});

test("edits are resolved against ORIGINAL offsets, not shifted ones", () => {
  // Replacement lengths differ from the original span lengths on purpose: if the implementation
  // naively applied edits front-to-back against a mutating string, the second edit's offsets
  // would land in the wrong place once the first edit changes the string's length.
  const text = "0123456789";
  const edits = [
    { start: 0, end: 2, replacement: "AAAAAAAAAA" }, // grows
    { start: 8, end: 10, replacement: "Z" }, // shrinks
  ];
  expect(applyEdits(text, edits)).toBe("AAAAAAAAAA234567Z");
});

test("adjacent edits (end of one === start of next) are allowed", () => {
  const text = "abcdef";
  const edits = [
    { start: 0, end: 3, replacement: "XYZ" },
    { start: 3, end: 6, replacement: "123" },
  ];
  expect(applyEdits(text, edits)).toBe("XYZ123");
});

test("throws on overlapping edits instead of silently corrupting output", () => {
  const text = "abcdef";
  const edits = [
    { start: 0, end: 4, replacement: "X" },
    { start: 3, end: 6, replacement: "Y" },
  ];
  expect(() => applyEdits(text, edits)).toThrow(/overlap/);
});

test("throws on an edit whose end is before its start", () => {
  expect(() => applyEdits("abc", [{ start: 2, end: 1, replacement: "x" }])).toThrow();
});

test("throws on an edit range beyond the text length", () => {
  expect(() => applyEdits("abc", [{ start: 1, end: 10, replacement: "x" }])).toThrow();
});

test("handles an empty file with an empty edit list", () => {
  expect(applyEdits("", [])).toBe("");
});

test("handles a single-byte file replaced entirely", () => {
  expect(applyEdits("x", [{ start: 0, end: 1, replacement: "yz" }])).toBe("yz");
});

test("handles inserting at the same offset for start and end (pure insertion) on an empty file", () => {
  expect(applyEdits("", [{ start: 0, end: 0, replacement: "abc" }])).toBe("abc");
});

test("does not mutate the input string reference-wise (pure function)", () => {
  const text = "hello";
  const copy = `${text}`;
  applyEdits(text, [{ start: 0, end: 1, replacement: "H" }]);
  expect(text).toBe(copy);
});
