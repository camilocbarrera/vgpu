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

// --- adversarial QA pre-push additions (T04-15) ----------------------------------------------
// A codemod's `replacement` is frequently built from an optional regex capture group / AST node
// that is absent on some call shape — these pin that such a bug throws loudly instead of writing
// the literal text "undefined"/a coerced number into the file.

test('throws instead of writing the literal text "undefined" for a missing replacement', () => {
  expect(() => applyEdits("abcdef", [{ start: 0, end: 3, replacement: undefined }])).toThrow(
    /replacement/,
  );
});

test("throws instead of silently coercing a non-string replacement (e.g. a number)", () => {
  expect(() => applyEdits("abcdef", [{ start: 0, end: 3, replacement: 42 }])).toThrow(
    /replacement/,
  );
});

test("throws on a null edit entry instead of crashing with an opaque TypeError", () => {
  expect(() => applyEdits("abc", [null])).toThrow(/replacement/);
});

// A total, deterministic sort order (start, then end) matters most for insertion-heavy codemods
// (T04-19's prepare-insertion): several edits routinely share the same `start`.

test("same-start edits (a zero-length insertion + a same-start replacement) apply deterministically, regardless of input order", () => {
  const text = "0123456789";
  const insert = { start: 5, end: 5, replacement: "INS" };
  const replace = { start: 5, end: 8, replacement: "REPL" };
  const expected = "01234INSREPL89";
  expect(applyEdits(text, [insert, replace])).toBe(expected);
  expect(applyEdits(text, [replace, insert])).toBe(expected);
});

test("throws on duplicate edit ranges instead of silently picking whichever happened to sort first", () => {
  const edits = [
    { start: 1, end: 1, replacement: "A" },
    { start: 1, end: 1, replacement: "B" },
  ];
  expect(() => applyEdits("xy", edits)).toThrow(/duplicate/);
});

// QA re-validation (T04-15) pinned this: a same-start tie-break by `end` is NOT the same total
// order as sorting purely by `end` — they only coincide when starts are also tied. A zero-length
// insertion that abuts the END of an unrelated, non-overlapping replacement (`[3,5)` followed by
// `[5,5)`) has DIFFERENT starts but the SAME end (5), so a sort-by-end-only implementation puts
// them in input order instead of start order, and the overlap guard then misreads them as
// overlapping. 2274/20000 randomized valid edit sets diverge under that bug — this pins the one
// minimal repro.
test("abutting insertion at the END offset of a replacement applies in either input order", () => {
  const text = "0123456789";
  const replace = { start: 3, end: 5, replacement: "REPL" };
  const insert = { start: 5, end: 5, replacement: "INS" };
  const expected = "012REPLINS56789";
  expect(applyEdits(text, [replace, insert])).toBe(expected);
  expect(applyEdits(text, [insert, replace])).toBe(expected);
});
