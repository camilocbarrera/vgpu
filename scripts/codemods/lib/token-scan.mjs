// Shared AST helper for the 04/codemod-tooling harness (T04-15).
//
// This does NOT do a full AST rewrite. It uses the TS Compiler API only to find the offsets where
// a *real* identifier/`this` token starts in the source text, so that a regex-driven codemod can
// discard matches that merely happen to fall inside a string literal, a template chunk, or a
// comment (where the same characters can appear without being live code). This is the same
// pattern proven in the T202 train (`~/codemod-t202/codemod.mjs`, `tokenStarts()`), extracted here
// so every downstream codemod (T04-16..19) shares one implementation instead of re-deriving it.
import ts from "typescript";

/**
 * Picks the `ts.ScriptKind` for a file based on its extension. `.tsx`/`.jsx` need JSX parsing
 * enabled or the scanner mis-tokenizes `<` as a type assertion / generic instead of JSX.
 */
function scriptKindFor(fileName) {
  if (fileName.endsWith(".tsx")) return ts.ScriptKind.TSX;
  if (fileName.endsWith(".jsx")) return ts.ScriptKind.JSX;
  return ts.ScriptKind.TS;
}

/**
 * Returns the `Set` of offsets (into `text`) where a real identifier or `this` token starts.
 *
 * An offset that is NOT in this set but that a naive regex matched anyway means the regex matched
 * text that lives inside a string/template/comment/etc — the caller must skip it.
 */
export function tokenStarts(text, fileName = "source.ts") {
  const source = ts.createSourceFile(fileName, text, ts.ScriptTarget.ESNext, true, scriptKindFor(fileName));
  const starts = new Set();
  const visit = (node) => {
    if (ts.isIdentifier(node) || node.kind === ts.SyntaxKind.ThisKeyword) {
      starts.add(node.getStart(source));
    }
    ts.forEachChild(node, visit);
  };
  visit(source);
  return starts;
}

/** Convenience wrapper: is `offset` the start of a real token in `text`? */
export function isRealTokenStart(text, fileName, offset) {
  return tokenStarts(text, fileName).has(offset);
}
