// Shared text-splicing helper for the 04/codemod-tooling harness (T04-15).
//
// Every codemod in this train computes a list of `{start, end, replacement}` edits as OFFSETS
// INTO THE ORIGINAL TEXT (never recomputed as it goes), then applies them back-to-front so that
// earlier offsets stay valid. This is the same mechanic `codemod.mjs` in T202 used implicitly
// (see `rewriteCalls`/`renameShadows` there); this module extracts it into a small, pure,
// independently-testable function.

/**
 * Applies a list of non-overlapping edits to `text`.
 *
 * @param {string} text - the original source text. Never mutated.
 * @param {{start: number, end: number, replacement: string}[]} edits - offsets into `text`.
 *   `start`/`end` are absolute offsets against the ORIGINAL `text`, not against any
 *   intermediate/partial result. `replacement` must be a string.
 * @returns {string} the text with all edits applied.
 * @throws if `replacement` is missing/not a string, if any edit's range is invalid, if two edits
 *   have the exact same range (ambiguous — which replacement wins?), or if any two edits
 *   overlap — fail fast, never silently corrupt the output by applying an inconsistent set of
 *   edits or writing the literal text "undefined".
 */
export function applyEdits(text, edits) {
  if (edits.length === 0) return text;

  // Sort by `start` primarily, but give ties (same `start`) a TOTAL, deterministic order by
  // `end` — a plain `sort((a, b) => a.start - b.start)` only promises to preserve whatever order
  // the caller happened to pass same-start edits in (stable sort, not a total order over the
  // edits themselves). That made the result of applying the exact same edit *set* depend on
  // array order — e.g. a zero-length insertion and a same-start replacement would throw the
  // "overlapping edits" error in one order and apply cleanly in the other. Insertion-heavy
  // codemods (T04-19's prepare-insertion) hit same-start edits routinely.
  const sorted = [...edits].sort((a, b) => a.start - b.start || a.end - b.end);

  for (const edit of sorted) {
    if (edit == null || typeof edit.replacement !== "string") {
      throw new Error(
        `splice: edit.replacement must be a string, got ${JSON.stringify(edit)} — a codemod ` +
          `whose replacement is built from an optional capture group/AST node must not let it ` +
          `reach applyEdits() as undefined (it would otherwise be written into the file as the ` +
          `literal text "undefined").`,
      );
    }
    if (
      !Number.isInteger(edit.start)
      || !Number.isInteger(edit.end)
      || edit.start < 0
      || edit.end < edit.start
      || edit.end > text.length
    ) {
      throw new Error(
        `splice: invalid edit range [${edit.start}, ${edit.end}) for text of length ${text.length}`,
      );
    }
  }

  for (let i = 1; i < sorted.length; i++) {
    const prev = sorted[i - 1];
    const cur = sorted[i];
    if (cur.start === prev.start && cur.end === prev.end) {
      throw new Error(
        `splice: duplicate edit range [${cur.start}, ${cur.end}) — ambiguous result order`,
      );
    }
    // Adjacent edits (prev.end === cur.start) are fine — they do not overlap, they abut.
    if (cur.start < prev.end) {
      throw new Error(
        `splice: overlapping edits [${prev.start}, ${prev.end}) and [${cur.start}, ${cur.end})`,
      );
    }
  }

  // Apply back-to-front so offsets computed against the original text stay valid throughout.
  let out = text;
  for (let i = sorted.length - 1; i >= 0; i--) {
    const edit = sorted[i];
    out = out.slice(0, edit.start) + edit.replacement + out.slice(edit.end);
  }
  return out;
}
