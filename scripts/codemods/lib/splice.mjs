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
 *   intermediate/partial result.
 * @returns {string} the text with all edits applied.
 * @throws if any two edits overlap, or if an edit's range is invalid — fail fast, never silently
 *   corrupt the output by applying an inconsistent set of edits.
 */
export function applyEdits(text, edits) {
  if (edits.length === 0) return text;

  const sorted = [...edits].sort((a, b) => a.start - b.start);

  for (const edit of sorted) {
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
