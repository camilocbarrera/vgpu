// Shared `--dry-run` reporting for the 04/codemod-tooling harness (T04-15).
//
// The hard rule for every codemod in this train: it MUST run with `--dry-run` first, print a JSON
// report of every site it would touch, and that report gets attached to the PR. Adversarial QA
// then diffs "expected report" against "actual diff" before the codemod is ever run for real. This
// module is the one place that defines the report shape and the dry-run/apply gate so all four
// downstream codemods (T04-16..19) behave identically on this axis.
import { writeFileSync } from "node:fs";

/** `true` if `--dry-run` is present in the given argv (defaults to `process.argv.slice(2)`). */
export function isDryRun(argv = process.argv.slice(2)) {
  return argv.includes("--dry-run");
}

/** Non-flag arguments (file paths, typically) — everything in `argv` that isn't `--dry-run`. */
export function positionalArgs(argv = process.argv.slice(2)) {
  return argv.filter((a) => a !== "--dry-run");
}

/**
 * One reported change site. `classification` is a codemod-specific label (e.g.
 * `"auto-bytes"` / `"ambiguous-cross-file"` / `"excluded-test-subject"`) — this module does not
 * constrain its values, each codemod defines its own vocabulary.
 */
export function reportEntry({ file, line, before, after, classification }) {
  if (!file || !Number.isInteger(line)) {
    throw new Error("reportEntry: `file` and integer `line` are required");
  }
  return { file, line, before, after, classification };
}

/** Serializes a report to the same JSON shape every codemod prints, for diffing/piping. */
export function formatReport(entries) {
  return JSON.stringify(entries, null, 2);
}

export function printReport(entries) {
  process.stdout.write(formatReport(entries) + "\n");
}

/**
 * Writes `text` to `file` UNLESS `dryRun` is set — the single choke point every codemod must
 * funnel writes through, so "dry-run touches disk" is structurally impossible rather than a rule
 * each codemod has to remember to honor.
 */
export function writeUnlessDryRun({ dryRun, file, text }) {
  if (dryRun) return;
  writeFileSync(file, text);
}
