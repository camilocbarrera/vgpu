import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, test } from "vitest";
import {
  formatReport,
  isDryRun,
  positionalArgs,
  reportEntry,
  writeUnlessDryRun,
} from "./report.mjs";

test("isDryRun detects the --dry-run flag", () => {
  expect(isDryRun(["--dry-run"])).toBe(true);
  expect(isDryRun(["file.ts", "--dry-run"])).toBe(true);
  expect(isDryRun(["file.ts"])).toBe(false);
  expect(isDryRun([])).toBe(false);
});

test("positionalArgs strips the --dry-run flag", () => {
  expect(positionalArgs(["a.ts", "--dry-run", "b.ts"])).toEqual(["a.ts", "b.ts"]);
});

test("reportEntry requires file and an integer line", () => {
  expect(() => reportEntry({ file: "a.ts" })).toThrow();
  expect(() =>
    reportEntry({ file: "a.ts", line: 1, before: "x", after: "y", classification: "auto" }),
  ).not.toThrow();
});

test("formatReport produces stable, parseable JSON", () => {
  const entries = [reportEntry({ file: "a.ts", line: 3, before: "x", after: "y", classification: "auto" })];
  const json = formatReport(entries);
  expect(JSON.parse(json)).toEqual(entries);
});

test("writeUnlessDryRun never touches disk in dry-run mode", () => {
  const dir = mkdtempSync(join(tmpdir(), "codemod-report-test-"));
  const file = join(dir, "out.txt");
  try {
    writeUnlessDryRun({ dryRun: true, file, text: "should not be written" });
    expect(existsSync(file)).toBe(false);

    writeUnlessDryRun({ dryRun: false, file, text: "written for real" });
    expect(readFileSync(file, "utf8")).toBe("written for real");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
