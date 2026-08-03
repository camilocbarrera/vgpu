#!/usr/bin/env node
// Thin CLI around verifyTask().
//
// Usage: node verify/run-verify.mjs --workspace <dir> --task <taskId> [--out <path>] [--work-dir <dir>]
//
// The process exit code (1 if any gate failed) is a convenience for the
// standalone `pnpm verify` command and for manual sanity checks. Automated
// drivers (Layer 2) MUST read the Evidence JSON's `gates` object instead and
// must never gate on this exit code.
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";
import { verifyTask } from "./verify-task.mjs";

const { values } = parseArgs({
  options: {
    workspace: { type: "string" },
    task: { type: "string" },
    out: { type: "string" },
    "work-dir": { type: "string" },
    help: { type: "boolean", short: "h" },
  },
});

const usage =
  "Usage: node verify/run-verify.mjs --workspace <dir> --task <taskId> [--out <path>] [--work-dir <dir>]";

if (values.help) {
  console.log(usage);
  process.exit(0);
}
if (!values.workspace || !values.task) {
  console.error(usage);
  process.exit(64);
}

const evidence = await verifyTask({
  taskId: values.task,
  workspaceDir: resolve(values.workspace),
  workDir: values["work-dir"] ? resolve(values["work-dir"]) : undefined,
});

const json = `${JSON.stringify(evidence, null, 2)}\n`;
if (values.out) {
  const outPath = resolve(values.out);
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, json, "utf8");
  console.error(`evidence written to ${outPath}`);
} else {
  process.stdout.write(json);
}

process.exitCode = Object.values(evidence.gates).includes("fail") ? 1 : 0;
