#!/usr/bin/env node
// Entry point for `pnpm agent-evals`.
//
// Dependency-free on purpose (only node: builtins): its entire job is to run
// correctly *before* anything workspace-specific is guaranteed to be installed
// for the current Node version.
//
// `apps/agent-evals` is driven by `eve`, which requires Node.js >= 24, while
// this repo itself pins Node 22 (.nvmrc, root engines). Without this preflight
// the failure surfaces as an opaque syntax/engine error from deep inside eve's
// bin; with it you get one actionable line.
import { spawn } from "node:child_process";
import process from "node:process";

const REQUIRED_MAJOR = 24;
// Exit 2, not 1, so CI logs and scripts can tell "wrong Node" apart from
// "the evals ran and failed" (which `eve eval` reports as exit 1).
const EXIT_WRONG_NODE = 2;

const major = Number.parseInt(process.versions.node.split(".")[0], 10);

if (!Number.isInteger(major) || major < REQUIRED_MAJOR) {
  process.stderr.write(
    [
      `pnpm agent-evals: Node.js >= ${REQUIRED_MAJOR} is required, but this is v${process.versions.node}.`,
      "",
      "  apps/agent-evals is driven by `eve`, which requires Node 24+. The rest of",
      "  this repo pins Node 22 on purpose, so switch Node just for this command:",
      "",
      `      nvm install ${REQUIRED_MAJOR} && nvm use ${REQUIRED_MAJOR} && pnpm agent-evals`,
      "",
      "  You also need an AI Gateway credential (AI_GATEWAY_API_KEY or",
      "  VERCEL_OIDC_TOKEN); without one the evals skip instead of failing.",
      "  See apps/agent-evals/README.md.",
      "",
    ].join("\n"),
  );
  process.exit(EXIT_WRONG_NODE);
}

const child = spawn(
  "pnpm",
  ["--filter", "@vgpu/agent-evals", "exec", "eve", "eval", ...process.argv.slice(2)],
  { stdio: "inherit" },
);

child.on("error", (error) => {
  process.stderr.write(`pnpm agent-evals: failed to start pnpm: ${error.message}\n`);
  process.exit(1);
});

child.on("close", (code, signal) => {
  if (signal) {
    process.stderr.write(`pnpm agent-evals: terminated by signal ${signal}\n`);
    process.exit(1);
  }
  process.exit(code ?? 1);
});
