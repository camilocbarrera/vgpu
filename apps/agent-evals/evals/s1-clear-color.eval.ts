import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { defineEval } from "eve/evals";
import { equals, satisfies } from "eve/evals/expect";
import {
  evidencePath,
  snapshotTarPath,
  snapshotWorkspaceDir,
  workDir,
} from "../agent/hooks/snapshot-path.ts";
// Layer 2 -> Layer 1: the grader is imported verbatim, never re-implemented.
import { verifyTask } from "../verify/verify-task.mjs";
import { requireCredentials } from "./lib/credentials-guard.ts";
import { deriveJourney, type MilestoneSpec } from "./lib/journey-milestones.ts";

const PACKAGE_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const TASK_ID = "s1-clear-color";
const TASKS_ROOT = join(PACKAGE_ROOT, "tasks");
const TASK_DIR = join(TASKS_ROOT, TASK_ID);

/**
 * Layer 1's Evidence, plus the Layer-2-only fields this eval adds. Layer 2
 * extends the schema, it never redefines it.
 */
interface Evidence {
  schemaVersion: number;
  taskId: string;
  env: Record<string, unknown>;
  gates: Record<string, string>;
  metrics: Record<string, unknown>;
  failures: string[];
  journey?: unknown;
}

interface TaskManifest {
  readonly taskId: string;
  readonly journeyMilestones: readonly MilestoneSpec[];
}

/**
 * The end-to-end walking skeleton: hand a neutral coding agent a neutral task,
 * export the workspace it leaves behind, and grade it with Layer 1's verifier.
 *
 * What is gated: `renders` and `colorExact`, both computed on the host from a
 * PNG this process re-rendered itself. What is NOT gated: the agent's reply,
 * and the whole journey (which tools it reached for, whether it ever ran
 * `vgpu doctor`). Gate the outcome, score the journey, never gate the ritual.
 */
export default defineEval({
  description: "s1: agent makes the rendered image solid red; graded by pixels, not by prose.",

  async test(t) {
    requireCredentials(t);

    const manifest = JSON.parse(
      readFileSync(join(TASK_DIR, "manifest.json"), "utf8"),
    ) as TaskManifest;

    // The prompt file is owned by Layer 1 and read, never duplicated.
    const turn = await t.send(readFileSync(join(TASK_DIR, "prompt.md"), "utf8"));

    // Harness-level success only: the run completed. This is not the pixel gate.
    t.succeeded();

    const sessionId = turn.sessionId;
    const tarPath = snapshotTarPath(sessionId);
    await t.require(
      existsSync(tarPath),
      satisfies((value: unknown) => value === true, `export hook wrote ${tarPath}`),
    );

    const extracted = snapshotWorkspaceDir(sessionId);
    rmSync(extracted, { force: true, recursive: true });
    mkdirSync(extracted, { recursive: true });
    execFileSync("tar", ["-xf", tarPath, "-C", extracted]);

    const evidence: Evidence = await verifyTask({
      tasksRoot: TASKS_ROOT,
      taskId: TASK_ID,
      workspaceDir: extracted,
      workDir: workDir(),
    });

    // Soft, report-only enrichment. Nothing below is gated on.
    evidence.journey = deriveJourney(turn.toolCalls, manifest.journeyMilestones);
    evidence.env.model = process.env.VGPU_EVALS_MODEL ?? "anthropic/claude-sonnet-5";
    evidence.env.sandboxBackend = process.env.VGPU_EVALS_SANDBOX ?? "docker";
    evidence.env.sessionId = sessionId;

    const artifact = evidencePath(sessionId);
    mkdirSync(dirname(artifact), { recursive: true });
    writeFileSync(artifact, `${JSON.stringify(evidence, null, 2)}\n`, "utf8");
    t.log(`evidence written to ${artifact}`);

    // The only two gates in this file.
    t.check(evidence.gates.renders, equals("pass")).gate();
    t.check(evidence.gates.colorExact, equals("pass")).gate();
  },
});
