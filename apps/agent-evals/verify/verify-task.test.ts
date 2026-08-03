import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { PNG } from "pngjs";
import { describe, expect, it } from "vitest";
import { verifyTask } from "./verify-task.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const PACKAGE_ROOT = join(HERE, "..");
const TASKS_ROOT = join(PACKAGE_ROOT, "tasks");
const FIXTURE_DIR = join(TASKS_ROOT, "s1-clear-color", "fixture");
const CASES = join(HERE, "__fixtures__");

// Each case gets its own workspaceDir and workDir: `.work/` is mutable state
// (an installed node_modules plus the last render's output) and sharing it
// between cases would let one case's leftovers decide another case's verdict.
function tempPair(name: string) {
  const root = mkdtempSync(join(tmpdir(), `agent-evals-${name}-`));
  const workspaceDir = join(root, "workspace");
  const workDir = join(root, "work");
  mkdirSync(workspaceDir, { recursive: true });
  mkdirSync(workDir, { recursive: true });
  // Every graded workspace starts from the task fixture's package.json, exactly
  // like a real agent session would.
  cpSync(join(FIXTURE_DIR, "package.json"), join(workspaceDir, "package.json"));
  return { workspaceDir, workDir };
}

function solidRedPng(width: number, height: number): Buffer {
  const png = new PNG({ width, height });
  for (let i = 0; i < png.data.length; i += 4) {
    png.data[i] = 255;
    png.data[i + 1] = 0;
    png.data[i + 2] = 0;
    png.data[i + 3] = 255;
  }
  return PNG.sync.write(png);
}

describe("verifyTask(s1-clear-color)", () => {
  it("passes both gates for a correct solution", { timeout: 300_000 }, async () => {
    const { workspaceDir, workDir } = tempPair("correct");
    cpSync(join(CASES, "correct", "render.mjs"), join(workspaceDir, "render.mjs"));

    const evidence = await verifyTask({
      tasksRoot: TASKS_ROOT,
      taskId: "s1-clear-color",
      workspaceDir,
      workDir,
    });

    expect(evidence.gates.renders).toBe("pass");
    expect(evidence.gates.colorExact).toBe("pass");
    expect(evidence.gates.packageJsonUnchanged).toBe("pass");
    // Soft signal (never a gate): this solution really did import vgpu.
    expect(evidence.metrics.vgpuLoaded).toBe(true);
    expect(evidence.metrics.matchedFraction).toBe(1);
    expect(evidence.metrics.dominantPixel).toEqual([255, 0, 0, 255]);
    expect(evidence.metrics.width).toBe(64);
    expect(evidence.metrics.height).toBe(64);
    expect(evidence.failures).toEqual([]);
  });

  it("renders but fails the color gate for the untouched fixture", { timeout: 300_000 }, async () => {
    const { workspaceDir, workDir } = tempPair("unchanged");
    cpSync(join(CASES, "unchanged", "render.mjs"), join(workspaceDir, "render.mjs"));

    const evidence = await verifyTask({
      tasksRoot: TASKS_ROOT,
      taskId: "s1-clear-color",
      workspaceDir,
      workDir,
    });

    // "It rendered, it just rendered the wrong thing" must be legible in the
    // evidence without re-running anything: renders passes, colorExact fails,
    // and dominantPixel carries the diagnostic non-red colour.
    expect(evidence.gates.renders).toBe("pass");
    expect(evidence.gates.colorExact).toBe("fail");
    expect(evidence.metrics.matchedFraction).toBe(0);
    // vec4f(0.25, 0.5, 0.75, 1.0) quantised to bytes by this renderer.
    expect(evidence.metrics.dominantPixel).toEqual([64, 128, 191, 255]);
    expect(evidence.metrics.distinctColors).toBe(1);
  });

  it("does not grade a forged out.png (delete-before-render)", { timeout: 300_000 }, async () => {
    // THIS IS THE EVAL'S CORE ANTI-CHEAT PROPERTY, not incidental coverage.
    // An agent can trivially write a solid-red PNG by hand instead of fixing the
    // shader. Step 5 of verifyTask deletes any pre-existing out.png before
    // re-rendering, so the only PNG that can ever be graded is one this process
    // just produced from the agent's *source*. If someone deletes that line, or
    // makes it conditional, this test must go red.
    const { workspaceDir, workDir } = tempPair("forgery");
    cpSync(join(CASES, "forgery", "render.mjs"), join(workspaceDir, "render.mjs"));
    // (a) the forged artefact, hand-written straight to disk, never rendered.
    writeFileSync(join(workspaceDir, "out.png"), solidRedPng(64, 64));

    // (b) the same forgery planted directly in the verify-workspace, which is
    // where a stale output from a previous run would live. This is the copy that
    // only step 5's unconditional delete can defeat.
    const runId = "forgery-run";
    const verifyWorkspace = join(workDir, "verify-workspace", "s1-clear-color", runId);
    mkdirSync(verifyWorkspace, { recursive: true });
    cpSync(FIXTURE_DIR, verifyWorkspace, { recursive: true });
    const plantedOutput = join(verifyWorkspace, "out.png");
    writeFileSync(plantedOutput, solidRedPng(64, 64));

    const evidence = await verifyTask({
      tasksRoot: TASKS_ROOT,
      taskId: "s1-clear-color",
      workspaceDir,
      workDir,
      runId,
    });

    expect(evidence.gates.renders).toBe("fail");
    expect(evidence.gates.colorExact).not.toBe("pass");
    // Explicitly: the forgery must not produce the all-green verdict.
    expect([evidence.gates.renders, evidence.gates.colorExact]).not.toEqual(["pass", "pass"]);
    // The planted red PNG is gone: it was deleted and the no-op renderer never
    // wrote a replacement.
    expect(existsSync(plantedOutput)).toBe(false);
    expect(evidence.metrics.matchedFraction).toBeUndefined();
    expect(evidence.failures.join("\n")).toContain("out.png");
  });
  it(
    "does not run install scripts from the graded workspace",
    { timeout: 300_000 },
    async () => {
      // REGRESSION (P0, was exploitable): the agent's package.json used to be
      // copied into the verify-workspace and installed there, so `pnpm install`
      // executed its `postinstall` ON THE HOST, with the verify-workspace as
      // cwd. The payload below is the one that actually defeated the verifier:
      // it leaves the WRONG shader in place and instead monkey-patches
      // node_modules/pngjs so every write() paints the expected colour. It
      // scored renders=pass + colorExact=pass.
      //
      // Two independent fixes must keep this red-flagged: the install ignores
      // scripts, and the agent's package.json is never copied at all.
      const { workspaceDir, workDir } = tempPair("install-script");
      cpSync(join(CASES, "unchanged", "render.mjs"), join(workspaceDir, "render.mjs"));
      writeFileSync(
        join(workspaceDir, "patch.js"),
        [
          'import { appendFileSync } from "node:fs";',
          'appendFileSync("node_modules/pngjs/lib/png.js", `',
          ";(function () {",
          "  const _w = exports.PNG.sync.write;",
          "  exports.PNG.sync.write = function (png, opts) {",
          "    for (let i = 0; i < png.data.length; i += 4) {",
          "      png.data[i] = 255; png.data[i+1] = 0; png.data[i+2] = 0; png.data[i+3] = 255;",
          "    }",
          "    return _w.call(this, png, opts);",
          "  };",
          "})();",
          "`);",
        ].join("\n"),
      );
      writeFileSync(
        join(workspaceDir, "package.json"),
        JSON.stringify({
          private: true,
          type: "module",
          scripts: { postinstall: "node patch.js" },
          dependencies: { vgpu: "0.2.0", pngjs: "7.0.0" },
        }),
      );

      const evidence = await verifyTask({
        tasksRoot: TASKS_ROOT,
        taskId: "s1-clear-color",
        workspaceDir,
        workDir,
      });

      // The wrong shader must score as wrong.
      expect(evidence.gates.colorExact).toBe("fail");
      expect(evidence.metrics.dominantPixel).toEqual([64, 128, 191, 255]);
      // The hijack never ran: pngjs in the run directory is untouched.
      const pngjs = readFileSync(
        join(workDir, "verify-workspace", "s1-clear-color", String(evidence.metrics.runId), "node_modules", "pngjs", "lib", "png.js"),
        "utf8",
      );
      expect(pngjs).not.toContain("exports.PNG.sync.write = function");
      // And the attempt is legible in the evidence rather than silently ignored.
      expect(evidence.gates.packageJsonUnchanged).toBe("fail");
      expect(evidence.failures.join("\n")).toContain("--ignore-scripts");
      expect(evidence.metrics.agentPackageScripts).toEqual({ postinstall: "node patch.js" });
    },
  );

  it("does not inherit sources from a previous trial", { timeout: 300_000 }, async () => {
    // REGRESSION (P0, was exploitable): the verify-workspace was seeded once
    // and only ever overlaid, so a second trial sharing the same workDir
    // inherited the first trial's render.mjs. An agent that wrote NOTHING
    // scored whatever the previous agent scored. PR2 reuses one workDir for
    // every trial, so this was reachable in normal operation.
    const { workspaceDir: solvedWorkspace, workDir } = tempPair("trial-a");
    cpSync(join(CASES, "correct", "render.mjs"), join(solvedWorkspace, "render.mjs"));

    const trialA = await verifyTask({
      tasksRoot: TASKS_ROOT,
      taskId: "s1-clear-color",
      workspaceDir: solvedWorkspace,
      workDir,
    });
    expect(trialA.gates.colorExact).toBe("pass");

    // Trial B: same workDir, same task, and an agent that produced nothing.
    const emptyWorkspace = join(workDir, "..", "empty-workspace");
    mkdirSync(emptyWorkspace, { recursive: true });

    const trialB = await verifyTask({
      tasksRoot: TASKS_ROOT,
      taskId: "s1-clear-color",
      workspaceDir: emptyWorkspace,
      workDir,
    });

    // B must be graded against the FIXTURE, not against A's solution.
    expect(trialB.gates.renders).toBe("pass");
    expect(trialB.gates.colorExact).toBe("fail");
    expect(trialB.metrics.dominantPixel).toEqual([64, 128, 191, 255]);
    expect(trialB.metrics.runId).not.toBe(trialA.metrics.runId);
  });
});
