// Layer 1 re-execution engine.
//
// The contract that matters: nothing produced inside the graded workspace is
// trusted. Source files are re-exported into a clean, separately-installed
// verify-workspace, any pre-existing output is deleted, and the render is run
// again in a fresh process before a single pixel is graded.
//
// IMPORTANT: no file in this directory may import `eve`. See ../README.md
// ("Layer boundary invariant").
//
// verifyTask(opts: {
//   tasksRoot: string,     // absolute path to apps/agent-evals/tasks
//   taskId: string,        // e.g. "s1-clear-color"
//   workspaceDir: string,  // absolute path to the (untrusted) agent workspace
//   workDir: string,       // absolute path to a scratch root, default apps/agent-evals/.work
// }) -> Promise<Evidence>
import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { gradePixels } from "./grade-pixels.mjs";

const PACKAGE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

/** Source extensions copied out of the untrusted workspace. */
const COPIED_EXTENSIONS = new Set([".mjs", ".js", ".ts", ".wgsl", ".json"]);
/** Directories never copied out of the untrusted workspace. */
const EXCLUDED_DIRS = new Set(["node_modules", ".git"]);
/** Files never copied out of the untrusted workspace (outputs must be re-produced). */
const EXCLUDED_FILES = new Set(["out.png"]);

const RENDER_TIMEOUT_MS = 120_000;

export const SCHEMA_VERSION = 1;

/**
 * @param {{ tasksRoot?: string, taskId: string, workspaceDir: string, workDir?: string }} opts
 */
export async function verifyTask(opts) {
  const tasksRoot = resolve(opts.tasksRoot ?? join(PACKAGE_ROOT, "tasks"));
  const workDir = resolve(opts.workDir ?? join(PACKAGE_ROOT, ".work"));
  const workspaceDir = resolve(opts.workspaceDir);
  const { taskId } = opts;

  const taskDir = join(tasksRoot, taskId);
  const manifestPath = join(taskDir, "manifest.json");
  if (!existsSync(manifestPath)) {
    // A missing manifest is a harness bug, not a grading outcome: throw.
    throw new Error(`verifyTask: manifest not found at ${manifestPath}`);
  }
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));

  // 2. Seed / reuse the verify-workspace.
  const verifyWorkspace = join(workDir, "verify-workspace", taskId);
  const hashFile = join(verifyWorkspace, ".install-hash");
  const fixtureDir = join(taskDir, "fixture");
  let seeded = false;
  if (!existsSync(verifyWorkspace)) {
    mkdirSync(verifyWorkspace, { recursive: true });
    cpSync(fixtureDir, verifyWorkspace, { recursive: true });
    seeded = true;
  }
  if (seeded || !existsSync(join(verifyWorkspace, "node_modules"))) {
    installDeps(verifyWorkspace);
    writeFileSync(hashFile, packageJsonHash(verifyWorkspace), "utf8");
  }

  // 3. Copy the agent's source files over the seeded workspace.
  copySources(workspaceDir, verifyWorkspace);

  // 4. Reinstall only if the (possibly agent-modified) package.json changed.
  const currentHash = packageJsonHash(verifyWorkspace);
  const previousHash = existsSync(hashFile) ? readFileSync(hashFile, "utf8") : "";
  if (currentHash !== previousHash) {
    installDeps(verifyWorkspace);
    writeFileSync(hashFile, currentHash, "utf8");
  }

  // 5. Delete any output *unconditionally* before rendering. This single line is
  //    what defeats a hand-forged PNG; do not make it conditional.
  const outPath = join(verifyWorkspace, "out.png");
  rmSync(outPath, { force: true });

  // 6. Re-render in a fresh process with a minimal environment.
  const render = await runRender(verifyWorkspace);

  const env = {
    vgpu: manifest.vgpu,
    node: process.version,
    gitSha: gitSha(),
    ...doctorEnv(verifyWorkspace),
    capturedAt: new Date().toISOString(),
  };

  /** @type {string[]} */
  const failures = [];
  /** @type {Record<string, unknown>} */
  const metrics = {
    renderExitCode: render.exitCode,
    renderDurationMs: render.durationMs,
  };

  // 7. renders gate.
  let renders = "fail";
  if (render.timedOut) {
    failures.push(`render timed out after ${RENDER_TIMEOUT_MS}ms`);
  } else if (render.exitCode !== 0) {
    failures.push(
      `render exited with code ${render.exitCode}: ${truncate(render.stderr)}`,
    );
  } else if (!existsSync(outPath)) {
    failures.push("render exited 0 but did not write out.png");
  } else {
    let graded;
    try {
      graded = gradePixels(readFileSync(outPath), manifest.expected);
    } catch (error) {
      failures.push(`could not decode out.png: ${String(error)}`);
    }
    if (graded) {
      metrics.width = graded.width;
      metrics.height = graded.height;
      metrics.distinctColors = graded.distinctColors;
      metrics.dominantPixel = graded.dominantPixel;
      metrics.matchedFraction = graded.matchedFraction;
      if (
        graded.width === manifest.expected.width &&
        graded.height === manifest.expected.height
      ) {
        renders = "pass";
      } else {
        failures.push(
          `out.png is ${graded.width}x${graded.height}, expected ${manifest.expected.width}x${manifest.expected.height}`,
        );
      }
    }
  }

  // 8. colorExact gate.
  let colorExact = "skip";
  if (renders === "pass") {
    colorExact = metrics.matchedFraction === 1 ? "pass" : "fail";
    if (colorExact === "fail") {
      failures.push(
        `only ${((metrics.matchedFraction ?? 0) * 100).toFixed(2)}% of pixels match ${JSON.stringify(
          manifest.expected.color,
        )}; dominant pixel was ${JSON.stringify(metrics.dominantPixel)}`,
      );
    }
  }

  // 9. Evidence. A bad render is data, never an exception.
  return {
    schemaVersion: SCHEMA_VERSION,
    taskId,
    env,
    gates: { renders, colorExact },
    metrics,
    failures,
  };
}

/**
 * `--ignore-workspace` is MANDATORY here: `apps/agent-evals` is a pnpm workspace
 * member, so without the flag pnpm resolves this nested directory to the root
 * workspace and silently no-ops the install, leaving no node_modules behind.
 * (The equivalent `.npmrc` setting does not work on pnpm 9.15.4 — it must be the
 * CLI flag.) Do not "clean this up".
 * @param {string} cwd
 */
function installDeps(cwd) {
  const result = spawnSyncish("pnpm", ["install", "--ignore-workspace"], cwd);
  if (result.status !== 0) {
    throw new Error(
      `verifyTask: pnpm install --ignore-workspace failed in ${cwd}: ${truncate(result.stderr)}`,
    );
  }
}

/** @param {string} verifyWorkspace */
function packageJsonHash(verifyWorkspace) {
  const pkgPath = join(verifyWorkspace, "package.json");
  const bytes = existsSync(pkgPath) ? readFileSync(pkgPath) : Buffer.alloc(0);
  return createHash("sha256").update(bytes).digest("hex");
}

/**
 * Copy every `**\/*.{mjs,js,ts,wgsl,json}` from the untrusted workspace into the
 * verify-workspace, excluding node_modules/, .git/ and out.png.
 * @param {string} from
 * @param {string} to
 */
function copySources(from, to) {
  if (!existsSync(from)) {
    throw new Error(`verifyTask: workspaceDir does not exist: ${from}`);
  }
  /** @param {string} dir */
  const walk = (dir) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (entry.isDirectory()) {
        if (EXCLUDED_DIRS.has(entry.name)) continue;
        walk(join(dir, entry.name));
        continue;
      }
      if (!entry.isFile()) continue;
      if (EXCLUDED_FILES.has(entry.name)) continue;
      const dot = entry.name.lastIndexOf(".");
      const ext = dot === -1 ? "" : entry.name.slice(dot);
      if (!COPIED_EXTENSIONS.has(ext)) continue;
      const abs = join(dir, entry.name);
      const rel = relative(from, abs);
      if (rel.split(sep).includes("node_modules")) continue;
      const dest = join(to, rel);
      mkdirSync(dirname(dest), { recursive: true });
      cpSync(abs, dest);
    }
  };
  walk(from);
}

/**
 * Fresh `node render.mjs`, hard timeout, explicit env allowlist (PATH + HOME
 * only — deps are already installed, nothing else from the parent process,
 * including any leaked secrets, is passed through).
 * @param {string} cwd
 */
function runRender(cwd) {
  return new Promise((resolvePromise) => {
    const startedAt = Date.now();
    const child = spawn(process.execPath, ["render.mjs"], {
      cwd,
      env: { PATH: process.env.PATH ?? "", HOME: process.env.HOME ?? "" },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGKILL");
    }, RENDER_TIMEOUT_MS);
    child.stdout.on("data", (chunk) => {
      stdout += String(chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderr += String(chunk);
    });
    child.on("error", (error) => {
      clearTimeout(timer);
      resolvePromise({
        exitCode: -1,
        stdout,
        stderr: `${stderr}${String(error)}`,
        timedOut,
        durationMs: Date.now() - startedAt,
      });
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      resolvePromise({
        exitCode: timedOut ? -1 : (code ?? -1),
        stdout,
        stderr,
        timedOut,
        durationMs: Date.now() - startedAt,
      });
    });
  });
}

/**
 * `vgpu doctor` against the verify-workspace: ~1-2s, and it catches a broken
 * host before a bad render gets blamed on the agent. Never throws.
 *
 * NOTE: no `--json` flag. `vgpu@0.2.0`'s doctor writes JSON to stdout by
 * default (`Usage: vgpu doctor [--no-render] [--pretty]`); passing `--json`
 * fails with "Unknown doctor option: --json" and exit 1.
 * @param {string} cwd
 */
function doctorEnv(cwd) {
  const fallback = { adapterName: null, adapterType: null, doctorVerdict: null };
  const result = spawnSyncish("npx", ["vgpu", "doctor"], cwd);
  if (result.status !== 0 && !result.stdout) return fallback;
  try {
    const parsed = JSON.parse(extractJson(result.stdout));
    const adapter = parsed.adapter ?? parsed.gpu ?? {};
    return {
      adapterName: adapter.name ?? adapter.description ?? parsed.adapterName ?? null,
      adapterType: adapter.type ?? parsed.adapterType ?? null,
      doctorVerdict: parsed.verdict ?? null,
    };
  } catch {
    return fallback;
  }
}

/** @param {string} text */
function extractJson(text) {
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  return start === -1 || end === -1 ? text : text.slice(start, end + 1);
}

function gitSha() {
  const result = spawnSyncish("git", ["rev-parse", "--short", "HEAD"], PACKAGE_ROOT);
  return result.status === 0 ? result.stdout.trim() : null;
}

/**
 * @param {string} command
 * @param {string[]} args
 * @param {string} cwd
 */
function spawnSyncish(command, args, cwd) {
  const result = spawnSync(command, args, {
    cwd,
    encoding: "utf8",
    env: process.env,
    maxBuffer: 32 * 1024 * 1024,
  });
  return {
    status: result.status ?? -1,
    stdout: result.stdout ?? "",
    stderr: result.stderr ?? "",
  };
}

/** @param {string} text */
function truncate(text, max = 2000) {
  const value = String(text ?? "");
  return value.length > max ? `${value.slice(0, max)}…` : value;
}
