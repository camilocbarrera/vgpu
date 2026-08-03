// Layer 1 re-execution engine.
//
// THE TRUST MODEL, precisely (read this before changing anything below):
//
//   The agent's workspace contributes SOURCE TEXT and nothing else. Every other
//   input to the verdict — the dependency manifest, the installed
//   node_modules, the output PNG — is produced by this file from the task
//   fixture. Concretely, per call:
//     * a run directory is seeded from `tasks/<id>/fixture/` and its source
//       tree is wiped and re-seeded, so no trial can inherit another's files;
//     * dependencies are installed from the FIXTURE's package.json with
//       `--ignore-scripts`, so nothing the agent wrote executes on the host at
//       install time and node_modules cannot be monkey-patched;
//     * the agent's own package.json / lockfiles are never copied — a changed
//       manifest (dependencies or scripts) is reported as the
//       `packageJsonUnchanged` gate instead of being honoured;
//     * `out.png` is deleted unconditionally and re-rendered in a fresh
//       process with an env allowlist (PATH + HOME) before a pixel is graded.
//
// What this is NOT: a security boundary. The graded render still executes
// agent-authored code on the host, as the agent's own user. It defeats
// accidental and opportunistic cheating (a forged PNG, a hijacked dependency,
// leftovers from a previous trial); it does not contain a determined attacker.
// Running the render inside a pinned container is tracked as follow-up work.
//
// IMPORTANT: no file in this directory may import `eve`. See ../README.md
// ("Layer boundary invariant").
//
// verifyTask(opts: {
//   tasksRoot: string,     // absolute path to apps/agent-evals/tasks
//   taskId: string,        // e.g. "s1-clear-color"
//   workspaceDir: string,  // absolute path to the (untrusted) agent workspace
//   workDir: string,       // absolute path to a scratch root, default apps/agent-evals/.work
//   runId: string,         // optional; isolates concurrent trials, default random
// }) -> Promise<Evidence>
import { spawn, spawnSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import {
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { gradePixels } from "./grade-pixels.mjs";

const PACKAGE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const VERIFY_DIR = join(PACKAGE_ROOT, "verify");

/** Source extensions copied out of the untrusted workspace. */
const COPIED_EXTENSIONS = new Set([".mjs", ".js", ".ts", ".wgsl", ".json"]);
/** Directories never copied out of the untrusted workspace. */
const EXCLUDED_DIRS = new Set(["node_modules", ".git"]);
/**
 * Files never copied out of the untrusted workspace, at ANY depth.
 *
 * `out.png` because outputs must be re-produced, never imported. Everything
 * else because it steers dependency resolution or install-time execution:
 * the dependency set is the fixture's, full stop. A nested `package.json` also
 * changes module resolution (`"type"`), so it is excluded too.
 */
const EXCLUDED_FILES = new Set([
  "out.png",
  "package.json",
  "pnpm-lock.yaml",
  "package-lock.json",
  "yarn.lock",
  "npm-shrinkwrap.json",
  ".npmrc",
  ".pnpmfile.cjs",
]);
/** Kept out of the wipe: reinstalling per trial would dominate the runtime. */
const PRESERVED_ON_RESEED = new Set(["node_modules"]);

const RENDER_TIMEOUT_MS = 120_000;

export const SCHEMA_VERSION = 1;

/**
 * @param {{ tasksRoot?: string, taskId: string, workspaceDir: string, workDir?: string, runId?: string }} opts
 */
export async function verifyTask(opts) {
  const tasksRoot = resolve(opts.tasksRoot ?? join(PACKAGE_ROOT, "tasks"));
  const workDir = resolve(opts.workDir ?? join(PACKAGE_ROOT, ".work"));
  const workspaceDir = resolve(opts.workspaceDir);
  const { taskId } = opts;
  // Concurrent trials of the same task must not share a run directory. The
  // default is unique per call; callers that want a stable, inspectable path
  // (and know their trials are serialised) can pass their own id — PR2 passes
  // the eve session id.
  const runId = opts.runId ?? randomUUID();

  const taskDir = join(tasksRoot, taskId);
  const manifestPath = join(taskDir, "manifest.json");
  if (!existsSync(manifestPath)) {
    // A missing manifest is a harness bug, not a grading outcome: throw.
    throw new Error(`verifyTask: manifest not found at ${manifestPath}`);
  }
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  const fixtureDir = join(taskDir, "fixture");

  // 2. Seed the run directory, or wipe its source tree and re-seed it.
  //
  //    The wipe is not an optimisation, it is the fix for cross-trial
  //    contamination: without it, a trial that writes NOTHING inherits the
  //    previous trial's render.mjs and scores whatever that one scored.
  const runDir = join(workDir, "verify-workspace", taskId, runId);
  if (existsSync(runDir)) {
    for (const entry of readdirSync(runDir)) {
      if (PRESERVED_ON_RESEED.has(entry)) continue;
      rmSync(join(runDir, entry), { force: true, recursive: true });
    }
  } else {
    mkdirSync(runDir, { recursive: true });
  }
  cpSync(fixtureDir, runDir, { recursive: true });

  // 3. Install the FIXTURE's dependencies. Never the agent's.
  if (!existsSync(join(runDir, "node_modules"))) {
    installDeps(runDir);
  }

  // 4. Overlay the agent's source files (no package.json, no lockfiles).
  copySources(workspaceDir, runDir);

  // 5. Delete any output *unconditionally* before rendering.
  const outPath = join(runDir, "out.png");
  rmSync(outPath, { force: true });

  // 6. Re-render in a fresh process with a minimal environment.
  const probeLog = join(workDir, "probes", `${runId}.log`);
  mkdirSync(dirname(probeLog), { recursive: true });
  const render = await runRender(runDir, probeLog);

  const env = {
    vgpu: manifest.vgpu,
    node: process.version,
    gitSha: gitSha(),
    ...doctorEnv(runDir),
    capturedAt: new Date().toISOString(),
  };

  /** @type {string[]} */
  const failures = [];
  /** @type {Record<string, unknown>} */
  const metrics = {
    renderExitCode: render.exitCode,
    renderDurationMs: render.durationMs,
    runId,
    // Soft signal, never a gate: did the graded render actually load vgpu?
    // `s1-clear-color` is passable with pngjs alone, so a `false` here means
    // "solved without the library under test" — worth seeing in the evidence,
    // not worth failing a run over. `null` = the probe could not report.
    vgpuLoaded: readVgpuLoaded(probeLog),
  };

  // 7. packageJsonUnchanged gate. Declared, reported, and never honoured: the
  //    install above already used the fixture's manifest, so this only tells a
  //    reader that the agent's answer assumed a different dependency set — or
  //    that it tried to get code executed at install time.
  const manifestCheck = compareManifests(fixtureDir, workspaceDir);
  metrics.agentPackageScripts = manifestCheck.agentScripts;
  if (!manifestCheck.unchanged) {
    failures.push(manifestCheck.reason ?? "agent changed package.json");
  }

  // 8. renders gate.
  let renders = "fail";
  if (render.timedOut) {
    failures.push(`render timed out after ${RENDER_TIMEOUT_MS}ms`);
  } else if (render.exitCode !== 0) {
    failures.push(`render exited with code ${render.exitCode}: ${truncate(render.stderr)}`);
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
      if (graded.width === manifest.expected.width && graded.height === manifest.expected.height) {
        renders = "pass";
      } else {
        failures.push(
          `out.png is ${graded.width}x${graded.height}, expected ${manifest.expected.width}x${manifest.expected.height}`,
        );
      }
    }
  }

  // 9. colorExact gate.
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

  // 10. Evidence. A bad render is data, never an exception.
  return {
    schemaVersion: SCHEMA_VERSION,
    taskId,
    env,
    gates: {
      renders,
      colorExact,
      packageJsonUnchanged: manifestCheck.unchanged ? "pass" : "fail",
    },
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
 *
 * `--ignore-scripts` is equally mandatory and is a correctness fix, not
 * hardening theatre: without it, a `postinstall` reachable from the graded
 * workspace runs on the host with the verify-workspace as cwd and can rewrite
 * `node_modules/pngjs` so that every render emits the expected colour. That
 * exact payload scored a false pass before this flag existed; it is a
 * regression test now.
 * @param {string} cwd
 */
function installDeps(cwd) {
  const result = spawnSyncish("pnpm", ["install", "--ignore-workspace", "--ignore-scripts"], cwd);
  if (result.status !== 0) {
    throw new Error(
      `verifyTask: pnpm install --ignore-workspace --ignore-scripts failed in ${cwd}: ${truncate(result.stderr)}`,
    );
  }
}

/**
 * Compare the agent's manifest against the fixture's — dependencies AND
 * scripts. Reported as the `packageJsonUnchanged` gate; the agent's manifest is
 * never installed and its scripts are never executed.
 *
 * `scripts` is part of the comparison because the payload this gate exists to
 * make visible did not touch dependencies at all: it kept them byte-identical
 * and added a `postinstall`.
 * @param {string} fixtureDir
 * @param {string} workspaceDir
 */
function compareManifests(fixtureDir, workspaceDir) {
  const fixture = readPackageJson(join(fixtureDir, "package.json"));
  const agentPath = join(workspaceDir, "package.json");
  if (!existsSync(agentPath)) {
    return {
      unchanged: false,
      reason: "agent workspace has no package.json (the fixture's was used)",
      agentScripts: {},
    };
  }
  const agent = readPackageJson(agentPath);
  if (agent === null) {
    return { unchanged: false, reason: "agent package.json is not valid JSON", agentScripts: {} };
  }

  const agentScripts = isRecord(agent.scripts) ? agent.scripts : {};
  const expectedDeps = normaliseDeps(fixture);
  const actualDeps = normaliseDeps(agent);
  const expectedScripts = JSON.stringify(normaliseScripts(fixture));
  const actualScripts = JSON.stringify(normaliseScripts(agent));

  if (expectedDeps !== actualDeps) {
    return {
      unchanged: false,
      reason: `agent changed the dependency set: expected ${expectedDeps}, workspace declares ${actualDeps} (the fixture's dependencies were installed instead)`,
      agentScripts,
    };
  }
  if (expectedScripts !== actualScripts) {
    return {
      unchanged: false,
      reason: `agent declared package.json scripts ${actualScripts}, expected ${expectedScripts} (they were NOT executed: the verify install runs with --ignore-scripts)`,
      agentScripts,
    };
  }
  return { unchanged: true, reason: null, agentScripts };
}

/** @param {string} path */
function readPackageJson(path) {
  try {
    const parsed = JSON.parse(readFileSync(path, "utf8"));
    return isRecord(parsed) ? parsed : {};
  } catch {
    return null;
  }
}

/** Stable, whitespace/ordering-insensitive view of a manifest's dependencies. */
function normaliseDeps(pkg) {
  const pick = (key) => {
    const value = pkg?.[key];
    if (!isRecord(value)) return [];
    return Object.keys(value)
      .sort()
      .map((name) => `${name}@${String(value[name])}`);
  };
  return JSON.stringify({ dependencies: pick("dependencies"), devDependencies: pick("devDependencies") });
}

/** Sorted view of a manifest's `scripts`, so ordering is not a difference. */
function normaliseScripts(pkg) {
  const scripts = pkg?.scripts;
  if (!isRecord(scripts)) return [];
  return Object.keys(scripts)
    .sort()
    .map((name) => `${name}=${String(scripts[name])}`);
}

/** @param {unknown} value */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Copy every `**\/*.{mjs,js,ts,wgsl,json}` from the untrusted workspace into the
 * run directory, excluding node_modules/, .git/, outputs and anything that
 * steers dependency resolution (see EXCLUDED_FILES).
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
 * `--import` (and `module.register`) landed in Node 20.6. Older runtimes still
 * render; they just report `vgpuLoaded: null`.
 */
function supportsModuleProbe() {
  const [major, minor] = process.versions.node.split(".").map((part) => Number.parseInt(part, 10));
  return major > 20 || (major === 20 && minor >= 6);
}

/**
 * Fresh `node render.mjs`, hard timeout, explicit env allowlist (PATH + HOME
 * only — deps are already installed, nothing else from the parent process,
 * including any leaked secrets, is passed through).
 *
 * The extra `--import` registers a resolve-hook that logs which specifiers the
 * render actually loaded. It is observational: the probe writes to a file
 * outside the run directory and never influences resolution.
 * @param {string} cwd
 * @param {string} probeLog
 */
function runRender(cwd, probeLog) {
  return new Promise((resolvePromise) => {
    const startedAt = Date.now();
    const args = [];
    if (supportsModuleProbe()) {
      args.push("--import", pathToFileURL(join(VERIFY_DIR, "probe-register.mjs")).href);
    }
    args.push("render.mjs");
    const child = spawn(process.execPath, args, {
      cwd,
      env: {
        PATH: process.env.PATH ?? "",
        HOME: process.env.HOME ?? "",
        VGPU_VERIFY_PROBE_OUT: probeLog,
      },
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
 * Did the graded render load the library under test? `null` when the probe did
 * not run (old Node, or the render died before importing anything).
 * @param {string} probeLog
 */
function readVgpuLoaded(probeLog) {
  if (!supportsModuleProbe() || !existsSync(probeLog)) return null;
  try {
    const text = readFileSync(probeLog, "utf8");
    if (text.trim() === "") return null;
    return /(^|[\t/])vgpu(\/|"|$)/mu.test(text) || text.includes("/node_modules/vgpu/");
  } catch {
    return null;
  }
}

/**
 * `vgpu doctor` against the run directory: ~1-2s, and it catches a broken host
 * before a bad render gets blamed on the agent. Never throws.
 *
 * Runs under the SAME env allowlist as the render (PATH + HOME): this spawns
 * `npx` in a directory the agent's sources were copied into, so the parent
 * environment — including any model/gateway credentials held by the eval
 * runner — must not be inherited.
 *
 * NOTE: no `--json` flag. `vgpu@0.2.0`'s doctor writes JSON to stdout by
 * default (`Usage: vgpu doctor [--no-render] [--pretty]`); passing `--json`
 * fails with "Unknown doctor option: --json" and exit 1.
 * @param {string} cwd
 */
function doctorEnv(cwd) {
  const fallback = { adapterName: null, adapterType: null, doctorVerdict: null };
  const result = spawnSyncish("npx", ["vgpu", "doctor"], cwd, {
    PATH: process.env.PATH ?? "",
    HOME: process.env.HOME ?? "",
  });
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

/**
 * Pull the JSON object out of a stream that may also carry warnings. Shared
 * with PR2's sandbox bootstrap, which parses the same command's output.
 * @param {string} text
 */
export function extractJson(text) {
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
 * @param {NodeJS.ProcessEnv} [env] defaults to the parent environment; pass an
 *   allowlist for anything that runs against agent-authored files.
 */
function spawnSyncish(command, args, cwd, env) {
  const result = spawnSync(command, args, {
    cwd,
    encoding: "utf8",
    env: env ?? process.env,
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

// Referenced by the JSDoc above; kept exported for tooling that inspects the
// verifier's layout without importing the whole module.
export const VERIFY_WORKSPACE_BASENAME = basename(VERIFY_DIR);
