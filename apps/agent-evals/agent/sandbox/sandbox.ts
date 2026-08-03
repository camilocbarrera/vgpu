import { defineSandbox } from "eve/sandbox";
import { evalSandboxBackend } from "./backend.ts";

/**
 * Sandbox for the agent-evals suite.
 *
 * `agent/sandbox/workspace/` is seeded into `/workspace` by eve. Its contents
 * are a materialised copy of `tasks/s1-clear-color/fixture/` — that directory
 * is the source of truth; re-sync this copy whenever it changes (eve's
 * `workspace/` is a static tree, not a build step, so it cannot be a symlink).
 *
 * `bootstrap` deliberately fails loudly when `vgpu doctor` is not healthy.
 * Whether a GPU/software renderer works inside the sandbox image is exactly
 * the kind of thing that would otherwise show up as a per-trial "the model
 * couldn't do it" flake. Gating it here turns it into one cached, obvious
 * infra failure instead — and the pixel verdict itself is still computed on
 * the host by Layer 1, so a broken sandbox can block a run but never corrupt
 * a result.
 *
 * Keep every command here backend-agnostic (plain shell, no docker flags, no
 * host paths) so switching to `VGPU_EVALS_SANDBOX=vercel` stays a one-line
 * env change.
 */
export default defineSandbox({
  backend: evalSandboxBackend(),
  // Bump on any bootstrap change so the cached template is rebuilt.
  revalidationKey: () => "agent-evals-s1-bootstrap-v1",
  async bootstrap({ use }) {
    const sandbox = await use();

    // The Vercel Sandbox spike found that the AL2023 runtimes need Mesa's
    // Vulkan driver before vgpu can fall back to the CPU renderer. This is a
    // no-op on images that already have it, and `|| true` keeps images without
    // dnf (e.g. the Debian-based eve default) working — the doctor gate below
    // is what actually decides whether the environment is usable.
    await sandbox.run({
      command: "command -v dnf >/dev/null && sudo dnf install -y mesa-vulkan-drivers vulkan-loader || true",
    });

    const install = await sandbox.run({ command: "pnpm install" });
    if (install.exitCode !== 0) {
      throw new Error(`bootstrap: pnpm install failed: ${install.stderr}`);
    }

    // No `--json`: vgpu@0.2.0's doctor already writes JSON to stdout
    // (`Usage: vgpu doctor [--no-render] [--pretty]`), and `--json` exits 1
    // with "Unknown doctor option".
    const doctor = await sandbox.run({ command: "npx vgpu doctor" });
    let verdict: unknown;
    try {
      verdict = (JSON.parse(doctor.stdout) as { verdict?: unknown }).verdict;
    } catch {
      throw new Error(`bootstrap: could not parse vgpu doctor output: ${doctor.stdout}`);
    }
    if (verdict !== "healthy") {
      throw new Error(
        `bootstrap: vgpu doctor verdict is "${String(verdict)}", expected "healthy": ${doctor.stdout}`,
      );
    }
  },
});
