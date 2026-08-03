import type { SandboxBackend } from "eve/sandbox";
import { docker } from "eve/sandbox/docker";
import { vercel } from "eve/sandbox/vercel";

/**
 * The ONLY place in this repo allowed to construct a sandbox backend.
 *
 * Rules (enforced in review, not just by this comment):
 * 1. No other file may import `eve/sandbox/docker`, `eve/sandbox/vercel` or
 *    `eve/sandbox/microsandbox`.
 * 2. Nothing may use `defaultBackend()` from `eve/sandbox`. Its fallback
 *    cascade can silently degrade to `just-bash`, which has no real binaries —
 *    that turns an infra problem into a fake "the agent failed" result.
 * 3. An unknown `VGPU_EVALS_SANDBOX` value throws loudly at startup. Never
 *    fall back silently.
 */
export function evalSandboxBackend(): SandboxBackend {
  const kind = process.env.VGPU_EVALS_SANDBOX ?? "docker"; // provisional default

  if (kind === "vercel") {
    // Landing spot for the Vercel Sandbox spike's output. The spike
    // (see the runbook linked from README.md) established the golden path:
    // runtime `node22`/`node24`, 1-2 vCPUs, and one
    // `sudo dnf install -y mesa-vulkan-drivers vulkan-loader` before
    // `vgpu doctor` reports healthy — that dnf step belongs in `bootstrap`
    // (sandbox.ts), not here, so it stays backend-agnostic. Only the
    // resource/runtime options belong in this object, and they are
    // deliberately not guessed here: fill them in when the spike lands.
    return vercel({
      /* pending the Vercel Sandbox spike — see README.md */
    });
  }

  if (kind === "docker") {
    // Image intentionally unpinned for the walking skeleton; pin by digest in
    // a later PR once the suite grows past s1.
    return docker({});
  }

  throw new Error(`VGPU_EVALS_SANDBOX invalid: ${kind} (docker|vercel)`);
}
