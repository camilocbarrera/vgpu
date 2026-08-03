import type { EveEvalContext } from "eve/evals";

/**
 * MUST be the first statement inside every `test(t)` in this package, `h0`
 * included. Without it a credential-less run does not skip cleanly: it fails
 * with a gateway auth error that reads like a harness bug and sends whoever
 * runs it debugging the wrong thing.
 *
 * `t.skip()` returns `never` — it aborts the test body before any model call
 * or sandbox provisioning happens.
 */
export function requireCredentials(t: EveEvalContext): void {
  const backend = process.env.VGPU_EVALS_SANDBOX ?? "docker";

  if (!process.env.AI_GATEWAY_API_KEY && !process.env.VERCEL_OIDC_TOKEN) {
    t.skip("no AI Gateway credentials (AI_GATEWAY_API_KEY | VERCEL_OIDC_TOKEN)");
  }
  if (backend === "vercel" && !process.env.VERCEL_OIDC_TOKEN && !process.env.VERCEL_TOKEN) {
    t.skip("VGPU_EVALS_SANDBOX=vercel requires Vercel credentials");
  }
}
