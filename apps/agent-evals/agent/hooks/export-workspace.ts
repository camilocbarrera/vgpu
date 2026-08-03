import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { defineHook } from "eve/hooks";
import { snapshotTarPath } from "./snapshot-path.ts";

/**
 * Out-of-band workspace export.
 *
 * Every completed turn, this hook tars `/workspace` (minus `node_modules/` and
 * `.git/`) straight out of the sandbox and writes it to the HOST at
 * `.work/snapshots/<sessionId>/workspace.tar`. The evals then grade that tar
 * with Layer 1's verifier. Nothing the agent *says* is ever used as evidence —
 * only bytes we extracted ourselves.
 *
 * WHY A HOOK AND NOT A CHANNEL: the plan for this PR specified an
 * `agent/channels/harness.ts` with a `GET /export` route calling
 * `ctx.getSandbox()`. That API does not exist: a channel route handler's
 * second argument is `RouteHandlerArgs` (`send`, `resolveActiveSession`,
 * `cancel`, `reset`, `getSession`, `receive`, `params`, `waitUntil`,
 * `requestIp`) — verified against both the local eve checkout (0.29.2) and the
 * pinned registry build (0.29.5, `dist/src/channel/routes.d.ts`). `getSandbox()`
 * lives on `SessionContext`, which `HookContext` extends
 * (`dist/src/public/definitions/callback-context.d.ts`), so an authored hook is
 * the supported way to reach the sandbox out of band. This is NOT the retired
 * `docker cp` fallback: it is backend-agnostic (plain `tar` over
 * `sandbox.run`), so `VGPU_EVALS_SANDBOX=vercel` keeps working unchanged.
 *
 * Assumption that `h0-harness-export.eval.ts` exists to prove or disprove on a
 * real run: the hook executes in the eve runtime process, i.e. on the same host
 * (and same filesystem) as the eval that reads the tar. That holds for a local
 * `eve eval`; it would not hold against a remotely deployed target.
 */
export default defineHook({
  events: {
    "turn.completed": async (_event, ctx) => {
      const sandbox = await ctx.getSandbox();

      // base64 because `run()` hands back stdout as a string: raw tar bytes
      // would not survive UTF-8 decoding. `-w0` keeps it on one line.
      const exported = await sandbox.run({
        command:
          "tar -cf - --exclude=./node_modules --exclude=./.git -C /workspace . | base64 -w0",
      });
      if (exported.exitCode !== 0) {
        throw new Error(`export-workspace: tar failed (${exported.exitCode}): ${exported.stderr}`);
      }

      const tarPath = snapshotTarPath(ctx.session.id);
      mkdirSync(dirname(tarPath), { recursive: true });
      writeFileSync(tarPath, Buffer.from(exported.stdout.trim(), "base64"));
    },
  },
});
