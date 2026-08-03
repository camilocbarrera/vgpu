import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { defineEval } from "eve/evals";
import { equals, satisfies } from "eve/evals/expect";
import { snapshotDir, snapshotTarPath } from "../agent/hooks/snapshot-path.ts";
import { requireCredentials } from "./lib/credentials-guard.ts";

const PACKAGE_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const SEED_DIR = join(PACKAGE_ROOT, "agent", "sandbox", "workspace");

/**
 * Infra self-test. No rendering, no GPU, no vgpu vocabulary: it only proves
 * that the harness can pull the sandbox's `/workspace` back to the host
 * byte-for-byte. This is the first eval that must go green on a real run —
 * if it fails, every `s1` result after it is meaningless.
 *
 * It is also the runtime check on the one assumption this PR could not verify
 * statically: that the export hook (which reaches the sandbox via
 * `ctx.getSandbox()`, the only supported way — channel route handlers have no
 * such accessor) runs on the same host filesystem as this eval.
 */
export default defineEval({
  description: "h0: the harness can export the sandbox workspace to the host, byte-exact.",

  async test(t) {
    requireCredentials(t);

    // A deliberately trivial turn: it provisions the sandbox and produces the
    // `turn.completed` the export hook listens for. Nothing vgpu-specific.
    const turn = await t.send("Reply with the single word: ready.");
    t.succeeded();

    const sessionId = turn.sessionId;
    const tarPath = snapshotTarPath(sessionId);
    await t.require(
      existsSync(tarPath),
      satisfies((value: unknown) => value === true, `export hook wrote ${tarPath}`),
    );

    const extracted = join(snapshotDir(sessionId), "h0-extract");
    rmSync(extracted, { force: true, recursive: true });
    mkdirSync(extracted, { recursive: true });
    execFileSync("tar", ["-xf", tarPath, "-C", extracted]);

    for (const name of ["package.json", "render.mjs"]) {
      const exported = readFileSync(join(extracted, name));
      const seeded = readFileSync(join(SEED_DIR, name));
      t.check(exported.equals(seeded), equals(true)).gate().label(`${name} round-trips byte-exact`);
    }
  },
});
