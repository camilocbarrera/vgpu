# @vgpu/agent-evals

Agent evals for `vgpu`: measures how well a coding agent (with no vgpu-specific
prompting) can complete a real rendering task using the published `vgpu`
package.

The package has two layers:

- **Layer 1 — `tasks/` + `verify/` (this PR).** Task fixtures plus a
  deterministic, driver-agnostic grader. No knowledge of any agent framework.
- **Layer 2 — `agent/` + `evals/` (next PR).** The `eve`-based driver that runs
  an agent against a task and hands the resulting workspace to Layer 1.

## Intentionally outside the root test/typecheck configs

`apps/agent-evals` is a workspace member (`apps/*` is already a member glob),
but it is **deliberately absent** from the root `vitest.config.ts` `include`
list and from the root `tsconfig.json` `references`. Do not add it there.

The verifier's tests install a real `node_modules` into `.work/` and spawn
`node render.mjs`; coupling them to the library's own fast CI would make that CI
slow and flaky. Run this package's tests explicitly instead:

```bash
pnpm --filter @vgpu/agent-evals test
pnpm --filter @vgpu/agent-evals exec tsc --noEmit
```

## Node.js version

Layer 1 (`tasks/`, `verify/`) runs on this repo's own Node (22).

Running the eve-driven evals (Layer 2, added in the next PR) requires
**Node.js >= 24** and an AI Gateway credential (`AI_GATEWAY_API_KEY` or
`VERCEL_OIDC_TOKEN`). This repo pins Node 22 (`.nvmrc`, root `engines`) and that
does not change — so this host may well have neither Node 24 nor a credential,
which is expected. Use `pnpm agent-evals` from the repo root (added in PR3),
which preflights the Node version and fails fast with an actionable message.

## Layer boundary invariant

**Nothing under `tasks/` or `verify/` may import `eve`.** That is what keeps
Layer 1 promotable to a standalone `packages/eval-tasks` (for a second, non-eve
driver) via a mechanical `git mv`. Layer 2 depending on Layer 1 is fine; the
reverse direction is a boundary violation.

## Usage (Layer 1)

```bash
node verify/run-verify.mjs --workspace <dir> --task s1-clear-color [--out evidence.json] [--work-dir <dir>]
```

### What the verifier trusts

The agent's workspace contributes **source text and nothing else**. Per call the
verifier:

1. seeds a run directory from `tasks/<id>/fixture/` and **wipes its source tree
   first**, so no trial can inherit another trial's files (run directories are
   keyed by `runId`, so concurrent trials cannot collide either);
2. installs the **fixture's** dependencies with `--ignore-scripts`, so nothing
   the agent wrote runs on the host at install time and `node_modules` cannot be
   monkey-patched;
3. never copies the agent's `package.json` or any lockfile — a changed manifest
   is *reported* via the `packageJsonUnchanged` gate, never honoured;
4. deletes any pre-existing `out.png` unconditionally and re-renders in a fresh
   process with a `PATH`+`HOME`-only environment;
5. grades only the PNG that run produced.

So a hand-forged PNG cannot pass, a hijacked dependency cannot pass, and a
stale trial cannot pass. Each of those is a named regression test in
`verify/verify-task.test.ts`, and each of them **did** pass at some point during
review — the claims above are the fixes, not the original design.

### What the verifier does NOT give you

It is **not a security boundary.** The graded render executes agent-authored
code on the host, as your user. It defeats accidental and opportunistic cheating;
it does not contain a determined attacker.

It also grades **on the host**, not inside the pinned container the parent design
calls for. That is a deliberate skeleton-stage tradeoff: it is what makes Layer 1
runnable in any dev environment and in CI without Docker. The consequences you
must know about:

- `evidence.env` records `node`, `vgpu`, `gitSha` and the `vgpu doctor` adapter,
  but **no `imageDigest`** — results are only as comparable as the hosts that
  produced them. Fine for an exact-colour gate; **not** fine for the
  pixel-ratio/tolerance tasks planned later. Tracked as follow-up: grade inside
  a pinned image and stamp its digest into `evidence.env` before any task whose
  verdict depends on a ratio.
- `metrics.vgpuLoaded` is a **soft signal, never a gate**: `s1-clear-color` is
  passable with `pngjs` alone (six lines, no vgpu). The evidence records whether
  the graded render actually loaded the library under test so that a
  library-free "solution" is visible to a reader instead of silently scoring
  green. Making it a gate is a task-design decision for a later PR, not a
  verifier change.

## Known friction (recorded on purpose, do not "fix")

The getting-started snippet does not run out of the box after `pnpm add vgpu`
alone: `pngjs` must be declared explicitly, because it is a transitive
dependency and pnpm's strict `node_modules` layout does not hoist it. That is
why `pngjs` appears twice in this package — once in
`tasks/s1-clear-color/fixture/package.json` (for the code that runs *inside* the
graded workspace) and once in this package's own `devDependencies` (for the
host-side grader in `verify/grade-pixels.mjs`). Those two are deliberately
separate. This is a real product finding that the eval exists to surface; do not
paper over it beyond declaring the dependency.

## Future work — skill payload excision clause

Once the skill-distribution payload (Amendment 3 of the parent design) is
vendored into this package in a later PR, it **must** be excised of
`packages/ infra/ examples/ apps/ scripts/ docs/` and of any lockfiles before
being trusted as a fixture. A vendored payload that still contains the repo's
own sources would let an agent read the answer instead of discovering it. Do not
shortcut this later.
