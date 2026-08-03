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

The verifier never trusts what it finds in `<dir>`: it copies the source files
into a clean, separately-installed workspace, **deletes any pre-existing
`out.png`**, re-runs `node render.mjs` in a fresh process, and grades the pixels
of the PNG that run produced. A hand-forged output PNG therefore cannot pass.

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
