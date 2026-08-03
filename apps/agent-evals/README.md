# @vgpu/agent-evals

Agent evals for `vgpu`: measures how well a coding agent (with no vgpu-specific
prompting) can complete a real rendering task using the published `vgpu`
package.

The package has two layers:

- **Layer 1 — `tasks/` + `verify/`.** Task fixtures plus a deterministic,
  driver-agnostic grader. No knowledge of any agent framework.
- **Layer 2 — `agent/` + `evals/`.** The `eve`-based driver that runs an agent
  against a task and hands the resulting workspace to Layer 1.

## Running the evals (Layer 2)

```bash
pnpm agent-evals                       # from the repo root (preflights Node >= 24)
pnpm --filter @vgpu/agent-evals exec eve eval evals/h0-harness-export.eval.ts
```

Environment:

| Variable | Default | Meaning |
| --- | --- | --- |
| `AI_GATEWAY_API_KEY` / `VERCEL_OIDC_TOKEN` | — | required; without either, every eval **skips** (never fails) |
| `VGPU_EVALS_MODEL` | `anthropic/claude-sonnet-5` | model under test |
| `VGPU_EVALS_SANDBOX` | `docker` | `docker` or `vercel`; anything else throws at startup |
| `VGPU_EVALS_WORK_DIR` | `<package>/.work` | where snapshots and evidence land |

Run `h0-harness-export` first: it is the infra self-test for the workspace
export. An `s1` result recorded while `h0` is red is not trustworthy.

### Credentials

Local development — one token covers both the sandbox and the model gateway:

```bash
npx vercel link --yes --scope vercel-labs --project vgpu
npx vercel env pull                 # writes .env.local with VERCEL_OIDC_TOKEN
node --env-file .env.local ../../scripts/agent-evals.mjs
```

`VERCEL_OIDC_TOKEN` authenticates `@vercel/sandbox` **and** the AI Gateway (eve
accepts it in place of `AI_GATEWAY_API_KEY`). It expires after 12 hours; re-run
`vercel env pull`.

> **Footgun:** `vercel env pull` writes the value **wrapped in double quotes**.
> Extracting it with `grep`/`cut` keeps the quotes and the gateway answers
> `403 invalidToken`, which looks exactly like a permissions problem. Always
> load the file with a real dotenv reader (`node --env-file .env.local`), never
> with hand-rolled shell parsing.

CI / non-interactive: a 12-hour OIDC token is useless for a scheduled run. Use
`VERCEL_TOKEN` + `VERCEL_TEAM_ID` + `VERCEL_PROJECT_ID` for the sandbox and a
separate `AI_GATEWAY_API_KEY` for the gateway — a Vercel access token does
**not** work as a Gateway bearer token.

### Sandbox backend selection

`agent/sandbox/backend.ts` is the only file in the repo allowed to construct a
`SandboxBackend`, and nothing may call `defaultBackend()` (its cascade can
silently degrade to `just-bash`, which has no real binaries — an infra problem
would then look like an agent failure).

The Vercel Sandbox backend is already selectable (`VGPU_EVALS_SANDBOX=vercel`);
wiring its concrete `vercel({...})` options is pending the sandbox spike — see
`backend.ts`. The spike verified the golden path: **x86_64 only**, runtime
`node22`/`node24`, one `sudo dnf install -y mesa-vulkan-drivers vulkan-loader`,
after which `vgpu doctor` reports healthy in ~22-30 s. That dnf step is already
in `bootstrap` as a backend-agnostic, no-op-if-absent shell step, so only the
runtime/resource options remain to be filled in.

### How the workspace gets out of the sandbox

`agent/hooks/export-workspace.ts` tars `/workspace` out of the sandbox on every
`turn.completed` and writes it to `.work/snapshots/<sessionId>/workspace.tar` on
the host. The evals grade that tar; nothing the agent *says* is ever evidence.

This is a hook rather than the originally-planned `GET /export` channel route
because a channel route handler's context (`RouteHandlerArgs`) has **no**
`getSandbox()` — verified against `eve@0.29.2` (local checkout) and `eve@0.29.5`
(the pinned build). `getSandbox()` lives on `SessionContext`, which
`HookContext` extends. The hook stays backend-agnostic (plain `tar` over
`sandbox.run`), so it is explicitly *not* the retired `docker cp` fallback.

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

Running the eve-driven evals (Layer 2) requires
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
