# Contributing

## Prerequisites

- Node.js 22 (the workspace engine is `>=22 <23`)
- pnpm

## Agent evals

`apps/agent-evals/` measures how well a coding agent uses the `vgpu` package,
driven by [`eve`](https://eve.dev/). It requires Node.js >= 24 and an AI Gateway
credential (`AI_GATEWAY_API_KEY` or `VERCEL_OIDC_TOKEN`), independent of this
repo's own Node 22 requirement. Run it with:

```bash
pnpm agent-evals
```

The command preflights the Node version and exits with code `2` and an
actionable message if it is too old. Without a credential the evals skip (exit
0) rather than fail. See `apps/agent-evals/README.md` for details, the
credential recipe and current limitations.

## Making changes

If your PR changes published package behavior, add a changeset before opening it:

```bash
pnpm changeset
```

Choose each affected `@vgpu/*` package, select the appropriate semver bump (`patch`, `minor`, or `major`), and write a short summary. That summary becomes the changelog entry for the release.

## Bundle budgets

`pnpm bundle-check` enforces gzip budgets stored in each package's `package.json`. Budgets are tiered by audience:

- `"client"` (default when unclassified) — browser-facing entries. **Hard gate**: one byte over budget fails.
- `"tooling"` — loaders, the Node runtime, the CLI and package tarballs. **Soft gate**: over budget warns, and only fails past `vgpuBundleBudgetGrowthThreshold` (default 5%).

Classify with `vgpuBundleAudience` (package-wide) or `vgpuExportBundleAudiences` (per export subpath). Tarball budgets measure published dist bytes: `*.docs.md` files, sourcemap `sourcesContent` and the budget metadata itself are excluded, so documenting the API never competes with the size gate.

When growth is intentional, re-baseline instead of hand-editing numbers:

```bash
pnpm bundle-check --update   # budget = next 512 B multiple at least 512 B above measured
```

Run `pnpm build` first, since budgets are measured from `dist`.

## PR checklist

- [ ] Code changes to a published package include a `.changeset/*.md` file.
- [ ] Docs-only and CI-only PRs may skip a changeset.
- [ ] `pnpm typecheck` passes locally.
- [ ] `pnpm test:fast` passes locally.

## Releasing

Releases are cut by hand and published by CI. There is no bot and no automatic
version-packages PR: `.github/workflows/release.yml` runs on a **published GitHub
Release** whose tag starts with `v`, and that is the only thing that publishes to npm.

### 1. Version the packages (a PR of its own)

From an up-to-date `main`, on a release branch:

```bash
pnpm changeset status   # what will be bumped, and why
pnpm changeset version  # applies the bumps, writes CHANGELOGs, consumes .changeset/*.md
pnpm install            # refresh the lockfile with the new internal versions
```

`changeset version` rewrites every `package.json` version, folds each `.changeset/*.md`
into the matching `CHANGELOG.md`, and deletes the changesets it consumed. Review the
diff — the changelog text is the public release note — then open a PR titled something
like `chore(release): 0.2.0` and merge it to `main`.

Private packages (`@vgpu/cli`, the docs app) are versioned so they get changelog entries,
but they are never published. `@vgpu/cli` ships *inside* the `vgpu` tarball: `copy-cli.mjs`
writes a synthetic `package.json` stamped with `vgpu`'s version, so its own version field
is internal bookkeeping only.

### 2. Tag and publish

Once the versioning PR is on `main`, create a **GitHub Release** on that commit with the
tag `vX.Y.Z` matching the new `vgpu` version. Publishing the release triggers
`release.yml`, which checks out the tag, builds, runs the release gates (typecheck, the
test suites that run on a plain runner, and `pnpm bundle-check`) and then runs
`pnpm -r publish --access public` with npm Trusted Publishing (OIDC).

Only tags starting with `v` publish. Binary-asset releases such as `dawn-*` are ignored by
the workflow's `if:` gate.

If a gate fails the release publishes nothing: fix `main`, then delete and recreate the
release/tag.

### Release candidates

Run `pnpm changeset version` as usual, then append `-rc.N` by hand to the version of each
package you intend to publish. Tag the merge commit `vX.Y.Z-rc.N` and create the GitHub
Release with **Set as a pre-release** ticked: the workflow reads that flag and publishes
under the `next` dist-tag, so `latest` — and therefore a plain `npm install vgpu` — keeps
pointing at the last stable. Testers opt in with `npm install vgpu@next`.

Forgetting the checkbox is the one mistake that would push a release candidate to `latest`,
so the workflow refuses any tag containing a hyphen unless the pre-release flag is set.

Promoting an RC is a clean re-release: version the packages to stable `vX.Y.Z`, tag, and
publish a normal (non-pre-release) GitHub Release. Do **not** use `npm dist-tag add` — in a
monorepo it has to be repeated for every package, and one forgotten package leaves `latest`
silently pointing at a release candidate.

### npm Trusted Publishing

Publishing uses OIDC, not a token — there is no `NPM_TOKEN` secret. Each published package
has a Trusted Publisher configured on npm (provider GitHub Actions, owner `vercel-labs`,
repository `vgpu`, workflow `release.yml`, no environment). A **new** package has to be
published manually once before Trusted Publishing can be configured for it.

## Prereleases (future)

If you need a prerelease cycle later:

```bash
pnpm changeset pre enter beta
```

Work on a dedicated prerelease branch such as `next`, then exit prerelease mode when you are ready to return to stable releases:

```bash
pnpm changeset pre exit
```
