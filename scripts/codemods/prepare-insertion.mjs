// Codemod 3 (T04-19) — `prepare()` insertion, SEMI-automatic by design.
//
// What this tool automates and what it deliberately does NOT:
//
//   MECHANICAL (this file):
//     1. Find every renderable construction in the corpus — `draw(gpu, …)`, `effect(gpu, …)`,
//        `compute(gpu, …)`, `bundle(gpu, …)` — by TYPE (via the shared corpus checker), not by
//        callee spelling, so a locally-aliased import or a re-exported helper is still seen.
//     2. Find every encode site that consumes one — `p.draw(x)` / `p.bundles(x)` inside
//        `f.pass(target, …)`, and `f.compute(c, …)` — and pair each renderable with the TARGET
//        EXPRESSION it is encoded against. That pairing is the `PrepareRequest` the design asks
//        for: readiness is a property of a COMBINATION, never of an object (prepare.ts's own
//        doc-comment).
//     3. Find the natural setup boundary of the enclosing function: the statement that holds the
//        first `frame(…)` / `frameLoop(…)` / `renderOnce(…)` call. That is where a `prepare()`
//        would go.
//     4. Flag the cases where step 2 or 3 cannot answer honestly: a renderable created inside a
//        loop/`Array.from` (dynamic multi-pass), a target that is re-created on resize, a
//        renderable whose only encode site is in another file, an enclosing function that is not
//        `async` and whose callers would have to change.
//
//   CRITERION (NOT automated — a human decides, per example, and writes down the decision):
//     • WHICH combinations to prepare, and how to GROUP them (one `await prepare(gpu, [...])`
//       with an array, vs. several). Grouping is the whole ergonomic point of the array form.
//     • WHERE exactly the await lands when the setup is not linear — a resource re-created on
//       resize prepares where the object is BORN, not once at setup; an effect created lazily
//       prepares at its creation site or is a documented skip.
//     • Whether a combination is already covered by `renderOnce()` / `dispatchOnce()` (those
//       carry their own async readiness path, contract #20) and must therefore NOT get a
//       redundant `prepare()`.
//
// Hence: this script's `--dry-run` report is the INPUT to the human pass, and the branch's
// `T04-19-DECISIONS.md` is its OUTPUT. There is no `--apply`: an automatic writer would have to
// invent the grouping and the async propagation it cannot decide, and this train has already paid
// for one codemod that "looked right" and moved a call site into a shape nobody had read
// (T04-17's five unpinned resources). Insertions are made by hand, from this report, and the
// report is re-run afterwards as a CHECKER (`--verify`) that every combination it found is
// covered by some `prepare()` in the same file.
import path from "node:path";
import { readFileSync } from "node:fs";
import ts from "typescript";
import { createCorpusProgram } from "./lib/corpus-program.mjs";
import { getCorpusFiles } from "./lib/glob-corpus.mjs";
import { isDryRun, positionalArgs, printReport, reportEntry } from "./lib/report.mjs";

/**
 * Scope of T04-19: the *application* corpus — the 20 `apps/docs/examples` directories, the 15
 * `examples/` projects and the hero. `packages/<pkg>/tests` is deliberately OUT: those files are
 * unit tests of the library, many of them are the pinned SUBJECT of a readiness behaviour (they
 * assert what happens WITHOUT `prepare()`), and inserting `prepare()` there would delete the
 * very coverage T04-21's flip depends on. `experiments/**` is out for a different reason: it is
 * in no repo gate (T04-18's finding), so a change there is unverifiable by CI.
 */
const SCOPE_RE = /^(apps\/docs\/examples\/|apps\/docs\/components\/hero\/|examples\/[^/]+\/src\/)/u;

/** Factory names whose call produces something `prepare()` can take. */
const RENDERABLE_FACTORIES = new Set(["draw", "effect", "compute", "bundle"]);

/** Encode sites that consume a prepared combination. */
const PASS_DRAW = new Set(["draw", "bundles"]);

export function inScope(relPath) {
  return SCOPE_RE.test(relPath) && !/\.(test|spec)\.tsx?$/u.test(relPath);
}

/** Repo-relative path of a SourceFile. */
function rel(repoRoot, sourceFile) {
  return path.relative(repoRoot, sourceFile.fileName).split(path.sep).join("/");
}

function lineOf(sourceFile, node) {
  return sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile)).line + 1;
}

/** The nearest enclosing function-like node, or the SourceFile for top-level code. */
function enclosingFunction(node) {
  let cur = node.parent;
  while (cur) {
    if (
      ts.isFunctionDeclaration(cur) || ts.isFunctionExpression(cur) || ts.isArrowFunction(cur)
      || ts.isMethodDeclaration(cur) || ts.isConstructorDeclaration(cur) || ts.isSourceFile(cur)
    ) return cur;
    cur = cur.parent;
  }
  return undefined;
}

/** `true` when `node` sits inside a loop / `Array.from` callback / `.map()` callback. */
function dynamicContext(node) {
  let cur = node.parent;
  while (cur) {
    if (
      ts.isForStatement(cur) || ts.isForOfStatement(cur) || ts.isForInStatement(cur)
      || ts.isWhileStatement(cur) || ts.isDoStatement(cur)
    ) return "loop";
    if (ts.isCallExpression(cur)) {
      const callee = cur.expression;
      if (ts.isPropertyAccessExpression(callee) && (callee.name.text === "map" || callee.name.text === "from" || callee.name.text === "flatMap")) {
        return "iterated-callback";
      }
    }
    cur = cur.parent;
  }
  return undefined;
}

/** Text of an expression, whitespace-collapsed, for the report. */
function textOf(node) {
  return node.getText().replace(/\s+/gu, " ").slice(0, 120);
}

/** Resolved (alias-followed) symbol of a callee, or undefined. */
function calleeSymbol(checker, callee) {
  const target = ts.isPropertyAccessExpression(callee) ? callee.name : callee;
  let symbol = checker.getSymbolAtLocation(target);
  if (!symbol) return undefined;
  if (symbol.flags & ts.SymbolFlags.Alias) symbol = checker.getAliasedSymbol(symbol);
  return symbol;
}

/** `true` when a symbol is declared inside the vgpu library sources (not in the app corpus). */
function declaredInVgpu(symbol) {
  return (symbol.getDeclarations() ?? []).some((d) => /[/\\]packages[/\\](vgpu-api|vgpu|render|scene|core)[/\\]/u.test(d.getSourceFile().fileName));
}

/**
 * Is `call` a call to one of the vgpu renderable factories? Resolved through the checker's symbol
 * for the callee, so `import { effect as fx }` and `import * as api` + `api.effect(gpu, …)` both
 * resolve — the callee's *declaration* is what is matched, never the local spelling at the call
 * site. The declaration-file check is what keeps a corpus-local helper that happens to be named
 * `compute` out of the construction list.
 */
function renderableFactoryName(checker, call) {
  if (!ts.isCallExpression(call)) return undefined;
  const callee = call.expression;
  if (!ts.isIdentifier(callee) && !ts.isPropertyAccessExpression(callee)) return undefined;
  const symbol = calleeSymbol(checker, callee);
  if (!symbol) return undefined;
  const name = symbol.getName();
  if (!RENDERABLE_FACTORIES.has(name) || !declaredInVgpu(symbol)) return undefined;
  // `frame.compute(...)` resolves to the same NAME as the `compute(gpu, …)` factory but to a
  // different declaration: a method on `Frame`, not a free function. Only free functions build.
  const isMethod = (symbol.getDeclarations() ?? []).some((d) => ts.isMethodDeclaration(d) || ts.isMethodSignature(d) || ts.isPropertyDeclaration(d));
  return isMethod ? undefined : name;
}

/** `true` when `X.m(…)` resolves to a method declared on a vgpu class (Frame / FramePass / …). */
function isVgpuMethod(checker, callee, methodName) {
  if (!ts.isPropertyAccessExpression(callee) || callee.name.text !== methodName) return false;
  const symbol = calleeSymbol(checker, callee);
  return Boolean(symbol && declaredInVgpu(symbol));
}

/**
 * The target expression a `f.pass(…)` renders into. The pass options are an object literal in
 * every corpus site (`f.pass({ target: t, clear: … }, cb)`), so the honest `PrepareRequest` target
 * is that object's `target` property — NOT the options object itself. When the options are not a
 * literal (a variable, a spread), the whole expression is reported with a `.target` suffix so the
 * human sees that the pairing needs a look.
 */
function targetExpressionOf(passOptions) {
  if (ts.isObjectLiteralExpression(passOptions)) {
    for (const prop of passOptions.properties) {
      if (ts.isPropertyAssignment(prop) && ts.isIdentifier(prop.name) && prop.name.text === "target") return { text: textOf(prop.initializer), literal: true };
      if (ts.isShorthandPropertyAssignment(prop) && prop.name.text === "target") return { text: "target", literal: true };
    }
    return { text: "<pass options without a target property>", literal: false };
  }
  return { text: `${textOf(passOptions)}.target`, literal: false };
}

/** The variable/property a construction is bound to, if any. */
function bindingNameOf(call) {
  const parent = call.parent;
  if (ts.isVariableDeclaration(parent) && ts.isIdentifier(parent.name)) return parent.name.text;
  if (ts.isPropertyDeclaration(parent) && ts.isIdentifier(parent.name)) return `this.${parent.name.text}`;
  if (ts.isPropertyAssignment(parent) && ts.isIdentifier(parent.name)) return parent.name.text;
  if (ts.isBinaryExpression(parent) && parent.operatorToken.kind === ts.SyntaxKind.EqualsToken) return textOf(parent.left);
  if (ts.isReturnStatement(parent)) return "<returned>";
  if (ts.isPropertyAccessExpression(parent)) return `${textOf(parent)}`;
  return undefined;
}

/**
 * Walks one source file and collects: constructions, encode combinations, frame boundaries and
 * the readiness paths that make a `prepare()` redundant.
 */
export function scanFile(checker, sourceFile, relPath) {
  const constructions = [];
  const combinations = [];
  const frameBoundaries = [];
  const oneShots = [];
  const existingPrepares = [];
  const resizeHosts = [];

  /** Stack of `{ targetText, node }` for the `f.pass(target, cb)` we are currently inside. */
  const passStack = [];

  const visit = (node) => {
    if (ts.isCallExpression(node)) {
      const factory = renderableFactoryName(checker, node);
      if (factory) {
        constructions.push({
          factory,
          binding: bindingNameOf(node) ?? "<unbound>",
          line: lineOf(sourceFile, node),
          dynamic: dynamicContext(node),
          fn: enclosingFunction(node),
          node,
        });
      }

      const callee = node.expression;
      if (ts.isIdentifier(callee)) {
        const name = callee.text;
        if (name === "frame" || name === "frameLoop") {
          frameBoundaries.push({ kind: name, line: lineOf(sourceFile, node), node, fn: enclosingFunction(node) });
        }
        if (name === "renderOnce") {
          oneShots.push({ kind: "renderOnce", line: lineOf(sourceFile, node), text: textOf(node) });
        }
      }
      // `prepare()` is matched by RESOLVED SYMBOL, not by the local spelling: three corpus files
      // import it aliased (`prepare as prepareCombinations`, `api.prepare`), and a textual match
      // silently reported those files as unprepared — a false gap is as damaging as a missed one,
      // because it trains the reader to ignore the report.
      {
        const symbol = (ts.isIdentifier(callee) || ts.isPropertyAccessExpression(callee)) ? calleeSymbol(checker, callee) : undefined;
        if (symbol && symbol.getName() === "prepare" && declaredInVgpu(symbol)) {
          existingPrepares.push({ line: lineOf(sourceFile, node), text: node.getText() });
        }
      }
      if (ts.isPropertyAccessExpression(callee)) {
        const method = callee.name.text;
        if (method === "dispatchOnce" && isVgpuMethod(checker, callee, "dispatchOnce")) {
          oneShots.push({ kind: "dispatchOnce", line: lineOf(sourceFile, node), text: textOf(callee.expression) });
        }
        if (method === "pass" && isVgpuMethod(checker, callee, "pass")) {
          const [passOptions, body] = node.arguments;
          if (passOptions && body) {
            const target = targetExpressionOf(passOptions);
            // `f.pass(target, renderable)` — the SHORTHAND form, where the second argument is a
            // Draw/Effect instead of a callback (`fft-ocean-surface` uses it for its composite).
            // It is exactly as much a `{ draw, target }` combination as the callback form; missing
            // it was a real under-report, not a harmless one, so it is matched explicitly rather
            // than left to the `p.draw()` walk that never fires for it.
            if (!ts.isArrowFunction(body) && !ts.isFunctionExpression(body)) {
              combinations.push({
                kind: "draw",
                renderable: textOf(body),
                target: target.text,
                targetLiteral: target.literal,
                line: lineOf(sourceFile, node),
                fn: enclosingFunction(node),
              });
            }
            passStack.push({ target, line: lineOf(sourceFile, node) });
            ts.forEachChild(node, visit);
            passStack.pop();
            return;
          }
        }
        if (method === "compute" && isVgpuMethod(checker, callee, "compute") && node.arguments.length >= 1) {
          // `f.compute(c, …)` — a compute combination has no target.
          combinations.push({
            kind: "compute",
            renderable: textOf(node.arguments[0]),
            target: undefined,
            targetLiteral: true,
            line: lineOf(sourceFile, node),
            fn: enclosingFunction(node),
          });
        }
        if (PASS_DRAW.has(method) && passStack.length && isVgpuMethod(checker, callee, method)) {
          const { target } = passStack[passStack.length - 1];
          for (const arg of node.arguments) {
            if (ts.isObjectLiteralExpression(arg)) continue; // per-call options, not a renderable
            combinations.push({
              kind: method === "bundles" ? "bundle" : "draw",
              renderable: textOf(arg),
              target: method === "bundles" ? undefined : target.text,
              targetLiteral: method === "bundles" ? true : target.literal,
              line: lineOf(sourceFile, node),
              fn: enclosingFunction(node),
            });
          }
        }
      }
    }

    if ((ts.isMethodDeclaration(node) || ts.isPropertyDeclaration(node) || ts.isFunctionDeclaration(node))
      && node.name && ts.isIdentifier(node.name) && /^(resize|onResize|handleResize|setSize)$/u.test(node.name.text)) {
      resizeHosts.push({ name: node.name.text, line: lineOf(sourceFile, node) });
    }

    ts.forEachChild(node, visit);
  };

  ts.forEachChild(sourceFile, visit);

  return { relPath, constructions, combinations, frameBoundaries, oneShots, existingPrepares, resizeHosts };
}

/**
 * De-duplicated `{renderable, target}` pairs, in first-encounter order — the shape a
 * `prepare(gpu, [...])` array mirrors.
 */
export function requestsOf(scan) {
  const seen = new Map();
  for (const combo of scan.combinations) {
    const key = `${combo.kind}|${combo.renderable}|${combo.target ?? ""}`;
    if (!seen.has(key)) seen.set(key, combo);
  }
  return [...seen.values()];
}

/**
 * Classification of a file for the dry-run report. `auto` means: every combination pairs a
 * statically-named renderable with a statically-named target, there is exactly one frame
 * boundary, and nothing is constructed in a loop — a mechanical insertion would be correct.
 * Everything else is `manual-reviewed`: the report says WHY, and the human decision is recorded
 * in `T04-19-DECISIONS.md`.
 */
export function classify(scan) {
  const reasons = [];
  if (scan.constructions.some((c) => c.dynamic)) reasons.push("dynamic-construction");
  if (scan.frameBoundaries.length === 0 && scan.combinations.length > 0) reasons.push("no-frame-boundary");
  if (scan.frameBoundaries.length > 1) reasons.push("multiple-frame-boundaries");
  if (scan.combinations.length === 0 && scan.constructions.length > 0) reasons.push("encode-site-in-another-file");
  if (scan.oneShots.length) reasons.push("one-shot-readiness-present");
  if (scan.resizeHosts.length) reasons.push("resize-recreates-resources");
  if (scan.combinations.some((c) => /[[\].]/u.test(c.renderable))) reasons.push("computed-renderable-expression");
  return { classification: reasons.length ? "manual-reviewed" : "auto", reasons };
}

/** Builds the full report for the scoped corpus. */
export function buildReport({ repoRoot = process.cwd(), files } = {}) {
  const corpus = (files ?? getCorpusFiles(repoRoot)).filter(inScope);
  const { checker, sourceFileFor } = createCorpusProgram(repoRoot, { corpus });
  const entries = [];
  for (const relPath of corpus) {
    const sourceFile = sourceFileFor(relPath);
    if (!sourceFile) {
      entries.push(reportEntry({ file: relPath, line: 1, before: "", after: "", classification: "unresolved-source-file" }));
      continue;
    }
    const scan = scanFile(checker, sourceFile, relPath);
    if (!scan.constructions.length && !scan.combinations.length) continue;
    const { classification, reasons } = classify(scan);
    const requests = requestsOf(scan);
    entries.push({
      ...reportEntry({
        file: relPath,
        line: scan.frameBoundaries[0]?.line ?? scan.constructions[0]?.line ?? 1,
        before: scan.existingPrepares.length ? `${scan.existingPrepares.length} existing prepare() call(s)` : "no prepare()",
        after: requests.length
          ? `await prepare(gpu, [${requests.map((r) => (r.kind === "draw" ? `{ draw: ${r.renderable}, target: ${r.target} }` : `{ ${r.kind}: ${r.renderable} }`)).join(", ")}])`
          : "(no encode site in this file)",
        classification,
      }),
      reasons,
      constructions: scan.constructions.map((c) => ({ factory: c.factory, binding: c.binding, line: c.line, dynamic: c.dynamic ?? null })),
      combinations: requests.map((r) => ({ kind: r.kind, renderable: r.renderable, target: r.target ?? null, line: r.line })),
      frameBoundaries: scan.frameBoundaries.map((f) => ({ kind: f.kind, line: f.line })),
      oneShots: scan.oneShots,
      resizeHosts: scan.resizeHosts,
      existingPrepares: scan.existingPrepares,
    });
  }
  return entries;
}

/**
 * Combinations whose `prepare()` is real but lives behind an INDIRECTION the textual matcher below
 * cannot follow: the request names a local (`background`, `g.present`, `drawable`) inside a
 * `prewarm()` / `createScene()` / `prepareScene()` helper, while the encode site names the same
 * object through the struct that helper returned (`scene.background`, `scene.bundle`). Every entry
 * was checked BY HAND against the prepare call it points at — this table is the written form of
 * that check, not a mute suppression list. A gap that is not in here is a real gap.
 *
 * Keyed `file` -> renderable text at the encode site -> where the preparation actually happens.
 */
export const COVERED_INDIRECTLY = {
  "apps/docs/examples/agent-radiance-cascades/simulation.ts": {
    "pass.effect": "prepareScene() prepares every effect by iterating scene.effects.{jfaSteps,cascade}; `pass.effect` is an element of the pass list built from those same arrays.",
  },
  "apps/docs/examples/radiance-cascades/simulation.ts": {
    "pass.effect": "prepareScene() prepares every effect by iterating scene.effects.{jfaSteps,cascade}; `pass.effect` is an element of the pass list built from those same arrays.",
  },
  "apps/docs/examples/batch-rendering/renderer.ts": {
    "scene.bundle": "createScene() prepares { bundle: recorded } AFTER bundle() records it, which is the only edge out of pending-pipelines; it warms the four recorded draws and encodes the native bundle.",
  },
  "apps/docs/examples/instanced-rendering/renderer.ts": {
    "scene.bundle": "createScene() prepares { bundle: recorded } after bundle() records it; that request subsumes the drawable and is awaited before the scene is published to the loop.",
  },
  "apps/docs/examples/environment-map/renderer.ts": {
    "scene.cube": "createScene() prepares the local `cube` against `hdr` and returns it as `scene.cube`.",
    "scene.present": "createScene() prepares the local `present` against `output` and returns it as `scene.present`.",
  },
  "apps/docs/examples/fft-ocean-surface/renderer.ts": {
    "scene!.skydome": "The frame-loop arm of the same pair prepared two lines above; `scene!.x` and `scene.x` are one object, the `!` is a narrowing artifact.",
    "scene!.ocean": "Same as skydome: prepared in the initialize() call directly above the frameLoop.",
  },
  "apps/docs/examples/fft-ocean/renderer.ts": {
    "stage.effect": "prewarm() prepares `...g.ifft.map((s) => ({ draw: s.effect, target: s.output }))`; `stage` iterates that same g.ifft array.",
  },
  "apps/docs/examples/fluid/simulation.ts": {
    "fluid.bundles![fluid.step & 1]": "Both parities are prepared as { bundle } while held in a local, and fluid.bundles is only assigned after that prepare resolves, so the index can only ever select a ready bundle.",
  },
  "apps/docs/examples/transmission/renderer.ts": {
    "scene.background": "createScene() prepares the local `background` against targets.hdr and returns it on the scene struct.",
    "scene.floor": "createScene() prepares the local `floor` against targets.hdr.",
    "scene.glass": "createScene() prepares the local `glass` against targets.hdr.",
    "scene.backface": "createScene() prepares the local `backface` against targets.backface.",
    "scene.present": "createScene() prepares the local `present` against `output`.",
    "scene.blurs[index].horizontal": "createScene() prepares `...blurs.flatMap(...)` over the same pyramid array this indexes.",
    "scene.blurs[index].vertical": "createScene() prepares `...blurs.flatMap(...)` over the same pyramid array this indexes.",
  },
  "apps/docs/examples/triangle-led-front/scene-renderer.ts": {
    "currentParts.raycastBundle": "prewarm() prepares { bundle: parts.raycastBundle }, and renderFrame() is gated on preparedGeneration === sceneGeneration so a rebuild() that swaps in a new raycastBundle cannot be replayed before its prewarm resolves.",
    "currentTheme === 'light' ? floorBundles.light : floorBundles.dark": "prewarm() eagerly calls recordFloorBundles() and prepares BOTH arms as { bundle }; renderFrame() is gated on the readiness generation, so a resize that re-records cannot be replayed before its prewarm resolves.",
  },
};

/**
 * `--verify` mode: every combination the scanner found must be covered by a `prepare()` (or by a
 * one-shot readiness path) somewhere in the same file. This is the mechanical half of the
 * completeness criterion "under a `throw` default, no example hits VGPU-PIPELINE-PENDING"; the
 * behavioural half is `apps/docs/examples/**` + `examples/**` running under a forced-`throw` gpu
 * (see `packages/vgpu-api/tests/prepare-corpus-coverage.test.ts`).
 *
 * Coverage is checked TEXTUALLY against the prepare() call's argument text, because the renderable
 * a `p.draw(x)` names and the one a `prepare()` names are the same source expression when the
 * insertion is right — a rename between the two is exactly the mistake worth catching.
 */
export function verify({ repoRoot = process.cwd(), files } = {}) {
  const corpus = (files ?? getCorpusFiles(repoRoot)).filter(inScope);
  const { checker, sourceFileFor } = createCorpusProgram(repoRoot, { corpus });
  const gaps = [];
  for (const relPath of corpus) {
    const sourceFile = sourceFileFor(relPath);
    if (!sourceFile) continue;
    const scan = scanFile(checker, sourceFile, relPath);
    if (!scan.combinations.length) continue;
    const prepareText = scan.existingPrepares.map((p) => p.text).join("\n").replace(/\s+/gu, "");
    for (const combo of requestsOf(scan)) {
      const name = combo.renderable.replace(/\s+/gu, "");
      const covered = prepareText.includes(name)
        || scan.oneShots.some((o) => o.text.includes(name));
      const indirect = COVERED_INDIRECTLY[relPath]?.[combo.renderable];
      if (!covered && !indirect) gaps.push({ file: relPath, line: combo.line, kind: combo.kind, renderable: combo.renderable, target: combo.target ?? null });
    }
  }
  return gaps;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const args = positionalArgs();
  const repoRoot = process.cwd();
  if (args.includes("--verify")) {
    const gaps = verify({ repoRoot });
    printReport(gaps);
    process.exitCode = gaps.length ? 1 : 0;
  } else if (isDryRun() || args.length === 0) {
    printReport(buildReport({ repoRoot }));
  } else {
    process.stderr.write("prepare-insertion: this codemod has no --apply mode (see the header comment). Use --dry-run or --verify.\n");
    process.exitCode = 2;
  }
}
