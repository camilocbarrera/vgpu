import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { type BashCall, docsUsage, sourceFiles } from "./transcript.ts";

/**
 * Deterministic signals shared by `n2-ship-hero` and its control
 * `n3-explore-hero`. Both start from the SAME seed, so the same code reads
 * "did the agent run the pre-PR checklist" in both — and the pair only means
 * something if it is the same code.
 *
 * Each signal is a literal, verifiable fact about the files or the commands,
 * never a semantic reading of code (those go to the judges). The seed is
 * deliberately built so every signal here is false before the agent touches
 * it: no `compile(`, no `bundle(`, three `rgba32float` targets, no PR.md.
 */

/** The guide the finishing agent is supposed to find. */
export const SHIPPING_GUIDE_SLUG = "shipping-to-production";

const APP_EXTENSIONS = new Set([".ts", ".tsx", ".mjs", ".js", ".wgsl"]);

/** The seed's HDR format, counted rather than searched so a partial downgrade still registers. */
const SEED_FORMAT = "rgba32float";

export interface HeroSignals {
  /** `.compile(` / `.compileSync(` appeared in shipped app code (pipeline pre-warm). */
  prewarmAdded: boolean;
  /** `bundle(` appeared in shipped app code (render bundles). */
  bundlesAdded: boolean;
  /** Fewer `rgba32float` mentions than the seed: a target format was changed. */
  formatChanged: boolean;
  /** Any seed `.wgsl` file differs or is gone. */
  shadersChanged: boolean;
  /** Any seed `.ts`/`.tsx` under app/ differs or is gone. */
  appCodeChanged: boolean;
  /** A non-empty PR.md exists in the shipped workspace. */
  prMdWritten: boolean;
  /** PR.md content, empty string when absent. */
  prMd: string;
  /** PR.md carries at least one unit-bearing number (ms, fps, KB, %, bytes). */
  prHasMeasurement: boolean;
  /** PR.md talks about texture formats or resolution (the seed's visible knobs). */
  prMentionsVisualKnobs: boolean;
  /** `vgpu check` was run at least once. */
  ranVgpuCheck: boolean;
  /** `next build` was run by the agent itself. */
  ranNextBuild: boolean;
  /** The agent wrote timing code: `performance.now(`, `timer(`, or `timestamp-query`. */
  wroteMeasurement: boolean;
  /** `vgpu docs` usage counters and guide discovery. */
  docs: { calls: number; invocations: number; openedShippingGuide: boolean; shippingGuideSurfaced: boolean };
  /** Union of the checklist's visible footprints; the control task expects this to be false. */
  checklistFootprint: boolean;
}

export function heroSignals(input: {
  seedDir: string;
  shippedDir: string;
  calls: readonly BashCall[];
  written: string;
}): HeroSignals {
  const seed = sourceFiles(join(input.seedDir, "app"), APP_EXTENSIONS);
  const shipped = sourceFiles(join(input.shippedDir, "app"), APP_EXTENSIONS);
  const shippedByPath = new Map(shipped.map((file) => [file.path, file.content]));
  const shippedCode = shipped.map((file) => file.content).join("\n");
  const seedCode = seed.map((file) => file.content).join("\n");

  const changed = (predicate: (path: string) => boolean) =>
    seed.filter((file) => predicate(file.path)).some((file) => shippedByPath.get(file.path) !== file.content);

  const count = (haystack: string, needle: string) => haystack.split(needle).length - 1;
  const commands = input.calls.map((call) => call.command).join("\n");
  const prPath = join(input.shippedDir, "PR.md");
  const prMd = existsSync(prPath) ? readFileSync(prPath, "utf8") : "";
  const docs = docsUsage(input.calls);

  const prewarmAdded = /\.compile(?:Sync)?\(/.test(shippedCode) && !/\.compile(?:Sync)?\(/.test(seedCode);
  const bundlesAdded = /\bbundle\(/.test(shippedCode) && !/\bbundle\(/.test(seedCode);
  const formatChanged = count(shippedCode, SEED_FORMAT) < count(seedCode, SEED_FORMAT);
  const prMdWritten = prMd.trim().length > 0;

  return {
    prewarmAdded,
    bundlesAdded,
    formatChanged,
    shadersChanged: changed((path) => path.endsWith(".wgsl")),
    appCodeChanged: changed((path) => /\.tsx?$/.test(path)),
    prMdWritten,
    prMd,
    prHasMeasurement: /\b\d+(?:\.\d+)?\s*(?:ms|fps|kib|kb|mb|bytes|%)\b/i.test(prMd),
    prMentionsVisualKnobs: /rgba16float|rgba32float|format|resolution|half[- ]res|dpr|low tier|low-tier|quality tier/i.test(prMd),
    ranVgpuCheck: /vgpu\s+check\b/.test(commands),
    ranNextBuild: /\bnext\s+build\b/.test(commands),
    wroteMeasurement: /performance\.now\(|\btimer\(|timestamp-query/.test(input.written),
    docs: {
      calls: docs.docsCalls.length,
      invocations: docs.invocations,
      openedShippingGuide: docs.opened(SHIPPING_GUIDE_SLUG),
      shippingGuideSurfaced: docs.surfaced(SHIPPING_GUIDE_SLUG),
    },
    checklistFootprint: prewarmAdded || bundlesAdded || formatChanged || prMdWritten,
  };
}

/** One line per signal, for the eval log. */
export function logHeroSignals(log: (line: string) => void, signals: HeroSignals): void {
  const flags: [string, boolean][] = [
    ["prewarm_added", signals.prewarmAdded],
    ["bundles_added", signals.bundlesAdded],
    ["format_changed", signals.formatChanged],
    ["shaders_changed", signals.shadersChanged],
    ["app_code_changed", signals.appCodeChanged],
    ["pr_md_written", signals.prMdWritten],
    ["pr_has_measurement", signals.prHasMeasurement],
    ["pr_mentions_visual_knobs", signals.prMentionsVisualKnobs],
    ["ran_vgpu_check", signals.ranVgpuCheck],
    ["ran_next_build", signals.ranNextBuild],
    ["wrote_measurement", signals.wroteMeasurement],
    ["opened_shipping_guide", signals.docs.openedShippingGuide],
    ["shipping_guide_surfaced", signals.docs.shippingGuideSurfaced],
    ["checklist_footprint", signals.checklistFootprint],
  ];
  for (const [name, value] of flags) log(`funnel: ${name}=${value}`);
  log(`funnel: docs_cmd_count=${signals.docs.calls}`);
  log(`funnel: docs_invocations_total=${signals.docs.invocations}`);
}
