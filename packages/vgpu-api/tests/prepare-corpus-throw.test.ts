// T04-19's completeness criterion, executed rather than asserted on paper.
//
// This ticket inserts `prepare()` while the `pendingPipelines` default is still `"sync"`, which
// makes every insertion SEMANTICALLY INERT today: the corpus behaves identically with or without
// it. That is the whole safety property of the additive phase — and it is also what makes the work
// unverifiable by the normal suite. A missing `prepare()` looks exactly like a present one until
// T04-21 flips the default, at which point it becomes a crash in an example nobody re-ran.
//
// So the criterion is not "the examples still pass". It is: **under the default T04-21 will ship,
// every example runs without `VGPU-PIPELINE-PENDING`.** This file simulates that flip by mocking
// the one constant T04-21 will edit (`DEFAULT_PENDING_PIPELINES` in `pending-pipelines.ts`), then
// runs the real, shipped example modules against it. No hand-written re-implementation of an
// example: the module under test is the file the repo publishes, imported through the same
// `vgpu/node` alias it uses in production.
//
// When T04-21 lands, this mock becomes a no-op and the file keeps passing — it is a regression
// gate for both sides of the flip, not scaffolding to delete.
import { expect, test, vi } from "vitest";

vi.mock("../src/pending-pipelines.ts", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/pending-pipelines.ts")>();
  return { ...actual, DEFAULT_PENDING_PIPELINES: "throw" as const };
});

/** Every `examples/*` entry point that encodes something, with the gpu it hands back for cleanup. */
const RUNNERS: readonly { name: string; run: () => Promise<{ gpu: { dispose(): void } }> }[] = [
  { name: "s02-fullscreen", run: async () => (await import("../../../examples/by-example-s02-fullscreen/src/example.ts")).runFullscreenExample() },
  { name: "s03-sharing", run: async () => (await import("../../../examples/by-example-s03-sharing/src/example.ts")).runSharingExample() },
  { name: "s04-shared-uniforms", run: async () => (await import("../../../examples/by-example-s04-shared-uniforms/src/example.ts")).runSharedUniformsExample() },
  { name: "s06-scene", run: async () => (await import("../../../examples/by-example-s06-scene/src/example.ts")).runSceneExample() },
  { name: "s07-hdr-post", run: async () => (await import("../../../examples/by-example-s07-hdr-post/src/example.ts")).runHdrPostExample() },
  { name: "s08-ping-pong", run: async () => (await import("../../../examples/by-example-s08-ping-pong/src/example.ts")).runPingPongExample() },
  { name: "s09-bundles", run: async () => (await import("../../../examples/by-example-s09-bundles/src/example.ts")).runBundlesExample() },
  { name: "s10-group-claim", run: async () => (await import("../../../examples/by-example-s10-group-claim/src/example.ts")).runGroupClaimExample() },
  { name: "s11-compute", run: async () => (await import("../../../examples/by-example-s11-compute/src/example.ts")).runComputeExample() },
  { name: "s12-scheduling-resize", run: async () => (await import("../../../examples/by-example-s12-scheduling-resize/src/example.ts")).runSchedulingResizeExample() },
  { name: "s13-headless", run: async () => (await import("../../../examples/by-example-s13-headless/src/example.ts")).renderGradientHeadless() },
];

test("the mocked flip is actually in effect", async () => {
  const { DEFAULT_PENDING_PIPELINES } = await import("../src/pending-pipelines.ts");
  // Guards the guard: if the mock silently stopped applying, every assertion below would pass for
  // the wrong reason (the corpus is green under "sync" by construction).
  expect(DEFAULT_PENDING_PIPELINES).toBe("throw");
});

test.each(RUNNERS)("$name runs clean under a throw default", async ({ run }) => {
  const result = await run();
  expect(result.gpu).toBeTruthy();
  result.gpu.dispose();
});

// s05 is the odd one out and gets its own assertion rather than a place in the table: its entire
// job is to COLLECT error messages, so "it did not throw" is not evidence of anything. What must
// hold is that the fix-it messages it demonstrates are still the fix-it messages — not
// `VGPU-PIPELINE-PENDING` shadowing them from the pipeline-resolution step that now runs first.
test("s05-fixits still reports fix-its, not a pending-pipeline error", async () => {
  const { collectFixitMessages } = await import("../../../examples/by-example-s05-fixits/src/example.ts");
  const messages = await collectFixitMessages();
  expect(messages.length).toBeGreaterThan(0);
  expect(messages.join("\n")).not.toContain("VGPU-PIPELINE-PENDING");
});
