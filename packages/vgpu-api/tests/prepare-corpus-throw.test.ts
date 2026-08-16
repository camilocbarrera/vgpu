// T04-19's completeness criterion, executed rather than asserted on paper — now under the REAL
// default.
//
// While `pendingPipelines` defaulted to `"sync"` every `prepare()` insertion was semantically inert
// and a missing one was invisible, so this file forced the future default by mocking the one
// constant T04-21 would edit (`DEFAULT_PENDING_PIPELINES`). T04-21 edited it: the default IS
// `"throw"`, the mock would now be a no-op asserting nothing, and the guard "is the mocked flip in
// effect?" is replaced below by an assertion on the shipped constant itself. What the file does has
// not changed — run the real, shipped example modules and require that none of them raises
// `VGPU-PIPELINE-PENDING` — except that it is no longer a simulation.
//
// HERMETICITY (fixed after CI, second lesson of the same shape). The first version of this file
// acquired a device through the REAL `vgpu/node`, i.e. Dawn via `@vgpu/adapter-node`. It passed
// locally only because this sandbox happens to have a software renderer installed from earlier
// dogfooding; `test-fast` has no adapter, so all twelve runners died on VGPU-NODE-NO-ADAPTER. The
// test was reading the ENVIRONMENT, not the code.
//
// The property under test — "does a synchronous encode meet a pipeline that is not ready?" — is
// decided by `pending-pipelines.ts`, `prepare.ts`, `pipeline-store.ts` and `bundle.ts`. Not one of
// them can tell a Dawn device from a mock one. So the device acquisition boundary is swapped and
// nothing else: `vgpu/node` resolves to `src/mock.ts`, which is the SAME public API re-exported
// over the mock adapter — identical `prepare`, `frame`, `bundle`, `draw`, `pipeline-store`; only
// `init()` differs. That is the pattern the T04-19 QA harness used to execute the docs corpus, and
// it is why this file is now a policy test rather than a hardware test.
//
// Real-device execution is not lost, it is just not THIS file's job: `examples/*/example.test.ts`
// render and read back pixels under `skipIf(VGPU_DOCKER_TEST !== "1")`, which is the repo's
// existing convention for work that needs an adapter, and those run in ci.yml's `docker-gpu`.
import { expect, test, vi } from "vitest";

// The device boundary, and only it. `src/mock.ts` re-exports the same modules `src/node.ts` does.
vi.mock("vgpu/node", async () => await import("../src/mock.ts"));

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

test("the shipped default is the one this file runs the corpus under", async () => {
  const { DEFAULT_PENDING_PIPELINES } = await import("../src/pending-pipelines.ts");
  // Guards the guard, exactly as the mock-detection assertion it replaces did: if the default ever
  // went back to "sync", every runner below would go green for the wrong reason (the corpus is
  // green under "sync" by construction), so the premise of the whole file is pinned first.
  expect(DEFAULT_PENDING_PIPELINES).toBe("throw");
});

// The second half of that guard, and the one the adapter swap makes necessary. The constant being
// "throw" proves the policy; it does NOT prove the throw still REACHES an encode once the device is
// a mock. If the mock adapter resolved pipelines eagerly, or never consulted the policy,
// all twelve runners below would go green by construction and this file would be decoration.
//
// So: a combination that nobody prepared, over the very same mock gpu the runners use, must still
// raise VGPU-PIPELINE-PENDING. This is the mutation test made permanent — deleting a `prepare()`
// from any example has to fail, and this proves the mechanism that would make it fail is live.
test("an unprepared combination still throws over the mock adapter", async () => {
  const { init, effect, frame, target } = await import("../src/mock.ts");
  const gpu = await init();
  try {
    const colorTarget = target(gpu, { size: [8, 8], format: "rgba8unorm" });
    const never = effect(gpu, {
      shader: "@fragment fn main() -> @location(0) vec4f { return vec4f(1.0, 0.0, 0.0, 1.0); }",
      label: "never-prepared",
    });
    let thrown: unknown;
    try {
      frame(gpu, (currentFrame) => currentFrame.pass({ target: colorTarget }, (p) => p.draw(never)));
    } catch (error) {
      thrown = error;
    }
    // Asserted on the CODE, not on the message. `pipelinePendingError()` puts the code on
    // `error.code` and never repeats it in the prose -- which is exactly the trap the s05
    // assertion below had fallen into, and the reason this control had to be written to find out.
    expect((thrown as { code?: string } | undefined)?.code).toBe("VGPU-PIPELINE-PENDING");
  } finally {
    gpu.dispose();
  }
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
  const joined = messages.join("\n");
  // The negative half. This line used to read `.not.toContain("VGPU-PIPELINE-PENDING")`, which was
  // VACUOUS: the example collects `error.message`, and `pipelinePendingError()` carries that string
  // as `error.code` while its message never repeats it, so the assertion could not fail for any
  // input. Rewritten against the message the error actually produces, it failed immediately and
  // exposed a real defect -- under the throw default the pending error fired FIRST and shadowed the
  // sampler fix-it -- which is why s05 now has a prepare() of its own.
  expect(joined).not.toMatch(/has no pipeline ready for target signature/u);
  // The positive half, which is the assertion that should have been here from the start: naming the
  // fix-its that must survive is what makes shadowing detectable, rather than trusting an absence.
  expect(joined).toContain("Unset `samp`");
  expect(joined).toContain("is lib-owned by its first JS set()");
});
