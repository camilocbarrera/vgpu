// T04-21 — the one class-C site of the corpus, executed under the real `"throw"` default.
//
// `light-sources-raw.ts` used to re-record its clear bundle INSIDE the synchronous frame callback
// whenever the bake key changed (occluder toggle / clip-inset tunable), and replay it on the spot.
// A bundle born there can never be prepared, so once `DEFAULT_PENDING_PIPELINES` became `"throw"`
// the first toggle raised `VGPU-PIPELINE-PENDING` at `p.bundles(clearBundle)`. T04-19 found it,
// could not fix it inside a purely additive ticket, and handed it to this one as its only known
// blocker. The fix was to delete the re-record: the recording never depended on the key (the key
// travels through `.set("cfg", ...)`, rewritten every encode), so the prepared bundle is replayed.
//
// The assertions below are the OBSERVABLE behaviour of that, not its implementation: a bake-key
// change must not throw, and the bake it asks for must land in the SAME frame — a fix that deferred
// the bake, dropped it, or stopped baking after some state got stuck would satisfy "does not throw"
// perfectly. The bundle/pipeline bookkeeping that produces it is deliberately not pinned.
//
// Hermetic in CI's sense (the device is the mock adapter, the same swap `prepare-corpus-throw.test.ts`
// uses; `prepare`, `frame`, `bundle` and the pipeline store are the REAL ones). It fails on the
// pre-fix code: reinstate `clearBundle = recordClearBundle()` in `encode()` and the toggle throws.
import { expect, test, vi } from 'vitest';
import { getMockGPUDeviceInstrumentation } from '@vgpu/core';
import { init } from 'vgpu/mock';
import { frame, storage } from 'vgpu';
import { createLightSourcesRaw } from './light-sources-raw';
import { DEFAULT_BRUSH, TUNABLE_DEFAULTS, TUNABLE_RANGES } from './settings';
import type { BrushState } from './light-sources-pass';

const SIM_SIZE = [96, 64] as const;
const BRUSH: BrushState = { ...DEFAULT_BRUSH, x: 10, y: 10, active: false };

interface EncodedPass {
  /** `"clear"` is the bake (the pass asks for a cleared target); `"load"` is the incremental pass. */
  readonly loadOp: string;
  /** Whether the pass actually replayed a bundle, i.e. whether anything was drawn into it. */
  bundles: boolean;
}

/** Records what each render pass of each frame asked the device for. */
function encodedPasses(device: GPUDevice): EncodedPass[] {
  const passes: EncodedPass[] = [];
  const createCommandEncoder = device.createCommandEncoder.bind(device);
  vi.spyOn(device, 'createCommandEncoder').mockImplementation((desc?: GPUCommandEncoderDescriptor) => {
    const encoder = createCommandEncoder(desc);
    const beginRenderPass = encoder.beginRenderPass.bind(encoder);
    encoder.beginRenderPass = (passDesc: GPURenderPassDescriptor) => {
      const pass = beginRenderPass(passDesc);
      const attachment = [...passDesc.colorAttachments][0];
      const record: EncodedPass = { loadOp: String(attachment?.loadOp), bundles: false };
      passes.push(record);
      const executeBundles = pass.executeBundles.bind(pass);
      pass.executeBundles = (bundles: Iterable<GPURenderBundle>) => {
        record.bundles = true;
        executeBundles(bundles);
      };
      return pass;
    };
    return encoder;
  });
  return passes;
}

test('a bake-key change bakes in the same frame under the throw default', async () => {
  // No `pendingPipelines` anywhere: this runs under the shipped default, which is the whole point.
  const gpu = await init();
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
  const ledStorage = storage(gpu, 72 * 8 * 4);
  const raw = createLightSourcesRaw(gpu, { size: SIM_SIZE, ledStorage });
  const passes = encodedPasses(gpu.device.gpu);

  const encode = (renderBlackOccluder: boolean, clipInsetPx: number, time: number) => {
    passes.length = 0;
    frame(gpu, (currentFrame) => raw.encode({
      frame: currentFrame,
      brush: BRUSH,
      time,
      tunables: { ...TUNABLE_DEFAULTS, ledRaycastClipInsetPx: clipInsetPx },
      renderBlackOccluder,
    }));
    return passes;
  };
  const baked = (encoded: EncodedPass[]) => encoded.some(({ loadOp, bundles }) => loadOp === 'clear' && bundles);

  // The setup boundary every caller goes through (`scene-renderer.ts` awaits it inside `prewarm()`).
  await raw.ready;

  // First frame: the first bake, unchanged by this ticket.
  expect(baked(encode(true, 0, 0))).toBe(true);
  // Same key: the incremental pass, which must NOT clear (that is what makes the bake worth caching).
  const unchanged = encode(true, 0, 0.016);
  expect(baked(unchanged)).toBe(false);
  expect(unchanged.some(({ loadOp, bundles }) => loadOp === 'load' && bundles)).toBe(true);

  // The toggle. Pre-fix this threw VGPU-PIPELINE-PENDING; the bake must land here, in this frame.
  expect(baked(encode(false, 0, 0.032))).toBe(true);
  expect(baked(encode(false, 0, 0.048))).toBe(false); // and then settle back to incremental
  // Toggling back is not a one-way door, and a key seen before is not blacklisted by anything.
  expect(baked(encode(true, 0, 0.064))).toBe(true);
  expect(baked(encode(false, 0, 0.08))).toBe(true);

  // A slider drag: the clip-inset tunable is continuous, so the key can change on EVERY frame. Each
  // one bakes, none throws, and nothing accumulates that would stop it.
  const { min, step } = TUNABLE_RANGES.ledRaycastClipInsetPx;
  for (let i = 0; i < 12; i++) {
    expect(baked(encode(true, min + i * step, 0.1 + i * 0.016)), `drag frame ${i}`).toBe(true);
  }

  // Nothing was ever compiled inline: under `"throw"` a stall is impossible, not merely unlikely.
  expect(mock.calls.createRenderPipeline).toBe(0);

  vi.restoreAllMocks();
  raw.destroy();
  gpu.dispose();
});
