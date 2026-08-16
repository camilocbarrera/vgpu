// T04-21 (the `pendingPipelines` default is now "throw"): the tests below name `"sync"` at init().
// Their subject is what the DEVICE does -- pixels, blend, resolve, readback, filterability -- not
// readiness, and `"sync"` is exactly the eager compile-on-encode they were written against. The
// default is covered on a real device by by-example-gpu.test.ts, which prepares instead.
import { resolve } from "node:path";
import { expect, test } from "vitest";
import { init, effect, frame, target } from "../../src/node.ts";
import { comparePixelSnapshot } from "../../test-utils/snapshot.ts";
import { REPRESENTATIVE_GRADIENT_WGSL, SNAPSHOT_SIZE } from "../fixtures/representative-gradient.ts";

const BASELINE = resolve(import.meta.dirname, "../__snapshots__/representative-gradient.png");

test.skipIf(process.env.VGPU_DOCKER_TEST !== "1")("representative gradient matches committed pixel baseline", async () => {
  const gpu = await init({ pendingPipelines: "sync" });
  try {
    const colorTarget = target(gpu, { size: SNAPSHOT_SIZE, format: "rgba8unorm", label: "representative-gradient" });
    const shader = effect(gpu, { shader: REPRESENTATIVE_GRADIENT_WGSL, label: "representative-gradient", set: { speed: 2 } });
    shader.set("params", { time: Math.PI / 4 });
    frame(gpu, (currentFrame) => currentFrame.pass({ target: colorTarget }, (encoder) => encoder.draw(shader)));
    const result = await comparePixelSnapshot(BASELINE, await colorTarget.read(), SNAPSHOT_SIZE[0], SNAPSHOT_SIZE[1]);
    expect(result).toMatchObject({ status: "matched", mismatchedPixels: 0, ratio: 0 });
  } finally {
    gpu.dispose();
  }
});
