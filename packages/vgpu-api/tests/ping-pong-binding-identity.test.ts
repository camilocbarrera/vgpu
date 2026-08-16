// Does the construction form the by-example ping-pong sample SHIPS bind the same half the flat bag
// bound? Runs on the mock adapter, so it guards a GPU-only behaviour without a GPU.
//
// This exists because T04-17's codemod once moved `buf.read` into `bindings` at construction.
// `pingPong()` returns `.read`/`.write` as GETTERS over a parity flag that `swap()` flips
// (ping-pong.ts), so pinning one at construction freezes that half: the copy pass then samples the
// very target it is drawing into, and the feedback loop stops accumulating. Nothing failed. The
// assertions that would have caught it live in `tests/gpu/*.test.ts` behind
// `describe.skipIf(process.env.VGPU_DOCKER_TEST !== "1")`, so on a normal run the corpus is not an
// oracle for this class of defect at all — which is exactly what this file is for.
//
// The `as-shipped` arm is READ OFF DISK rather than hardcoded. A probe that hardcodes the migrated
// spelling demonstrates the defect but cannot observe a fix: it fails the same on a tree that never
// had the bug. Reading the example means this test tracks whatever the example actually says.
import { readFileSync } from "node:fs";
import path from "node:path";
import { expect, test } from "vitest";
import { getMockGPUDeviceInstrumentation } from "@vgpu/core";
import { init } from "../src/mock.ts";
import { pingPong } from "../src/ping-pong.ts";
import { effect } from "../src/effect.ts";
import { frame } from "../src/frame.ts";

const REPO_ROOT = path.resolve(import.meta.dirname, "../../..");
const EXAMPLE = "examples/by-example-s08-ping-pong/src/example.ts";

const FILL = `@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f { return vec4f(uv, 0.5, 1.0); }`;
const COPY = `
struct Params { texel: vec2f }
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
  return textureLoad(src, vec2u(vec2f(uv) / params.texel), 0);
}`;

/** Makes each half self-identifying so the recorded bind group says which one it captured. */
function tagHalves(buf: { read: unknown; write: unknown }) {
  for (const [half, t] of [["ping", buf.read], ["pong", buf.write]] as const) {
    (t as { color: { gpu: { createView: () => unknown } } }).color.gpu.createView = () => ({ __half: half });
  }
}

function sampledHalf(mock: { createBindGroupDescriptors: { entries: { resource?: { __half?: string } }[] }[] }): string {
  let out = "none";
  for (const d of mock.createBindGroupDescriptors) {
    for (const e of d.entries) if (e.resource?.__half) out = e.resource.__half;
  }
  return out;
}

/** How the shipped example declares the copy pass: pinned at construction, or set per frame? */
function shippedShape() {
  const src = readFileSync(path.join(REPO_ROOT, EXAMPLE), "utf8");
  const ctor = /const copy = effect\([^;]*?\);/su.exec(src)?.[0] ?? "";
  const setCall = /copy\.set\(([^;]*?)\);/su.exec(src)?.[0] ?? "";
  return {
    pinsReadIntoBindings: /bindings:[^}]*\bbuf\.read\b/su.test(ctor),
    setPassesSrc: /\bsrc:\s*buf\.read\b/u.test(setCall),
  };
}

async function runCopyPass(mode: "flat-bag" | "as-shipped") {
  const shape = shippedShape();
  const gpu = await init();
  const buf = pingPong(gpu, 8, 8, { format: "rgba8unorm", label: "buf" });
  tagHalves(buf);
  const fill = effect(gpu, { shader: FILL, label: "fill" });
  const pin = mode === "as-shipped" && shape.pinsReadIntoBindings;
  const copy = pin
    ? effect(gpu, { shader: COPY, label: "copy", bindings: { src: buf.read } })
    : effect(gpu, { shader: COPY, label: "copy" });

  frame(gpu, (f) => f.pass({ target: buf.write }, (p) => p.draw(fill)));
  buf.swap();

  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
  mock.createBindGroupDescriptors.length = 0;
  frame(gpu, (f) => f.pass({ target: buf.write }, (p) => {
    if (mode === "flat-bag" || shape.setPassesSrc) copy.set({ src: buf.read, texel: buf.read.texelSize });
    else copy.set("params", { texel: buf.read.texelSize });
    p.draw(copy);
  }));

  const result = {
    sampled: sampledHalf(mock),
    renderTarget: (buf.write as { color: { view: { __half: string } } }).color.view.__half,
  };
  gpu.dispose();
  return result;
}

test("the shipped ping-pong example samples the same half the flat bag did", async () => {
  const flatBag = await runCopyPass("flat-bag");
  const shipped = await runCopyPass("as-shipped");
  expect(shipped.sampled).toBe(flatBag.sampled);
});

test("the shipped ping-pong example never samples the target it is drawing into", async () => {
  // The defect's signature, stated directly: if the read half is pinned at construction it stays
  // pinned across `swap()`, and the copy pass reads its own render target.
  const shipped = await runCopyPass("as-shipped");
  expect(shipped.sampled).not.toBe(shipped.renderTarget);
});
