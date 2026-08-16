/**
 * Contract #22 — transient-attachment safety predicate, as a **negative** contract.
 *
 * "No attachment reachable through a public accessor is ever allocated with `TRANSIENT_ATTACHMENT`:
 * a single-sample `depth: true` target's depth attachment, a target's `.color`/`.colors`, and a
 * surface's canvas texture never carry the flag. Where the implementation does apply it to a private
 * discard-only multisample attachment, it is feature-detected and its absence changes nothing
 * observable. Applying the flag is **not** a required behavior — the required behavior is that it
 * never appears on an observable attachment."
 *
 * This tree applies the flag NOWHERE (see the normative comment in `target-utils.ts` for why the one
 * candidate — the intermediate MSAA color attachment — cannot receive it safely while #323 stands),
 * so the contract is verified in both directions: statically (no source line applies the flag) and
 * at runtime (every texture vgpu allocates carries only standard WebGPU usage bits, and every
 * publicly reachable attachment is sampleable/copyable, which a transient attachment can never be).
 */
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, vi } from "vitest";
import { init, surface, target } from "../src/mock.ts";

/** Every texture usage bit standard WebGPU defines. A transient-attachment flag is by definition NOT one of them. */
const STANDARD_TEXTURE_USAGE = 1 /* COPY_SRC */ | 2 /* COPY_DST */ | 4 /* TEXTURE_BINDING */ | 8 /* STORAGE_BINDING */ | 16 /* RENDER_ATTACHMENT */;
/** The usage names `packages/core` can even express; `transient_attachment` is deliberately not among them. */
const KNOWN_USAGE_NAMES = new Set(["copy_src", "copy_dst", "texture_binding", "storage_binding", "render_attachment"]);

function canvasLike(width = 8, height = 4): HTMLCanvasElement {
  const context = {
    configure: vi.fn(),
    unconfigure: vi.fn(),
    getCurrentTexture: () => ({ createView: () => ({}) }),
  };
  const canvas: Record<string, unknown> = {
    width,
    height,
    getContext: (kind: string) => (kind === "webgpu" ? { ...context, canvas } : null),
  };
  return canvas as unknown as HTMLCanvasElement;
}

/** Every `GPUTextureDescriptor` the device is asked to allocate, in creation order. */
function recordTextureDescriptors(device: GPUDevice): GPUTextureDescriptor[] {
  const created: GPUTextureDescriptor[] = [];
  const createTexture = device.createTexture.bind(device);
  vi.spyOn(device, "createTexture").mockImplementation((descriptor: GPUTextureDescriptor) => {
    created.push(descriptor);
    return createTexture(descriptor);
  });
  return created;
}

function sourceFiles(root: string): string[] {
  return readdirSync(root).flatMap((entry) => {
    const path = join(root, entry);
    if (statSync(path).isDirectory()) return sourceFiles(path);
    return path.endsWith(".ts") ? [path] : [];
  });
}

/** Strips line and block comments, so the normative prose that NAMES the flag is not mistaken for applying it. */
function withoutComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^[ \t]*\/\/.*$/gm, "");
}

// --- Static half: nothing in the shipped source applies the flag ------------------------------

test("contract #22: no source line in vgpu-api or core applies TRANSIENT_ATTACHMENT", () => {
  const roots = [
    fileURLToPath(new URL("../src", import.meta.url)),
    fileURLToPath(new URL("../../core/src", import.meta.url)),
  ];
  const offenders = roots
    .flatMap((root) => sourceFiles(root))
    .filter((path) => /transient[_-]?attachment/i.test(withoutComments(readFileSync(path, "utf8"))));
  expect(offenders, `these files apply a transient-attachment flag in code: ${offenders.join(", ")}`).toEqual([]);
});

// --- Runtime half: every allocated texture carries only standard usage bits --------------------

test("contract #22: an MSAA target allocates no texture with a non-standard usage bit", async () => {
  const gpu = await init();
  const descriptors = recordTextureDescriptors(gpu.device.gpu);
  try {
    const msaa = target(gpu, { size: [4, 4], depth: true, msaa: true, label: "msaa" });
    // Resolved color + intermediate MSAA color + depth: the whole allocation set of the design's
    // one transient candidate.
    expect(descriptors.length).toBe(3);
    for (const descriptor of descriptors) expect(Number(descriptor.usage) & ~STANDARD_TEXTURE_USAGE).toBe(0);
    // The intermediate MSAA attachment is the only render-attachment-only texture, and it is not
    // reachable from any public accessor — but it still carries no transient flag.
    expect(msaa.colors.length).toBe(1);
  } finally {
    gpu.dispose();
  }
});

test("contract #22: an MSAA surface allocates no texture with a non-standard usage bit", async () => {
  const gpu = await init();
  const descriptors = recordTextureDescriptors(gpu.device.gpu);
  try {
    const screen = surface(gpu, canvasLike(), { depth: true, sampleCount: 4, size: [8, 4] });
    // The surface allocates its private attachments lazily, when the pass descriptor is built.
    screen.renderPassDescriptor();
    expect(descriptors.length).toBeGreaterThan(0);
    for (const descriptor of descriptors) expect(Number(descriptor.usage) & ~STANDARD_TEXTURE_USAGE).toBe(0);
  } finally {
    gpu.dispose();
  }
});

// --- Runtime half: every publicly reachable attachment stays observable ------------------------

test("contract #22: publicly reachable attachments of an MSAA target are sampleable/copyable, never transient", async () => {
  const gpu = await init();
  try {
    const msaa = target(gpu, { size: [4, 4], depth: true, msaa: true, label: "reachable" });
    const reachable = [msaa.color, ...msaa.colors, msaa.depth!];
    for (const texture of reachable) {
      for (const usage of texture.usage) expect(KNOWN_USAGE_NAMES.has(usage)).toBe(true);
      // A transient attachment can be neither sampled nor copied; every publicly reachable
      // attachment here is at least one of the two, which is what makes it observable.
      expect(texture.usage.includes("texture_binding") || texture.usage.includes("copy_src")).toBe(true);
    }
    // `.color` is the resolved single-sample texture, so `read()`/`readFloats()` stay legal.
    expect(msaa.color.sampleCount ?? 1).toBe(1);
    await expect(msaa.read()).resolves.toBeInstanceOf(Uint8Array);
    await expect(msaa.readFloats()).resolves.toBeInstanceOf(Float32Array);
  } finally {
    gpu.dispose();
  }
});

test("contract #22: a single-sample depth: true attachment is public and bindable, so it is never transient", async () => {
  const gpu = await init();
  try {
    const plain = target(gpu, { size: [4, 4], depth: true, label: "plain" });
    expect(plain.depth).toBeDefined();
    expect(plain.depth!.usage).toContain("texture_binding");
    for (const usage of plain.depth!.usage) expect(KNOWN_USAGE_NAMES.has(usage)).toBe(true);
  } finally {
    gpu.dispose();
  }
});
