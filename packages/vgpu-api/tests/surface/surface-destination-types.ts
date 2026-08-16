/**
 * Type-level fixture for contract #23 / design §4c: `RenderDestination` is the shared supertype,
 * `Surface` no longer extends `Target`, and every destination-shaped entry point still accepts both.
 *
 * Compiled by `pnpm typecheck:fixtures` (tsconfig.types.json next to this file): every
 * `@ts-expect-error` below must be a real error, and every other line must compile clean.
 *
 * This fixture pins the TYPE surface only. It is deliberately NOT the enforcement mechanism for the
 * not-bindable rule: `bindings` is `Record<string, unknown>`, so the rejection is nominal and at runtime
 * (see tests/surface-split.test.ts). Structural typing cannot carry that contract.
 */
import type { CompileTarget, Frame, RenderDestination, Surface, Target, TargetOptions } from "../../src/index.ts";
import { bundle, effect, frame, prepare, target } from "../../src/index.ts";
import type { Gpu } from "../../src/kernel.ts";

declare const gpu: Gpu;
declare const screen: Surface;
declare const scene: Target;
declare const fx: ReturnType<typeof effect>;
declare const options: TargetOptions;

// --- Both halves are render destinations ---------------------------------------------------------

const asDestination: RenderDestination = screen;
const targetAsDestination: RenderDestination = scene;
const compileFromSurface: CompileTarget = screen;
const compileFromTarget: CompileTarget = scene;
void asDestination; void targetAsDestination; void compileFromSurface; void compileFromTarget;
void target(gpu, options);

// --- Every destination-shaped entry point keeps accepting a Surface ------------------------------

frame(gpu, (f: Frame) => {
  f.pass(screen, () => undefined);
  f.pass({ target: screen }, () => undefined);
  f.pass(scene, () => undefined);
  f.pass({ target: scene, clear: false }, () => undefined);
  // The union the corpus actually writes when a renderer can output to either.
  const output: Surface | Target = screen;
  f.pass(output, () => undefined);
  f.pass({ target: output }, () => undefined);
});

void prepare(gpu, [{ draw: fx, target: screen }]);
void prepare(gpu, [{ draw: fx, target: scene }]);
void prepare(gpu, [{ draw: fx, target: { colors: ["rgba8unorm"], sampleCount: 4 } }]);
void fx.compile(screen);
void fx.compileSync(scene);
frame(gpu, () => { bundle(gpu, { target: screen }, (b) => b.draw(fx)); });

// --- A Surface is NOT a Target -------------------------------------------------------------------

// @ts-expect-error a Surface has no persistent attachments: it is not a Target (design §4c).
const notATarget: Target = screen;
void notATarget;

// @ts-expect-error `.color` is Target-only: a surface's presentation texture is frame-scoped.
void screen.color;

// @ts-expect-error `.colors` is Target-only for the same reason.
void screen.colors;

// @ts-expect-error `.depth` is Target-only for the same reason.
void screen.depth;

// A `RenderDestination` alone does not expose attachments either.
declare const destination: RenderDestination;
// @ts-expect-error attachments belong to the Target half of the split.
void destination.color;

// --- The shared members really are shared --------------------------------------------------------

void screen.size; void scene.size;
void screen.texelSize; void scene.texelSize;
void screen.format; void scene.format;
void screen.sampleCount; void scene.sampleCount;
void screen.clearColor; void scene.clearColor;
void screen.resourceIdentity; void scene.resourceIdentity;
void screen.renderPassDescriptor(); void scene.renderPassDescriptor();
screen.resize([2, 2]); scene.resize([2, 2]);
void screen.read(); void scene.read();
void screen.readFloats(); void scene.readFloats();

// --- Multisampling vocabulary --------------------------------------------------------------------

void target(gpu, { size: [2, 2], sampleCount: 4 });
void target(gpu, { size: [2, 2], sampleCount: 1 });
// The legacy spelling stays valid — the corpus uses it and this branch does not retire it.
void target(gpu, { size: [2, 2], msaa: true });
void target(gpu, { size: [2, 2], msaa: 4 });
// @ts-expect-error WebGPU render targets are 1 or 4.
void target(gpu, { size: [2, 2], sampleCount: 2 });
