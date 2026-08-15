/**
 * Type-level fixture for contract #24 / #25: `frame()` and `frameLoop()` accept synchronous
 * callbacks only, and the encoder `f.raw()` lends out cannot be finished.
 *
 * Compiled by `pnpm typecheck:fixtures` (tsconfig.types.json next to this file): every
 * `@ts-expect-error` below must be a real error, and every other line must compile clean.
 */
import type { BorrowedCommandEncoder, Frame, Gpu, Target } from "../../src/index.ts";
import { frame, frameLoop } from "../../src/index.ts";

declare const gpu: Gpu;
declare const screen: Target;
declare const loadAssets: () => Promise<void>;

// --- Accepted: every synchronous shape the corpus uses today ------------------------------------

// Block-bodied arrow.
frame(gpu, (f) => { f.pass(screen, () => undefined); });
// Expression-bodied arrow returning void.
frame(gpu, (f) => f.pass(screen, () => undefined));
// Expression-bodied arrow returning a value (not a thenable).
frame(gpu, (f) => (f.pass(screen, () => undefined), 42));
// Function expression form.
frame(gpu, function record(f: Frame) { f.pass(screen, () => undefined); });
// Early return inside a block body.
frame(gpu, (f) => { if (Math.random() > 0.5) return; f.pass(screen, () => undefined); });
// No callback at all: the manual frame form.
frame(gpu);
// Options bag still accepted alongside the callback.
frame(gpu, (f) => { f.pass(screen, () => undefined); }, { pendingPipelines: "sync" });

frameLoop(gpu, (f) => { f.pass(screen, () => undefined); });
frameLoop(gpu, (f) => f.pass(screen, () => undefined));
frameLoop(gpu, (f) => { f.pass(screen, () => undefined); }, { fps: 30 });

// --- Rejected: any callback whose inferred return extends PromiseLike ---------------------------

// @ts-expect-error an async callback returns Promise<void>, which FrameCallback rejects (contract #24)
frame(gpu, async (f) => { await loadAssets(); f.pass(screen, () => undefined); });
// @ts-expect-error an expression-bodied callback returning a promise is rejected too
frame(gpu, () => loadAssets());
// @ts-expect-error frameLoop() uses the same constraint
frameLoop(gpu, async (f) => { await loadAssets(); f.pass(screen, () => undefined); });
// @ts-expect-error frameLoop() rejects an expression-bodied promise return as well
frameLoop(gpu, () => loadAssets());

// --- f.raw(): finish() is unrepresentable, not merely forbidden (contract #25) ------------------

declare const stats: GPUQuerySet;
declare const resolved: GPUBuffer;

frame(gpu, (f) => {
  f.raw((encoder: BorrowedCommandEncoder) => {
    encoder.pushDebugGroup("simulation");
    encoder.clearBuffer(resolved, 0, 16);
    encoder.resolveQuerySet(stats, 0, 2, resolved, 0);
    encoder.popDebugGroup();
    // @ts-expect-error finish() is omitted from the borrowed encoder: submission stays the frame's job
    encoder.finish();
  });
});

// The callback parameter is inferred, so an annotation is never required.
frame(gpu, (f) => {
  f.raw((encoder) => { encoder.pushDebugGroup("x"); encoder.popDebugGroup(); });
});

// --- f.copyBuffer(): structural, so any { gpu, size } pair is accepted (contract #18) ----------

declare const source: { readonly gpu: GPUBuffer; readonly size: number };
declare const destination: { readonly gpu: GPUBuffer; readonly size: number };

frame(gpu, (f) => {
  f.copyBuffer({ source, destination });
  f.copyBuffer({ source, destination, size: 4, sourceOffset: 4, destinationOffset: 8 });
  // @ts-expect-error a buffer-less object is not a copyable buffer
  f.copyBuffer({ source: { size: 4 }, destination });
});
