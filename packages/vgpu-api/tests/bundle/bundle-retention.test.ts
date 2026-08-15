import { expect, test } from "vitest";

import { bundle, effect, init, prepare, target } from "../../src/mock.ts";

/**
 * The retention half of contract #15's `dispose()` row, imported from the adversarial QA probe that
 * found the leak this file now guards.
 *
 * Why a GC test at all: `dispose()`'s own unit test can only assert what `dispose()` sets (status
 * `disposed`, every replay throwing) — it is tautological about the reference graph, and it passed
 * happily while a disposed bundle stayed reachable forever from every draw it had recorded. The
 * draw→bundle back-reference registry is add-only by construction, so the ONLY thing that observes
 * `unregisterDrawBundle()` being wired into `dispose()` is a collectability measurement. If someone
 * touches `dispose()` or that registry, this is the net.
 *
 * Read the shape of the test carefully before editing it: the bundle MUST be created inside its own
 * function, with only the `WeakRef` escaping. Creating it in a block scope of the test body and
 * awaiting there instead reports a false positive — V8 keeps block-scoped bindings alive in the
 * enclosing async context, so the bundle looks retained even when the registry is clean. That false
 * positive is exactly what the first version of the QA probe measured.
 */
const gc = () => (globalThis as { gc?: () => void }).gc;

test.skipIf(typeof gc() !== "function")("a disposed bundle is collectable while the draws it recorded stay alive; skipped without --expose-gc", async () => {
  const gpu = await init();
  try {
    const scene = target(gpu, { size: [4, 4] });
    // The draw outlives every bundle here: that is the whole point. It is what holds the registry.
    const fx = effect(gpu, SOLID, { label: "retentionFx" });

    const makeDisposedBundle = async (): Promise<WeakRef<object>> => {
      const recorded = bundle(gpu, { target: scene, label: "retentionDisposed" }, (b) => b.draw(fx));
      await prepare(gpu, { bundle: recorded });
      recorded.dispose();
      return new WeakRef(recorded as object);
    };

    const disposed = await makeDisposedBundle();
    await forceCollection();
    expect(disposed.deref()).toBeUndefined();

    // And the cost of an identity change must not grow with the number of bundles that ever recorded
    // the draw: a registry that never drops entries makes every .set() walk the graveyard.
    const samples: WeakRef<object>[] = [];
    for (let i = 0; i < 500; i++) {
      const churn = bundle(gpu, { target: scene, label: `retentionChurn${i}` }, (b) => b.draw(fx));
      churn.dispose();
      if (i % 125 === 0) samples.push(new WeakRef(churn as object));
    }
    await forceCollection();
    expect(samples.filter((sample) => sample.deref() !== undefined)).toEqual([]);
  } finally {
    gpu.dispose();
  }
});

async function forceCollection(): Promise<void> {
  const runGc = gc();
  if (!runGc) return;
  for (let i = 0; i < 8; i++) {
    runGc();
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
}

const SOLID = `
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f { return vec4f(uv, 0.0, 1.0); }
`;
