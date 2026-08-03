// Module-resolution recorder for the graded render. See probe-register.mjs.
//
// Runs on the loader thread. It appends `<specifier>\t<resolved url>` per
// resolution to the file named by VGPU_VERIFY_PROBE_OUT and otherwise defers
// entirely to the default resolver.
import { appendFileSync } from "node:fs";

const OUT = process.env.VGPU_VERIFY_PROBE_OUT;

export async function resolve(specifier, context, nextResolve) {
  const result = await nextResolve(specifier, context);
  if (OUT) {
    try {
      appendFileSync(OUT, `${specifier}\t${result.url}\n`);
    } catch {
      // Never let bookkeeping break a render.
    }
  }
  return result;
}
