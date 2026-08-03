// Loaded via `node --import` by the graded render (see verify-task.mjs).
//
// Observational only: it registers a resolve hook that records which module
// specifiers the render loaded, so the evidence can report whether the solution
// actually used the library under test. It must never change resolution, and a
// failure here must never fail the render — hence the try/catch.
import { register } from "node:module";

try {
  register(new URL("./probe-loader.mjs", import.meta.url));
} catch {
  // Old runtime or a restricted loader policy: the evidence reports
  // `vgpuLoaded: null` and the render proceeds untouched.
}
