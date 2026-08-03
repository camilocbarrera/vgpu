import { defineEvalConfig } from "eve/evals";

// No `judge`: nothing in this skeleton uses `t.judge.*` (the whole point is
// that the verdict comes from pixels, not from prose). No `reporters` either —
// wiring Braintrust/JUnit can wait until there is more than one eval worth
// aggregating.
export default defineEvalConfig({
  timeoutMs: 180_000,
});
