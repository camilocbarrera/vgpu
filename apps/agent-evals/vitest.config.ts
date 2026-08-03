import { defineConfig } from "vitest/config";

// Local config on purpose: this package is intentionally absent from the root
// `vitest.config.ts` include list (see README.md). Without a config of its own,
// `vitest run` here would walk up and pick the repo root's config, which does
// not match any file in this package.
//
// The verifier's tests install a real node_modules and spawn a renderer, so the
// timeouts are generous compared to the library's unit tests.
export default defineConfig({
  test: {
    include: ["verify/**/*.test.ts"],
    pool: "forks",
    testTimeout: 300_000,
    hookTimeout: 300_000,
  },
});
