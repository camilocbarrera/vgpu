import { defineAgent } from "eve";

// No `tools:` field on purpose. eve's defaults (bash, read_file, write_file,
// glob, grep) are exactly the representative coding-agent proxy this suite
// measures; adding a vgpu-aware tool here would hand the agent part of the
// answer and silently invalidate every discoverability metric. Do not add one.
export default defineAgent({
  model: process.env.VGPU_EVALS_MODEL ?? "anthropic/claude-sonnet-5",
});
