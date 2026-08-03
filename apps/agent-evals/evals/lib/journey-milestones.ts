import type { EveEvalToolCall } from "eve/evals";

/**
 * Soft, non-gating journey adapter: it turns raw tool calls into "did the agent
 * ever run `vgpu doctor`?"-style counters.
 *
 * Gate the outcome, score the journey, never gate the ritual — nothing derived
 * from this module may end up inside a `.gate()`. It is recorded in the
 * evidence artifact and read by humans, that is all.
 *
 * Layer boundary: this module belongs to Layer 2. Nothing under `verify/` or
 * `tasks/` may import it (Layer 2 -> Layer 1 is fine; the reverse is not).
 */
export interface MilestoneSpec {
  readonly id: string;
  readonly tool: string;
  readonly commandPattern?: string;
  readonly pathPattern?: string;
}

export interface Journey {
  readonly milestones: Record<string, number>;
  readonly order: string[];
  readonly turnsToFirstRender: number | null;
  readonly toolCalls: number;
}

export function deriveJourney(
  toolCalls: readonly EveEvalToolCall[],
  milestones: readonly MilestoneSpec[],
): Journey {
  const counts: Record<string, number> = {};
  for (const milestone of milestones) counts[milestone.id] = 0;

  const order: string[] = [];
  let turnsToFirstRender: number | null = null;

  for (const call of toolCalls) {
    for (const milestone of milestones) {
      if (call.name !== milestone.tool) continue;

      const subject =
        milestone.commandPattern !== undefined
          ? stringField(call.input, "command")
          : stringField(call.input, "path");
      const pattern = milestone.commandPattern ?? milestone.pathPattern;
      if (subject === null || pattern === undefined) continue;

      if (!new RegExp(pattern, "u").test(subject)) continue;

      counts[milestone.id] = (counts[milestone.id] ?? 0) + 1;
      order.push(milestone.id);
      if (milestone.id === "ranRender" && turnsToFirstRender === null) {
        turnsToFirstRender = call.turnIndex;
      }
    }
  }

  return {
    milestones: counts,
    order,
    turnsToFirstRender,
    toolCalls: toolCalls.length,
  };
}

function stringField(input: unknown, key: string): string | null {
  if (typeof input !== "object" || input === null) return null;
  const value = (input as Record<string, unknown>)[key];
  return typeof value === "string" ? value : null;
}
