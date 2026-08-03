/**
 * Why a turn ended without completing, in one line.
 *
 * Kept out of the eval, and kept as a plain module, so it can be replayed
 * against an archived transcript without a model call — the shape below was
 * derived from a real run, not from the docs, and it is worth being able to
 * re-check it when eve's event payloads move.
 *
 * eve stamps the cause on its `*.failed` events (`step.failed`, `turn.failed`,
 * `session.failed`) as a `code` plus a nested `details` object. Verified against
 * the transcript of a run that died on a restricted provider:
 *
 *   turn.failed -> data.code: "MODEL_CALL_FAILED"
 *                  data.details.statusCode: 403
 *                  data.details.message: "Your team has restricted access ..."
 */

/** @typedef {{ type?: unknown, data?: { code?: unknown, details?: Record<string, unknown> } }} FailureEventLike */

/**
 * @param {readonly unknown[]} events the turn's stream events
 * @returns {string} one-line cause, or a stated fallback when nothing explains it
 */
export function turnFailure(events) {
  for (const event of events) {
    const failure = /** @type {FailureEventLike} */ (event);
    if (typeof failure.type !== "string" || !failure.type.endsWith(".failed")) continue;
    const data = failure.data ?? {};
    const details = data.details ?? {};
    const status = details.statusCode ?? details.upstreamStatusCode;
    const message = details.message ?? details.apiErrorMessage ?? details.upstreamMessage;
    const line = [data.code, status === undefined ? null : `HTTP ${status}`, message]
      .filter((part) => part !== null && part !== undefined && part !== "")
      .join(" ");
    // The contract above says "one line", so honour it here rather than hoping
    // upstream messages cooperate: gateway errors arrive multi-line and can run
    // to thousands of characters, and this string goes straight onto a reporter
    // summary line.
    if (line !== "") return line.replace(/\s+/g, " ").trim().slice(0, 300);
  }
  return "no failure event in the transcript";
}
