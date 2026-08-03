import { join, resolve } from "node:path";

/**
 * Where the workspace export for a session lands on the HOST.
 *
 * Shared by the export hook (writer) and the evals (readers) so the two can
 * never drift. `eve eval` runs with the package root as cwd; override with
 * `VGPU_EVALS_WORK_DIR` if you drive the evals from somewhere else.
 */
export function workDir(): string {
  return resolve(process.env.VGPU_EVALS_WORK_DIR ?? join(process.cwd(), ".work"));
}

/** Directory holding everything captured for one session. */
export function snapshotDir(sessionId: string): string {
  return join(workDir(), "snapshots", sessionId);
}

/** The tar produced by the export hook for one session. */
export function snapshotTarPath(sessionId: string): string {
  return join(snapshotDir(sessionId), "workspace.tar");
}

/** Where a snapshot tar is extracted for grading. */
export function snapshotWorkspaceDir(sessionId: string): string {
  return join(snapshotDir(sessionId), "workspace");
}

/** Final evidence artifact for one session. */
export function evidencePath(sessionId: string): string {
  return join(workDir(), "evidence", `${sessionId}.json`);
}
