/**
 * Per-frame clock and default clear color, as a lazy kernel service.
 *
 * This is the small "frame state" the core deliberately does not carry as public fields:
 * a program that never opens a frame never creates it. Registered through a token so the
 * kernel keeps no static reference to it.
 */
import { serviceToken, type Kernel } from "./kernel.ts";
import { frameReentrantError, VGPUError } from "./errors.ts";
import type { ClearColor } from "./target-utils.ts";

export interface FrameState {
  /** Seconds since the first frame. */
  time: number;
  /** Seconds between the last two frames. */
  deltaTime: number;
  frameCount: number;
  /** Default clear color used by passes that clear. Validated on assignment. */
  clearColor: ClearColor;
  /** Advances the clock and runs the registered per-frame hooks. Throws `VGPU-FRAME-REENTRANT` if re-entered. */
  advance(): void;
  /** Runs right after the clock advances, before the frame callback (surface auto-resize lives here). */
  onAdvance(hook: () => void): () => void;
}

export const frameStateToken = serviceToken<FrameState>("frame-state");

/** Lazily creates the frame state of this kernel; repeated calls return the same instance. */
export function frameState(kernel: Kernel): FrameState {
  return kernel.service(frameStateToken, createFrameState);
}

function createFrameState(): FrameState {
  const hooks = new Set<() => void>();
  let lastTimeMs = nowMs();
  let advancing = false;
  let clearColor: ClearColor = [0, 0, 0, 1];
  const state: FrameState = {
    time: 0,
    deltaTime: 0,
    frameCount: 0,
    get clearColor(): ClearColor { return clearColor; },
    set clearColor(value: ClearColor) {
      const o = value as any, n = Array.isArray(value) ? value : [o?.r, o?.g, o?.b, o?.a];
      if (n.length !== 4 || !n.every(Number.isFinite)) throw new VGPUError({ code: "VGPU-CLEAR-COLOR-INVALID", message: "invalid gpu.clearColor.", where: "gpu.clearColor" });
      clearColor = value;
    },
    advance(): void {
      if (advancing) throw frameReentrantError();
      advancing = true;
      try {
        const next = nowMs();
        state.deltaTime = Math.max(0, (next - lastTimeMs) / 1000);
        state.time += state.deltaTime;
        lastTimeMs = next;
        state.frameCount += 1;
        for (const hook of [...hooks]) hook();
      } finally {
        advancing = false;
      }
    },
    onAdvance(hook: () => void): () => void {
      hooks.add(hook);
      return () => { hooks.delete(hook); };
    },
  };
  return state;
}

function nowMs(): number { return globalThis.performance?.now?.() ?? Date.now(); }
