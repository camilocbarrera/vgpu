/**
 * The frame clock of a gpu, as a free function over the lazy frame-state service.
 *
 * `clock(gpu)` is the only public clock: `Frame` does not duplicate it. Reading it never creates
 * anything a frame would not have created anyway, and a program that never opens a frame and never
 * calls `clock()` never allocates the service.
 */
import { frameState } from "./frame-state.ts";
import { liveKernel } from "./live-kernel.ts";
import { clockDeltaInvalidError } from "./errors.ts";
import { serviceToken, type Gpu, type Kernel } from "./kernel.ts";

export interface Clock {
  /** Seconds since the first frame. */
  readonly time: number;
  /** Seconds between the last two ticks: wall-clock, or the value the last `advance()` was given. */
  readonly deltaTime: number;
  /** Frames opened on this gpu. Counted by `frame()`/`frameLoop()`, never by `advance()`. */
  readonly frameCount: number;
  /**
   * Moves the clock forward by `dtSeconds` right now, and claims this frame's tick: the next
   * `frame()` counts the frame but does not advance the clock again. One advance per frame, with
   * the manual one winning — so an external ticker (GSAP, Motion, an XR frame callback), a
   * timescale (`advance(dt * 0.5)`) or a fixed timestep drives vgpu without a second clock racing it.
   */
  advance(dtSeconds: number): void;
}

/** One clock facade per gpu, so `clock(gpu) === clock(gpu)`. */
const clockToken = serviceToken<Clock>("clock");

/**
 * The frame clock of `gpu`.
 *
 * Without `advance()`, every `frame()` moves `time` forward by the wall-clock delta since the last
 * frame — the default 0.1.x behavior. With `advance(dt)`, you own the clock for that frame.
 *
 * Throws `VGPU-GPU-DISPOSED` after `gpu.dispose()`.
 */
export function clock(gpu: Gpu): Clock {
  return createClock(liveKernel(gpu, "clock"));
}

function createClock(kernel: Kernel): Clock {
  return kernel.service(clockToken, (self) => {
    const state = frameState(self);
    return {
      get time(): number { return state.time; },
      get deltaTime(): number { return state.deltaTime; },
      get frameCount(): number { return state.frameCount; },
      advance(dtSeconds: number): void {
        if (typeof dtSeconds !== "number" || !Number.isFinite(dtSeconds) || dtSeconds < 0) throw clockDeltaInvalidError(dtSeconds);
        state.advanceBy(dtSeconds);
      },
    };
  });
}
