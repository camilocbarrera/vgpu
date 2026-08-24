/**
 * Codec-free thumbnail for the video-to-texture example.
 *
 * The Node path has no video decoder and no `HTMLVideoElement`, so it uploads the
 * same deterministic test pattern the browser shows before its first decoded frame
 * and renders one frame of the real scene. Everything except the source of the
 * bytes is the production path.
 */
import type { Gpu, Target } from 'vgpu';
import { frame } from 'vgpu';

import type { ThumbnailOptions } from '../../lib/example-renderer';
import { createScene, destroyScene, renderScene, SPIN_RATE, uploadTestPattern } from './scene';

/** 16:9, matching the committed clip, so the face crop matches the live example. */
const PATTERN_SIZE = { width: 640, height: 360 } as const;
/**
 * Three-quarter view: two faces visible, so the thumbnail reads as a cube rather than
 * a flat picture. Expressed as the pose it wants and converted to a time, so changing
 * `SPIN_RATE` cannot silently rotate the thumbnail.
 */
const THUMBNAIL_POSE = 0.405;

export async function renderThumbnail(
  gpu: Gpu,
  target: Target,
  options: ThumbnailOptions = {},
): Promise<void> {
  const scene = createScene(gpu, { ...PATTERN_SIZE, label: 'video-to-texture-thumb' });
  try {
    uploadTestPattern(gpu, scene);
    frame(gpu, (currentFrame) =>
      renderScene(currentFrame, scene, target, options.time ?? THUMBNAIL_POSE / SPIN_RATE),
    );
  } finally {
    await Promise.allSettled([
      Promise.resolve().then(() => gpu.gpu.queue.onSubmittedWorkDone()),
      Promise.resolve().then(() => gpu.settled()),
    ]);
    destroyScene(scene);
  }
}
