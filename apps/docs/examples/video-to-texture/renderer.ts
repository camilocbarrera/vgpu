/**
 * Browser lifecycle: load the clip, allocate a texture the size of its picture, and
 * keep the two clocks apart.
 *
 * `requestVideoFrameCallback` is the upload clock — it fires once per presented
 * frame, and each firing bumps a token. `frameLoop` is the render clock, running at
 * the display's refresh rate so the cube's rotation is smooth regardless of the
 * clip's frame rate. The loop copies into the texture only when the token has
 * moved, which is what keeps a 30 fps clip from paying for 120 uploads a second.
 */
import { clock, frameLoop, init, surface } from 'vgpu';

import {
  createScene,
  destroyScene,
  renderScene,
  uploadTestPattern,
  uploadVideoFrame,
  type VideoCubeScene,
} from './scene';
import { loadVideo, type VideoFrameInfo, type VideoSource } from './video-source';

/**
 * Big Buck Bunny, © 2008 Blender Foundation — CC BY 3.0 (peach.blender.org).
 *
 * A 10.4 second, 640×360, 30 fps excerpt is committed alongside the example so the
 * demo has no network dependency beyond its own origin and no licence ambiguity. The
 * cut is the flying-squirrel glide, chosen because it survives being a texture: a
 * large subject that stays inside the square centre crop of each cube face, saturated
 * colour, and continuous motion, so a stalled upload is obvious rather than plausible.
 *
 * Reproduce it from the Wikimedia Commons transcode of `Big Buck Bunny 4K.webm`:
 *
 *   ffmpeg -ss 398.95 -t 10.40 -i source.webm -an \
 *     -vf "scale=640:360:flags=lanczos,setsar=1,fps=30" \
 *     -c:v libx264 -profile:v high -crf 22 -preset veryslow -movflags +faststart out.mp4
 *
 * 398.95s and 409.35s are shot boundaries in the film, so the clip contains no
 * half-cut shot and loops from empty sky back to the opening close-up.
 */
export const VIDEO_URL = '/examples/video-to-texture/big-buck-bunny-360p-glide.mp4';

export interface VideoStatus {
  readonly phase: 'loading' | 'playing' | 'failed';
  readonly frame?: VideoFrameInfo;
  /** Textures uploaded so far: one per presented frame, never one per rendered frame. */
  readonly uploads: number;
  readonly rendered: number;
}

export interface VideoRendererOptions {
  readonly onStatus?: (status: VideoStatus) => void;
  readonly onError?: (error: unknown) => void;
}

export interface VideoRenderer {
  dispose(): void;
}

export async function createRenderer(
  canvas: HTMLCanvasElement,
  options: VideoRendererOptions = {},
): Promise<VideoRenderer> {
  let disposed = false;
  let video: VideoSource | undefined;
  let scene: VideoCubeScene | undefined;
  let uploads = 0;
  let rendered = 0;
  let lastReport = 0;

  options.onStatus?.({ phase: 'loading', uploads, rendered });

  const gpu = await init();
  try {
    video = await loadVideo({ url: VIDEO_URL });
    if (disposed) throw new Error('Renderer disposed during startup.');

    const output = surface(gpu, canvas, { dpr: [1, 2] });
    scene = createScene(gpu, { width: video.width, height: video.height });
    // The first display frame can arrive before the first decoded one; the test
    // pattern means the cube is never a black box waiting on the codec.
    uploadTestPattern(gpu, scene);

    let pendingToken = 0;
    let uploadedToken = -1;
    video.start((token) => {
      pendingToken = token;
    });

    const time = clock(gpu);
    frameLoop(gpu, (currentFrame) => {
      if (!scene || !video) return;
      if (pendingToken !== uploadedToken) {
        uploadVideoFrame(gpu, scene, video.frame);
        uploadedToken = pendingToken;
        uploads += 1;
      }
      rendered += 1;
      renderScene(currentFrame, scene, output, time.time);
      // Status drives React text; four updates a second is enough to read and
      // cheap enough not to matter next to the render itself.
      if (options.onStatus && time.time - lastReport > 0.25) {
        lastReport = time.time;
        options.onStatus({ phase: 'playing', frame: video.info, uploads, rendered });
      }
    });

    return {
      dispose: () => {
        if (disposed) return;
        disposed = true;
        video?.dispose();
        if (scene) destroyScene(scene);
        gpu.dispose();
      },
    };
  } catch (error) {
    disposed = true;
    video?.dispose();
    if (scene) destroyScene(scene);
    gpu.dispose();
    options.onStatus?.({ phase: 'failed', uploads, rendered });
    throw error;
  }
}
