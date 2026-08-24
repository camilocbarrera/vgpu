import type { ExampleMetaDefinition } from '../../lib/example-meta';

export const meta = {
  slug: 'video-to-texture',
  title: 'Video to Texture',
  description:
    'Drive a texture from a playing video with requestVideoFrameCallback, so a copy into the GPU happens once per decoded frame while the cube keeps spinning at the display refresh rate.',
  tags: ['video', 'animation', '3d', 'rendering'],
  capabilities: [
    'webgpu',
    'textures',
    'video-input',
    'external-device',
    'continuous-rendering',
    'responsive-canvas',
  ],
  // The thumbnail uploads the committed test pattern: no codec, no network, no time
  // dependence beyond the cube's fixed pose. `time` is seconds, so it is scaled by
  // SPIN_RATE in scene.ts; 2.7s holds the same three-quarter view the thumbnail has
  // always used (0.405 rad) now that the cube turns three times slower.
  thumb: { warmupFrames: 1, dt: 0, time: 2.7, note: 'Deterministic test-pattern upload; no video decoding in the Node path.' },
  files: ['index.tsx', 'renderer.ts', 'video-source.ts', 'scene.ts', 'test-pattern.ts', 'cube.wgsl'],
} as const satisfies ExampleMetaDefinition;
