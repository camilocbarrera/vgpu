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
  // dependence beyond the cube's fixed pose.
  thumb: { warmupFrames: 1, dt: 0, time: 0.9, note: 'Deterministic test-pattern upload; no video decoding in the Node path.' },
  files: ['index.tsx', 'renderer.ts', 'video-source.ts', 'scene.ts', 'test-pattern.ts', 'cube.wgsl'],
} as const satisfies ExampleMetaDefinition;
