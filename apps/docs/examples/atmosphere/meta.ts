export const meta = {
  slug: 'atmosphere',
  title: 'Atmosphere',
  description: 'A physically based sky after Hillaire 2020 — transmittance, multiple-scattering, sky-view and aerial-perspective lookup tables rebuilt every frame with compute and storage textures, with ozone, a limb-darkened sun and AgX tone mapping, from sea level up to the stratosphere.',
  tags: ['atmosphere', 'volumetric', 'compute', 'storage-textures'],
  capabilities: ['webgpu', 'compute-shader', 'multi-pass', 'continuous-rendering', 'responsive-canvas'],
  thumb: { warmupFrames: 1, time: 0 },
  files: [
    'example.ts', 'tuning.ts', 'camera.ts', 'controls.ts',
    'atmosphere-common.wgsl', 'transmittance-lut.wgsl', 'multiscatter-lut.wgsl', 'sky-view-lut.wgsl', 'aerial-lut.wgsl',
    'terrain.wgsl', 'scene.wgsl', 'present.wgsl', 'lut-preview.wgsl',
  ],
} as const;
