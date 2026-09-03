export const meta = {
  slug: 'atmosphere',
  title: 'Atmosphere',
  description: 'A physically based sky and volumetric clouds — Hillaire 2020 transmittance, multiple-scattering, sky-view and aerial-perspective lookup tables built with compute and storage textures, ozone, a limb-darkened sun, and Nubis-style clouds raymarched through tileable 3D Perlin-Worley noise and lit by the same tables, from sea level up to the stratosphere.',
  tags: ['atmosphere', 'volumetric', 'compute', 'storage-textures'],
  capabilities: ['webgpu', 'compute-shader', 'multi-pass', 'continuous-rendering', 'responsive-canvas'],
  thumb: { warmupFrames: 1, time: 0 },
  files: [
    'example.ts', 'tuning.ts', 'camera.ts', 'controls.ts',
    'atmosphere-common.wgsl', 'transmittance-lut.wgsl', 'multiscatter-lut.wgsl', 'sky-view-lut.wgsl', 'aerial-lut.wgsl',
    'terrain.wgsl', 'terrain-heightmap.wgsl', 'scene.wgsl', 'noise-common.wgsl', 'cloud-shape-noise.wgsl', 'cloud-detail-noise.wgsl', 'weather-map.wgsl',
    'clouds-common.wgsl', 'clouds.wgsl', 'present.wgsl', 'lut-preview.wgsl',
  ],
} as const;
