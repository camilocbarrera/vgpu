import type { Compute, Effect, Frame, Gpu, PingPongTargets, SharedUniforms, Surface, Target, Texture } from 'vgpu';
import { cameraUniforms, sunDirection, type CameraUniformValues } from './camera';
import { ATMOSPHERE_PHYSICS, CLOUD_TUNING, DEFAULT_PRESET, LUT_SIZES, PRESETS, TONEMAPS, type AtmosphereState } from './tuning';
import transmittanceLutWgsl from './transmittance-lut.wgsl';
import multiScatterLutWgsl from './multiscatter-lut.wgsl';
import skyViewLutWgsl from './sky-view-lut.wgsl';
import aerialLutWgsl from './aerial-lut.wgsl';
import sceneWgsl from './scene.wgsl';
import presentWgsl from './present.wgsl';
import lutPreviewWgsl from './lut-preview.wgsl';
import cloudShapeNoiseWgsl from './cloud-shape-noise.wgsl';
import cloudDetailNoiseWgsl from './cloud-detail-noise.wgsl';
import weatherMapWgsl from './weather-map.wgsl';
import cloudsWgsl from './clouds.wgsl';
import terrainHeightmapWgsl from './terrain-heightmap.wgsl';

type Output = Surface | Target;
type Vec3 = readonly [number, number, number];
export type DebugView = 'transmittance' | 'multiscatter' | 'sky-view' | 'weather' | 'terrain';

type AtmosphereUniformValues = {
  rayleighScattering: Vec3; rayleighScaleHeight: number;
  mieScattering: Vec3; mieScaleHeight: number;
  mieAbsorption: Vec3; mieG: number;
  ozoneAbsorption: Vec3; ozoneCenter: number;
  groundAlbedo: Vec3; ozoneWidth: number;
  sunIlluminance: Vec3; groundRadius: number;
  sunDirection: Vec3; atmosphereRadius: number;
}

type ReprojectionUniformValues = {
  forward: Vec3; frame: number;
  right: Vec3; tanHalfFov: number;
  up: Vec3; aspect: number;
  valid: number; pad0: number; pad1: number; pad2: number;
};

type CloudUniformValues = {
  bottom: number; top: number; coverage: number; density: number;
  shapeScale: number; detailScale: number; weatherScale: number; wind: number;
  detailStrength: number; groundRadius: number; pad0: number; pad1: number;
};

export interface AtmosphereGraph {
  readonly atmosphere: SharedUniforms<AtmosphereUniformValues>;
  readonly camera: SharedUniforms<CameraUniformValues>;
  readonly clouds: SharedUniforms<CloudUniformValues>;
  readonly shapeNoise: Texture;
  readonly detailNoise: Texture;
  readonly weatherMap: Texture;
  readonly terrainMap: Texture;
  /** Ping-pong cloud buffers: `write` receives this frame, `read` is last frame's history for reprojection. */
  readonly cloudsTargets: PingPongTargets;
  readonly reprojection: SharedUniforms<ReprojectionUniformValues>;
  readonly transmittance: Target;
  readonly multiScatter: Texture;
  readonly skyView: Target;
  readonly aerial: Texture;
  readonly scene: Target;
  readonly transmittanceEffect: Effect;
  readonly multiScatterCompute: Compute;
  readonly skyViewEffect: Effect;
  readonly aerialCompute: Compute;
  readonly sceneEffect: Effect;
  readonly cloudsEffect: Effect;
  readonly presentEffect: Effect;
  readonly lutPreview: Effect;
  readonly sampler: GPUSampler;
  /** stale: medium changed; transmittance: transmittance pass encoded, multi-scatter dispatch pending; ready: both tables valid. */
  lutPhase: 'stale' | 'transmittance' | 'ready';
  bakedHaze: number;
  frame: number;
  currentCamera?: CameraUniformValues;
  previousCamera?: CameraUniformValues;
}

/** Frames needed for every cloud texel to be re-marched at least once. */
export const CLOUD_CONVERGENCE_FRAMES = 16;

interface ThumbOptions {
  time?: number;
  onVariantRendered?: (variant: 'noon', pixels: Uint8Array, size: readonly [number, number]) => void | Promise<void>;
}

const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
const CLEAR = [0, 0, 0, 1] as const;
const AERIAL_WORKGROUP = 4;
const NOISE_WORKGROUP = 4;
const WEATHER_WORKGROUP = 8;
/** Keep in sync with TERRAIN_MAP_SIZE in terrain.wgsl. */
const TERRAIN_MAP_SIZE = 2048;

export async function run(canvas: HTMLCanvasElement): Promise<() => void> {
  const { init } = await import('vgpu');
  const { installControls } = await import('./controls');
  const gpu = await init();
  const surface = gpu.surface(canvas, { dpr: [1, 1.5] });
  const graph = await createGraph(gpu, surface, 'atmosphere-live');
  const controls = installControls(canvas, { ...PRESETS[DEFAULT_PRESET] });
  let disposed = false;
  let sawInitialResize = false;
  const unsubscribeResize = surface.onResize(() => {
    if (!sawInitialResize) { sawInitialResize = true; return; }
    if (disposed) return;
    resizeGraph(graph, surface.size);
  });
  const loop = gpu.frame.loop((frame) => {
    const state = { ...controls.getState(), time: gpu.time };
    applyState(graph, state, surface.size);
    renderGraph(frame, graph, surface);
  });
  return () => {
    if (disposed) return;
    disposed = true;
    loop.stop();
    unsubscribeResize();
    controls.dispose();
    destroyGraph(graph);
    surface.dispose();
    gpu.dispose();
  };
}

/** Docs thumbnail: golden hour, plus a noon variant so the thumbnail check can compare sky colour. */
export async function renderThumb(gpu: Gpu, output: Target, opts: ThumbOptions = {}): Promise<void> {
  const graph = await createGraph(gpu, output, 'atmosphere-thumb');
  renderState(gpu, graph, output, PRESETS.noon);
  await gpu.gpu.queue.onSubmittedWorkDone();
  await opts.onVariantRendered?.('noon', await output.read(), output.size);
  renderState(gpu, graph, output, PRESETS[DEFAULT_PRESET]);
  await gpu.gpu.queue.onSubmittedWorkDone();
  await gpu.settled();
  destroyGraph(graph);
}

/** Headless still for scripts: one state, one target, optional LUT debug view instead of the scene. */
export async function renderStill(gpu: Gpu, output: Target, state: AtmosphereState, debug?: DebugView): Promise<void> {
  const graph = await createGraph(gpu, output, 'atmosphere-still');
  if (debug) {
    applyState(graph, state, output.size);
    bakeLuts(gpu, graph);
    gpu.frame((frame) => frame.pass({ target: graph.skyView, clear: CLEAR }, (pass) => pass.draw(graph.skyViewEffect)));
    const sources = { transmittance: graph.transmittance, multiscatter: graph.multiScatter, weather: graph.weatherMap, terrain: graph.terrainMap, 'sky-view': graph.skyView } as const;
    const gains = { transmittance: 1, multiscatter: 1, weather: 1, terrain: 0.3, 'sky-view': 2 ** state.exposureEv } as const;
    graph.lutPreview.set({ preview: { gain: gains[debug], channel: 0, pad: [0, 0] }, lut: sources[debug], linearSampler: graph.sampler });
    await graph.lutPreview.compile(output);
    gpu.frame((frame) => frame.pass({ target: output, clear: CLEAR }, (pass) => pass.draw(graph.lutPreview)));
  } else {
    renderState(gpu, graph, output, state);
  }
  await gpu.gpu.queue.onSubmittedWorkDone();
  await gpu.settled();
  destroyGraph(graph);
}

export async function createGraph(gpu: Gpu, output: Output, label: string): Promise<AtmosphereGraph> {
  const sampler = gpu.sampler({ minFilter: 'linear', magFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge' });
  const atmosphere = gpu.uniforms<AtmosphereUniformValues>({ ...ATMOSPHERE_PHYSICS, sunDirection: [0, 1, 0] });
  const camera = gpu.uniforms<CameraUniformValues>(cameraUniforms(PRESETS[DEFAULT_PRESET], output.size));
  const clouds = gpu.uniforms<CloudUniformValues>(cloudUniforms(PRESETS[DEFAULT_PRESET]));
  const noiseSampler = gpu.sampler({ minFilter: 'linear', magFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat', addressModeW: 'repeat' });
  const transmittance = gpu.target({ size: LUT_SIZES.transmittance, format: HDR_FORMAT, label: `${label}-transmittance` });
  const multiScatter = gpu.texture({ size: [LUT_SIZES.multiScatter, LUT_SIZES.multiScatter], format: HDR_FORMAT, label: `${label}-multiscatter` });
  const skyView = gpu.target({ size: LUT_SIZES.skyView, format: HDR_FORMAT, label: `${label}-sky-view` });
  const aerial = gpu.texture({ size: [LUT_SIZES.aerial, LUT_SIZES.aerial, LUT_SIZES.aerial], format: HDR_FORMAT, dimension: '3d', label: `${label}-aerial` });
  const scene = gpu.target({ size: output.size, format: HDR_FORMAT, label: `${label}-scene` });
  const cloudSize = cloudSizeFor(output.size);
  const cloudsTargets = gpu.pingPong(cloudSize[0], cloudSize[1], { format: HDR_FORMAT, label: `${label}-clouds` });
  const reprojection = gpu.uniforms<ReprojectionUniformValues>(reprojectionUniforms(undefined, 0));
  const noise = CLOUD_TUNING.noise;
  const shapeNoise = gpu.texture({ size: [noise.shape, noise.shape, noise.shape], format: 'rgba8unorm', dimension: '3d', label: `${label}-cloud-shape` });
  const detailNoise = gpu.texture({ size: [noise.detail, noise.detail, noise.detail], format: 'rgba8unorm', dimension: '3d', label: `${label}-cloud-detail` });
  const weatherMap = gpu.texture({ size: [noise.weather, noise.weather], format: 'rgba8unorm', label: `${label}-weather` });
  const terrainMap = gpu.texture({ size: [TERRAIN_MAP_SIZE, TERRAIN_MAP_SIZE], format: HDR_FORMAT, label: `${label}-terrain` });

  const transmittanceEffect = gpu.effect(transmittanceLutWgsl, { label: `${label}-transmittance`, set: { atmosphere } });
  const multiScatterCompute = gpu.compute(multiScatterLutWgsl, { label: `${label}-multiscatter`, set: { atmosphere, transmittanceLut: transmittance, lutSampler: sampler, multiScatterLut: multiScatter } });
  const skyViewEffect = gpu.effect(skyViewLutWgsl, { label: `${label}-sky-view`, set: { atmosphere, camera, transmittanceLut: transmittance, multiScatterLut: multiScatter, lutSampler: sampler } });
  const aerialCompute = gpu.compute(aerialLutWgsl, { label: `${label}-aerial`, set: { atmosphere, camera, transmittanceLut: transmittance, multiScatterLut: multiScatter, lutSampler: sampler, aerialLut: aerial } });
  const sceneEffect = gpu.effect(sceneWgsl, { label: `${label}-scene`, set: { atmosphere, camera, transmittanceLut: transmittance, skyViewLut: skyView, aerialLut: aerial, lutSampler: sampler, clouds, weatherMap, noiseSampler, terrainMap } });
  const cloudsEffect = gpu.effect(cloudsWgsl, { label: `${label}-clouds`, set: {
    atmosphere, camera, clouds, transmittanceLut: transmittance, skyViewLut: skyView, aerialLut: aerial,
    shapeNoise, detailNoise, weatherMap, sceneHdr: scene, lutSampler: sampler, noiseSampler, history: cloudsTargets.read, reprojection,
  } });
  const presentEffect = gpu.effect(presentWgsl, { label: `${label}-present`, set: { present: { exposure: 1, tonemap: 0, dither: 1, pad: 0 }, sceneHdr: scene, cloudsHdr: cloudsTargets.write, linearSampler: sampler } });
  // Cloud noise and weather are static: generate them once with compute into storage textures.
  gpu.compute(cloudShapeNoiseWgsl, { label: `${label}-cloud-shape-noise`, set: { shapeNoise } }).dispatch(noise.shape / NOISE_WORKGROUP, noise.shape / NOISE_WORKGROUP, noise.shape / NOISE_WORKGROUP);
  gpu.compute(cloudDetailNoiseWgsl, { label: `${label}-cloud-detail-noise`, set: { detailNoise } }).dispatch(noise.detail / NOISE_WORKGROUP, noise.detail / NOISE_WORKGROUP, noise.detail / NOISE_WORKGROUP);
  gpu.compute(weatherMapWgsl, { label: `${label}-weather-map`, set: { weatherMap } }).dispatch(noise.weather / WEATHER_WORKGROUP, noise.weather / WEATHER_WORKGROUP, 1);
  // The heightfield is baked once too: the terrain march then costs one texture tap per step instead of a 6-octave fbm.
  gpu.compute(terrainHeightmapWgsl, { label: `${label}-terrain-heightmap`, set: { terrainMap } }).dispatch(TERRAIN_MAP_SIZE / WEATHER_WORKGROUP, TERRAIN_MAP_SIZE / WEATHER_WORKGROUP, 1);
  const lutPreview = gpu.effect(lutPreviewWgsl, { label: `${label}-lut-preview` });

  const graph: AtmosphereGraph = {
    atmosphere, camera, clouds, shapeNoise, detailNoise, weatherMap, terrainMap, cloudsTargets, reprojection, transmittance, multiScatter, skyView, aerial, scene,
    transmittanceEffect, multiScatterCompute, skyViewEffect, aerialCompute, sceneEffect, cloudsEffect, presentEffect, lutPreview, sampler,
    lutPhase: 'stale', bakedHaze: 1, frame: 0,
  };
  await Promise.all([
    transmittanceEffect.compile(transmittance),
    skyViewEffect.compile(skyView),
    sceneEffect.compile(scene),
    cloudsEffect.compile(cloudsTargets.write),
    presentEffect.compile({ colors: [output.format] }),
  ]);
  return graph;
}

/** Transmittance and multi-scattering only depend on the medium: bake both up front outside a frame loop. */
export function bakeLuts(gpu: Gpu, graph: AtmosphereGraph): void {
  gpu.frame((frame) => encodeTransmittance(frame, graph));
  dispatchMultiScatter(graph);
}

function encodeTransmittance(frame: Frame, graph: AtmosphereGraph): void {
  frame.pass({ target: graph.transmittance, clear: CLEAR }, (pass) => pass.draw(graph.transmittanceEffect));
  graph.lutPhase = 'transmittance';
}

/** Reads the transmittance table, so it must run after the frame that encoded it has been submitted. */
function dispatchMultiScatter(graph: AtmosphereGraph): void {
  graph.multiScatterCompute.dispatch(LUT_SIZES.multiScatter, LUT_SIZES.multiScatter, 1);
  graph.lutPhase = 'ready';
}

export function applyState(graph: AtmosphereGraph, state: AtmosphereState, size: readonly [number, number]): void {
  const haze = Math.max(0.01, state.haze);
  graph.atmosphere.set({
    sunDirection: sunDirection(state),
    mieScattering: scale(ATMOSPHERE_PHYSICS.mieScattering, haze),
    mieAbsorption: scale(ATMOSPHERE_PHYSICS.mieAbsorption, haze),
  });
  graph.camera.set(cameraUniforms(state, size));
  graph.clouds.set(cloudUniforms(state));
  graph.currentCamera = cameraUniforms(state, size);
  graph.presentEffect.set({ present: { exposure: 2 ** state.exposureEv, tonemap: TONEMAPS.indexOf(state.tonemap), dither: 1, pad: 0 } });
  // The medium changed, so the baked transmittance and multi-scattering tables are stale.
  if (graph.bakedHaze !== haze) graph.lutPhase = 'stale';
  graph.bakedHaze = haze;
}

/**
 * Per-frame work: compute dispatches submit immediately, so they run before this frame's passes.
 * A stale medium re-encodes transmittance in this frame and dispatches multi-scatter on the next one.
 */
export function renderGraph(frame: Frame, graph: AtmosphereGraph, output: Output): void {
  if (graph.lutPhase === 'transmittance') dispatchMultiScatter(graph);
  const groups = LUT_SIZES.aerial / AERIAL_WORKGROUP;
  graph.aerialCompute.dispatch(groups, groups, groups);
  if (graph.lutPhase === 'stale') encodeTransmittance(frame, graph);
  frame.pass({ target: graph.skyView, clear: CLEAR }, (pass) => pass.draw(graph.skyViewEffect));
  frame.pass({ target: graph.scene, clear: CLEAR }, (pass) => pass.draw(graph.sceneEffect));
  // Sixteenth-rate cloud update: this frame's texels are marched, the rest are reprojected from last frame's buffer.
  graph.reprojection.set(reprojectionUniforms(graph.previousCamera, graph.frame));
  graph.cloudsEffect.set({ history: graph.cloudsTargets.read });
  graph.presentEffect.set({ cloudsHdr: graph.cloudsTargets.write });
  frame.pass({ target: graph.cloudsTargets.write, clear: [0, 0, 0, 1] }, (pass) => pass.draw(graph.cloudsEffect));
  frame.pass({ target: output, clear: CLEAR }, (pass) => pass.draw(graph.presentEffect));
  graph.cloudsTargets.swap();
  graph.previousCamera = graph.currentCamera;
  graph.frame += 1;
}

/** Stills render enough frames for the temporal cloud update to touch every texel. */
function renderState(gpu: Gpu, graph: AtmosphereGraph, output: Target, state: AtmosphereState): void {
  applyState(graph, state, output.size);
  if (graph.lutPhase !== 'ready') bakeLuts(gpu, graph);
  for (let i = 0; i < CLOUD_CONVERGENCE_FRAMES; i++) gpu.frame((frame) => renderGraph(frame, graph, output));
}

function reprojectionUniforms(previous: CameraUniformValues | undefined, frame: number): ReprojectionUniformValues {
  return {
    forward: previous?.forward ?? [0, 0, 1], frame,
    right: previous?.right ?? [1, 0, 0], tanHalfFov: previous?.tanHalfFov ?? 1,
    up: previous?.up ?? [0, 1, 0], aspect: previous?.aspect ?? 1,
    valid: previous ? 1 : 0, pad0: 0, pad1: 0, pad2: 0,
  };
}

function resizeGraph(graph: AtmosphereGraph, size: readonly [number, number]): void {
  graph.scene.resize(size);
  const cloudSize = cloudSizeFor(size);
  graph.cloudsTargets.read.resize(cloudSize);
  graph.cloudsTargets.write.resize(cloudSize);
  // The history no longer matches the new size; re-march every texel on the next frame.
  graph.previousCamera = undefined;
}

function cloudSizeFor(size: readonly [number, number]): readonly [number, number] {
  return [Math.max(1, Math.round(size[0] / CLOUD_TUNING.renderScale)), Math.max(1, Math.round(size[1] / CLOUD_TUNING.renderScale))];
}

function cloudUniforms(state: AtmosphereState): CloudUniformValues {
  return {
    bottom: CLOUD_TUNING.bottom, top: CLOUD_TUNING.top, coverage: Math.min(1, Math.max(0, state.cloudCoverage)), density: CLOUD_TUNING.density,
    shapeScale: CLOUD_TUNING.shapeScale, detailScale: CLOUD_TUNING.detailScale, weatherScale: CLOUD_TUNING.weatherScale, wind: state.time * CLOUD_TUNING.windSpeed,
    detailStrength: CLOUD_TUNING.detailStrength, groundRadius: ATMOSPHERE_PHYSICS.groundRadius, pad0: 0, pad1: 0,
  };
}

function scale(v: Vec3, factor: number): Vec3 { return [v[0] * factor, v[1] * factor, v[2] * factor]; }

function destroyGraph(graph: AtmosphereGraph): void {
  for (const target of [graph.transmittance, graph.skyView, graph.scene, graph.cloudsTargets.read, graph.cloudsTargets.write]) target.color.destroy();
  for (const texture of [graph.multiScatter, graph.aerial, graph.shapeNoise, graph.detailNoise, graph.weatherMap, graph.terrainMap]) texture.destroy();
}
