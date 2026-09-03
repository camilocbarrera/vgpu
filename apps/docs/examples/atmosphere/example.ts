import type { Compute, Effect, Frame, Gpu, SharedUniforms, Surface, Target, Texture } from 'vgpu';
import { cameraUniforms, sunDirection, type CameraUniformValues } from './camera';
import { ATMOSPHERE_PHYSICS, DEFAULT_PRESET, LUT_SIZES, PRESETS, TONEMAPS, type AtmosphereState } from './tuning';
import transmittanceLutWgsl from './transmittance-lut.wgsl';
import multiScatterLutWgsl from './multiscatter-lut.wgsl';
import skyViewLutWgsl from './sky-view-lut.wgsl';
import aerialLutWgsl from './aerial-lut.wgsl';
import sceneWgsl from './scene.wgsl';
import presentWgsl from './present.wgsl';
import lutPreviewWgsl from './lut-preview.wgsl';

type Output = Surface | Target;
type Vec3 = readonly [number, number, number];
export type DebugView = 'transmittance' | 'multiscatter' | 'sky-view';

type AtmosphereUniformValues = {
  rayleighScattering: Vec3; rayleighScaleHeight: number;
  mieScattering: Vec3; mieScaleHeight: number;
  mieAbsorption: Vec3; mieG: number;
  ozoneAbsorption: Vec3; ozoneCenter: number;
  groundAlbedo: Vec3; ozoneWidth: number;
  sunIlluminance: Vec3; groundRadius: number;
  sunDirection: Vec3; atmosphereRadius: number;
}

export interface AtmosphereGraph {
  readonly atmosphere: SharedUniforms<AtmosphereUniformValues>;
  readonly camera: SharedUniforms<CameraUniformValues>;
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
  readonly presentEffect: Effect;
  readonly lutPreview: Effect;
  readonly sampler: GPUSampler;
  /** stale: medium changed; transmittance: transmittance pass encoded, multi-scatter dispatch pending; ready: both tables valid. */
  lutPhase: 'stale' | 'transmittance' | 'ready';
  bakedHaze: number;
}

interface ThumbOptions {
  time?: number;
  onVariantRendered?: (variant: 'noon', pixels: Uint8Array, size: readonly [number, number]) => void | Promise<void>;
}

const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
const CLEAR = [0, 0, 0, 1] as const;
const AERIAL_WORKGROUP = 4;

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
    const state = controls.getState();
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
    const source = debug === 'transmittance' ? graph.transmittance : debug === 'multiscatter' ? graph.multiScatter : graph.skyView;
    graph.lutPreview.set({ preview: { gain: debug === 'sky-view' ? 2 ** state.exposureEv : 1, channel: 0, pad: [0, 0] }, lut: source, linearSampler: graph.sampler });
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
  const transmittance = gpu.target({ size: LUT_SIZES.transmittance, format: HDR_FORMAT, label: `${label}-transmittance` });
  const multiScatter = gpu.texture({ size: [LUT_SIZES.multiScatter, LUT_SIZES.multiScatter], format: HDR_FORMAT, label: `${label}-multiscatter` });
  const skyView = gpu.target({ size: LUT_SIZES.skyView, format: HDR_FORMAT, label: `${label}-sky-view` });
  const aerial = gpu.texture({ size: [LUT_SIZES.aerial, LUT_SIZES.aerial, LUT_SIZES.aerial], format: HDR_FORMAT, dimension: '3d', label: `${label}-aerial` });
  const scene = gpu.target({ size: output.size, format: HDR_FORMAT, label: `${label}-scene` });

  const transmittanceEffect = gpu.effect(transmittanceLutWgsl, { label: `${label}-transmittance`, set: { atmosphere } });
  const multiScatterCompute = gpu.compute(multiScatterLutWgsl, { label: `${label}-multiscatter`, set: { atmosphere, transmittanceLut: transmittance, lutSampler: sampler, multiScatterLut: multiScatter } });
  const skyViewEffect = gpu.effect(skyViewLutWgsl, { label: `${label}-sky-view`, set: { atmosphere, camera, transmittanceLut: transmittance, multiScatterLut: multiScatter, lutSampler: sampler } });
  const aerialCompute = gpu.compute(aerialLutWgsl, { label: `${label}-aerial`, set: { atmosphere, camera, transmittanceLut: transmittance, multiScatterLut: multiScatter, lutSampler: sampler, aerialLut: aerial } });
  const sceneEffect = gpu.effect(sceneWgsl, { label: `${label}-scene`, set: { atmosphere, camera, transmittanceLut: transmittance, skyViewLut: skyView, aerialLut: aerial, lutSampler: sampler } });
  const presentEffect = gpu.effect(presentWgsl, { label: `${label}-present`, set: { present: { exposure: 1, tonemap: 0, dither: 1, pad: 0 }, sceneHdr: scene, linearSampler: sampler } });
  const lutPreview = gpu.effect(lutPreviewWgsl, { label: `${label}-lut-preview` });

  const graph: AtmosphereGraph = {
    atmosphere, camera, transmittance, multiScatter, skyView, aerial, scene,
    transmittanceEffect, multiScatterCompute, skyViewEffect, aerialCompute, sceneEffect, presentEffect, lutPreview, sampler,
    lutPhase: 'stale', bakedHaze: 1,
  };
  await Promise.all([
    transmittanceEffect.compile(transmittance),
    skyViewEffect.compile(skyView),
    sceneEffect.compile(scene),
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
  frame.pass({ target: output, clear: CLEAR }, (pass) => pass.draw(graph.presentEffect));
}

function renderState(gpu: Gpu, graph: AtmosphereGraph, output: Target, state: AtmosphereState): void {
  applyState(graph, state, output.size);
  if (graph.lutPhase !== 'ready') bakeLuts(gpu, graph);
  gpu.frame((frame) => renderGraph(frame, graph, output));
}

function resizeGraph(graph: AtmosphereGraph, size: readonly [number, number]): void {
  graph.scene.resize(size);
}

function scale(v: Vec3, factor: number): Vec3 { return [v[0] * factor, v[1] * factor, v[2] * factor]; }

function destroyGraph(graph: AtmosphereGraph): void {
  for (const target of [graph.transmittance, graph.skyView, graph.scene]) target.color.destroy();
  graph.multiScatter.destroy();
  graph.aerial.destroy();
}
