import {
  clock as createClock,
  compute as createCompute,
  draw as createDraw,
  effect as createEffect,
  frame as createFrame,
  frameLoop,
  pingPong,
  sampler as createSampler,
  storage as createStorage,
  surface as createSurface,
  target as createTarget,
  texture as createTexture,
  uniforms as createUniforms,
  type Compute,
  type Draw,
  type Effect,
  type Frame,
  type Gpu,
  type PingPongTargets,
  type SharedUniforms,
  type StorageBuffer,
  type Surface,
  type Target,
  type Texture,
} from 'vgpu';
import { cameraUniforms, sunDirection, terrainSector, type CameraUniformValues } from './camera';
import { ATMOSPHERE_PHYSICS, CLOUD_TUNING, DEFAULT_PRESET, LUT_SIZES, PRESETS, TONEMAPS, type AtmosphereState } from './tuning';
import transmittanceLutWgsl from './transmittance-lut.wgsl';
import multiScatterLutWgsl from './multiscatter-lut.wgsl';
import skyViewLutWgsl from './sky-view-lut.wgsl';
import aerialLutWgsl from './aerial-lut.wgsl';
import sceneWgsl from './scene.wgsl';
import terrainDepthWgsl from './terrain-depth.wgsl';
import presentWgsl from './present.wgsl';
import lutPreviewWgsl from './lut-preview.wgsl';
import cloudShapeNoiseWgsl from './cloud-shape-noise.wgsl';
import cloudDetailNoiseWgsl from './cloud-detail-noise.wgsl';
import weatherMapWgsl from './weather-map.wgsl';
import cloudsWgsl from './clouds.wgsl';
import terrainHeightmapWgsl from './terrain-heightmap.wgsl';
import curlNoiseWgsl from './curl-noise.wgsl';
import frameConstantsWgsl from './frame-constants.wgsl';
import terrainShadowWgsl from './terrain-shadow.wgsl';

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
  position: Vec3; valid: number;
  blend: number; refreshPeriod: number; jitter: readonly [number, number];
};

type TerrainMeshUniformValues = { columnOffset: number; columns: number };

type CloudUniformValues = {
  bottom: number; top: number; coverage: number; density: number;
  shapeScale: number; detailScale: number; weatherScale: number; wind: number;
  detailStrength: number; groundRadius: number; curlStrength: number; detailLodDistance: number;
  typeBias: number; seed: number; pad0: number; pad1: number;
};

export interface AtmosphereGraph {
  readonly atmosphere: SharedUniforms<AtmosphereUniformValues>;
  readonly camera: SharedUniforms<CameraUniformValues>;
  readonly clouds: SharedUniforms<CloudUniformValues>;
  /** Which columns of the terrain ring grid this frame draws. */
  readonly terrainMesh: SharedUniforms<TerrainMeshUniformValues>;
  readonly shapeNoise: Texture;
  readonly detailNoise: Texture;
  readonly weatherMap: Texture;
  readonly curlNoise: Texture;
  readonly terrainMap: Texture;
  readonly terrainAlbedoMap: Texture;
  /** Altitude below which the air over each heightmap texel is in terrain shadow; depends only on the sun. */
  readonly terrainShadowMap: Texture;
  /** Ping-pong cloud buffers: `write` receives this frame, `read` is last frame's history for reprojection. */
  readonly cloudsTargets: PingPongTargets;
  readonly reprojection: SharedUniforms<ReprojectionUniformValues>;
  readonly transmittance: Target;
  readonly multiScatter: Texture;
  readonly skyView: Target;
  readonly aerial: Texture;
  /** Single scattering the aerial pass removed as terrain-shadowed, taken out of the sky-view LUT by sky pixels. */
  readonly aerialLoss: Texture;
  /** Depth prepass of the terrain ring grid (reversed-Z, depth32float); its color is a masked-out dummy. */
  readonly terrainDepth: Target;
  readonly scene: Target;
  readonly transmittanceEffect: Effect;
  readonly multiScatterCompute: Compute;
  readonly skyViewEffect: Effect;
  readonly aerialCompute: Compute;
  readonly terrainShadowCompute: Compute;
  /** Per-frame constants (sky ambient, sun disc trig, horizon terms) baked by a one-thread compute into a storage buffer. */
  readonly frameConstants: StorageBuffer;
  readonly frameConstantsCompute: Compute;
  /** Terrain ring grid, one triangle strip per visible column, depth only. */
  readonly terrainDraw: Draw;
  /** Shades every pixel once: terrain where the prepass left depth, sky and bare sphere elsewhere. */
  readonly sceneEffect: Effect;
  readonly cloudsEffect: Effect;
  readonly presentEffect: Effect;
  readonly lutPreview: Effect;
  readonly sampler: GPUSampler;
  /** stale: medium changed; transmittance: transmittance pass encoded, multi-scatter dispatch pending; ready: both tables valid. */
  lutPhase: 'stale' | 'transmittance' | 'ready';
  bakedHaze: number;
  sunDirection: Vec3;
  /** Instances of the terrain strip to draw this frame (terrainSector); 0 when the frustum looks above all terrain. */
  terrainColumns: number;
  /** Sun the terrain shadow map was last built for; undefined until the first frame. */
  bakedSunDirection?: Vec3;
  frame: number;
  /** Live rendering blends re-marched cloud texels into their jittered history; stills keep it off to stay deterministic. */
  accumulate: boolean;
  /**
   * Frames left of fast cloud refresh: a lighting, cloud or altitude change stales the whole history, so the next
   * CLOUD_FAST_REFRESH_PERIOD frames re-march one texel in two (a checkerboard) with full blend instead of one in
   * sixteen accumulated. Two frames of a fine checkerboard read as a quick crossfade; four read as a dot pattern.
   */
  cloudChangeFrames: number;
  cloudStateKey?: string;
  currentCamera?: CameraUniformValues;
  previousCamera?: CameraUniformValues;
}

/** Frames needed for every cloud texel to be re-marched at least once, at rest and right after a change. */
export const CLOUD_CONVERGENCE_FRAMES = 16;
export const CLOUD_FAST_REFRESH_PERIOD = 2;

export interface ThumbOptions {
  time?: number;
  onVariantRendered?: (variant: 'noon', pixels: Uint8Array, size: readonly [number, number]) => void | Promise<void>;
}

const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
const MAX_DPR = 1;
const MAX_FPS = 60;
const CLEAR = [0, 0, 0, 1] as const;
const AERIAL_WORKGROUP = 4;
const NOISE_WORKGROUP = 4;
const WEATHER_WORKGROUP = 8;
/** Keep in sync with TERRAIN_MAP_SIZE in terrain.wgsl. */
const TERRAIN_MAP_SIZE = 2048;
/** Keep in sync with TERRAIN_SHADOW_MAP_SIZE in terrain.wgsl. */
const TERRAIN_SHADOW_MAP_SIZE = 512;
/** Keep in sync with TERRAIN_MESH_COLUMNS and TERRAIN_MESH_RINGS in terrain.wgsl. */
const TERRAIN_MESH_COLUMNS = 4096;
const TERRAIN_MESH_RINGS = 512;
/** Keep in sync with SIZE in curl-noise.wgsl. */
const CURL_SIZE = 128;
/** Size of FrameConstants in atmosphere-common.wgsl: four 16-byte rows plus the 64-entry terrain transmittance table. */
const FRAME_CONSTANTS_BYTES = 64 + 64 * 16;

export async function run(canvas: HTMLCanvasElement): Promise<() => void> {
  const { init } = await import('vgpu');
  const { installControls } = await import('./controls');
  // `?bench` in the URL times the passes of a frame on this GPU (bench.ts) before the live loop starts.
  const bench = typeof location !== 'undefined' && new URLSearchParams(location.search).has('bench') ? await import('./bench') : undefined;
  const gpu = await init();
  // Device pixels and frame rate are capped: the frame cost is linear in pixels and this is a laptop demo.
  const surface = createSurface(gpu, canvas, { dpr: MAX_DPR });
  const graph = await createGraph(gpu, surface, 'atmosphere-live');
  graph.accumulate = true;
  if (bench) await bench.mountBenchReport(canvas, await bench.runBench(gpu, [canvas.clientWidth, canvas.clientHeight]));
  const controls = installControls(canvas, { ...PRESETS[DEFAULT_PRESET] });
  let disposed = false;
  let sawInitialResize = false;
  const unsubscribeResize = surface.onResize(() => {
    if (!sawInitialResize) { sawInitialResize = true; return; }
    if (disposed) return;
    resizeGraph(graph, surface.size);
  });
  const timeline = createClock(gpu);
  let fpsWindowStart = performance.now();
  let fpsWindowFrames = 0;
  const loop = frameLoop(gpu, (frame) => {
    const state = { ...controls.getState(), time: timeline.time };
    applyState(graph, state, surface.size);
    renderGraph(frame, graph, surface);
    // Frame rate over half-second windows, so the cap and the cost of a change are visible in the panel.
    fpsWindowFrames += 1;
    const elapsed = performance.now() - fpsWindowStart;
    if (elapsed >= 500) {
      controls.setFps(fpsWindowFrames * 1000 / elapsed);
      fpsWindowStart += elapsed;
      fpsWindowFrames = 0;
    }
  }, { fps: MAX_FPS });
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

type RendererRun = (canvas: HTMLCanvasElement) => Promise<() => void>;

/**
 * A canvas context is shared even when separate vgpu instances configure it.
 * Keep each renderer's complete lifetime serialized so a late Strict Mode
 * cleanup cannot unconfigure the context owned by the replacement renderer.
 */
const canvasRendererLifetimes = new WeakMap<HTMLCanvasElement, Promise<void>>();

export function createRenderer(
  { canvas }: { readonly canvas: HTMLCanvasElement },
  start: RendererRun = run,
) {
  let cleanup: (() => void) | undefined;
  let disposed = false;
  let released = false;
  let releaseLifetime!: () => void;

  const previousLifetime = canvasRendererLifetimes.get(canvas) ?? Promise.resolve();
  const lifetime = new Promise<void>((resolve) => {
    releaseLifetime = resolve;
  });
  const queuedLifetime = previousLifetime.catch(() => undefined).then(() => lifetime);
  canvasRendererLifetimes.set(canvas, queuedLifetime);

  const release = () => {
    if (released) return;
    released = true;
    releaseLifetime();
    if (canvasRendererLifetimes.get(canvas) === queuedLifetime) {
      canvasRendererLifetimes.delete(canvas);
    }
  };

  const ready = previousLifetime
    .catch(() => undefined)
    .then(async () => {
      if (disposed) {
        release();
        return;
      }

      const nextCleanup = await start(canvas);
      if (disposed) {
        try {
          nextCleanup();
        } finally {
          release();
        }
      } else {
        cleanup = nextCleanup;
      }
    })
    .catch((error: unknown) => {
      release();
      if (!disposed) throw error;
    });

  return {
    ready,
    dispose(): void {
      if (disposed) return;
      disposed = true;
      if (!cleanup) return;
      const disposeRenderer = cleanup;
      cleanup = undefined;
      try {
        disposeRenderer();
      } finally {
        release();
      }
    },
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
    createFrame(gpu, (frame) => encodeSkyView(frame, graph));
    const sources = { transmittance: graph.transmittance, multiscatter: graph.multiScatter, weather: graph.weatherMap, terrain: graph.terrainMap, 'sky-view': graph.skyView } as const;
    const gains = { transmittance: 1, multiscatter: 1, weather: 1, terrain: 0.3, 'sky-view': 2 ** state.exposureEv } as const;
    graph.lutPreview.set({ preview: { gain: gains[debug], channel: 0, pad: [0, 0] }, lut: sources[debug], linearSampler: graph.sampler });
    await graph.lutPreview.compile(output);
    createFrame(gpu, (frame) => frame.pass({ target: output, clear: CLEAR }, (pass) => pass.draw(graph.lutPreview)));
  } else {
    renderState(gpu, graph, output, state);
  }
  await gpu.gpu.queue.onSubmittedWorkDone();
  await gpu.settled();
  destroyGraph(graph);
}

export async function createGraph(gpu: Gpu, output: Output, label: string): Promise<AtmosphereGraph> {
  const sampler = createSampler(gpu, { minFilter: 'linear', magFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge' });
  const atmosphere = createUniforms<AtmosphereUniformValues>(gpu, { ...ATMOSPHERE_PHYSICS, sunDirection: [0, 1, 0] });
  const camera = createUniforms<CameraUniformValues>(gpu, cameraUniforms(PRESETS[DEFAULT_PRESET], output.size));
  const clouds = createUniforms<CloudUniformValues>(gpu, cloudUniforms(PRESETS[DEFAULT_PRESET]));
  const terrainMesh = createUniforms<TerrainMeshUniformValues>(gpu, { columnOffset: 0, columns: 0 });
  const noiseSampler = createSampler(gpu, { minFilter: 'linear', magFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat', addressModeW: 'repeat' });
  const transmittance = createTarget(gpu, { size: LUT_SIZES.transmittance, format: HDR_FORMAT, label: `${label}-transmittance` });
  const multiScatter = createTexture(gpu, { size: [LUT_SIZES.multiScatter, LUT_SIZES.multiScatter], format: HDR_FORMAT, label: `${label}-multiscatter` });
  const skyView = createTarget(gpu, { size: LUT_SIZES.skyView, format: HDR_FORMAT, label: `${label}-sky-view` });
  const aerial = createTexture(gpu, { size: [LUT_SIZES.aerial, LUT_SIZES.aerial, LUT_SIZES.aerial], format: HDR_FORMAT, dimension: '3d', label: `${label}-aerial` });
  const aerialLoss = createTexture(gpu, { size: [LUT_SIZES.aerial, LUT_SIZES.aerial, LUT_SIZES.aerial], format: HDR_FORMAT, dimension: '3d', label: `${label}-aerial-loss` });
  const terrainDepth = createTarget(gpu, { size: output.size, format: 'r8unorm', depth: 'depth32float', label: `${label}-terrain-depth` });
  const scene = createTarget(gpu, { size: output.size, format: HDR_FORMAT, label: `${label}-scene` });
  const cloudSize = cloudSizeFor(output.size);
  // Two attachments: premultiplied luminance + transmittance, and the mean cloud depth the reprojection needs.
  const cloudsTargets = pingPong(gpu, cloudSize[0], cloudSize[1], { colors: [{ format: HDR_FORMAT }, { format: 'r16float' }], label: `${label}-clouds` });
  const reprojection = createUniforms<ReprojectionUniformValues>(gpu, reprojectionUniforms(undefined, 0, false, false));
  const noise = CLOUD_TUNING.noise;
  const shapeNoise = createTexture(gpu, { size: [noise.shape, noise.shape, noise.shape], format: 'rgba8unorm', dimension: '3d', label: `${label}-cloud-shape` });
  const detailNoise = createTexture(gpu, { size: [noise.detail, noise.detail, noise.detail], format: 'rgba8unorm', dimension: '3d', label: `${label}-cloud-detail` });
  const weatherMap = createTexture(gpu, { size: [noise.weather, noise.weather], format: 'rgba8unorm', label: `${label}-weather` });
  const terrainMap = createTexture(gpu, { size: [TERRAIN_MAP_SIZE, TERRAIN_MAP_SIZE], format: HDR_FORMAT, label: `${label}-terrain` });
  const terrainAlbedoMap = createTexture(gpu, { size: [TERRAIN_MAP_SIZE, TERRAIN_MAP_SIZE], format: 'rgba8unorm', label: `${label}-terrain-albedo` });
  const terrainShadowMap = createTexture(gpu, { size: [TERRAIN_SHADOW_MAP_SIZE, TERRAIN_SHADOW_MAP_SIZE], format: HDR_FORMAT, label: `${label}-terrain-shadow` });
  const curlNoise = createTexture(gpu, { size: [CURL_SIZE, CURL_SIZE], format: 'rgba8unorm', label: `${label}-curl` });

  const transmittanceEffect = createEffect(gpu, transmittanceLutWgsl, { label: `${label}-transmittance`, set: { atmosphere } });
  const multiScatterCompute = createCompute(gpu, multiScatterLutWgsl, { label: `${label}-multiscatter`, set: { atmosphere, transmittanceLut: transmittance, lutSampler: sampler, multiScatterLut: multiScatter } });
  const skyViewEffect = createEffect(gpu, skyViewLutWgsl, { label: `${label}-sky-view`, set: { atmosphere, camera, transmittanceLut: transmittance, multiScatterLut: multiScatter, lutSampler: sampler } });
  const aerialCompute = createCompute(gpu, aerialLutWgsl, { label: `${label}-aerial`, set: { atmosphere, camera, transmittanceLut: transmittance, multiScatterLut: multiScatter, lutSampler: sampler, aerialLut: aerial, terrainShadowMap, aerialLossLut: aerialLoss } });
  const terrainShadowCompute = createCompute(gpu, terrainShadowWgsl, { label: `${label}-terrain-shadow`, set: { atmosphere, terrainMap, lutSampler: sampler, terrainShadowMap } });
  const frameConstants = createStorage(gpu, FRAME_CONSTANTS_BYTES, 'read-write');
  const frameConstantsCompute = createCompute(gpu, frameConstantsWgsl, { label: `${label}-frame-constants`, set: { atmosphere, camera, transmittanceLut: transmittance, skyViewLut: skyView.color, lutSampler: sampler, frameConstants, terrainMap } });
  const terrainDraw = createDraw(gpu, {
    shader: terrainDepthWgsl,
    label: `${label}-terrain-depth`,
    geometry: { topology: 'triangle-strip', vertexCount: 2 * (TERRAIN_MESH_RINGS + 1) },
    depth: { compare: 'greater' },
    writeMask: [],
    set: { atmosphere, camera, mesh: terrainMesh, terrainMap, lutSampler: sampler },
  });
  const sceneEffect = createEffect(gpu, sceneWgsl, { label: `${label}-scene`, set: { atmosphere, camera, transmittanceLut: transmittance, skyViewLut: skyView, aerialLut: aerial, lutSampler: sampler, clouds, weatherMap, noiseSampler, terrainMap, terrainAlbedoMap, frame: frameConstants, aerialLossLut: aerialLoss, terrainDepth } });
  const cloudsEffect = createEffect(gpu, cloudsWgsl, { label: `${label}-clouds`, set: {
    atmosphere, camera, clouds, transmittanceLut: transmittance, skyViewLut: skyView, aerialLut: aerial,
    shapeNoise, detailNoise, weatherMap, curlNoise, sceneHdr: scene, lutSampler: sampler, noiseSampler, history: cloudsTargets.read, historyDepth: cloudsTargets.read.colors[1], reprojection, frame: frameConstants,
  } });
  const presentEffect = createEffect(gpu, presentWgsl, { label: `${label}-present`, set: { present: { exposure: 1, tonemap: 0, dither: 1, pad: 0 }, sceneHdr: scene, cloudsHdr: cloudsTargets.write, linearSampler: sampler } });
  // Cloud noise and weather are static: generate them once with compute into storage textures.
  createCompute(gpu, cloudShapeNoiseWgsl, { label: `${label}-cloud-shape-noise`, set: { shapeNoise } }).dispatch(noise.shape / NOISE_WORKGROUP, noise.shape / NOISE_WORKGROUP, noise.shape / NOISE_WORKGROUP);
  createCompute(gpu, cloudDetailNoiseWgsl, { label: `${label}-cloud-detail-noise`, set: { detailNoise } }).dispatch(noise.detail / NOISE_WORKGROUP, noise.detail / NOISE_WORKGROUP, noise.detail / NOISE_WORKGROUP);
  createCompute(gpu, weatherMapWgsl, { label: `${label}-weather-map`, set: { weatherMap } }).dispatch(noise.weather / WEATHER_WORKGROUP, noise.weather / WEATHER_WORKGROUP, 1);
  createCompute(gpu, curlNoiseWgsl, { label: `${label}-curl-noise`, set: { curlNoise } }).dispatch(CURL_SIZE / WEATHER_WORKGROUP, CURL_SIZE / WEATHER_WORKGROUP, 1);
  // The heightfield is baked once too: the terrain march then costs one texture tap per step instead of a 6-octave fbm.
  createCompute(gpu, terrainHeightmapWgsl, { label: `${label}-terrain-heightmap`, set: { terrainMap, albedoMap: terrainAlbedoMap } }).dispatch(TERRAIN_MAP_SIZE / WEATHER_WORKGROUP, TERRAIN_MAP_SIZE / WEATHER_WORKGROUP, 1);
  const lutPreview = createEffect(gpu, lutPreviewWgsl, { label: `${label}-lut-preview` });

  const graph: AtmosphereGraph = {
    atmosphere, camera, clouds, terrainMesh, shapeNoise, detailNoise, weatherMap, curlNoise, terrainMap, terrainAlbedoMap, terrainShadowMap, cloudsTargets, reprojection, transmittance, multiScatter, skyView, aerial, aerialLoss, terrainDepth, scene,
    transmittanceEffect, multiScatterCompute, skyViewEffect, aerialCompute, terrainShadowCompute, frameConstants, frameConstantsCompute, terrainDraw, sceneEffect, cloudsEffect, presentEffect, lutPreview, sampler,
    lutPhase: 'stale', bakedHaze: 1, frame: 0, accumulate: false, cloudChangeFrames: 0, sunDirection: sunDirection(PRESETS[DEFAULT_PRESET]), terrainColumns: 0,
  };
  await Promise.all([
    transmittanceEffect.compile(transmittance),
    skyViewEffect.compile(skyView),
    terrainDraw.compile(terrainDepth),
    sceneEffect.compile(scene),
    cloudsEffect.compile(cloudsTargets.write),
    presentEffect.compile({ colors: [output.format] }),
  ]);
  return graph;
}

/** Transmittance and multi-scattering only depend on the medium: bake both up front outside a frame loop. */
export function bakeLuts(gpu: Gpu, graph: AtmosphereGraph): void {
  createFrame(gpu, (frame) => encodeTransmittance(frame, graph));
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
  graph.sunDirection = sunDirection(state);
  graph.atmosphere.set({
    sunDirection: graph.sunDirection,
    mieScattering: scale(ATMOSPHERE_PHYSICS.mieScattering, haze),
    mieAbsorption: scale(ATMOSPHERE_PHYSICS.mieAbsorption, haze),
  });
  graph.currentCamera = cameraUniforms(state, size);
  graph.camera.set(graph.currentCamera);
  graph.clouds.set(cloudUniforms(state));
  const sector = terrainSector(graph.currentCamera, TERRAIN_MESH_COLUMNS);
  graph.terrainMesh.set({ columnOffset: sector.first, columns: sector.count });
  graph.terrainColumns = sector.count;
  graph.presentEffect.set({ present: { exposure: 2 ** state.exposureEv, tonemap: TONEMAPS.indexOf(state.tonemap), dither: 1, pad: 0 } });
  // The medium changed, so the baked transmittance and multi-scattering tables are stale.
  if (graph.bakedHaze !== haze) graph.lutPhase = 'stale';
  graph.bakedHaze = haze;
  // Anything that changes how a cloud texel looks (not where it is: rotation reprojects) stales the cloud history.
  const cloudStateKey = [state.sunElevation, state.sunAzimuth, state.altitudeKm, haze, state.cloudCoverage, state.cloudDetail, state.cloudType, state.cloudSeed].join(',');
  if (graph.cloudStateKey !== undefined && graph.cloudStateKey !== cloudStateKey) graph.cloudChangeFrames = CLOUD_FAST_REFRESH_PERIOD;
  graph.cloudStateKey = cloudStateKey;
}

/**
 * Per-frame work: compute dispatches submit immediately, so they run before this frame's passes.
 * A stale medium re-encodes transmittance in this frame and dispatches multi-scatter on the next one.
 * Each pass is its own function so bench.ts can time them one at a time.
 */
export function renderGraph(frame: Frame, graph: AtmosphereGraph, output: Output): void {
  if (graph.lutPhase === 'transmittance') dispatchMultiScatter(graph);
  encodeTerrainShadow(graph);
  encodeAerial(graph);
  encodeFrameConstants(graph);
  if (graph.lutPhase === 'stale') encodeTransmittance(frame, graph);
  encodeSkyView(frame, graph);
  encodeScene(frame, graph);
  encodeClouds(frame, graph);
  encodePresent(frame, graph, output);
  finishFrame(graph);
}

/** The terrain shadow map depends only on the sun (the heightmap is static): rebuilt when the sun moves. */
export function encodeTerrainShadow(graph: AtmosphereGraph): void {
  if (sameDirection(graph.bakedSunDirection, graph.sunDirection)) return;
  const groups = TERRAIN_SHADOW_MAP_SIZE / WEATHER_WORKGROUP;
  graph.terrainShadowCompute.dispatch(groups, groups, 1);
  graph.bakedSunDirection = graph.sunDirection;
}

export function encodeAerial(graph: AtmosphereGraph): void {
  const groups = LUT_SIZES.aerial / AERIAL_WORKGROUP;
  graph.aerialCompute.dispatch(groups, groups, groups);
}

/** Reads the sky-view LUT of the previous frame; stills pre-render one sky-view pass so it is already current. */
export function encodeFrameConstants(graph: AtmosphereGraph): void {
  graph.frameConstantsCompute.dispatch(1);
}

export function encodeSkyView(frame: Frame, graph: AtmosphereGraph): void {
  frame.pass({ target: graph.skyView, clear: CLEAR }, (pass) => pass.draw(graph.skyViewEffect));
}

/** Terrain depth prepass (reversed-Z, cleared to 0 = no terrain), then one fullscreen pass shades every pixel once. */
export function encodeScene(frame: Frame, graph: AtmosphereGraph): void {
  frame.pass({ target: graph.terrainDepth, clear: [0, 0, 0, 0], clearDepth: 0 }, (pass) => {
    if (graph.terrainColumns > 0) pass.draw(graph.terrainDraw, { instances: graph.terrainColumns });
  });
  frame.pass({ target: graph.scene, clear: CLEAR }, (pass) => pass.draw(graph.sceneEffect));
}

/**
 * Temporal cloud update: this frame's texels are marched (one in sixteen at rest, one in two for two frames after a
 * change), the rest are reprojected from last frame's buffers.
 */
export function encodeClouds(frame: Frame, graph: AtmosphereGraph): void {
  graph.reprojection.set(reprojectionUniforms(graph.previousCamera, graph.frame, graph.accumulate, graph.cloudChangeFrames > 0));
  graph.cloudsEffect.set({ history: graph.cloudsTargets.read, historyDepth: graph.cloudsTargets.read.colors[1] });
  frame.pass({ target: graph.cloudsTargets.write, clear: [0, 0, 0, 1] }, (pass) => pass.draw(graph.cloudsEffect));
}

export function encodePresent(frame: Frame, graph: AtmosphereGraph, output: Output): void {
  graph.presentEffect.set({ cloudsHdr: graph.cloudsTargets.write });
  frame.pass({ target: output, clear: CLEAR }, (pass) => pass.draw(graph.presentEffect));
}

/** Swaps the cloud history and advances the temporal sequence; once per frame, after the passes. */
export function finishFrame(graph: AtmosphereGraph): void {
  graph.cloudsTargets.swap();
  graph.previousCamera = graph.currentCamera;
  graph.frame += 1;
  if (graph.cloudChangeFrames > 0) graph.cloudChangeFrames -= 1;
}

/** Stills render enough frames for the temporal cloud update to touch every texel. */
function renderState(gpu: Gpu, graph: AtmosphereGraph, output: Target, state: AtmosphereState): void {
  applyState(graph, state, output.size);
  if (graph.lutPhase !== 'ready') bakeLuts(gpu, graph);
  // The per-frame constants read the sky-view LUT before this frame's pass writes it: make it current first.
  createFrame(gpu, (frame) => encodeSkyView(frame, graph));
  for (let i = 0; i < CLOUD_CONVERGENCE_FRAMES; i++) createFrame(gpu, (frame) => renderGraph(frame, graph, output));
}

/** 4x4 Bayer sequence, centred: the sub-texel offsets a texel cycles through while accumulating. */
const JITTER_SEQUENCE: readonly (readonly [number, number])[] = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]
  .map((index) => [((index % 4) + 0.5) / 4 - 0.5, (Math.floor(index / 4) + 0.5) / 4 - 0.5] as const);

/** `fast` (right after a change) refreshes one texel in two with full blend; at rest, one in sixteen accumulated with jitter. */
function reprojectionUniforms(previous: CameraUniformValues | undefined, frame: number, accumulate: boolean, fast: boolean): ReprojectionUniformValues {
  const accumulating = accumulate && !fast;
  return {
    forward: previous?.forward ?? [0, 0, 1], frame,
    right: previous?.right ?? [1, 0, 0], tanHalfFov: previous?.tanHalfFov ?? 1,
    up: previous?.up ?? [0, 1, 0], aspect: previous?.aspect ?? 1,
    position: previous?.position ?? [0, ATMOSPHERE_PHYSICS.groundRadius, 0], valid: previous ? 1 : 0,
    blend: accumulating ? 0.5 : 1, refreshPeriod: fast ? CLOUD_FAST_REFRESH_PERIOD : CLOUD_CONVERGENCE_FRAMES,
    jitter: accumulating ? JITTER_SEQUENCE[Math.floor(frame / 16) % 16]! : [0, 0],
  };
}

function resizeGraph(graph: AtmosphereGraph, size: readonly [number, number]): void {
  graph.scene.resize(size);
  graph.terrainDepth.resize(size);
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
    detailStrength: CLOUD_TUNING.detailStrength * Math.max(0, state.cloudDetail), groundRadius: ATMOSPHERE_PHYSICS.groundRadius,
    curlStrength: CLOUD_TUNING.curlStrength * Math.max(0, state.cloudDetail), detailLodDistance: CLOUD_TUNING.detailLodDistance,
    typeBias: state.cloudType * 0.5, seed: state.cloudSeed, pad0: 0, pad1: 0,
  };
}

function scale(v: Vec3, factor: number): Vec3 { return [v[0] * factor, v[1] * factor, v[2] * factor]; }

function sameDirection(a: Vec3 | undefined, b: Vec3): boolean {
  return a !== undefined && a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
}

export function destroyGraph(graph: AtmosphereGraph): void {
  for (const target of [graph.transmittance, graph.skyView, graph.scene, graph.terrainDepth, graph.cloudsTargets.read, graph.cloudsTargets.write]) for (const color of target.colors) color.destroy();
  graph.terrainDepth.depth?.destroy();
  for (const texture of [graph.multiScatter, graph.aerial, graph.aerialLoss, graph.shapeNoise, graph.detailNoise, graph.weatherMap, graph.curlNoise, graph.terrainMap, graph.terrainAlbedoMap, graph.terrainShadowMap]) texture.destroy();
}
