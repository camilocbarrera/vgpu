import { describe, expect, it, vi } from 'vitest';
import { frame, init, target } from 'vgpu/mock';
import { cameraUniforms, sunDirection } from './camera';
import { CLOUD_CONVERGENCE_FRAMES, applyState, bakeLuts, createGraph, createRenderer, renderGraph } from './renderer';
import { LUT_SIZES, PRESETS } from './tuning';

const dot = (a: readonly number[], b: readonly number[]) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

describe('atmosphere camera', () => {
  it('builds an orthonormal basis that looks along yaw/pitch', () => {
    const camera = cameraUniforms({ ...PRESETS.noon, yaw: 90, pitch: 0 }, [1280, 720]);
    expect(camera.forward[0]).toBeCloseTo(1, 6);
    expect(camera.forward[1]).toBeCloseTo(0, 6);
    expect(dot(camera.forward, camera.right)).toBeCloseTo(0, 6);
    expect(dot(camera.forward, camera.up)).toBeCloseTo(0, 6);
    expect(dot(camera.right, camera.up)).toBeCloseTo(0, 6);
    expect(camera.up[1]).toBeGreaterThan(0.99);
    expect(camera.aspect).toBeCloseTo(1280 / 720, 6);
    expect(camera.position[1]).toBeCloseTo(6360 + PRESETS.noon.altitudeKm, 6);
  });

  it('places the sun from elevation and azimuth', () => {
    const zenith = sunDirection({ ...PRESETS.noon, sunElevation: 90 });
    expect(zenith[1]).toBeCloseTo(1, 6);
    const horizon = sunDirection({ ...PRESETS.noon, sunElevation: 0, sunAzimuth: 0 });
    expect(horizon).toEqual([0, 0, 1]);
  });

  it('clamps the altitude to the atmosphere', () => {
    const camera = cameraUniforms({ ...PRESETS.noon, altitudeKm: 500 }, [64, 64]);
    expect(camera.position[1]).toBeLessThan(6460);
  });
});

describe('atmosphere graph on the mock adapter', () => {
  it('creates the storage LUTs, bakes and renders one frame without binding errors', async () => {
    const gpu = await init();
    try {
      const output = target(gpu, { size: [96, 54], format: 'rgba8unorm' });
      const graph = await createGraph(gpu, output, 'atmosphere-test');
      expect(graph.aerial.dimension).toBe('3d');
      expect(graph.aerial.size).toEqual([LUT_SIZES.aerial, LUT_SIZES.aerial, LUT_SIZES.aerial]);
      expect([...graph.multiScatter.usage]).toContain('storage_binding');
      expect(graph.shapeNoise.dimension).toBe('3d');
      expect(graph.shapeNoise.format).toBe('rgba8unorm');
      expect(graph.cloudsTargets.write.size).toEqual([96, 54]);
      expect(graph.aerialMie.dimension).toBe('3d');
      expect(graph.curlNoise.format).toBe('rgba8unorm');
      expect(graph.terrainMap.size).toEqual([2048, 2048]);
      expect([...graph.terrainMap.usage]).toContain('storage_binding');
      applyState(graph, PRESETS['golden-hour'], output.size);
      expect(graph.lutPhase).toBe('stale');
      bakeLuts(gpu, graph);
      expect(graph.lutPhase).toBe('ready');
      expect(() => frame(gpu, (current) => renderGraph(current, graph, output))).not.toThrow();
      // Changing the haze invalidates the medium-dependent tables; the next frame re-encodes them.
      applyState(graph, { ...PRESETS['golden-hour'], haze: 4 }, output.size);
      expect(graph.lutPhase).toBe('stale');
      frame(gpu, (current) => renderGraph(current, graph, output));
      expect(graph.lutPhase).toBe('transmittance');
      frame(gpu, (current) => renderGraph(current, graph, output));
      expect(graph.lutPhase).toBe('ready');
      // The temporal cloud update alternates the ping-pong buffers and counts frames.
      const before = graph.cloudsTargets.write;
      frame(gpu, (current) => renderGraph(current, graph, output));
      expect(graph.cloudsTargets.read).toBe(before);
      expect(graph.frame).toBe(4);
      expect(CLOUD_CONVERGENCE_FRAMES).toBe(16);
      await gpu.settled();
    } finally {
      gpu.dispose();
    }
  });
});

describe('atmosphere renderer lifecycle', () => {
  it('finishes a stale Strict Mode cleanup before reconfiguring the same canvas', async () => {
    const canvas = {} as HTMLCanvasElement;
    const firstInitialization = deferred<() => void>();
    const firstCleanup = vi.fn();
    const secondCleanup = vi.fn();
    const firstStart = vi.fn(() => firstInitialization.promise);
    const secondStart = vi.fn(async () => secondCleanup);

    const first = createRenderer({ canvas }, firstStart);
    await vi.waitFor(() => expect(firstStart).toHaveBeenCalledOnce());
    first.dispose();

    const second = createRenderer({ canvas }, secondStart);
    await Promise.resolve();
    expect(secondStart).not.toHaveBeenCalled();

    firstInitialization.resolve(firstCleanup);
    await first.ready;
    await second.ready;

    expect(firstCleanup).toHaveBeenCalledOnce();
    expect(secondStart).toHaveBeenCalledOnce();
    expect(firstCleanup.mock.invocationCallOrder[0]).toBeLessThan(secondStart.mock.invocationCallOrder[0]!);

    second.dispose();
    expect(secondCleanup).toHaveBeenCalledOnce();
  });

  it('holds the canvas until an initialized renderer is disposed', async () => {
    const canvas = {} as HTMLCanvasElement;
    const firstCleanup = vi.fn();
    const secondCleanup = vi.fn();
    const first = createRenderer({ canvas }, async () => firstCleanup);
    const secondStart = vi.fn(async () => secondCleanup);
    const second = createRenderer({ canvas }, secondStart);

    await first.ready;
    await Promise.resolve();
    expect(secondStart).not.toHaveBeenCalled();

    first.dispose();
    await second.ready;
    expect(secondStart).toHaveBeenCalledOnce();

    second.dispose();
  });
});
