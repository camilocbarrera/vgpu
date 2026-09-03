import { describe, expect, it } from 'vitest';
import { init } from 'vgpu/mock';
import { cameraUniforms, sunDirection } from './camera';
import { CLOUD_CONVERGENCE_FRAMES, applyState, bakeLuts, createGraph, renderGraph } from './example';
import { LUT_SIZES, PRESETS } from './tuning';

const dot = (a: readonly number[], b: readonly number[]) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;

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
      const target = gpu.target({ size: [96, 54], format: 'rgba8unorm' });
      const graph = await createGraph(gpu, target, 'atmosphere-test');
      expect(graph.aerial.dimension).toBe('3d');
      expect(graph.aerial.size).toEqual([LUT_SIZES.aerial, LUT_SIZES.aerial, LUT_SIZES.aerial]);
      expect([...graph.multiScatter.usage]).toContain('storage_binding');
      expect(graph.shapeNoise.dimension).toBe('3d');
      expect(graph.shapeNoise.format).toBe('rgba8unorm');
      expect(graph.cloudsTargets.write.size).toEqual([96, 54]);
      expect(graph.curlNoise.format).toBe('rgba8unorm');
      expect(graph.terrainMap.size).toEqual([2048, 2048]);
      expect([...graph.terrainMap.usage]).toContain('storage_binding');
      applyState(graph, PRESETS['golden-hour'], target.size);
      expect(graph.lutPhase).toBe('stale');
      bakeLuts(gpu, graph);
      expect(graph.lutPhase).toBe('ready');
      expect(() => gpu.frame((frame) => renderGraph(frame, graph, target))).not.toThrow();
      // Changing the haze invalidates the medium-dependent tables; the next frame re-encodes them.
      applyState(graph, { ...PRESETS['golden-hour'], haze: 4 }, target.size);
      expect(graph.lutPhase).toBe('stale');
      gpu.frame((frame) => renderGraph(frame, graph, target));
      expect(graph.lutPhase).toBe('transmittance');
      gpu.frame((frame) => renderGraph(frame, graph, target));
      expect(graph.lutPhase).toBe('ready');
      // The temporal cloud update alternates the ping-pong buffers and counts frames.
      const before = graph.cloudsTargets.write;
      gpu.frame((frame) => renderGraph(frame, graph, target));
      expect(graph.cloudsTargets.read).toBe(before);
      expect(graph.frame).toBe(4);
      expect(CLOUD_CONVERGENCE_FRAMES).toBe(16);
      await gpu.settled();
    } finally {
      gpu.dispose();
    }
  });
});
