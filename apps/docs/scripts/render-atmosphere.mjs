// Headless renders of the atmosphere example for visual verification.
//   node scripts/render-atmosphere.mjs [--out dir] [--preset name|all] [--size WxH] [--debug transmittance|multiscatter|sky-view|weather|terrain] [--bench N] [--accumulate N]
//   overrides: --sun <deg> --azimuth <deg> --altitude <km> --yaw <deg> --pitch <deg> --ev <stops> --haze <x> --coverage <0..1> --detail <x> --type <-1..1> --seed <n> --time <s> --tonemap agx|aces|neutral|none
import { mkdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { frame, init, target as createTarget } from 'vgpu/node';
import { writePng } from '@vgpu/cli/lib/snapshot/png.js';
import { transformWgsl } from '@vgpu/wgsl/loader-vite';

const docsDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = parseArgs(process.argv.slice(2));
const outDir = path.resolve(args.out ?? path.join(docsDir, '..', '..', 'artifacts', 'atmosphere'));
const cacheDir = path.join(docsDir, '.atmosphere-cache');
const entry = path.join(cacheDir, 'entry.ts');
const bundle = path.join(cacheDir, 'atmosphere.mjs');

await mkdir(cacheDir, { recursive: true });
await writeFile(entry, "export { renderStill, createGraph, applyState, bakeLuts, renderGraph } from '../examples/atmosphere/renderer.ts';\nexport { PRESETS } from '../examples/atmosphere/tuning.ts';\n");
await build({ entryPoints: [entry], outfile: bundle, bundle: true, platform: 'node', format: 'esm', sourcemap: false, external: ['vgpu', 'vgpu/node'], plugins: [wgslPlugin()], logLevel: 'silent' });
const { renderStill, createGraph, applyState, bakeLuts, renderGraph, PRESETS } = await import(pathToFileURL(bundle).href);
await rm(cacheDir, { recursive: true, force: true });

const size = args.size ?? [960, 540];
const presetNames = args.preset === 'all' || !args.preset ? Object.keys(PRESETS) : [args.preset];
await mkdir(outDir, { recursive: true });
for (const name of presetNames) {
  const base = PRESETS[name];
  if (!base) throw new Error(`Unknown preset '${name}'. Known: ${Object.keys(PRESETS).join(', ')}`);
  const state = { ...base, ...args.overrides };
  const gpu = await init();
  try {
    const target = createTarget(gpu, { size, format: 'rgba8unorm', label: `atmosphere-${name}` });
    if (args.bench) {
      console.log(`- ${name}: ${await bench(gpu, target, state, args.bench)}`);
      continue;
    }
    const started = performance.now();
    if (args.accumulate) await renderAccumulated(gpu, target, state, args.accumulate);
    else await renderStill(gpu, target, state, args.debug);
    const pixels = await target.read();
    const suffix = args.debug ? `.${args.debug}` : '';
    const file = path.join(outDir, `${name}${suffix}.png`);
    await writePng(file, pixels, size[0], size[1]);
    console.log(`- ${name}${suffix}: ${path.relative(process.cwd(), file)} (${(performance.now() - started).toFixed(0)} ms) ${describe(pixels, size)}`);
  } finally {
    gpu.dispose();
  }
}

/** Live-loop path: history accumulation with jitter over `frames` frames, as the browser example renders it. */
async function renderAccumulated(gpu, target, state, frames) {
  const graph = await createGraph(gpu, target, 'atmosphere-accumulate');
  graph.accumulate = true;
  applyState(graph, state, target.size);
  bakeLuts(gpu, graph);
  for (let i = 0; i < frames; i++) frame(gpu, (current) => renderGraph(current, graph, target));
  await gpu.gpu.queue.onSubmittedWorkDone();
}

/** Wall-clock ms per frame over `frames` steady-state frames on one graph (first frame bakes and warms up). */
async function bench(gpu, target, state, frames) {
  const graph = await createGraph(gpu, target, 'atmosphere-bench');
  applyState(graph, state, target.size);
  bakeLuts(gpu, graph);
  frame(gpu, (current) => renderGraph(current, graph, target));
  await gpu.gpu.queue.onSubmittedWorkDone();
  const started = performance.now();
  for (let i = 0; i < frames; i++) {
    applyState(graph, { ...state, time: state.time + i / 60 }, target.size);
    frame(gpu, (current) => renderGraph(current, graph, target));
  }
  await gpu.gpu.queue.onSubmittedWorkDone();
  const perFrame = (performance.now() - started) / frames;
  const cloudless = { ...state, cloudCoverage: 0 };
  const startedCloudless = performance.now();
  for (let i = 0; i < frames; i++) {
    applyState(graph, { ...cloudless, time: state.time + i / 60 }, target.size);
    frame(gpu, (current) => renderGraph(current, graph, target));
  }
  await gpu.gpu.queue.onSubmittedWorkDone();
  const perFrameCloudless = (performance.now() - startedCloudless) / frames;
  return `${perFrame.toFixed(0)} ms/frame (${perFrameCloudless.toFixed(0)} ms without clouds, ${(perFrame - perFrameCloudless).toFixed(0)} ms clouds) at ${target.size.join('x')} over ${frames} frames`;
}

/** Mean sRGB of the top band (zenith-ish), middle band (horizon) and bottom band (ground) as a quick sanity readout. */
function describe(pixels, [width, height]) {
  const band = (y0, y1) => {
    const sum = [0, 0, 0];
    let count = 0;
    for (let y = y0; y < y1; y++) for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      sum[0] += pixels[i]; sum[1] += pixels[i + 1]; sum[2] += pixels[i + 2];
      count++;
    }
    return sum.map((v) => Math.round(v / count));
  };
  const top = band(0, Math.floor(height * 0.1));
  const middle = band(Math.floor(height * 0.45), Math.floor(height * 0.55));
  const bottom = band(Math.floor(height * 0.9), height);
  return `top=${top.join(',')} mid=${middle.join(',')} bottom=${bottom.join(',')}`;
}

function parseArgs(argv) {
  const parsed = { out: undefined, preset: undefined, size: undefined, debug: undefined, bench: 0, accumulate: 0, overrides: {} };
  const numeric = { sun: 'sunElevation', azimuth: 'sunAzimuth', altitude: 'altitudeKm', yaw: 'yaw', pitch: 'pitch', ev: 'exposureEv', haze: 'haze', coverage: 'cloudCoverage', detail: 'cloudDetail', type: 'cloudType', seed: 'cloudSeed', time: 'time' };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--out') parsed.out = argv[++i];
    else if (arg === '--preset') parsed.preset = argv[++i];
    else if (arg === '--size') parsed.size = argv[++i].split('x').map(Number);
    else if (arg === '--debug') parsed.debug = argv[++i];
    else if (arg === '--bench') parsed.bench = Number(argv[++i]);
    else if (arg === '--accumulate') parsed.accumulate = Number(argv[++i]);
    else if (arg === '--tonemap') parsed.overrides.tonemap = argv[++i];
    else if (arg.startsWith('--') && numeric[arg.slice(2)]) parsed.overrides[numeric[arg.slice(2)]] = Number(argv[++i]);
    else throw new Error(`Unknown argument '${arg}'.`);
  }
  return parsed;
}

function wgslPlugin() {
  return {
    name: 'docs-wgsl',
    setup(build) {
      build.onLoad({ filter: /\.wgsl$/ }, async (file) => {
        const source = await import('node:fs/promises').then(({ readFile }) => readFile(file.path, 'utf8'));
        const result = await transformWgsl({ source, id: file.path });
        return { contents: result.code, loader: 'js', resolveDir: path.dirname(file.path) };
      });
    },
  };
}
