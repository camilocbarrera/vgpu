// Headless renders of the atmosphere example for visual verification.
//   node scripts/render-atmosphere.mjs [--out dir] [--preset name|all] [--size WxH] [--debug transmittance|multiscatter|sky-view]
//   overrides: --sun <deg> --azimuth <deg> --altitude <km> --yaw <deg> --pitch <deg> --ev <stops> --tonemap agx|aces|neutral|none
import { mkdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { init } from 'vgpu/node';
import { writePng } from '@vgpu/cli/lib/snapshot/png.js';
import { transformWgsl } from '@vgpu/wgsl/loader-vite';

const docsDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = parseArgs(process.argv.slice(2));
const outDir = path.resolve(args.out ?? path.join(docsDir, '..', '..', 'artifacts', 'atmosphere'));
const cacheDir = path.join(docsDir, '.atmosphere-cache');
const entry = path.join(cacheDir, 'entry.ts');
const bundle = path.join(cacheDir, 'atmosphere.mjs');

await mkdir(cacheDir, { recursive: true });
await writeFile(entry, "export { renderStill } from '../examples/atmosphere/example.ts';\nexport { PRESETS } from '../examples/atmosphere/tuning.ts';\n");
await build({ entryPoints: [entry], outfile: bundle, bundle: true, platform: 'node', format: 'esm', sourcemap: false, external: ['vgpu', 'vgpu/node'], plugins: [wgslPlugin()], logLevel: 'silent' });
const { renderStill, PRESETS } = await import(pathToFileURL(bundle).href);
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
    const target = gpu.target({ size, format: 'rgba8unorm', label: `atmosphere-${name}` });
    const started = performance.now();
    await renderStill(gpu, target, state, args.debug);
    const pixels = await target.read();
    const suffix = args.debug ? `.${args.debug}` : '';
    const file = path.join(outDir, `${name}${suffix}.png`);
    await writePng(file, pixels, size[0], size[1]);
    console.log(`- ${name}${suffix}: ${path.relative(process.cwd(), file)} (${(performance.now() - started).toFixed(0)} ms) ${describe(pixels, size)}`);
  } finally {
    gpu.dispose();
  }
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
  const parsed = { out: undefined, preset: undefined, size: undefined, debug: undefined, overrides: {} };
  const numeric = { sun: 'sunElevation', azimuth: 'sunAzimuth', altitude: 'altitudeKm', yaw: 'yaw', pitch: 'pitch', ev: 'exposureEv' };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--out') parsed.out = argv[++i];
    else if (arg === '--preset') parsed.preset = argv[++i];
    else if (arg === '--size') parsed.size = argv[++i].split('x').map(Number);
    else if (arg === '--debug') parsed.debug = argv[++i];
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
