import { writeFileSync } from "node:fs";
import { PNG } from "pngjs";
import { init, effect, target } from "vgpu/node";

const SHADER = `
  @fragment fn main() -> @location(0) vec4f {
    return vec4f(0.25, 0.5, 0.75, 1.0);
  }
`;
const width = 64;
const height = 64;

const gpu = await init();
const colorTarget = target(gpu, { size: [width, height] });
effect(gpu, SHADER).draw(colorTarget);
const pixels = await colorTarget.read();

const png = new PNG({ width, height });
png.data.set(pixels);
writeFileSync("out.png", PNG.sync.write(png));
gpu.dispose();
