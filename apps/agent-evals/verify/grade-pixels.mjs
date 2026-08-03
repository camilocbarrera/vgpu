// Pure pixel grading. Knows nothing about tasks, workspaces or processes, so it
// is testable with an in-memory PNG buffer alone.
//
// gradePixels(pngBuffer: Buffer, expected: { width: number, height: number, color: [number,number,number,number] })
//   -> { width, height, distinctColors, dominantPixel, matchedFraction }
import { PNG } from "pngjs";

/**
 * @param {Buffer} pngBuffer raw bytes of a PNG file
 * @param {{ width: number, height: number, color: [number, number, number, number] }} expected
 */
export function gradePixels(pngBuffer, expected) {
  const png = PNG.sync.read(pngBuffer);
  const { width, height, data } = png;

  /** @type {Map<string, number>} */
  const counts = new Map();
  const [er, eg, eb, ea] = expected.color;
  let matched = 0;
  let total = 0;

  for (let i = 0; i + 3 < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const a = data[i + 3];
    total += 1;

    const key = `${r},${g},${b},${a}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);

    if (r === er && g === eg && b === eb && a === ea) matched += 1;
  }

  let dominantKey = "0,0,0,0";
  let dominantCount = -1;
  for (const [key, count] of counts) {
    if (count > dominantCount) {
      dominantCount = count;
      dominantKey = key;
    }
  }

  return {
    width,
    height,
    distinctColors: counts.size,
    dominantPixel: /** @type {[number, number, number, number]} */ (
      dominantKey.split(",").map(Number)
    ),
    matchedFraction: total === 0 ? 0 : matched / total,
  };
}
