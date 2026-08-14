import type { ShaderSource } from "@vgpu/wgsl";
import { malformedShaderSourceError, unsupportedError } from "./errors.ts";

/** Normalizes public ring-1 shader inputs to raw WGSL. Strings remain first-class; ShaderSource is the loader artifact. */
export function toWgsl(input: string | ShaderSource): string {
  if (typeof input === "string") return input;
  if (!isObject(input)) throw malformedShaderSourceError(input);
  if (!("version" in input)) throw malformedShaderSourceError(input);
  const version = (input as { readonly version: unknown }).version;
  if (version !== 1) throw malformedShaderSourceError(input);
  const wgsl = (input as { readonly wgsl?: unknown }).wgsl;
  if (typeof wgsl !== "string") throw malformedShaderSourceError(input);
  return wgsl;
}

/**
 * Splits the additive single-object form (`effect(gpu, { shader, ... })` / `compute(gpu, { shader, ... })`)
 * from the standing two-argument form (`fn(gpu, source, opts)`) and the ShaderSource/string shorthand.
 * Shared by `effect()` and `compute()` so both stay byte-identical in behavior and error text.
 *
 * `shader` wins over `version`: a real `ShaderSource` artifact is exactly `{ version, wgsl }` and can
 * never legitimately carry a `shader` property, while an options bag can carry a stray `version`
 * (directly, or via a spread like `{ ...artifact, shader, blend }`). Checking `"shader" in input`
 * first means that spread still honors `shader`/`blend` instead of silently reverting to the artifact
 * and dropping every other option. An object with neither `shader` nor `version` is rejected with an
 * actionable error instead of the generic malformed-shader-source message.
 */
export function resolveShaderInput<Opts extends object>(where: string, input: string | ShaderSource | Opts, opts: Opts): readonly [string | ShaderSource, Opts] {
  if (!isObject(input)) return [input as string | ShaderSource, opts];
  if ("shader" in input) return [(input as Opts & { readonly shader: string | ShaderSource }).shader, input as Opts];
  if ("version" in input) return [input as unknown as ShaderSource, opts];
  throw unsupportedError(where, `${where}(gpu, options) requires options.shader; use ${where}(gpu, source, opts) for the two-argument form, or ${where}(gpu, { shader, ... }).`);
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
