import { effect, frame, init, surface } from "vgpu";

import shaderSource from "./shader-code-scales/main.wgsl";

export async function renderShaderPreview(
  canvas: HTMLCanvasElement
): Promise<() => void> {
  const gpu = await init();

  try {
    const output = surface(gpu, canvas, { autoResize: false, dpr: 1 });
    const shader = effect(gpu, {
      shader: shaderSource,
      label: "homepage-shader-code-scales-preview",
    });
    let resizeFrame = 0;

    const render = () => {
      resizeFrame = 0;
      const size = [
        Math.max(1, Math.round(canvas.clientWidth)),
        Math.max(1, Math.round(canvas.clientHeight)),
      ] as const;

      if (output.size[0] !== size[0] || output.size[1] !== size[1]) {
        output.resize(size);
      }
      shader.set({ uniforms: { resolution: output.size } });
      frame(gpu, (currentFrame) => currentFrame.pass(output, shader));
    };

    const observer = new ResizeObserver(() => {
      if (!resizeFrame) resizeFrame = requestAnimationFrame(render);
    });

    observer.observe(canvas);
    render();

    return () => {
      observer.disconnect();
      if (resizeFrame) cancelAnimationFrame(resizeFrame);
      gpu.dispose();
    };
  } catch (error) {
    gpu.dispose();
    throw error;
  }
}
