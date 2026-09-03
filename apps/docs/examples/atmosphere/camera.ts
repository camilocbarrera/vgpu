import { ATMOSPHERE_PHYSICS, CAMERA_TUNING, type AtmosphereState } from './tuning';

type Vec3 = readonly [number, number, number];

export type CameraUniformValues = {
  position: Vec3; tanHalfFov: number;
  forward: Vec3; aspect: number;
  right: Vec3; sunAngularRadius: number;
  up: Vec3; pixelAngle: number;
}

const DEG = Math.PI / 180;

/** Direction from elevation/azimuth angles in degrees; azimuth 0 points toward +Z. */
export function directionFromAngles(elevationDeg: number, azimuthDeg: number): Vec3 {
  const elevation = elevationDeg * DEG;
  const azimuth = azimuthDeg * DEG;
  return [Math.cos(elevation) * Math.sin(azimuth), Math.sin(elevation), Math.cos(elevation) * Math.cos(azimuth)];
}

export function sunDirection(state: AtmosphereState): Vec3 {
  return directionFromAngles(state.sunElevation, state.sunAzimuth);
}

/** Planet-centric camera basis: the camera sits on the +Y axis at ground radius + altitude. */
export function cameraUniforms(state: AtmosphereState, size: readonly [number, number]): CameraUniformValues {
  const altitude = Math.min(Math.max(state.altitudeKm, 0.001), CAMERA_TUNING.maxAltitudeKm);
  const forward = directionFromAngles(state.pitch, state.yaw);
  const right = normalize(cross([0, 1, 0], forward));
  const up = cross(forward, right);
  const tanHalfFov = Math.tan(CAMERA_TUNING.fovDegrees * DEG / 2);
  return {
    position: [0, ATMOSPHERE_PHYSICS.groundRadius + altitude, 0], tanHalfFov,
    forward, aspect: size[0] / Math.max(1, size[1]),
    right, sunAngularRadius: ATMOSPHERE_PHYSICS.sunAngularRadius,
    up, pixelAngle: 2 * tanHalfFov / Math.max(1, size[1]),
  };
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function normalize(v: Vec3): Vec3 {
  const length = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / length, v[1] / length, v[2] / length];
}
