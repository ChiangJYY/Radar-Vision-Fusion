"""Radar-to-camera calibration from 3D radar / 2D image correspondences.

Uses the original ZYX Euler rigid transform and robust reprojection objective.
No correspondence data, installation parameters or preset coordinate bounds.
"""
import numpy as np
from scipy.optimize import least_squares

from ..geometry.projection import CameraIntrinsics, RigidTransform


def calibrate_extrinsics(radar_xyz, image_uv, intrinsics: CameraIntrinsics,
                         initial_parameters, *, loss="huber", f_scale=1.0):
    xyz, uv = np.asarray(radar_xyz, float), np.asarray(image_uv, float)
    initial = np.asarray(initial_parameters, float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or uv.shape != (len(xyz), 2) or len(xyz) < 6:
        raise ValueError("Need at least six 3D radar / 2D pixel correspondences")
    if initial.shape != (6,) or not all(np.isfinite(a).all() for a in (xyz, uv, initial)):
        raise ValueError("Inputs must be finite; initial parameters are tx,ty,tz,rx,ry,rz")
    if f_scale <= 0:
        raise ValueError("f_scale must be positive")

    def residual(parameters):
        camera = RigidTransform.from_euler(parameters[:3], *parameters[3:]).apply(xyz)
        # Fixed residual dimension even if an optimization step moves a point behind the camera.
        z = np.maximum(camera[:, 2], 1e-6)
        projected = np.column_stack((camera[:, 0] * intrinsics.fx / z + intrinsics.cx,
                                     camera[:, 1] * intrinsics.fy / z + intrinsics.cy))
        return (projected - uv).ravel()

    result = least_squares(residual, initial, loss=loss, f_scale=f_scale)
    if not result.success:
        raise RuntimeError(f"Extrinsic calibration did not converge: {result.message}")
    transform = RigidTransform.from_euler(result.x[:3], *result.x[3:])
    if np.any(transform.apply(xyz)[:, 2] <= 0):
        raise ValueError("Calibration places a correspondence behind the camera; check initialization")
    return transform, {"rmse_px": float(np.sqrt(np.mean(residual(result.x) ** 2))),
                       "correspondences": len(xyz), "parameters": result.x.tolist()}
