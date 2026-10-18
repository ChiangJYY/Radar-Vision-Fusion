"""Pinhole projection and explicit rigid coordinate transforms.

Lengths use a common unit (normally metres), pixels use the depth-map coordinate
system, and Euler angles use radians. No installation calibration is bundled.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera parameters in pixels, without lens distortion."""

    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self):
        if not np.isfinite([self.fx, self.fy, self.cx, self.cy]).all():
            raise ValueError("Camera intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Focal lengths must be positive")


@dataclass(frozen=True)
class RigidTransform:
    """Transform source to target: ``target = source @ rotation.T + translation``.

    For the camera/radar helpers, supply ``camera_from_radar``: its rotation and
    translation map radar coordinates into camera coordinates.
    """

    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self):
        rotation = np.array(self.rotation, dtype=np.float64, copy=True)
        translation = np.array(self.translation, dtype=np.float64, copy=True)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("Expected a (3, 3) rotation and (3,) translation")
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("Transform must be finite")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("Rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError("Rotation must have determinant +1")
        rotation.setflags(write=False)
        translation.setflags(write=False)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    @classmethod
    def identity(cls):
        """Return an identity transform for synthetic examples."""
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_euler(cls, translation, rx: float, ry: float, rz: float):
        """Build ``Rz(rz) @ Ry(ry) @ Rx(rx)`` using radian Euler angles."""
        sx, cx = np.sin(rx), np.cos(rx)
        sy, cy = np.sin(ry), np.cos(ry)
        sz, cz = np.sin(rz), np.cos(rz)
        rotation = np.array([
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ])
        return cls(rotation, translation)

    def apply(self, points):
        """Transform an array with final dimension 3 to the target frame."""
        points = np.asarray(points, dtype=np.float64)
        if points.shape[-1:] != (3,):
            raise ValueError("points must have final dimension 3")
        return points @ self.rotation.T + self.translation

    def inverse(self):
        """Return the target-to-source transform."""
        return RigidTransform(self.rotation.T, -self.rotation.T @ self.translation)


def pixel_to_camera(uv, depth, intrinsics: CameraIntrinsics):
    """Back-project pixels ``(..., 2)`` with optical-axis depth ``(...)``.

    The camera convention is x right, y down, z forward. Depth is z, not
    Euclidean distance to the camera. Invalid/nonpositive depths are rejected.
    """
    uv = np.asarray(uv, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    if uv.shape[-1:] != (2,) or depth.shape != uv.shape[:-1]:
        raise ValueError("Expected uv (..., 2) and matching depth (...)")
    if not np.isfinite(uv).all() or not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("Pixels must be finite and depths must be finite and positive")
    return np.stack([
        (uv[..., 0] - intrinsics.cx) / intrinsics.fx * depth,
        (uv[..., 1] - intrinsics.cy) / intrinsics.fy * depth,
        depth,
    ], axis=-1)


def camera_to_pixel(points, intrinsics: CameraIntrinsics):
    """Project camera coordinates ``(..., 3)`` with positive z into pixels."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("Expected finite points (..., 3)")
    if np.any(points[..., 2] <= 0):
        raise ValueError("Projection requires positive camera z")
    return np.stack([
        intrinsics.fx * points[..., 0] / points[..., 2] + intrinsics.cx,
        intrinsics.fy * points[..., 1] / points[..., 2] + intrinsics.cy,
    ], axis=-1)


def camera_to_radar(uv, depth, intrinsics: CameraIntrinsics, camera_from_radar: RigidTransform):
    """Back-project pixels and return all three radar-coordinate components."""
    return camera_from_radar.inverse().apply(pixel_to_camera(uv, depth, intrinsics))


def radar_to_pixel(points, intrinsics: CameraIntrinsics, camera_from_radar: RigidTransform):
    """Transform radar points into the camera and project them into pixels."""
    return camera_to_pixel(camera_from_radar.apply(points), intrinsics)
