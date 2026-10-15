"""Camera depth sampling and calibrated coordinate conversion."""

from .camera_depth import (
    DepthFusionConfig,
    DepthFusionResult,
    DepthStats,
    bbox_sample_point,
    depth_fusion_weight,
    fuse_person_depth,
    person_height_prior_depth,
    sample_depth_stats,
)
from .projection import (
    CameraIntrinsics,
    RigidTransform,
    camera_to_pixel,
    camera_to_radar,
    pixel_to_camera,
    radar_to_pixel,
)

__all__ = [
    "CameraIntrinsics", "RigidTransform", "pixel_to_camera", "camera_to_pixel",
    "camera_to_radar", "radar_to_pixel", "DepthStats", "DepthFusionConfig",
    "DepthFusionResult", "bbox_sample_point", "sample_depth_stats",
    "person_height_prior_depth", "depth_fusion_weight", "fuse_person_depth",
]
