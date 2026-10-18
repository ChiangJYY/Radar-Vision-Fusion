"""Convert detector boxes and metric depths into calibrated camera observations."""
import numpy as np

from ..geometry.camera_depth import DepthFusionConfig, bbox_sample_point, sample_depth_stats, fuse_person_depth
from ..geometry.projection import pixel_to_camera


def observations_from_image(detections, depth_map, intrinsics, camera_from_radar, *, config=None, person_class=0):
    config = config or DepthFusionConfig()
    depth = np.asarray(depth_map, dtype=float)
    if depth.ndim != 2:
        raise ValueError("Metric depth map must have shape [height,width]")
    result = []
    for number, detection in enumerate(detections):
        if int(detection["label"]) != person_class:
            continue
        box = np.asarray(detection["bbox"], dtype=float)
        u, v = bbox_sample_point(box, depth.shape, mode="foot", foot_up_px=5)
        stats = sample_depth_stats(depth, u, v, window=9)
        fused = fuse_person_depth(stats.median, stats.mad, box[3] - box[1], intrinsics.fy, config)
        xyz = pixel_to_camera([u, v], fused.depth, intrinsics)
        shared_xyz = camera_from_radar.inverse().apply(xyz)
        result.append({"id": number, "xy": shared_xyz[:2].tolist(), "bbox": box.tolist(),
                       "center_uv": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                       "sample_uv": [u, v], "score": float(detection["score"]),
                       "depth_m": fused.depth, "xyz_camera": xyz.tolist()})
    return result
