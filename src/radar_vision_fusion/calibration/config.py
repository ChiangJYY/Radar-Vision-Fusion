"""Load caller-owned camera intrinsics and camera-from-radar calibration."""
from ..geometry.projection import CameraIntrinsics, RigidTransform
from ..io import load_json


def load_calibration(path):
    config = load_json(path)
    intrinsics = config.get("intrinsics")
    transform = config.get("camera_from_radar")
    if not isinstance(intrinsics, dict) or any(intrinsics.get(k) is None for k in ("fx", "fy", "cx", "cy")):
        raise ValueError("Fill intrinsics.fx/fy/cx/cy with your camera calibration")
    if not isinstance(transform, dict) or transform.get("rotation") is None or transform.get("translation") is None:
        raise ValueError("Fill camera_from_radar.rotation and translation with your rigid calibration")
    return CameraIntrinsics(**intrinsics), RigidTransform(transform["rotation"], transform["translation"])
