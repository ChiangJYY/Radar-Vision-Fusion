"""Portable JSON input/output. Sensor xy values already use the shared frame."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .tracking.math_utils import Vec2d, Vec4d
from .tracking.sensors_detection import CameraDetection, CameraFrame, RadarDetection, RadarFrame


def _vector(value, size: int, label: str) -> list[float]:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain {size} finite numbers")
    return array.tolist()


def load_json(path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, value) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def detection_frames(payload: dict) -> list[tuple[CameraFrame, RadarFrame]]:
    """Parse ``frames`` containing timestamps and camera/radar detection lists.

    A missing sensor is represented by an empty list, while the timestamp is
    retained. Ground-truth positions are deliberately outside this interface.
    """
    result = []
    previous = -float("inf")
    for index, item in enumerate(payload["frames"]):
        timestamp = float(item["timestamp"])
        if not np.isfinite(timestamp) or timestamp <= previous:
            raise ValueError("frame timestamps must be finite and strictly increasing")
        previous = timestamp
        camera, radar = [], []
        for number, detection in enumerate(item.get("camera", [])):
            xy = _vector(detection["xy"], 2, "camera xy")
            bbox = _vector(detection["bbox"], 4, "camera bbox")
            uv = detection.get("center_uv", [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            score = float(detection.get("score", 1.0))
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("camera score must be in [0,1]")
            camera.append(CameraDetection(int(detection.get("id", number)), timestamp, Vec2d(*xy),
                                          Vec4d(*bbox), Vec2d(*_vector(uv, 2, "center_uv")), score))
        for number, detection in enumerate(item.get("radar", [])):
            xy = _vector(detection["xy"], 2, "radar xy")
            velocity = _vector(detection.get("velocity", [0, 0]), 2, "radar velocity")
            acceleration = _vector(detection.get("acceleration", [0, 0]), 2, "radar acceleration")
            confidence = float(detection.get("confidence", 1.0))
            if not np.isfinite(confidence):
                raise ValueError("radar confidence must be finite")
            radar.append(RadarDetection(int(detection.get("id", number)), timestamp, Vec2d(*xy),
                                       Vec2d(*velocity), Vec2d(*acceleration),
                                       float(detection.get("g", 0.0)), confidence))
        result.append((CameraFrame(index, timestamp, camera), RadarFrame(index, timestamp, radar)))
    return result


def load_detection_sequence(path) -> list[tuple[CameraFrame, RadarFrame]]:
    return detection_frames(load_json(path))
