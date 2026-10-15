from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .math_utils import Vec2d, Vec4d


@dataclass(frozen=True)
class CameraDetection:
    id: int
    timestamp: float
    position: Vec2d  # x, y
    bbox: Vec4d  # x1, y1, x2, y2
    center_uv: Vec2d  # center_u, center_v
    score: float


@dataclass(frozen=True)
class CameraFrame:
    index: int
    timestamp: float
    detections: List[CameraDetection]


@dataclass(frozen=True)
class RadarDetection:
    id: int
    timestamp: float
    position: Vec2d  # X, Y
    velocity: Vec2d  # Vx, Vy
    acceleration: Vec2d  # Ax, Ay
    g: float
    confidence_level: float


@dataclass(frozen=True)
class RadarFrame:
    index: int
    timestamp: float
    detections: List[RadarDetection]



