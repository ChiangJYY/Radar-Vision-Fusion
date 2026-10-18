from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Vec2d:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: "Vec2d") -> "Vec2d":
        return Vec2d(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2d") -> "Vec2d":
        return Vec2d(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2d":
        return Vec2d(self.x * scalar, self.y * scalar)

    __rmul__ = __mul__

    def length(self) -> float:
        return math.hypot(self.x, self.y)


@dataclass(frozen=True)
class Vec3d:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def xy(self) -> Vec2d:
        return Vec2d(self.x, self.y)

    def __add__(self, other: "Vec3d") -> "Vec3d":
        return Vec3d(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3d") -> "Vec3d":
        return Vec3d(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3d":
        return Vec3d(self.x * scalar, self.y * scalar, self.z * scalar)

    __rmul__ = __mul__

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)


@dataclass(frozen=True)
class Vec4d:
    a1: float = 0.0
    a2: float = 0.0
    a3: float = 0.0
    a4: float = 0.0

    def xy(self) -> Vec2d:
        return Vec2d(self.a1, self.a2)

    def __add__(self, other: "Vec4d") -> "Vec4d":
        return Vec4d(
            self.a1 + other.a1,
            self.a2 + other.a2,
            self.a3 + other.a3,
            self.a4 + other.a4,
        )

    def __sub__(self, other: "Vec4d") -> "Vec4d":
        return Vec4d(
            self.a1 - other.a1,
            self.a2 - other.a2,
            self.a3 - other.a3,
            self.a4 - other.a4,
        )

    def __mul__(self, scalar: float) -> "Vec4d":
        return Vec4d(
            self.a1 * scalar,
            self.a2 * scalar,
            self.a3 * scalar,
            self.a4 * scalar,
        )

    __rmul__ = __mul__

    def length(self) -> float:
        return math.sqrt(
            self.a1 * self.a1
                + self.a2 * self.a2
                + self.a3 * self.a3
                + self.a4 * self.a4
        )


@dataclass(frozen=True)
class Transformation2d:
    translation: Vec2d = Vec2d()
    theta: float = 0.0

    def transform(self, point: Vec2d) -> Vec2d:
        p = point - self.translation
        sin_theta = math.sin(self.theta)
        cos_theta = math.cos(self.theta)
        return Vec2d(
            p.x * cos_theta + p.y * sin_theta,
            p.y * cos_theta - p.x * sin_theta,
        )

