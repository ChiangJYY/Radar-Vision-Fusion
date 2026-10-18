"""Robust foot-point sampling and person-height depth fusion.

These helpers accept an existing metric depth map and bounding box. Detection
models, depth-model weights, and image preprocessing are intentionally external.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class DepthStats:
    """Median optical-axis depth, unscaled MAD, and valid sample count."""

    median: float
    mad: float
    valid: int
    window: int


@dataclass(frozen=True)
class DepthFusionConfig:
    """Configurable height prior and source-method gating parameters.

    ``person_height_m`` is an assumed height, not a measured installation value.
    Defaults retain the source algorithm's heuristic choices.
    """

    person_height_m: float = 1.7
    height_bias_px: float = 80.0
    mad_scale_m: float = 0.5
    relative_error_gate: float = 0.1
    max_depth_weight: float = 0.3

    def __post_init__(self):
        values = [self.person_height_m, self.height_bias_px, self.mad_scale_m,
                  self.relative_error_gate, self.max_depth_weight]
        if not np.isfinite(values).all():
            raise ValueError("Fusion parameters must be finite")
        if self.person_height_m <= 0 or self.height_bias_px < 0 or self.mad_scale_m <= 0:
            raise ValueError("Height/MAD scale must be positive; height bias must be nonnegative")
        if self.relative_error_gate < 0 or not 0 <= self.max_depth_weight <= 1:
            raise ValueError("Invalid fusion gate or depth weight")


@dataclass(frozen=True)
class DepthFusionResult:
    """Fused depth and the information used to make the fusion decision."""

    depth: float
    prior_depth: float
    depth_weight: float
    relative_error: float
    mode: str


def bbox_sample_point(box_xyxy, image_shape, *, mode="foot", foot_up_px=5.0):
    """Return clipped (u, v) for a foot point or centre of an xyxy box.

    ``image_shape`` is (height, width), matching the depth map's resolution.
    """
    box = np.asarray(box_xyxy, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("box_xyxy must contain four finite values")
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Bounding box must have positive width and height")
    height, width = image_shape[:2]
    if height <= 0 or width <= 0 or foot_up_px < 0:
        raise ValueError("Invalid image dimensions or foot offset")
    if mode not in {"foot", "center"}:
        raise ValueError("mode must be 'foot' or 'center'")
    u = (x1 + x2) / 2
    v = (y1 + y2) / 2 if mode == "center" else y2 - foot_up_px
    return float(np.clip(u, 0, width - 1)), float(np.clip(v, 0, height - 1))


def sample_depth_stats(depth_map, u: float, v: float, window: int = 9):
    """Sample finite positive depths, returning median and median absolute deviation.

    ``window=0`` samples one pixel. Otherwise an odd square window is clipped to
    image bounds. An empty valid patch returns NaN statistics and ``valid=0``.
    """
    depth = np.asarray(depth_map, dtype=np.float64)
    if depth.ndim != 2 or min(depth.shape) == 0:
        raise ValueError("depth_map must be a nonempty H x W array")
    if not np.isfinite([u, v]).all():
        raise ValueError("Sample coordinates must be finite")
    if not isinstance(window, int) or window < 0 or (window > 0 and window % 2 == 0):
        raise ValueError("window must be zero or a positive odd integer")
    height, width = depth.shape
    ui = int(np.clip(round(u), 0, width - 1))
    vi = int(np.clip(round(v), 0, height - 1))
    radius = window // 2
    patch = depth[max(0, vi-radius):min(height, vi+radius+1),
                  max(0, ui-radius):min(width, ui+radius+1)]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return DepthStats(float("nan"), float("nan"), 0, window)
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median))) if window else float("nan")
    return DepthStats(median, mad, int(valid.size), window)


def person_height_prior_depth(fy_px: float, person_height_m: float, bbox_height_px: float):
    """Estimate z = fy * assumed person height / box height; invalid inputs yield NaN."""
    values = [fy_px, person_height_m, bbox_height_px]
    if not np.isfinite(values).all() or min(values) <= 0:
        return float("nan")
    return float(fy_px * person_height_m / bbox_height_px)


def depth_fusion_weight(bbox_height_px: float, mad_m: float, config: DepthFusionConfig):
    """Compute h/(h+bias) * exp(-MAD/scale), then apply the depth-weight cap."""
    if not np.isfinite(bbox_height_px) or bbox_height_px <= 0:
        return 0.0
    if np.isfinite(mad_m) and mad_m < 0:
        raise ValueError("MAD must be nonnegative")
    dispersion = math.exp(-mad_m / config.mad_scale_m) if np.isfinite(mad_m) else 1.0
    raw = bbox_height_px / (bbox_height_px + config.height_bias_px) * dispersion
    return float(np.clip(raw, 0, config.max_depth_weight))


def fuse_person_depth(depth_m: float, mad_m: float, bbox_height_px: float,
                      fy_px: float, config: DepthFusionConfig | None = None):
    """Fuse metric depth with a height prior using a relative-disagreement gate.

    When both depths are valid, disagreements above the gate use the prior.
    A single valid source is used on its own; neither valid source yields NaN.
    """
    config = config or DepthFusionConfig()
    prior = person_height_prior_depth(fy_px, config.person_height_m, bbox_height_px)
    valid_depth = np.isfinite(depth_m) and depth_m > 0
    valid_prior = np.isfinite(prior) and prior > 0
    if not valid_prior:
        return DepthFusionResult(float(depth_m) if valid_depth else float("nan"), prior,
                                 1.0 if valid_depth else 0.0, float("nan"),
                                 "depth_only" if valid_depth else "invalid")
    if not valid_depth:
        return DepthFusionResult(prior, prior, 0.0, float("nan"), "prior_only")
    relative_error = float(abs(depth_m - prior) / prior)
    weight = depth_fusion_weight(bbox_height_px, mad_m, config)
    if relative_error > config.relative_error_gate:
        return DepthFusionResult(prior, prior, 0.0, relative_error, "prior_only")
    return DepthFusionResult(float(weight * depth_m + (1 - weight) * prior), prior,
                             weight, relative_error, "fused")
