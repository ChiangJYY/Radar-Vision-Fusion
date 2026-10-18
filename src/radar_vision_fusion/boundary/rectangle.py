"""Minimum-area initialization and Cauchy-robust rectangle reconstruction.

Fits describe boundary walks using distance to the perimeter, rather than a
constraint that encloses every observation. Bounds use rectangle parameters
[centre_x, centre_y, log_half_width, log_half_height, angle_radians].
"""

from collections.abc import Mapping

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, QhullError


def _points(points, *, minimum=0):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum:
        raise ValueError(f"Expected at least {minimum} points of shape (N, 2)")
    if not np.isfinite(points).all():
        raise ValueError("Rectangle points must be finite")
    return points


def minimum_area_rectangle(points):
    """Return four counterclockwise corners of an enclosing oriented rectangle."""
    points = _points(points, minimum=3)
    try:
        hull = points[ConvexHull(points).vertices]
    except QhullError as error:
        raise ValueError("Need noncollinear points for a rectangle") from error
    best, best_area = None, np.inf
    for first, second in zip(hull, np.roll(hull, -1, axis=0)):
        axis0 = (second - first) / np.linalg.norm(second - first)
        axis1 = np.array([-axis0[1], axis0[0]])
        rotation = np.stack([axis0, axis1], axis=1)
        local = hull @ rotation
        lower, upper = local.min(axis=0), local.max(axis=0)
        area = float(np.prod(upper - lower))
        if area < best_area:
            best_area = area
            best = np.array([
                [lower[0], lower[1]], [upper[0], lower[1]],
                [upper[0], upper[1]], [lower[0], upper[1]],
            ]) @ rotation.T
    if best is None or best_area <= 1e-12:
        raise ValueError("Degenerate rectangle input")
    return best


def rectangle_parameters(corners):
    """Convert ordered rectangle corners into centre, log half-sizes, and angle."""
    corners = _points(corners, minimum=4)
    if corners.shape != (4, 2):
        raise ValueError("Expected four ordered rectangle corners")
    edge0, edge1 = corners[1] - corners[0], corners[2] - corners[1]
    length0, length1 = np.linalg.norm(edge0), np.linalg.norm(edge1)
    if min(length0, length1) <= 0:
        raise ValueError("Rectangle side lengths must be positive")
    tolerance = 1e-6 * length0 * length1
    if abs(np.dot(edge0, edge1)) > tolerance:
        raise ValueError("Adjacent rectangle edges must be orthogonal")
    if not np.allclose(corners[0] + corners[2], corners[1] + corners[3]):
        raise ValueError("Opposite rectangle corners must share a centre")
    return np.array([
        *corners.mean(axis=0), np.log(length0 / 2), np.log(length1 / 2),
        np.arctan2(edge0[1], edge0[0]),
    ])


def corners_from_parameters(parameters):
    """Reconstruct four counterclockwise rectangle corners from five parameters."""
    parameters = np.asarray(parameters, dtype=np.float64)
    if parameters.shape != (5,) or not np.isfinite(parameters).all():
        raise ValueError("Expected five finite rectangle parameters")
    centre, half0, half1, theta = parameters[:2], np.exp(parameters[2]), np.exp(parameters[3]), parameters[4]
    axis0 = np.array([np.cos(theta), np.sin(theta)])
    axis1 = np.array([-np.sin(theta), np.cos(theta)])
    return np.array([
        centre - half0 * axis0 - half1 * axis1,
        centre + half0 * axis0 - half1 * axis1,
        centre + half0 * axis0 + half1 * axis1,
        centre - half0 * axis0 + half1 * axis1,
    ])


def perimeter_distance(parameters, points):
    """Return nonnegative distance to the rectangle perimeter for each point."""
    centre = np.asarray(parameters[:2], dtype=np.float64)
    half0, half1, theta = np.exp(parameters[2]), np.exp(parameters[3]), parameters[4]
    shifted = np.asarray(points, dtype=np.float64) - centre
    cosine, sine = np.cos(theta), np.sin(theta)
    local0 = np.abs(shifted[:, 0] * cosine + shifted[:, 1] * sine)
    local1 = np.abs(-shifted[:, 0] * sine + shifted[:, 1] * cosine)
    outside = np.hypot(np.maximum(local0 - half0, 0), np.maximum(local1 - half1, 0))
    inside = np.minimum(half0 - local0, half1 - local1)
    return np.where((local0 <= half0) & (local1 <= half1), inside, outside)


def robust_rectangle_fit(points, candidates=None, *, f_scale=0.1, bounds=None, max_nfev=10000):
    """Fit a Cauchy-robust perimeter rectangle from one or more initial rectangles.

    ``candidates`` is a sequence of (name, corners) pairs; the default is the
    minimum-area rectangle. Returns ``(corners, trials, selected_trial)``.
    ``f_scale`` uses point-coordinate units. Default bounds impose a numerical
    minimum half-size of 1e-6, no site-specific position or size range, and an
    angle range [-2*pi, 2*pi]. Supply (lower, upper) five-element arrays to bound
    a known application. Only successful optimization trials are considered.
    """
    points = _points(points, minimum=3)
    if not np.isfinite(f_scale) or f_scale <= 0 or max_nfev < 1:
        raise ValueError("f_scale and max_nfev must be positive")
    if bounds is None:
        lower = np.array([-np.inf, -np.inf, np.log(1e-6), np.log(1e-6), -2 * np.pi])
        upper = np.array([np.inf, np.inf, np.inf, np.inf, 2 * np.pi])
    else:
        lower, upper = (np.asarray(side, dtype=np.float64) for side in bounds)
        if lower.shape != (5,) or upper.shape != (5,) or np.any(lower >= upper):
            raise ValueError("bounds must be ordered five-element lower/upper arrays")
        if np.isnan(lower).any() or np.isnan(upper).any():
            raise ValueError("bounds cannot contain NaN")
    if candidates is None:
        candidates = [("minimum_area_start", minimum_area_rectangle(points))]
    trials = []
    for name, candidate in candidates:
        initial = rectangle_parameters(candidate)
        if np.any(initial < lower) or np.any(initial > upper):
            raise ValueError(f"Initial rectangle {name!r} lies outside the configured bounds")
        result = least_squares(
            perimeter_distance, initial, args=(points,), bounds=(lower, upper),
            loss="cauchy", f_scale=f_scale, max_nfev=max_nfev,
            ftol=1e-12, xtol=1e-12, gtol=1e-12,
        )
        distances = perimeter_distance(result.x, points)
        trials.append({
            "start": str(name), "cost": float(result.cost), "parameters": result.x.tolist(),
            "median_point_distance": float(np.median(distances)),
            "p90_point_distance": float(np.percentile(distances, 90)),
            "success": bool(result.success),
        })
    successful = [trial for trial in trials if trial["success"]]
    if not successful:
        raise RuntimeError("All robust rectangle fits failed")
    best = min(successful, key=lambda item: (item["cost"], item["start"]))
    return corners_from_parameters(best["parameters"]), trials, best


def fit_pooled_rectangle(points_by_scene: Mapping, *, f_scale=0.1, bounds=None, audit=None):
    """Fit one rectangle to scene-pooled boundary observations.

    Every scene needs at least three noncollinear points. First fit each scene;
    then fit pooled observations using both its enclosing rectangle and each
    individual robust result as candidates. Returns (pooled_rectangle,
    individual_rectangles, pooled_points). An optional dict ``audit`` receives
    per-scene and pooled optimizer trials. Scene keys/order belong to the caller.
    """
    if not points_by_scene:
        raise ValueError("At least one scene is required")
    individual = {}
    values = []
    for name, points in points_by_scene.items():
        points = _points(points, minimum=3)
        fitted, trials, selected = robust_rectangle_fit(points, f_scale=f_scale, bounds=bounds)
        individual[name] = fitted
        values.append(points)
        if audit is not None:
            audit[f"individual_{name}"] = {"trials": trials, "selected": selected}
    pooled = np.vstack(values)
    candidates = [("pooled_minimum_area", minimum_area_rectangle(pooled))]
    candidates.extend((f"{name}_robust", corners) for name, corners in individual.items())
    fitted, trials, selected = robust_rectangle_fit(pooled, candidates, f_scale=f_scale, bounds=bounds)
    if audit is not None:
        audit["pooled"] = {"trials": trials, "selected": selected}
    return fitted, individual, pooled


def sample_perimeter(corners, samples=400):
    """Sample an exact number of points uniformly by rectangle perimeter length."""
    corners = _points(corners, minimum=4)
    rectangle_parameters(corners)
    if not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    cumulative = np.r_[0, np.cumsum(lengths)]
    distances = np.arange(samples) * cumulative[-1] / samples
    indices = np.searchsorted(cumulative[1:], distances, side="right")
    fractions = (distances - cumulative[indices]) / lengths[indices]
    return corners[indices] + fractions[:, None] * edges[indices]
