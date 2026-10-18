"""Distance and crossing rules extracted from the joint monitoring experiments.

Distances are negative inside a convex polygon and positive outside. Crossing
events are extracted from stable runs in a sequence, so their timestamp is the
first sample of the new state, rather than the time an online alarm is issued.
"""

from __future__ import annotations

import numpy as np


def _convex_polygon(polygon) -> np.ndarray:
    vertices = np.asarray(polygon, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise ValueError("polygon must contain at least three ordered xy vertices")
    if not np.isfinite(vertices).all():
        raise ValueError("polygon vertices must be finite")
    edges = np.roll(vertices, -1, axis=0) - vertices
    if np.any(np.linalg.norm(edges, axis=1) <= 1e-12):
        raise ValueError("polygon must not contain repeated consecutive vertices")
    next_edges = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    if not ((turns >= -1e-12).all() or (turns <= 1e-12).all()):
        raise ValueError("signed_distance supports ordered convex polygons")
    area_twice = np.sum(vertices[:, 0] * np.roll(vertices[:, 1], -1)
                        - vertices[:, 1] * np.roll(vertices[:, 0], -1))
    if abs(area_twice) <= 1e-12:
        raise ValueError("polygon must have positive area")
    # Every vertex must lie in each edge's interior half-plane. This also
    # rejects self-intersecting vertex orders with locally consistent turns.
    relative = vertices[None, :, :] - vertices[:, None, :]
    cross = edges[:, None, 0] * relative[:, :, 1] - edges[:, None, 1] * relative[:, :, 0]
    if not ((cross >= -1e-12).all() or (cross <= 1e-12).all()):
        raise ValueError("polygon must be convex and ordered without self-intersection")
    return vertices


def signed_distance(points, polygon) -> np.ndarray:
    """Return signed Euclidean distances in the same units as the input xy."""
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        points = points.reshape(0, 2)
    if points.ndim == 1 and points.shape == (2,):
        points = points.reshape(1, 2)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("points must be a finite array with shape [N,2]")
    vertices = _convex_polygon(polygon)
    edges = np.roll(vertices, -1, axis=0) - vertices
    relative = points[:, None, :] - vertices
    fraction = np.clip(np.sum(relative * edges, axis=2) / np.sum(edges * edges, axis=1), 0, 1)
    distance = np.linalg.norm(relative - fraction[:, :, None] * edges, axis=2).min(axis=1)
    cross = edges[:, 0] * relative[:, :, 1] - edges[:, 1] * relative[:, :, 0]
    inside = (cross >= -1e-12).all(axis=1) | (cross <= 1e-12).all(axis=1)
    return np.where(inside, -distance, distance)


def region_relation(xy, polygon) -> dict:
    """Compute one track's distance and inside state without ground truth."""
    point = np.asarray(xy, dtype=float)
    if point.shape != (2,):
        raise ValueError("xy must contain exactly two coordinates")
    distance = float(signed_distance(point, polygon)[0])
    return {"signed_distance": distance, "inside": distance <= 0}


def event_list(frame, time, distance, minimum_run: int = 3, maximum_gap: float = 0.35) -> list[dict]:
    """Extract entry/exit events between sufficiently long consecutive runs.

    Missing distances and nonconsecutive frame indices split the sequence. Run
    lengths count samples. ``maximum_gap`` is expressed in seconds.
    """
    frame = np.asarray(frame, dtype=int)
    time = np.asarray(time, dtype=float)
    distance = np.asarray(distance, dtype=float)
    if frame.ndim != 1 or time.shape != frame.shape or distance.shape != frame.shape:
        raise ValueError("frame, time and distance must be equally sized 1D arrays")
    if minimum_run < 1 or maximum_gap <= 0:
        raise ValueError("minimum_run and maximum_gap must be positive")
    if not np.isfinite(time).all() or np.any(np.diff(frame) <= 0) or np.any(np.diff(time) <= 0):
        raise ValueError("frame indices and finite timestamps must increase strictly")
    indices = np.flatnonzero(np.isfinite(distance))
    if not len(indices):
        return []
    segments, current = [], [indices[0]]
    for index in indices[1:]:
        previous = current[-1]
        if frame[index] == frame[previous] + 1 and time[index] - time[previous] <= maximum_gap:
            current.append(index)
        else:
            segments.append(current)
            current = [index]
    segments.append(current)
    output = []
    for segment in segments:
        state = distance[segment] <= 0
        starts = [0] + [i for i in range(1, len(segment)) if state[i] != state[i - 1]] + [len(segment)]
        runs = [(starts[i], starts[i + 1], bool(state[starts[i]])) for i in range(len(starts) - 1)]
        for old, new in zip(runs, runs[1:]):
            if old[1] - old[0] >= minimum_run and new[1] - new[0] >= minimum_run:
                index = segment[new[0]]
                output.append({"time": float(time[index]), "event": "entry" if new[2] else "exit"})
    return output
