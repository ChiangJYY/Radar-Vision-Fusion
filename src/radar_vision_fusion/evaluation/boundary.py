"""Point errors and convex stored-region metrics, independent of experiments."""
import numpy as np
from scipy.spatial import cKDTree

from ..boundary.rectangle import sample_perimeter
from ..monitoring.region import _convex_polygon


def point_errors(predicted_xy, reference_xy):
    predicted, reference = np.asarray(predicted_xy, float), np.asarray(reference_xy, float)
    if predicted.shape != reference.shape or predicted.ndim != 2 or predicted.shape[1] != 2 or not len(predicted):
        raise ValueError("Corresponding point arrays must have matching nonempty [N,2] shapes")
    if not np.isfinite(predicted).all() or not np.isfinite(reference).all():
        raise ValueError("Point arrays must be finite")
    error = np.linalg.norm(predicted - reference, axis=1)
    return {"mae_m": float(error.mean()), "rmse_m": float(np.sqrt(np.mean(error ** 2))), "count": len(error)}


def _area(polygon):
    if len(polygon) < 3:
        return 0.
    p = np.asarray(polygon, float)
    return abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))) / 2


def polygon_iou(first, second):
    subject, clip = _convex_polygon(first), _convex_polygon(second)
    signed = np.sum(clip[:, 0] * np.roll(clip[:, 1], -1) - clip[:, 1] * np.roll(clip[:, 0], -1))
    if signed < 0:
        clip = clip[::-1]
    output = list(subject)
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        current, output = output, []
        if not current:
            break
        previous = current[-1]
        edge = b - a
        cross = lambda p: float(edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]))
        for point in current:
            before, after = cross(previous), cross(point)
            if (before >= -1e-10) != (after >= -1e-10):
                fraction = before / (before - after)
                output.append(previous + fraction * (point - previous))
            if after >= -1e-10:
                output.append(point)
            previous = point
    intersection = _area(output)
    return intersection / (_area(subject) + _area(clip) - intersection)


def boundary_metrics(predicted_corners, reference_corners, *, samples=400):
    """ASSD and HD95 on arc-length perimeter samples, plus convex polygon IoU.

    This evaluates caller-provided stored rectangles; it does not rerun paper
    preprocessing, alignment, point grouping or confidence-interval analysis.
    """
    first, second = sample_perimeter(predicted_corners, samples), sample_perimeter(reference_corners, samples)
    forward = cKDTree(second).query(first)[0]
    backward = cKDTree(first).query(second)[0]
    return {"assd_m": float((forward.mean() + backward.mean()) / 2),
            "hd95_m": float(max(np.percentile(forward, 95), np.percentile(backward, 95))),
            "iou": float(polygon_iou(predicted_corners, reference_corners)), "samples": samples}
