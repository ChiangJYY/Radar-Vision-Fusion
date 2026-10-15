"""Scene-local normalized k-nearest-neighbour graphs for ECDR-GAT."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
import torch


@dataclass(frozen=True)
class GraphNormalization:
    """Mean and standard deviation for [u, v, metric depth], fitted on training data.

    Identity defaults permit synthetic use. Real applications must supply their
    own training statistics and retain them with the model checkpoint.
    """

    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self):
        mean, std = np.asarray(self.mean), np.asarray(self.std)
        if mean.shape != (3,) or std.shape != (3,):
            raise ValueError("mean and std must each contain three values")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("Statistics must be finite and standard deviations positive")
        object.__setattr__(self, "mean", tuple(float(x) for x in mean))
        object.__setattr__(self, "std", tuple(float(x) for x in std))

    @classmethod
    def fit(cls, raw_uvz, *, minimum_std: float = 1e-6):
        """Fit population statistics on nonempty training rows [u, v, depth]."""
        values = _feature_array(raw_uvz)
        if len(values) == 0 or not np.isfinite(minimum_std) or minimum_std <= 0:
            raise ValueError("Need nonempty features and positive minimum_std")
        return cls(tuple(values.mean(axis=0)), tuple(np.maximum(values.std(axis=0), minimum_std)))

    def normalize(self, raw_uvz):
        """Normalize rows into a float32 NumPy array."""
        values = _feature_array(raw_uvz)
        return ((values - np.asarray(self.mean)) / np.asarray(self.std)).astype(np.float32)


def _feature_array(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Expected finite features of shape (N, 3)")
    return values


def build_knn_edge_index(features_n, k: int = 8):
    """Build a directed graph from normalized features (N, 3).

    Returns CPU tensors: ``edge_index`` (2, E), ``edge_attr`` (E, 4).
    Row 0 is the receiver/query node; row 1 is its neighbour. Edge features are
    [neighbour-minus-query u, v, z, 1/(distance+1e-6)]. Nondegenerate graphs
    exclude self edges. A single-node graph gets one zero-feature self edge.
    Construct each scene separately to avoid connections between scenes.
    """
    features = _feature_array(features_n).astype(np.float32)
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    count = len(features)
    if count < 2:
        indices = np.stack([np.arange(count), np.arange(count)], axis=0)
        attributes = np.zeros((count, 4), dtype=np.float32)
    else:
        neighbours = min(k, count - 1)
        _, nearest = cKDTree(features).query(features, k=neighbours + 1)
        # Truncate after removing self, including coincident feature rows.
        sources, destinations = [], []
        for index, candidates in enumerate(nearest):
            selected = candidates[candidates != index][:neighbours]
            sources.extend([index] * len(selected))
            destinations.extend(selected.tolist())
        source = np.asarray(sources, dtype=np.int64)
        destination = np.asarray(destinations, dtype=np.int64)
        indices = np.stack([source, destination], axis=0)
        delta = features[destination] - features[source]
        distance = np.linalg.norm(delta, axis=-1) + 1e-6
        attributes = np.concatenate([delta, (1 / distance)[:, None]], axis=-1).astype(np.float32)
    return torch.from_numpy(indices.astype(np.int64)), torch.from_numpy(attributes)


def build_scene_graph(raw_uvz, normalization: GraphNormalization | None = None,
                      *, k: int = 8, device=None):
    """Return ``(node_features, edge_index, edge_attr)`` for one scene.

    ``raw_uvz`` contains pixel u, pixel v, and metric optical-axis depth. All
    tensors are float32 except long edge indices; ``device`` defaults to CPU.
    """
    normalization = normalization or GraphNormalization()
    features = normalization.normalize(raw_uvz)
    edge_index, edge_attr = build_knn_edge_index(features, k=k)
    return (torch.from_numpy(features).to(device=device),
            edge_index.to(device=device), edge_attr.to(device=device))
