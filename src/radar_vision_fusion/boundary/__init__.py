"""ECDR-GAT depth correction and robust monitored-region reconstruction."""

from .ecdr_gat import (
    GATLayer,
    GATResidualNet,
    correct_depth,
    project_to_radar_torch,
    radar_supervised_loss,
)
from .graph import GraphNormalization, build_knn_edge_index, build_scene_graph
from .rectangle import (
    corners_from_parameters,
    fit_pooled_rectangle,
    minimum_area_rectangle,
    perimeter_distance,
    rectangle_parameters,
    robust_rectangle_fit,
    sample_perimeter,
)

__all__ = [
    "GATLayer", "GATResidualNet", "correct_depth", "project_to_radar_torch",
    "radar_supervised_loss", "GraphNormalization", "build_knn_edge_index",
    "build_scene_graph", "minimum_area_rectangle", "rectangle_parameters",
    "corners_from_parameters", "perimeter_distance", "robust_rectangle_fit",
    "fit_pooled_rectangle", "sample_perimeter",
]
