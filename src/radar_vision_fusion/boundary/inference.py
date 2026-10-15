"""Explicit checkpoint-backed ECDR depth correction and stored-region reconstruction."""
import numpy as np
import torch

from .ecdr_gat import GATResidualNet
from .graph import GraphNormalization, build_scene_graph
from .rectangle import robust_rectangle_fit
from ..geometry.projection import camera_to_radar


def load_boundary_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state"), dict):
        raise ValueError("ECDR checkpoint requires state, config and normalization dictionaries")
    config, normalization = payload.get("config"), payload.get("normalization")
    if not isinstance(config, dict) or not isinstance(normalization, dict):
        raise ValueError("ECDR checkpoint must include architecture config and normalization")
    if any(key not in config for key in ("hidden", "heads", "layers")):
        raise ValueError("ECDR checkpoint config requires hidden, heads and layers")
    stats = GraphNormalization(tuple(normalization["mean"]), tuple(normalization["std"]))
    model = GATResidualNet(**{key: config[key] for key in ("hidden", "heads", "layers")})
    model.load_state_dict(payload["state"], strict=True)
    if any(not torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
        raise ValueError("ECDR checkpoint contains nonfinite parameters")
    return model.to(device).eval(), stats


def reconstruct_boundary(raw_uvz, checkpoint, intrinsics, camera_from_radar, *, k=8, device="cpu"):
    model, stats = load_boundary_checkpoint(checkpoint, device)
    raw = np.asarray(raw_uvz, dtype=float)
    graph = [tensor.to(device) for tensor in build_scene_graph(raw, stats, k=k)]
    with torch.inference_mode():
        residual = model(*graph).cpu().numpy()
    depth = raw[:, 2] + residual
    if np.any(depth <= 0) or not np.isfinite(depth).all():
        raise ValueError("ECDR produced invalid corrected optical-axis depths")
    xy = camera_to_radar(raw[:, :2], depth, intrinsics, camera_from_radar)[:, :2]
    corners, _, audit = robust_rectangle_fit(xy)
    return {"region": corners.tolist(), "corrected_xy": xy.tolist(),
            "corrected_depth": depth.tolist(), "depth_residual": residual.tolist(), "fit": audit}
