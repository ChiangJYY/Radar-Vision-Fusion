"""Edge-conditioned depth-residual GAT and its radar-supervised objective.

Extracted from the formal ECDR training path. The published configuration uses
hidden=32, heads=4, layers=2, k=8. Feature statistics, calibration, and learned
weights must be supplied by the caller.
"""

import torch
from torch import nn
from torch.nn import functional as F

from ..geometry.projection import CameraIntrinsics, RigidTransform


class GATLayer(nn.Module):
    """Multihead attention with four edge channels injected into attention scores.

    The first edge-index row receives messages from the second row, matching
    :func:`build_knn_edge_index`. Attention normalizes over each receiver's
    neighbours, independently for each head.
    """

    def __init__(self, in_dim, out_dim, edge_dim=4, heads=4, dropout=0.0, concat=True):
        super().__init__()
        if heads < 1 or out_dim < 1 or out_dim % heads != 0:
            raise ValueError("out_dim must be positive and divisible by heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.heads = heads
        self.head_dim = out_dim // heads
        self.concat = concat
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.W_e = nn.Linear(edge_dim, heads, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        for parameter in (self.W.weight, self.W_e.weight, self.a_src, self.a_dst):
            nn.init.xavier_uniform_(parameter)
        self.lrelu = nn.LeakyReLU(0.2)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        """Map node features (N, in_dim) to (N, out_dim) when concat=True."""
        count = x.size(0)
        heads, dimension = self.heads, self.head_dim
        projected = self.W(x).view(count, heads, dimension)
        source, destination = edge_index[0], edge_index[1]
        receiver, neighbour = projected[source], projected[destination]
        logits = (receiver * self.a_src).sum(-1) + (neighbour * self.a_dst).sum(-1)
        logits = self.lrelu(logits + self.W_e(edge_attr))
        scatter_index = source[:, None].expand(-1, heads)
        maxima = torch.full((count, heads), -torch.inf, dtype=logits.dtype, device=x.device)
        maxima = maxima.scatter_reduce(0, scatter_index, logits, reduce="amax", include_self=False)
        exponentials = torch.exp(logits - maxima[source])
        denominator = torch.zeros((count, heads), dtype=logits.dtype, device=x.device)
        denominator.scatter_add_(0, scatter_index, exponentials)
        attention = exponentials / denominator.clamp_min(1e-12)[source]
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        messages = neighbour * attention.unsqueeze(-1)
        output = torch.zeros((count, heads, dimension), dtype=messages.dtype, device=x.device)
        output.scatter_add_(0, source[:, None, None].expand(-1, heads, dimension), messages)
        return output.reshape(count, heads * dimension) if self.concat else output.mean(dim=1)


class GATResidualNet(nn.Module):
    """Predict one additive metric depth residual per [u_n, v_n, z_n] node.

    A linear input projection is followed by residual GAT blocks, layer norms,
    and ReLU activations. A two-layer MLP head starts at zero residual. Parameter
    names retain compatibility with source ECDR state dictionaries.
    """

    def __init__(self, in_dim=3, edge_dim=4, hidden=32, heads=4, layers=2, dropout=0.0):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        self.input_proj = nn.Linear(in_dim, hidden)
        self.norms_in = nn.LayerNorm(hidden)
        self.layers = nn.ModuleList([
            GATLayer(hidden, hidden, edge_dim=edge_dim, heads=heads, dropout=dropout)
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, edge_index, edge_attr):
        """Return additive depth residuals of shape (N,) in depth units."""
        hidden = F.relu(self.norms_in(self.input_proj(x)))
        for layer, norm in zip(self.layers, self.norms):
            hidden = F.relu(norm(hidden + layer(hidden, edge_index, edge_attr)))
        return self.head(hidden).squeeze(-1)


def correct_depth(model, node_features, edge_index, edge_attr, raw_depth):
    """Return ``(corrected_depth, delta_z)`` without clamping learned residuals.

    The source objective uses raw depth + residual directly. Callers should
    reject nonpositive corrected depths before geometric inference.
    """
    delta_z = model(node_features, edge_index, edge_attr)
    if raw_depth.shape != delta_z.shape:
        raise ValueError("raw_depth must match the model's (N,) output")
    return raw_depth + delta_z, delta_z


def project_to_radar_torch(raw_uvz, delta_z, intrinsics: CameraIntrinsics,
                           camera_from_radar: RigidTransform):
    """Differentiably back-project corrected depth to radar coordinates (N, 3).

    No depth clamp is applied, retaining the source training loss. Calibration
    is fixed and converted to the same device and dtype as the depth residual.
    """
    if raw_uvz.ndim != 2 or raw_uvz.shape[1] != 3 or delta_z.shape != raw_uvz.shape[:1]:
        raise ValueError("Expected raw_uvz (N, 3) and delta_z (N,)")
    raw_uvz = raw_uvz.to(device=delta_z.device, dtype=delta_z.dtype)
    depth = raw_uvz[:, 2] + delta_z
    camera = torch.stack([
        (raw_uvz[:, 0] - intrinsics.cx) / intrinsics.fx * depth,
        (raw_uvz[:, 1] - intrinsics.cy) / intrinsics.fy * depth,
        depth,
    ], dim=-1)
    rotation = torch.tensor(camera_from_radar.rotation, dtype=depth.dtype, device=depth.device)
    translation = torch.tensor(camera_from_radar.translation, dtype=depth.dtype, device=depth.device)
    return (camera - translation) @ rotation


def radar_supervised_loss(delta_z, raw_uvz, radar_xy, intrinsics: CameraIntrinsics,
                          camera_from_radar: RigidTransform, *, regularization=0.01):
    """Return ``(loss, detached_metrics, predicted_radar_xy)`` for one scene.

    The formal objective is mean |x_pred-x_radar| + mean |y_pred-y_radar|
    + 0.01 * mean |delta_z|. Ground truth is not part of this training objective.
    All observations must be valid, matched radar/camera rows in common units.
    """
    if delta_z.numel() == 0 or radar_xy.shape != (delta_z.numel(), 2):
        raise ValueError("Need nonempty residuals and matching radar_xy (N, 2)")
    if regularization < 0:
        raise ValueError("regularization must be nonnegative")
    prediction = project_to_radar_torch(raw_uvz, delta_z, intrinsics, camera_from_radar)[:, :2]
    radar_xy = radar_xy.to(device=prediction.device, dtype=prediction.dtype)
    radar_loss = (prediction - radar_xy).abs().mean(dim=0).sum()
    residual_loss = delta_z.abs().mean()
    loss = radar_loss + regularization * residual_loss
    metrics = {
        "loss": float(loss.detach()),
        "loss_radar": float(radar_loss.detach()),
        "loss_reg": float(residual_loss.detach()),
        "delta_z_mean": float(delta_z.detach().mean()),
    }
    return loss, metrics, prediction
