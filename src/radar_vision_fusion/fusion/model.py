"""Causal measurement-covariance network used by the Trans-DKF method.

The network takes the raw 23-dimensional tracking features, normalizes them
with training-set statistics and predicts six positive measurement variances.
Process noise is fixed by the filter, rather than predicted by this network.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CovarianceConfig:
    """The published architecture; smaller values are useful for examples."""

    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 16
    max_seq_len: int = 256
    dropout: float = 0.1
    variance_floor: float = 0.01

    def __post_init__(self) -> None:
        if self.hidden_size < 2 or self.num_heads < 1:
            raise ValueError("hidden_size >= 2 and num_heads >= 1 are required")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_layers < 1 or self.max_seq_len < 1:
            raise ValueError("num_layers and max_seq_len must be positive")
        if not 0 <= self.dropout < 1 or not self.variance_floor > 0:
            raise ValueError("dropout must be in [0, 1), variance_floor > 0")

    def to_dict(self) -> dict:
        return asdict(self)


class CovarianceNet(nn.Module):
    """Predict camera ``R[2]`` and radar ``R[4]`` from ``[B, T, 23]``.

    Parameter names match the formal full129 experiment's ``kind='trans'``
    network, so its state dictionary can be loaded strictly with the default
    architecture. Normalization statistics are part of the state dictionary.
    """

    input_size = 23
    kind = "trans"

    def __init__(
        self,
        mean: Sequence[float] | torch.Tensor | None = None,
        std: Sequence[float] | torch.Tensor | None = None,
        config: CovarianceConfig | Mapping | None = None,
    ) -> None:
        super().__init__()
        self.config = (
            CovarianceConfig(**dict(config)) if isinstance(config, Mapping)
            else config or CovarianceConfig()
        )
        mean_tensor = torch.zeros(23) if mean is None else torch.as_tensor(mean, dtype=torch.float32)
        std_tensor = torch.ones(23) if std is None else torch.as_tensor(std, dtype=torch.float32)
        if mean_tensor.shape != (23,) or std_tensor.shape != (23,):
            raise ValueError("mean and std must each have 23 entries")
        if not torch.isfinite(mean_tensor).all() or not torch.isfinite(std_tensor).all():
            raise ValueError("normalization statistics must be finite")
        if (std_tensor < 0).any():
            raise ValueError("std cannot be negative")
        self.register_buffer("mean", mean_tensor.clone())
        self.register_buffer("std", std_tensor.clone())
        c = self.config
        self.embedding = nn.Linear(23, c.hidden_size)
        self.pos = nn.Parameter(torch.randn(c.max_seq_len, c.hidden_size))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                c.hidden_size, c.num_heads, 4 * c.hidden_size,
                c.dropout, "gelu", batch_first=True,
            ),
            c.num_layers,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(c.hidden_size),
            nn.Linear(c.hidden_size, c.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_size // 2, 6),
        )

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.ndim != 3 or raw.shape[-1] != 23:
            raise ValueError("raw must have shape [batch, time, 23]")
        t = raw.shape[1]
        if not 1 <= t <= self.config.max_seq_len:
            raise ValueError(f"sequence length must be in [1, {self.config.max_seq_len}]")
        if not torch.isfinite(raw).all():
            raise ValueError("features must be finite; use validity flags for missing observations")
        normalized = (raw - self.mean) / (self.std + 1e-6)
        h = self.embedding(normalized) + self.pos[:t]
        causal_mask = torch.triu(
            torch.ones(t, t, dtype=torch.bool, device=h.device), diagonal=1
        )
        h = self.encoder(h, mask=causal_mask)
        return F.softplus(self.head(h)) + self.config.variance_floor
