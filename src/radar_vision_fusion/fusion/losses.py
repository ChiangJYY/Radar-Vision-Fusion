"""Position-supervision loss; no training loop or optimizer is distributed."""

from __future__ import annotations

import torch



def position_mse(
    predicted: torch.Tensor, target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared error per coordinate; optional mask has shape [B,T]."""
    if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[-1] != 2:
        raise ValueError("predicted and target must have matching [batch,time,2] shapes")
    if not torch.isfinite(predicted).all():
        raise ValueError("predicted positions must be finite")
    if mask is None:
        if not torch.isfinite(target).all():
            raise ValueError("target positions must be finite")
        return (predicted - target).square().mean()
    if mask.shape != predicted.shape[:2]:
        raise ValueError("mask must have shape [batch,time]")
    selected = mask.to(dtype=torch.bool)
    if not selected.any():
        raise ValueError("mask must select at least one target position")
    if not torch.isfinite(target[selected]).all():
        raise ValueError("selected target positions must be finite")
    return (predicted[selected] - target[selected]).square().mean()
